# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``PrefixCacheSessions`` and ``/encode`` validation on CPU, with a stand-in engine pass.

The stand-in records every pass (tokens, offsets, salt, prefix reads) and answers a hit the way stock vLLM's
scheduler lands one: for a salt that already encoded a prefix of these tokens, the last hit-grid point below that
prefix's length.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from vllm_arc.server import app as server  # noqa: E402
from vllm_arc.engine.sessions import KvPool, Pass, PrefixCacheSessions, Refused  # noqa: E402

GRID = 16


class Engine:
    def __init__(self, grid: int = GRID) -> None:
        self.grid, self.calls, self.cached_by_salt, self.fail, self.force_hit = grid, [], {}, False, None

    async def run(self, tokens, offsets, salt, skip, images):
        self.calls.append({"tokens": list(tokens), "offsets": list(offsets), "salt": salt, "skip": skip,
                           "images": images})
        if self.fail:
            raise RuntimeError("engine pass failed")
        prior = self.cached_by_salt.get(salt)
        cached = 0
        if not skip and prior is not None and tokens[:len(prior)] == prior:
            cached = (len(prior) - 1) // self.grid * self.grid
        if self.force_hit is not None and not skip:
            cached = self.force_hit
        self.cached_by_salt[salt] = list(tokens)
        rows = torch.tensor([[float(tokens[o]) if o >= cached else float("nan")] for o in offsets])
        return Pass(rows.reshape(len(offsets), 1), cached, len(tokens))


def _sessions(engine, blocks=1000, max_sessions=8, idle=900.0, clock=None):
    pool = KvPool(blocks=blocks, block_size=GRID, seq_overhead_blocks=1, max_model_len=320)
    return PrefixCacheSessions(engine.run, pool, max_sessions, idle, 2, GRID, **({"clock": clock} if clock else {}))


def run(coro):
    return asyncio.run(coro)


def test_a_session_appends_the_whole_sequence_under_its_salt_with_reads_on():
    engine = Engine()
    s = _sessions(engine)
    first = run(s.encode("a", 0, list(range(100)), [99, 0]))
    second = run(s.encode("a", 100, list(range(100, 140)), [139, 100]))
    one, two = engine.calls
    assert one["skip"] is True and two["skip"] is False and one["salt"] == two["salt"]
    assert two["tokens"] == list(range(140))
    assert first.rows[:, 0].tolist() == [99.0, 0.0] and second.rows[:, 0].tolist() == [139.0, 100.0]
    assert second.cached == 96 and not second.over_hit
    assert s.counters["appends"] == 1 and s.counters["append_cached_tokens"] == 96 and s.counters["over_hits"] == 0


def test_out_of_step_unknown_and_rebuilt_sessions():
    """Wrong base or an unknown session is ``None`` (409); a rebuild from 0 takes a fresh salt."""
    engine = Engine()
    s = _sessions(engine)
    run(s.encode("a", 0, list(range(50)), [1]))
    assert run(s.encode("a", 49, [1], [49])) is None
    assert run(s.encode("nope", 3, [1], [3])) is None
    run(s.encode("a", 0, list(range(50)), [1]))
    assert engine.calls[0]["salt"] != engine.calls[1]["salt"]


def test_a_failed_pass_drops_the_session():
    engine = Engine()
    s = _sessions(engine)
    run(s.encode("a", 0, list(range(50)), [1]))
    engine.fail = True
    with pytest.raises(RuntimeError):
        run(s.encode("a", 50, [7, 8], [50]))
    assert "a" not in s.entries and s.counters["failed_passes"] == 1


def test_an_over_hit_is_redone_with_reads_off_under_a_fresh_salt_the_session_keeps():
    engine = Engine()
    s = _sessions(engine)
    run(s.encode("a", 0, list(range(64)), [63]))
    engine.force_hit = 70  # covers offset 64
    out = run(s.encode("a", 64, list(range(64, 80)), [64, 79]))
    assert out.over_hit and torch.isfinite(out.rows).all() and out.rows[:, 0].tolist() == [64.0, 79.0]
    first, retry = engine.calls[1:]
    assert first["skip"] is False and retry["skip"] is True and retry["salt"] != first["salt"]
    assert s.entries["a"].salt == retry["salt"] and s.counters["over_hits"] == 1 and s.counters["rebuilds"] == 1


def test_one_shots_take_a_fresh_salt_with_reads_off_and_store_nothing():
    engine = Engine()
    s = _sessions(engine)
    run(s.encode_once(list(range(30)), [29]))
    run(s.encode_once(list(range(30)), [29]))
    a, b = engine.calls
    assert a["skip"] and b["skip"] and a["salt"] != b["salt"] and not s.entries
    with pytest.raises(Refused, match="exceeds_kv_budget"):
        run(s.encode_once(list(range(321)), [0]))


def test_a_prefix_session_refuses_images_and_a_one_shot_passes_them_to_its_pass():
    # Each append re-sends the whole session: an earlier input's images would be lost, so image sessions are
    # retained-only. A one-shot's images go to its pass.
    engine = Engine()
    s = _sessions(engine)
    with pytest.raises(Refused, match="images_need_retained"):
        run(s.encode("a", 0, list(range(30)), [29], images=["img"]))
    assert not s.entries and not engine.calls
    run(s.encode_once(list(range(30)), [29], images=["img"]))
    assert engine.calls[0]["images"] == ["img"]


def test_admission_forgets_least_recently_used_idle_sessions_to_fit_the_budget_and_count():
    """Budget: 60 blocks - vLLM's null block - headroom (20 + 1) = 38; a 160-token session costs 11, so three fit."""
    engine = Engine()
    s = _sessions(engine, blocks=60, max_sessions=8)
    for key in "abcd":
        run(s.encode(key, 0, list(range(160)), [0]))
    assert list(s.entries) == ["b", "c", "d"] and s.counters["evicted_sessions"] == 1
    run(s.encode("b", 160, [1], [160]))  # b becomes most recent
    run(s.encode("e", 0, list(range(160)), [0]))
    assert list(s.entries) == ["d", "b", "e"]
    s2 = _sessions(Engine(), max_sessions=2)
    for key in "xyz":
        run(s2.encode(key, 0, [1, 2], [0]))
    assert list(s2.entries) == ["y", "z"]
    with pytest.raises(Refused, match="exceeds_kv_budget"):
        run(s.encode("big", 0, list(range(700)), [0]))
    assert "big" not in s.entries


def test_a_rebuild_of_a_busy_session_and_a_sequence_past_max_model_len_are_refused():
    async def scenario():
        engine = Engine()
        gate = asyncio.Event()
        inner = engine.run

        async def slow(*args):
            await gate.wait()
            return await inner(*args)

        s = _sessions(engine)
        await s.encode("a", 0, [1, 2], [0])
        engine_run, s.run = s.run, slow
        append = asyncio.create_task(s.encode("a", 2, [3], [2]))
        await asyncio.sleep(0)
        assert await s.encode("a", 0, [9], [0]) is None  # busy: 409, the append keeps its session
        gate.set()
        assert (await append).rows[:, 0].tolist() == [3.0]
        s.run = engine_run
        with pytest.raises(Refused, match="exceeds_kv_budget"):
            await s.encode("a", 3, list(range(318)), [3])
        assert len(s.entries["a"].tokens) == 3

    run(scenario())


def test_idle_sessions_close_and_owners_close_theirs():
    now = [0.0]
    engine = Engine()
    s = _sessions(engine, idle=10, clock=lambda: now[0])
    run(s.encode("a", 0, [1, 2], [0], owner="o1"))
    run(s.encode("b", 0, [1, 2], [0], owner="o2"))
    now[0] = 20
    run(s.encode("c", 0, [1, 2], [0], owner="o1"))
    assert list(s.entries) == ["c"] and s.counters["idle_closed_sessions"] == 2
    assert run(s.close_owner("o1")) == (1, 0) and not s.entries


@pytest.mark.parametrize("body, status, code", [
    ({"tokens": [1, 2], "offsets": [2], "base": 0}, 422, "offsets_out_of_range"),
    ({"tokens": [1, 2], "offsets": [0], "base": 3, "retain": False}, 422, "invalid_retain"),
    ({"tokens": [1, 2], "offsets": [0], "base": 0, "images": [{}]}, 422, "images_not_served"),
    ({"tokens": [], "offsets": [], "base": 0}, 422, "offsets_out_of_range"),
    ({"tokens": [1], "offsets": ["x"], "base": 0}, 422, "invalid_body"),
    ({"tokens": [1.0], "offsets": [0], "base": 0}, 422, "invalid_body"),
    ({"tokens": [True], "offsets": [0], "base": 0, "retain": False}, 422, "invalid_body"),
    ({"tokens": [1, 2], "offsets": [0], "base": 0}, 422, "invalid_body"),
    ({"session": "s", "tokens": [1, 2], "offsets": [3], "base": 2}, 409, "session_unknown"),
    ({"session": "s", "tokens": [1, 2], "offsets": [0], "base": 0, "owner": ""}, 422, "invalid_owner"),
])
def test_encode_request_keeps_todays_refusals(body, status, code):
    state = {"sessions": _sessions(Engine())}
    assert run(server.encode_request(state, body)) == (status, {"error": code})
