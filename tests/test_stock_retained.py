# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``RetainedSessions`` (patch 0001's session route) on CPU, with stand-in live sessions: one live request per
session, appends that carry only new tokens, 409s, failure handling, and KV admission over retained blocks."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from vllm_arc.server import app as server  # noqa: E402
from vllm_arc.engine.pooling_session import PoolingSessionError  # noqa: E402
from vllm_arc.engine.retained import RetainedSessions  # noqa: E402
from vllm_arc.engine.sessions import KvPool, Pass, Refused  # noqa: E402

BLOCK = 16


class Live:
    def __init__(self, n: int) -> None:
        self.n, self.tokens, self.appends, self.closed = n, [], [], False

    async def close(self) -> None:
        self.closed = True


class Engine:
    def __init__(self) -> None:
        self.sessions: list[Live] = []
        self.one_shots: list[tuple] = []
        self.images: list = []
        self.fail = False
        self.gate: asyncio.Event | None = None

    def open(self) -> Live:
        self.sessions.append(Live(len(self.sessions)))
        return self.sessions[-1]

    async def append(self, session: Live, tokens, offsets, images) -> Pass:
        session.appends.append(list(tokens))
        self.images.append(images)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("append failed")
        session.tokens += tokens
        rows = torch.tensor([[float(session.tokens[o])] for o in offsets]).reshape(len(offsets), 1)
        return Pass(rows, 0, len(tokens))

    async def run(self, tokens, offsets, salt, skip, images) -> Pass:
        self.one_shots.append((len(tokens), salt, skip, images))
        return Pass(torch.tensor([[float(tokens[o])] for o in offsets]).reshape(len(offsets), 1), 0, len(tokens))


def _sessions(engine, blocks=1000, max_sessions=8, idle=900.0, clock=None):
    pool = KvPool(blocks=blocks, block_size=BLOCK, seq_overhead_blocks=1, max_model_len=320)
    return RetainedSessions(engine.open, engine.append, engine.run, pool, max_sessions, idle, 2, BLOCK,
                            **({"clock": clock} if clock else {}))


def run(coro):
    return asyncio.run(coro)


def test_a_session_keeps_one_live_request_and_an_append_sends_only_its_new_tokens():
    engine = Engine()
    s = _sessions(engine)
    first = run(s.encode("a", 0, list(range(100)), [99, 0]))
    second = run(s.encode("a", 100, [7, 8], [101, 100]))
    [live] = engine.sessions
    assert live.appends == [list(range(100)), [7, 8]] and not live.closed
    assert first.rows[:, 0].tolist() == [99.0, 0.0] and second.rows[:, 0].tolist() == [8.0, 7.0]
    assert (first.cached, first.encoded) == (0, 100) and (second.cached, second.encoded) == (100, 2)
    assert s.counters["appends"] == 1 and s.counters["append_cached_tokens"] == 100
    assert s.usage()["resident_tokens"] == 102 and s.usage()["resident_blocks"] == -(-102 // BLOCK) + 1


def test_out_of_step_unknown_or_busy_sessions_answer_none():
    async def go():
        engine = Engine()
        s = _sessions(engine)
        await s.encode("a", 0, [1, 2, 3], [2])
        assert await s.encode("a", 2, [4], [2]) is None
        assert await s.encode("b", 3, [4], [3]) is None
        engine.gate = asyncio.Event()
        busy = asyncio.create_task(s.encode("a", 3, [4], [3]))
        await asyncio.sleep(0.01)
        assert await s.encode("a", 3, [5], [3]) is None and await s.encode("a", 0, [5], [0]) is None
        engine.gate.set()
        assert (await busy).rows.tolist() == [[4.0]]

    run(go())


def test_base_zero_closes_the_old_live_request_and_opens_a_new_one():
    engine = Engine()
    s = _sessions(engine)
    run(s.encode("a", 0, [1, 2], [1]))
    run(s.encode("a", 0, [3], [0]))
    old, new = engine.sessions
    assert old.closed and not new.closed and new.appends == [[3]] and s.counters["closed_sessions"] == 1


class ExpansionError(PoolingSessionError):
    """As the image layer raises from ``append`` when the engine expanded an image other than declared."""


@pytest.mark.parametrize("fault", ["error", "cancel", "expansion"])
def test_a_failed_or_cancelled_append_closes_and_forgets_the_session(fault):
    async def go():
        engine = Engine()
        s = _sessions(engine)
        await s.encode("a", 0, [1, 2], [1])
        if fault == "error":
            engine.fail = True
            with pytest.raises(RuntimeError):
                await s.encode("a", 2, [3], [2])
        elif fault == "expansion":
            append = engine.append

            async def refuse(session, tokens, offsets, images):
                await append(session, tokens, offsets, images)  # the engine answered; the check then fails
                raise ExpansionError("image run not expanded as declared")

            s.append = refuse
            with pytest.raises(ExpansionError):
                await s.encode("a", 2, [3], [2], images=["img"])
            s.append = append
        else:
            engine.gate = asyncio.Event()
            task = asyncio.create_task(s.encode("a", 2, [3], [2]))
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        [live] = engine.sessions
        assert live.closed and "a" not in s.entries and s.counters["failed_passes"] == 1
        engine.fail, engine.gate = False, None
        assert await s.encode("a", 2, [3], [2]) is None  # the client rebuilds from 0

    run(go())


def test_admission_closes_idle_sessions_lru_first_to_fit_retained_blocks():
    engine = Engine()
    # 40 blocks: vLLM's null block and the headroom for one 320-token sequence (21 blocks) leave 18 retained.
    s = _sessions(engine, blocks=40)
    assert s.budget == 18
    run(s.encode("a", 0, list(range(100)), [0]))  # 8 blocks
    run(s.encode("b", 0, list(range(100)), [0]))  # 8 blocks
    run(s.encode("a", 100, [1], [100]))  # a is now most recently used
    run(s.encode("c", 0, list(range(40)), [0]))  # 4 blocks: b (least recently used) is closed
    a, b, c = engine.sessions
    assert b.closed and not a.closed and not c.closed and list(s.entries) == ["a", "c"]
    assert s.counters["evicted_sessions"] == 1
    with pytest.raises(Refused) as refused:
        run(s.encode("d", 0, list(range(320)), [0]))  # 21 blocks > the retained budget
    assert refused.value.status == 422 and refused.value.code == "exceeds_kv_budget" and "d" not in s.entries


def test_only_busy_sessions_in_the_way_refuses_503_and_leaves_no_entry():
    async def go():
        engine = Engine()
        s = _sessions(engine, blocks=40)
        engine.gate = asyncio.Event()
        busy = asyncio.create_task(s.encode("a", 0, list(range(200)), [0]))  # 14 blocks, in flight
        await asyncio.sleep(0.01)
        with pytest.raises(Refused) as refused:
            await s.encode("b", 0, list(range(100)), [0])  # 8 more blocks: over 18, only a busy one to close
        assert refused.value.status == 503 and refused.value.code == "kv_exhausted" and "b" not in s.entries
        assert engine.sessions[1].closed and not engine.sessions[0].closed
        engine.gate.set()
        await busy

    run(go())


def test_one_session_over_the_count_with_only_busy_ones_in_the_way_refuses_503():
    # A paused session keeps its runner slot: admitting one past max_sessions would leave it waiting for a slot
    # that only a close frees (and take a one-shot's scratch slot meanwhile).
    async def go():
        engine = Engine()
        s = _sessions(engine, max_sessions=1)
        engine.gate = asyncio.Event()
        busy = asyncio.create_task(s.encode("a", 0, [1, 2], [0]))
        await asyncio.sleep(0.01)
        with pytest.raises(Refused) as refused:
            await s.encode("b", 0, [3], [0])
        assert refused.value.status == 503 and refused.value.code == "sessions_exhausted" and "b" not in s.entries
        engine.gate.set()
        await busy
        engine.gate = None
        await s.encode("b", 0, [3], [0])  # a is idle now: it is closed to make room
        assert engine.sessions[0].closed and list(s.entries) == ["b"] and s.counters["sessions_exhausted"] == 1

    run(go())


def test_idle_sessions_close_after_the_idle_time():
    now = [0.0]
    engine = Engine()
    s = _sessions(engine, idle=10.0, clock=lambda: now[0])
    run(s.encode("a", 0, [1], [0]))
    now[0] = 11.0
    run(s.encode("b", 0, [1], [0]))
    assert engine.sessions[0].closed and list(s.entries) == ["b"] and s.counters["idle_closed_sessions"] == 1


def test_close_owner_and_close_all_close_the_live_requests():
    engine = Engine()
    s = _sessions(engine)
    run(s.encode("a", 0, [1], [0], owner="o"))
    run(s.encode("b", 0, [1], [0], owner="p"))
    assert run(s.close_owner("o")) == (1, 0) and engine.sessions[0].closed and list(s.entries) == ["b"]
    assert run(s.close_all()) == 1 and engine.sessions[1].closed and not s.entries


def test_one_shots_stay_plain_encodes_with_reads_off():
    engine = Engine()
    s = _sessions(engine)
    out = run(s.encode_once([5, 6, 7], [2]))
    assert out.rows.tolist() == [[7.0]] and engine.one_shots[0][2] is True and engine.sessions == []


def test_each_inputs_images_reach_its_own_pass_unread():
    # The server's image layer builds the engine prompt; sessions only carry an input's images to its pass.
    engine = Engine()
    s = _sessions(engine)
    run(s.encode("a", 0, [1, 2, 3], [2], images="first"))
    run(s.encode("a", 3, [4], [3]))
    run(s.encode("a", 4, [5, 6], [5], images="third"))
    run(s.encode_once([5, 6, 7], [2], images="once"))
    assert engine.images == ["first", None, "third"] and engine.one_shots[0][3] == "once"
    assert engine.sessions[0].appends == [[1, 2, 3], [4], [5, 6]]


def test_the_server_route_serves_retained_sessions_with_the_same_contract():
    engine = Engine()
    state = {"sessions": _sessions(engine)}
    assert run(server.encode_request(state, {"session": "s", "base": 0, "tokens": [1, 2], "offsets": [1]}))[0] == 200
    status, result = run(server.encode_request(state, {"session": "s", "base": 2, "tokens": [3], "offsets": [2]}))
    assert status == 200 and result.rows.tolist() == [[3.0]] and engine.sessions[0].appends == [[1, 2], [3]]
    assert run(server.encode_request(state, {"session": "s", "base": 1, "tokens": [3], "offsets": [1]})) == (
        409, {"error": "session_unknown"})
