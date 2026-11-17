# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""A wedged engine call never holds every later retained encode: a close the engine never answers (an evicted,
replaced or released session) runs outside the session lock and is aborted after ``ADMISSION_WAIT_S``, and a wait
for the lock or a scratch slot past it is refused 503 ``admission_timeout`` (the fork server's bound)."""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from vllm_arc.engine import sessions as stock_sessions  # noqa: E402
from vllm_arc.engine.retained import RetainedSessions  # noqa: E402
from vllm_arc.engine.sessions import KvPool, Pass, Refused  # noqa: E402

BLOCK = 16
WAIT = 0.2


class Hung:
    """A session whose close the engine never answers (``close`` awaits forever) until cancelled."""

    def __init__(self, engine) -> None:
        self.engine, self.closed, self.cancelled = engine, False, False

    async def close(self) -> None:
        self.closed = True
        if not self.engine.hang_closes:
            return
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True  # PoolingSession.close: its output stream is closed, the request aborted
            raise


class Engine:
    def __init__(self) -> None:
        self.sessions: list[Hung] = []
        self.hang_closes = False
        self.gate: asyncio.Event | None = None

    def open(self) -> Hung:
        self.sessions.append(Hung(self))
        return self.sessions[-1]

    async def append(self, session, tokens, offsets, images) -> Pass:
        if self.gate is not None:
            await self.gate.wait()
        return Pass(torch.zeros(len(offsets), 1), 0, len(tokens))

    async def run(self, tokens, offsets, salt, skip, images) -> Pass:
        if self.gate is not None:
            await self.gate.wait()
        return Pass(torch.zeros(len(offsets), 1), 0, len(tokens))


@pytest.fixture(autouse=True)
def short_wait(monkeypatch):
    monkeypatch.setattr(stock_sessions, "ADMISSION_WAIT_S", WAIT)


def run(coro, limit: float = 5.0) -> None:
    """``asyncio.run(coro)`` on a daemon thread, failed if it has not finished in ``limit`` seconds: a wedged session
    lock (what these tests catch) must fail the test, not hang the suite."""
    error: list = []

    def target():
        try:
            asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the test's thread
            error.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(limit)
    if thread.is_alive():
        pytest.fail(f"wedged: still running after {limit} s")
    if error:
        raise error[0]


def _sessions(engine, max_sessions=2):
    pool = KvPool(blocks=1000, block_size=BLOCK, seq_overhead_blocks=1, max_model_len=320)
    return RetainedSessions(engine.open, engine.append, engine.run, pool, max_sessions, 900.0, 1, BLOCK)


def test_a_hung_eviction_close_holds_no_other_session():
    """At the count cap a new session evicts the oldest idle one; its close never answers. The lock is free
    meanwhile (a refusal or a 409 answers at once), the hung close is aborted at the bound, and every append then
    runs: appends wait for closes in flight (`_drain`), never beyond the bound."""
    async def go():
        engine = Engine()
        s = _sessions(engine)
        await s.encode("a", 0, [1, 2], [1])
        await s.encode("b", 0, [1, 2], [1])
        engine.hang_closes = True
        evicting = asyncio.ensure_future(s.encode("c", 0, [1, 2], [1]))  # evicts "a"; its close hangs
        await asyncio.sleep(0.02)
        assert "a" not in s.entries and engine.sessions[0].closed and not evicting.done()
        assert await asyncio.wait_for(s.encode("b", 9, [3], [9]), 0.05) is None  # the lock is free: a 409 at once
        other = asyncio.ensure_future(s.encode("b", 2, [3], [2]))
        result = await asyncio.wait_for(evicting, 1.0)  # the close is aborted at WAIT; then "c" runs
        assert result is not None and engine.sessions[0].cancelled
        assert await asyncio.wait_for(other, 1.0) is not None
        assert s.counters["close_timeouts"] == 1 and s.counters["evicted_sessions"] == 1
    run(go())


def test_a_hung_replacement_or_release_close_is_bounded_too():
    async def go():
        engine = Engine()
        s = _sessions(engine, max_sessions=4)
        await s.encode("a", 0, [1, 2], [1], owner="o")
        await s.encode("b", 0, [1, 2], [1], owner="p")
        engine.hang_closes = True
        rebuild = asyncio.ensure_future(s.encode("a", 0, [5], [0]))  # replaces "a": its old session's close hangs
        await asyncio.sleep(0.02)
        assert await asyncio.wait_for(s.encode("b", 9, [3], [9]), 0.05) is None  # the lock is free
        assert await asyncio.wait_for(rebuild, 1.0) is not None
        release = asyncio.ensure_future(s.close_owner("p"))  # an episode's release, its close hangs
        await asyncio.sleep(0.02)
        assert await asyncio.wait_for(s.encode("a", 9, [6], [9]), 0.05) is None  # the lock is free
        assert await asyncio.wait_for(s.encode("a", 1, [6], [1]), 1.0) is not None
        assert await asyncio.wait_for(release, 1.0) == (1, 0)
        assert s.counters["close_timeouts"] == 2
    run(go())


def test_a_cancelled_caller_still_closes_what_it_evicted():
    """The client of an evicting encode goes away mid-close: the eviction's close runs on, shielded, to its bound,
    and the cancelled encode's own new session is dropped and closed, never left busy."""
    async def go():
        engine = Engine()
        s = _sessions(engine)
        await s.encode("a", 0, [1], [0])
        await s.encode("b", 0, [1], [0])
        engine.hang_closes = True
        evicting = asyncio.ensure_future(s.encode("c", 0, [1], [0]))
        await asyncio.sleep(0.02)
        evicting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(evicting, 1.0)
        assert engine.sessions[0].cancelled and "c" not in s.entries and not s._closing
        assert s.counters["close_timeouts"] == 2  # "a"'s and the dropped "c"'s
        assert not any(e.busy for e in s.entries.values())
        assert await s.encode("c", 0, [1], [0]) is not None
    run(go())


def test_an_append_starts_only_after_every_detached_session_has_ended():
    """Admission counts only live entries, so an append must not start while a session another caller detached
    (here an owner release of two) still holds its engine slot and blocks."""
    async def go():
        engine = Engine()
        s = _sessions(engine, max_sessions=2)
        await s.encode("a", 0, [1], [0], owner="o")
        await s.encode("b", 0, [1], [0], owner="o")
        engine.hang_closes = True
        release = asyncio.ensure_future(s.close_owner("o"))
        await asyncio.sleep(0.02)
        started = []
        append = engine.append

        async def recording(session, tokens, offsets, images):
            started.append([x.cancelled for x in engine.sessions[:2]])
            return await append(session, tokens, offsets, images)

        s.append = recording
        assert await asyncio.wait_for(s.encode("c", 0, [1], [0]), 2.0) is not None
        assert started == [[True, True]]  # both releases had ended (aborted at the bound) before "c" ran
        await release
    run(go())


def test_sleep_quiesces_without_holding_the_lock():
    """``/sleep`` (`quiesce`): new retained encodes are refused 503 ``sleeping`` until `resume`, an append in flight
    finishes and its session is then closed too, and the closes (here hung, aborted at the bound) run with the lock
    free."""
    async def go():
        engine = Engine()
        s = _sessions(engine, max_sessions=4)
        await s.encode("a", 0, [1], [0])
        engine.gate = asyncio.Event()
        busy = asyncio.ensure_future(s.encode("b", 0, [1], [0]))  # in flight when sleep starts
        await asyncio.sleep(0.02)
        engine.hang_closes = True
        sleeping = asyncio.ensure_future(s.quiesce())
        await asyncio.sleep(0.02)
        with pytest.raises(Refused, match="sleeping"):
            await asyncio.wait_for(s.encode("c", 0, [1], [0]), 0.05)
        assert not sleeping.done()  # waits for "b"'s append
        engine.gate.set()
        assert await busy is not None
        await asyncio.sleep(0.05)
        assert not s.lock.locked() and not s.entries  # both detached; their closes hang outside the lock
        assert await asyncio.wait_for(sleeping, 1.0) == 2 and s.counters["close_timeouts"] == 2
        engine.hang_closes = False
        s.resume()
        assert await s.encode("c", 0, [1], [0]) is not None
    run(go())


def test_sleep_gives_up_on_a_busy_append_past_the_bound_and_resumes():
    async def go():
        engine = Engine()
        engine.gate = asyncio.Event()
        s = _sessions(engine)
        busy = asyncio.ensure_future(s.encode("a", 0, [1], [0]))
        await asyncio.sleep(0.02)
        with pytest.raises(Refused) as exc:
            await s.quiesce()
        assert exc.value.detail == {"wait": "sleep_busy_sessions"} and not s.paused
        engine.gate.set()
        assert await busy is not None
    run(go())


def test_a_wait_for_the_lock_past_the_bound_is_refused_503():
    async def go():
        engine = Engine()
        s = _sessions(engine)
        await s.lock.acquire()  # whatever holds it, held past the bound
        with pytest.raises(Refused) as exc:
            await s.encode("a", 0, [1], [0])
        assert (exc.value.status, exc.value.code, exc.value.detail) == (503, "admission_timeout",
                                                                         {"wait": "session_lock"})
        with pytest.raises(Refused, match="admission_timeout"):
            await s.close_owner("o")
        s.lock.release()
        assert await s.encode("a", 0, [1], [0]) is not None
    run(go())


def test_a_wait_for_a_scratch_slot_past_the_bound_is_refused_503():
    async def go():
        engine = Engine()
        engine.gate = asyncio.Event()
        s = _sessions(engine)  # one scratch slot
        first = asyncio.ensure_future(s.encode_once([1], [0]))
        await asyncio.sleep(0.02)
        with pytest.raises(Refused) as exc:
            await s.encode_once([1], [0])
        assert exc.value.detail == {"wait": "scratch_slot"}
        engine.gate.set()
        assert (await first) is not None
    run(go())


def test_a_sleep_cancelled_during_its_closes_resumes_the_route():
    """A ``/sleep`` cancelled while its sessions close (the engine never went to sleep) leaves retained encodes served,
    not refused ``sleeping`` until a ``/wake_up`` that may never come."""
    async def go():
        engine = Engine()
        s = _sessions(engine)
        await s.encode("a", 0, [1], [0])
        engine.hang_closes = True
        sleeping = asyncio.ensure_future(s.quiesce())
        await asyncio.sleep(0.05)  # "a" detached, its close hanging (within the bound)
        assert s.paused and not sleeping.done()
        sleeping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sleeping
        assert not s.paused
        engine.hang_closes = False
        assert await s.encode("b", 0, [1], [0]) is not None
    run(go())
