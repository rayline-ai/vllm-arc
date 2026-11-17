# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``RetainedSessions`` (patch 0001): one retained streaming pooling request per session, as the fork ran.

An append sends only its new input (tokens, and the images inside them) to the session's live request
(``PoolingSession``); the engine keeps the session's KV blocks and GDN state, so nothing is re-sent, re-hashed, looked
up, restored from a checkpoint or replayed. Same interface and ``/encode`` contract as ``PrefixCacheSessions``, which
stays selectable (``ARC_STOCK_SESSIONS=prefix``) and still serves every one-shot (``retain: false``) encode. Lengths
and offsets are session positions as the engine counts them (an image's placeholder run at its expanded length).

- ``base == 0``: a new session under ``key`` (an idle one there is closed first); its first append encodes the tokens.
- ``base == length``: an append. Otherwise, or busy, or unknown: ``None`` (the caller answers 409, the client
  rebuilds from 0).
- a failed or cancelled append closes the session: the engine's state of it is unknown. That includes a
  ``PoolingSessionError`` the injected ``append`` raises itself after the engine answered (the image layer's
  expansion check): the key then answers ``None`` until the client rebuilds it from 0.

Admission (today's server's model): a paused session holds its blocks and the scheduler never preempts it, so the
pool is split once into a one-shot headroom (one ``max_model_len`` sequence) and the retained budget. Before an
append runs, sessions idle too long are closed, then the least recently used idle ones until the retained sessions
fit the budget and the count; if only busy sessions are left in the way it refuses 503 ``kv_exhausted`` (nothing
would free their blocks) or, over the count, 503 ``sessions_exhausted`` (a paused session keeps its runner slot, and
the engine has ``max_sessions`` of them beside the one-shots' scratch slots: one more would wait for a slot that only
a close frees). No engine wait is made under the session lock: closes (evicted, replaced, released sessions) run
after it is released, each bounded by ``ADMISSION_WAIT_S`` (then aborted), and a wait for the lock past that bound is
refused 503 ``admission_timeout``. No vLLM import here: the session factory and the append pass are injected.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from . import sessions as _sessions
from .sessions import KvPool, Pass, PrefixCacheSessions, Refused, Result, Run, _finite, admitted

#: ``append(session, tokens, offsets, images) -> Pass``: one append through a live session. ``images`` is the
#: server's, for this input, passed through unread (``None``: a text append).
Append = Callable[[Any, list[int], list[int], Any], Awaitable[Pass]]


@dataclass
class Live:
    session: Any  # PoolingSession
    owner: str | None
    length: int = 0  # tokens the engine holds
    reserved: int = 0  # tokens it holds once the append in flight finishes
    used: float = 0.0
    busy: bool = False


class RetainedSessions:
    def __init__(self, open_session: Callable[[], Any], append: Append, run: Run, pool: KvPool, max_sessions: int,
                 idle_seconds: float, scratch_slots: int, hit_alignment: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """``open_session()`` starts a ``PoolingSession``; ``append`` runs one append; ``run`` is the one-shot pass
        (``PrefixCacheSessions``' own, which serves ``encode_once``)."""
        self.open_session, self.append, self.pool, self.max_sessions = open_session, append, pool, max_sessions
        self.idle_seconds, self.clock = idle_seconds, clock
        self.once = PrefixCacheSessions(run, pool, max_sessions, idle_seconds, scratch_slots, hit_alignment, clock)
        self.headroom, self.budget = self.once.headroom, self.once.budget
        self.entries: OrderedDict[str, Live] = OrderedDict()
        self.lock = asyncio.Lock()
        self._closing: set = set()  # close tasks in flight (`_finish`)
        self.paused = False  # `quiesce` (``/sleep``) to `resume` (``/wake_up``): retained encodes refused 503
        # The prefix route's keys stay (/health reads them; hits and misses stay 0 here) beside this route's own.
        self.counters = self.once.counters
        self.counters.update(dict.fromkeys(("closed_sessions", "close_timeouts", "kv_exhausted", "sessions_exhausted"), 0))

    def resident(self) -> int:
        return sum(self.pool.cost(e.reserved) for e in self.entries.values())

    def usage(self) -> dict:
        return {"blocks": self.pool.blocks, "block_size": self.pool.block_size,
                "seq_overhead_blocks": self.pool.seq_overhead_blocks, "retained_budget_blocks": self.budget,
                "scratch_headroom_blocks": self.headroom, "resident_blocks": self.resident(),
                "resident_tokens": sum(e.length for e in self.entries.values()), "sessions": len(self.entries),
                "max_sessions": self.max_sessions}

    def _detach(self, key: str, closing: list) -> None:
        """Forget ``key`` (lock held) and queue its session on ``closing``, which the caller closes after releasing
        the lock (`_finish`): a close waits for the engine, and no engine wait is ever made under the lock."""
        entry = self.entries.pop(key, None)
        if entry is not None:
            self.counters["closed_sessions"] += 1
            closing.append(entry.session)

    async def _finish(self, closing: list) -> None:
        """Close the detached sessions (lock not held), each within ``ADMISSION_WAIT_S``: a close the engine does not
        answer in time is cancelled, which aborts the request (``PoolingSession.close`` closes its output stream).
        A failing or timed-out close is counted, never raised: the entry is gone either way. The closes run in a task
        of their own, shielded, so a caller cancelled meanwhile (a client gone) never leaves a session open."""
        if closing:
            sessions = list(closing)
            closing.clear()
            task = asyncio.ensure_future(self._close_each(sessions))
            self._closing.add(task)
            task.add_done_callback(self._closing.discard)
        await self._drain()

    async def _drain(self) -> None:
        """Wait for every close in flight, whoever detached it (each is bounded, so this is). An append runs only
        after the sessions detached before it have ended: admission counts only ``entries``, so the engine must not
        still hold a detached session's slot and blocks when an append it admitted against starts."""
        if self._closing:
            await asyncio.shield(asyncio.gather(*list(self._closing), return_exceptions=True))

    async def _close_each(self, sessions: list) -> None:
        for session in sessions:
            try:
                await asyncio.wait_for(session.close(), _sessions.ADMISSION_WAIT_S)
            except TimeoutError:
                self.counters["close_timeouts"] += 1
            except Exception:  # noqa: BLE001 -- the request is finished or aborted either way
                self.counters["failed_passes"] += 1

    async def close_all(self) -> int:
        """Close every idle session (``/sleep``): detached under the lock, closed after it (`_finish`, bounded). A
        busy one is left to its append, which closes it on failure or keeps it: the engine is put to sleep only
        between requests."""
        closing: list = []
        async with admitted(self.lock, "session_lock"):
            keys = [k for k, e in self.entries.items() if not e.busy]
            for key in keys:
                self._detach(key, closing)
        await self._finish(closing)
        return len(keys)

    async def quiesce(self) -> int:
        """``/sleep``: refuse new retained encodes (503 ``sleeping``), let the appends in flight finish, then close
        every session; ``ADMISSION_WAIT_S`` bounds the wait for busy appends (then 503 ``admission_timeout`` and the
        route resumes). The lock is held only to change state, never across an engine wait. Returns the sessions
        closed."""
        closing: list = []
        async with admitted(self.lock, "session_lock"):
            self.paused = True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _sessions.ADMISSION_WAIT_S
        try:
            while True:
                async with self.lock:  # bookkeeping only
                    if not any(e.busy for e in self.entries.values()):
                        n = len(self.entries)
                        for key in list(self.entries):
                            self._detach(key, closing)
                        break
                if loop.time() > deadline:
                    raise Refused(503, "admission_timeout", wait="sleep_busy_sessions")
                await asyncio.sleep(0.01)
            await self._finish(closing)
        except BaseException:
            self.paused = False
            raise
        return n

    def resume(self) -> None:
        """``/wake_up``: retained encodes are served again."""
        self.paused = False

    def _admit(self, entry: Live, closing: list) -> None:
        """Make room for ``entry`` (busy, lock held), or refuse 503; the sessions it evicts go on ``closing``."""
        now = self.clock()
        for key in [k for k, e in self.entries.items() if not e.busy and now - e.used > self.idle_seconds]:
            self._detach(key, closing)
            self.counters["idle_closed_sessions"] += 1
        while len(self.entries) > self.max_sessions or self.resident() > self.budget:
            victim = next((k for k, e in self.entries.items() if not e.busy), None)
            if victim is None:
                if self.resident() > self.budget:
                    self.counters["kv_exhausted"] += 1
                    raise Refused(503, "kv_exhausted", resident_blocks=self.resident(), budget_blocks=self.budget)
                self.counters["sessions_exhausted"] += 1
                raise Refused(503, "sessions_exhausted", sessions=len(self.entries), max_sessions=self.max_sessions)
            self._detach(victim, closing)
            self.counters["evicted_sessions"] += 1

    async def encode_once(self, tokens: list[int], offsets: list[int], images: Any = None) -> Result:
        return await self.once.encode_once(tokens, offsets, images)

    async def encode(self, key: str, base: int, tokens: list[int], offsets: list[int],
                     owner: str | None = None, images: Any = None) -> Result | None:
        closing: list = []
        admitted_entry = None
        try:
            async with admitted(self.lock, "session_lock"):
                if self.paused:
                    raise Refused(503, "sleeping")
                if base == 0:
                    if key in self.entries and self.entries[key].busy:
                        return None  # mid-append: 409, as for any busy session
                    self._detach(key, closing)
                    self.entries[key] = Live(session=self.open_session(), owner=owner, used=self.clock())
                entry = self.entries.get(key)
                if entry is None or entry.length != base or entry.busy:
                    return None
                total = base + len(tokens)
                if total > self.pool.max_model_len or self.pool.cost(total) > self.budget:
                    if base == 0:
                        self._detach(key, closing)
                    raise Refused(422, "exceeds_kv_budget", tokens=total,
                                  budget_tokens=self.budget * self.pool.block_size)
                entry.busy, entry.reserved = True, total
                self.entries.move_to_end(key)
                try:
                    self._admit(entry, closing)
                except BaseException:
                    entry.busy, entry.reserved = False, entry.length
                    if base == 0:
                        self._detach(key, closing)
                    raise
                admitted_entry = entry
        finally:
            if admitted_entry is None and closing:  # refused: its detached sessions still end, outside the lock
                await self._finish(closing)
        try:
            # Evicted and replaced sessions end before this append starts (their slots and blocks freed first),
            # outside the lock; a cancellation here is this append's failure: the entry is dropped, never left busy.
            await self._finish(closing)
            out = await self.append(entry.session, list(tokens), list(offsets), images)
        except BaseException:
            async with self.lock:  # bookkeeping only: no engine wait under it
                self.counters["failed_passes"] += 1
                if self.entries.get(key) is entry:
                    self._detach(key, closing)
            await self._finish(closing)
            raise
        self.counters["encodes"] += 1
        if base:
            self.counters["appends"] += 1
            self.counters["append_base_tokens"] += base
            self.counters["append_cached_tokens"] += base
        if not _finite(out.rows):
            self.counters["nonfinite_rows"] += 1
        async with self.lock:  # bookkeeping only, never refused: giving up here would leave the entry busy
            entry.length, entry.reserved, entry.used, entry.busy = total, total, self.clock(), False
        # Nothing before ``base`` was re-sent: the engine computed only this append's tokens.
        return Result(out.rows, base, len(tokens))

    async def close_owner(self, owner: str) -> tuple[int, int]:
        """Close every idle session of ``owner``; ``(closed, busy left)``."""
        closing: list = []
        async with admitted(self.lock, "session_lock"):
            owned = [(k, e.busy) for k, e in self.entries.items() if e.owner == owner]
            for key, busy in owned:
                if not busy:
                    self._detach(key, closing)
        await self._finish(closing)
        return sum(not b for _, b in owned), sum(b for _, b in owned)
