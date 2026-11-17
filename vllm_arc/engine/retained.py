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
a close frees). No vLLM import here: the session factory and the append pass are injected.
"""
from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .sessions import KvPool, Pass, PrefixCacheSessions, Refused, Result, Run, _finite

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
        # The prefix route's keys stay (/health reads them; hits and misses stay 0 here) beside this route's own.
        self.counters = self.once.counters
        self.counters.update(dict.fromkeys(("closed_sessions", "kv_exhausted", "sessions_exhausted"), 0))

    def resident(self) -> int:
        return sum(self.pool.cost(e.reserved) for e in self.entries.values())

    def usage(self) -> dict:
        return {"blocks": self.pool.blocks, "block_size": self.pool.block_size,
                "seq_overhead_blocks": self.pool.seq_overhead_blocks, "retained_budget_blocks": self.budget,
                "scratch_headroom_blocks": self.headroom, "resident_blocks": self.resident(),
                "resident_tokens": sum(e.length for e in self.entries.values()), "sessions": len(self.entries),
                "max_sessions": self.max_sessions}

    async def _close(self, key: str) -> None:
        """Close and forget ``key`` (lock held). A failing close is counted, never raised: the entry is gone."""
        entry = self.entries.pop(key, None)
        if entry is None:
            return
        self.counters["closed_sessions"] += 1
        try:
            await entry.session.close()
        except Exception:  # noqa: BLE001 -- the request is finished or aborted either way
            self.counters["failed_passes"] += 1

    async def close_all(self) -> int:
        """Close every session (``/reset_prefix_cache``, ``/sleep``; lock held by the caller)."""
        keys = list(self.entries)
        for key in keys:
            await self._close(key)
        return len(keys)

    async def _admit(self, entry: Live) -> None:
        """Make room for ``entry`` (busy, lock held), or refuse 503."""
        now = self.clock()
        for key in [k for k, e in self.entries.items() if not e.busy and now - e.used > self.idle_seconds]:
            await self._close(key)
            self.counters["idle_closed_sessions"] += 1
        while len(self.entries) > self.max_sessions or self.resident() > self.budget:
            victim = next((k for k, e in self.entries.items() if not e.busy), None)
            if victim is None:
                if self.resident() > self.budget:
                    self.counters["kv_exhausted"] += 1
                    raise Refused(503, "kv_exhausted", resident_blocks=self.resident(), budget_blocks=self.budget)
                self.counters["sessions_exhausted"] += 1
                raise Refused(503, "sessions_exhausted", sessions=len(self.entries), max_sessions=self.max_sessions)
            await self._close(victim)
            self.counters["evicted_sessions"] += 1

    async def encode_once(self, tokens: list[int], offsets: list[int], images: Any = None) -> Result:
        return await self.once.encode_once(tokens, offsets, images)

    async def encode(self, key: str, base: int, tokens: list[int], offsets: list[int],
                     owner: str | None = None, images: Any = None) -> Result | None:
        async with self.lock:
            if base == 0:
                if key in self.entries and self.entries[key].busy:
                    return None  # mid-append: 409, as for any busy session
                await self._close(key)
                self.entries[key] = Live(session=self.open_session(), owner=owner, used=self.clock())
            entry = self.entries.get(key)
            if entry is None or entry.length != base or entry.busy:
                return None
            total = base + len(tokens)
            if total > self.pool.max_model_len or self.pool.cost(total) > self.budget:
                if base == 0:
                    await self._close(key)
                raise Refused(422, "exceeds_kv_budget", tokens=total, budget_tokens=self.budget * self.pool.block_size)
            entry.busy, entry.reserved = True, total
            self.entries.move_to_end(key)
            try:
                await self._admit(entry)
            except BaseException:
                entry.busy, entry.reserved = False, entry.length
                if base == 0:
                    await self._close(key)
                raise
        try:
            out = await self.append(entry.session, list(tokens), list(offsets), images)
        except BaseException:
            async with self.lock:
                self.counters["failed_passes"] += 1
                if self.entries.get(key) is entry:
                    await self._close(key)
            raise
        self.counters["encodes"] += 1
        if base:
            self.counters["appends"] += 1
            self.counters["append_base_tokens"] += base
            self.counters["append_cached_tokens"] += base
        if not _finite(out.rows):
            self.counters["nonfinite_rows"] += 1
        async with self.lock:
            entry.length, entry.reserved, entry.used, entry.busy = total, total, self.clock(), False
        # Nothing before ``base`` was re-sent: the engine computed only this append's tokens.
        return Result(out.rows, base, len(tokens))

    async def close_owner(self, owner: str) -> tuple[int, int]:
        """Close every idle session of ``owner``; ``(closed, busy left)``."""
        async with self.lock:
            owned = [(k, e.busy) for k, e in self.entries.items() if e.owner == owner]
            for key, busy in owned:
                if not busy:
                    await self._close(key)
        return sum(not b for _, b in owned), sum(b for _, b in owned)
