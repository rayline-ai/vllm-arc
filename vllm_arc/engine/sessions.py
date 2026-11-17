# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``PrefixCacheSessions``: retained sessions as cache salts over stock prefix caching.

The fork kept a paused pooling request per session. Stock vLLM keeps nothing per session: the server holds each
session's full token sequence and a fresh cache salt, and every append re-sends the whole sequence under that salt
with prefix-cache reads on, so the engine recomputes only past its last cached GDN checkpoint.

- ``base == 0`` (retained): a new entry with a new salt; the tokens encode with prefix reads off.
- ``base == length``: ``stored + new`` under the entry's salt, reads on (an append).
- otherwise, or busy, or unknown: ``None`` (the caller answers 409, the client rebuilds from 0 under a new salt).
- ``retain: false``: a fresh salt, reads off, nothing stored.
- a failed or cancelled pass drops the entry: a retry under its salt could hit blocks cached past the new offsets.

Over-hit: a hit that covers a requested offset (``num_cached_tokens > min(offsets)``) or a non-finite row (the
pooler's mark for a covered offset). An append cannot over-hit while each prefix is encoded at most once per salt;
the check stays, and an over-hit pass is redone with reads off under a fresh salt that the entry then keeps.

Admission is the server's model of residency only: nothing a paused session holds pins engine blocks any more
(freed blocks age out of the engine's LRU), so it bounds the working set the server expects to hit, not memory.
Idle entries are forgotten least recently used first until the retained sessions fit the budget and the count.
No vLLM import here: the engine pass is injected, so this runs on CPU in tests.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


class Refused(Exception):
    """An encode the server will not run: ``status`` and ``code`` are its HTTP refusal."""

    def __init__(self, status: int, code: str, **detail) -> None:
        super().__init__(code)
        self.status, self.code, self.detail = status, code, detail


@dataclass
class Pass:
    """One engine pass: rows ``[len(offsets), dim]`` (a CPU float tensor), the engine's cached-token count, and the
    tokens sent. ``processed`` is the engine's processed prompt ids (for a retained append, the whole session's, not
    copied) and ``start`` where this input begins in them, so ``processed[start:start + encoded]`` is this input as
    the engine expanded it (the image layer checks its declared runs there)."""

    rows: Any
    cached: int
    encoded: int
    processed: Any = None
    start: int = 0


@dataclass
class Result:
    rows: Any
    cached: int
    encoded: int
    over_hit: bool = False


class KvPool:
    """The engine's KV cache in blocks, and how many of them one sequence of ``n`` tokens holds (today's model)."""

    def __init__(self, blocks: int, block_size: int, seq_overhead_blocks: int, max_model_len: int) -> None:
        self.blocks, self.block_size = blocks, block_size
        self.seq_overhead_blocks, self.max_model_len = seq_overhead_blocks, max_model_len

    @classmethod
    def from_engine(cls, engine) -> "KvPool":
        cache, max_len = engine.vllm_config.cache_config, engine.vllm_config.model_config.max_model_len
        blocks, size, tokens = cache.num_gpu_blocks, cache.block_size, getattr(cache, "kv_cache_size_tokens", None)
        if not (blocks and size and tokens):
            raise RuntimeError(f"the engine reports no KV cache capacity (blocks {blocks}, block size {size}, "
                               f"tokens {tokens})")
        per_sequence = round(blocks * max_len / tokens)
        return cls(blocks, size, max(0, per_sequence - -(-max_len // size)), max_len)

    def cost(self, tokens: int) -> int:
        return -(-tokens // self.block_size) + self.seq_overhead_blocks if tokens else 0


@dataclass
class Entry:
    salt: str
    owner: str | None
    tokens: list[int] = field(default_factory=list)
    reserved: int = 0  # tokens it holds once the pass in flight finishes
    used: float = 0.0
    busy: bool = False


def new_salt() -> str:
    return uuid.uuid4().hex


def _finite(rows) -> bool:
    import torch

    return bool(torch.isfinite(rows).all()) if rows.numel() else True


#: ``run(tokens, offsets, salt, skip_reading_prefix_cache, images) -> Pass``: one engine pass over ``tokens``.
#: ``images`` is the server's, passed through unread (``None``: a text pass).
Run = Callable[[list[int], list[int], str, bool, Any], Awaitable[Pass]]


class PrefixCacheSessions:
    def __init__(self, run: Run, pool: KvPool, max_sessions: int, idle_seconds: float, scratch_slots: int,
                 hit_alignment: int, clock: Callable[[], float] = time.monotonic) -> None:
        """``run(tokens, offsets, salt, skip_reading_prefix_cache, images) -> Pass`` is one engine pass;
        ``hit_alignment`` is the engine's hit grid, for the short-hit counter."""
        self.run, self.pool, self.max_sessions, self.idle_seconds = run, pool, max_sessions, idle_seconds
        self.hit_alignment, self.clock = hit_alignment, clock
        self.headroom = pool.cost(pool.max_model_len)
        # vLLM reserves one block of the pool for good (the block pool's null block): it is never allocatable.
        self.budget = pool.blocks - 1 - self.headroom
        self.entries: OrderedDict[str, Entry] = OrderedDict()
        self.lock = asyncio.Lock()
        self.scratch = asyncio.Semaphore(scratch_slots)
        self.counters = dict.fromkeys(("encodes", "one_shots", "appends", "append_base_tokens", "append_cached_tokens",
                                       "zero_hit_appends", "short_hit_appends", "over_hits", "nonfinite_rows",
                                       "rebuilds", "evicted_sessions", "idle_closed_sessions", "failed_passes"), 0)

    def resident(self) -> int:
        return sum(self.pool.cost(e.reserved) for e in self.entries.values())

    def usage(self) -> dict:
        return {"blocks": self.pool.blocks, "block_size": self.pool.block_size,
                "seq_overhead_blocks": self.pool.seq_overhead_blocks, "retained_budget_blocks": self.budget,
                "resident_blocks": self.resident(), "resident_tokens": sum(len(e.tokens) for e in self.entries.values()),
                "sessions": len(self.entries), "max_sessions": self.max_sessions}

    def forget(self, key: str) -> None:
        self.entries.pop(key, None)

    def forget_all(self) -> int:
        n = len(self.entries)
        self.entries.clear()
        return n

    def _admit(self) -> None:
        """Forget idle entries (lock held): those idle too long, then least recently used first until the retained
        sessions fit the budget and the count. Busy entries are never forgotten; with only busy ones left, admit."""
        now = self.clock()
        for key in [k for k, e in self.entries.items() if not e.busy and now - e.used > self.idle_seconds]:
            self.forget(key)
            self.counters["idle_closed_sessions"] += 1
        while len(self.entries) > self.max_sessions or self.resident() > self.budget:
            victim = next((k for k, e in self.entries.items() if not e.busy), None)
            if victim is None:
                break
            self.forget(victim)
            self.counters["evicted_sessions"] += 1

    async def _checked(self, tokens: list[int], offsets: list[int], salt: str, skip: bool) -> tuple[Result, str]:
        """One pass, redone with reads off under a fresh salt if it over-hit; ``(result, salt it ran under)``."""
        first = await self.run(tokens, offsets, salt, skip, None)
        covered = bool(offsets) and first.cached > min(offsets)
        finite = _finite(first.rows)
        if not finite:
            self.counters["nonfinite_rows"] += 1
        if not covered and finite:
            return Result(first.rows, first.cached, first.encoded), salt
        self.counters["over_hits"] += 1
        self.counters["rebuilds"] += 1
        salt = new_salt()
        redo = await self.run(tokens, offsets, salt, True, None)
        return Result(redo.rows, redo.cached, redo.encoded, over_hit=True), salt

    async def close_all(self) -> int:
        """``RetainedSessions``' interface: nothing here holds engine state, so closing is forgetting."""
        return self.forget_all()

    async def encode_once(self, tokens: list[int], offsets: list[int], images: Any = None) -> Result:
        """A one-shot (``retain: false``): a fresh salt, prefix reads off, nothing stored. ``images`` (the
        server's, for this input) go to ``run`` unread."""
        if len(tokens) > self.pool.max_model_len:
            raise Refused(422, "exceeds_kv_budget", tokens=len(tokens), budget_tokens=self.pool.max_model_len)
        async with self.scratch:
            self.counters["encodes"] += 1
            self.counters["one_shots"] += 1
            out = await self.run(tokens, offsets, new_salt(), True, images)
            if not _finite(out.rows):
                self.counters["nonfinite_rows"] += 1
            return Result(out.rows, out.cached, out.encoded)

    async def encode(self, key: str, base: int, tokens: list[int], offsets: list[int],
                     owner: str | None = None, images: Any = None) -> Result | None:
        if images:
            # Every append re-sends the whole session, so an earlier input's images would have to be re-sent and
            # re-processed each time: image sessions are served by ``RetainedSessions`` only.
            raise Refused(422, "images_need_retained")
        async with self.lock:
            if base == 0:
                if key in self.entries and self.entries[key].busy:
                    return None  # mid-append: 409, as for any busy session
                self.forget(key)
                self.entries[key] = Entry(salt=new_salt(), owner=owner, used=self.clock())
            entry = self.entries.get(key)
            if entry is None or len(entry.tokens) != base or entry.busy:
                return None
            total = base + len(tokens)
            if total > self.pool.max_model_len or self.pool.cost(total) > self.budget:
                if base == 0:
                    self.forget(key)
                raise Refused(422, "exceeds_kv_budget", tokens=total, budget_tokens=self.budget * self.pool.block_size)
            entry.busy, entry.reserved = True, total
            self.entries.move_to_end(key)
            self._admit()
        full = entry.tokens + list(tokens)
        try:
            self.counters["encodes"] += 1
            if base == 0:
                result, salt = await self._checked(full, offsets, entry.salt, True)
            else:
                result, salt = await self._checked(full, offsets, entry.salt, False)
                self.counters["appends"] += 1
                self.counters["append_base_tokens"] += base
                if not result.over_hit:
                    self.counters["append_cached_tokens"] += result.cached
                    if result.cached == 0:
                        self.counters["zero_hit_appends"] += 1
                    elif result.cached < base - self.hit_alignment:
                        self.counters["short_hit_appends"] += 1
        except BaseException:
            async with self.lock:
                self.counters["failed_passes"] += 1
                if self.entries.get(key) is entry:
                    self.forget(key)
            raise
        async with self.lock:
            entry.tokens, entry.salt, entry.reserved = full, salt, total
            entry.used, entry.busy = self.clock(), False
        return result

    async def close_owner(self, owner: str) -> tuple[int, int]:
        """Forget every idle entry of ``owner``; ``(forgotten, busy left)``."""
        async with self.lock:
            owned = [(k, e.busy) for k, e in self.entries.items() if e.owner == owner]
            for key, busy in owned:
                if not busy:
                    self.forget(key)
        return sum(not b for _, b in owned), sum(b for _, b in owned)
