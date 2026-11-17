# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""EQ's encodes: stateless one-shots over the 24-layer engine with hybrid prefix caching (runs in the vLLM venv).

The engine's checkpoint declares
``readout_depths`` [12, 24], so every GATHER row is ``norm(h_12) | norm(h_24)``, 2 x 1024 wide:

- `EqEncodes.encode` answers ``/encode``: columns ``[0, 1024)`` (``norm(h_12)``). A client calls it once per
  conversation turn, on the prefix's last token, to put the prefix in the engine's cache under that turn's salt.
- `EqEncodes.branches` answers ``/branches`` (the question branches): ``prefix + suffix`` per question, reading
  columns ``[1024, 2048)`` at suffix positions; each should hit the prefix in the cache, and a branch that recomputed
  more than one cache unit of it is counted (``branch_misses``).

Every turn has its own salt, so nothing one turn leaves in the engine's cache is reachable by another; the
blocks are reclaimed by the engine's LRU. Nothing is kept on disk or in this process. No token, salt or text is ever
logged.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable

import numpy as np

HIDDEN = 1024  # Qwen3.5-0.8B hidden size: one depth's readout width
READOUT_DEPTHS = [12, 24]
#: Hybrid prefix caching for the GATHER pooling path: align mode, bf16 state, vLLM's default ``mamba_block_size``
#: and hit unit, exactly the configuration the engine is qualified with. A ``prefix_match_unit`` of 16 is not served:
#: it shrinks the state blocks, and EQ's reference readouts then differ. ``CACHE_VARIANT`` stays empty unless a
#: later qualification covers another; the qualification check reads this same constant.
PREFIX_CACHING = {"enable_prefix_caching": True, "mamba_cache_mode": "align"}
CACHE_VARIANT: dict = {}


class Refused(Exception):
    def __init__(self, status: int, code: str, **detail) -> None:
        super().__init__(code)
        self.status, self.code, self.detail = status, code, detail


def engine_args() -> dict:
    return {**PREFIX_CACHING, **CACHE_VARIANT}


class EqEncodes:
    """Admission and the counters around ``engine.encode`` for one EQ engine.

    Admission bounds the tokens in flight (``budget_tokens``, a share of the KV pool) and the requests in flight
    (``max_inflight``): one that cannot start within ``admit_wait_s`` is refused 429 ``capacity`` at once, never
    queued past the caller's deadline. A cold long conversation is not refused here: the caller's own deadline cuts
    it, and the prefill it started finishes, so the next turn is warm."""

    def __init__(self, engine: Any, *, prompt: Callable, params: Callable, max_model_len: int, unit: int,
                 budget_tokens: int, max_inflight: int, admit_wait_s: float, addon: dict) -> None:
        self.engine, self.prompt, self.params = engine, prompt, params
        self.max_model_len, self.unit = max_model_len, unit
        self.budget_tokens, self.max_inflight, self.admit_wait_s = budget_tokens, max_inflight, admit_wait_s
        self.addon = addon
        self.inflight_tokens = self.inflight = 0
        self.changed = asyncio.Condition()
        self.counters = {"encodes": 0, "branches": 0, "branch_misses": 0, "capacity_refusals": 0,
                         "hit_over_readout": 0, "prefix_cold_tokens": 0}

    # ------------------------------------------------------------------------------------------- admission ----

    async def _admit(self, tokens: int) -> None:
        if tokens > self.budget_tokens:
            raise Refused(422, "exceeds_kv_budget", tokens=tokens, budget_tokens=self.budget_tokens)

        def fits() -> bool:
            return self.inflight < self.max_inflight and self.inflight_tokens + tokens <= self.budget_tokens

        async with self.changed:
            if not fits():
                try:
                    await asyncio.wait_for(self.changed.wait_for(fits), self.admit_wait_s)
                except TimeoutError:
                    self.counters["capacity_refusals"] += 1
                    raise Refused(429, "capacity") from None
            self.inflight += 1
            self.inflight_tokens += tokens

    async def _release(self, tokens: int) -> None:
        async with self.changed:
            self.inflight -= 1
            self.inflight_tokens -= tokens
            self.changed.notify_all()

    async def _one(self, tokens: list[int], offsets: list[int], salt: str, images: Any = None) -> tuple[Any, int]:
        final = None
        async for output in self.engine.encode(self.prompt(tokens, images, salt), self.params(offsets, images),
                                               request_id=uuid.uuid4().hex):
            final = output
        if final is None or final.outputs.data.shape[0] != len(offsets):
            raise RuntimeError("the engine returned the wrong number of readouts")
        data = final.outputs.data.float().numpy()
        if data.shape[1] != HIDDEN * len(READOUT_DEPTHS):
            raise RuntimeError(f"EQ rows are {data.shape[1]} wide, not {HIDDEN * len(READOUT_DEPTHS)}")
        return data, int(getattr(final, "num_cached_tokens", 0) or 0)

    # ---------------------------------------------------------------------------------------------- encode ----

    async def encode(self, tokens: list[int], offsets: list[int], *, salt: str | None,
                     images: Any = None) -> np.ndarray:
        """``norm(h_12)`` at ``offsets``, under ``salt`` (a fresh one without), so the prefix's blocks are cached for
        the same request's branches."""
        effective = salt if salt is not None else uuid.uuid4().hex
        await self._admit(len(tokens))
        try:
            data, hit = await self._one(tokens, offsets, effective, images)
            if hit > min(offsets):
                # Defence only: the engine caps a prefix hit at the smallest readout offset. Were it ever
                # past one, that row is not trusted: encode again under a fresh salt.
                self.counters["hit_over_readout"] += 1
                data, _ = await self._one(tokens, offsets, uuid.uuid4().hex, images)
            self.counters["encodes"] += 1
        finally:
            await self._release(len(tokens))
        return data[:, :HIDDEN]

    # -------------------------------------------------------------------------------------------- branches ----

    async def branches(self, prefix: list[int], suffixes: list[tuple[list[int], list[int]]], *,
                       salt: str | None, timing: dict | None = None) -> tuple[list[np.ndarray], list[int], int]:
        """``(rows per suffix, cached tokens per branch, misses)``: ``norm(h_24)`` at each suffix's offsets.
        ``timing``, when given, gets the stages' milliseconds: ``admit`` (waiting for admission), ``prefix`` (the
        prefix's encode) and ``branch`` (the branches, concurrently)."""
        if self.addon.get("verified") is not True:
            raise Refused(503, "question_head_unavailable", reason=self.addon.get("reason"))
        if not prefix:
            raise Refused(422, "empty_prefix")
        for tokens, offsets in suffixes:
            if not tokens or any(not 0 <= o < len(tokens) for o in offsets) or not offsets:
                raise Refused(422, "offsets_out_of_range")
            if len(prefix) + len(tokens) > self.max_model_len:
                raise Refused(422, "state_too_long", tokens=len(prefix) + len(tokens),
                              max_model_len=self.max_model_len)
        effective = salt if salt is not None else uuid.uuid4().hex
        need = len(prefix) + sum(len(tokens) for tokens, _ in suffixes)
        t0 = time.perf_counter()
        await self._admit(need)
        t1 = time.perf_counter()
        try:
            # The prefix first, once: every branch then hits it in the engine's cache instead of each one prefilling
            # it concurrently. Its hit is what the engine still held of it.
            _, prefix_hit = await self._one(prefix, [len(prefix) - 1], effective)
            self.counters["prefix_cold_tokens"] += len(prefix) - prefix_hit
            t2 = time.perf_counter()
            done = await asyncio.gather(*(self._branch(prefix, tokens, offsets, effective)
                                          for tokens, offsets in suffixes))
            if timing is not None:
                timing.update(admit=(t1 - t0) * 1000, prefix=(t2 - t1) * 1000,
                              branch=(time.perf_counter() - t2) * 1000)
        finally:
            await self._release(need)
        cached = [hit for _, hit in done]
        misses = sum(hit < len(prefix) - self.unit for hit in cached)
        self.counters["branches"] += len(suffixes)
        self.counters["branch_misses"] += misses
        return [rows for rows, _ in done], cached, misses

    async def _branch(self, prefix: list[int], tokens: list[int], offsets: list[int],
                      effective: str) -> tuple[np.ndarray, int]:
        """``norm(h_24)`` at one suffix's offsets. A hit past the suffix's first readout (a repeated question in one
        request) is never trusted: that branch alone is encoded again under a fresh salt."""
        readouts = [len(prefix) + o for o in offsets]
        data, hit = await self._one(prefix + tokens, readouts, effective)
        if hit > min(readouts):
            self.counters["hit_over_readout"] += 1
            data, hit = await self._one(prefix + tokens, readouts, uuid.uuid4().hex)
        return data[:, HIDDEN:], hit

    async def reset(self) -> None:
        """Before the engine sleeps (the snapshot): no cached state of any caller survives it."""
        await self.engine.reset_prefix_cache()

    def usage(self) -> dict:
        return {"addon": self.addon, "inflight": self.inflight,
                "inflight_tokens": self.inflight_tokens, "budget_tokens": self.budget_tokens,
                "max_inflight": self.max_inflight,
                **self.counters}
