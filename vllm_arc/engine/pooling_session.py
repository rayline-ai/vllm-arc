# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``PoolingSession`` (patch 0001): one retained streaming pooling request, one output per append.

A port of the fork's ``AsyncPoolingSession`` (``vllm/v1/engine/pooling_session.py`` at atlasfutures/vllm@66029e2d)
to stock vLLM 0.33.0 with patch 0001 (``engine.lock``). The engine keeps the request alive
between appends (a paused resumable request keeps its KV blocks, GDN state and runner slot), so an append sends only
its new input and the engine computes only it.

An append is any vLLM prompt (``{"prompt_token_ids": [...]}``, or with ``multi_modal_data``) plus the number of
session positions it covers once the engine has expanded it (``num_positions``): positions are counted as the
engine counts them, so an image's placeholder run counts as its expanded length. The engine's processed prompt must
then hold exactly the session's positions so far; anything else closes the session (the expansion check).

Differences from the fork: readout offsets travel in ``extra_kwargs["arc_gather"]`` (stock ``PoolingParams`` has no
``readout_offsets``), and no ``retain_pooling_state`` flag exists (the GATHER pooler keeps nothing between inputs).

Backpressure, as in the fork: the output collector keeps only the latest pooling output, so the next input is not
released to the engine until the caller has the previous output. Appends are serialized. ``close`` ends the input
stream, which finishes the request and frees its blocks. Any failure closes the session; the caller drops it.

No vLLM import at module load: the engine is injected, so this runs on CPU in tests.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import Any

#: ``pooling.KEY`` (that module imports vLLM at load; this one must not).
KEY = "arc_gather"


class PoolingSessionError(RuntimeError):
    """The retained request broke its append/output lifecycle; the session is closed."""


def _streaming_input(prompt: Any, params: Any) -> Any:
    from vllm.engine.protocol import StreamingInput

    return StreamingInput(prompt=prompt, sampling_params=params)


def offsets_of(params: Any) -> list[int] | None:
    spec = (getattr(params, "extra_kwargs", None) or {}).get(KEY)
    return None if spec is None else list(spec["offsets"])


class PoolingSession:
    """``engine.encode(input_stream, pooling_params, request_id)`` stays open; ``append`` feeds it."""

    def __init__(self, engine: Any, *, pooling_params: Any, request_id: str,
                 make_input: Callable[[Any, Any], Any] = _streaming_input) -> None:
        if not request_id:
            raise ValueError("pooling session request_id must be non-empty")
        if offsets_of(pooling_params):
            raise ValueError("readout offsets are per input: pass them to append(), not to the session")
        self.request_id, self._make_input = request_id, make_input
        self._inputs: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        self._ack: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        self._lock = asyncio.Lock()
        self._closed = self._started = False
        self._num_tokens = 0  # engine positions appended so far
        self._outputs = engine.encode(self._input_stream(), pooling_params, request_id)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    async def _input_stream(self) -> AsyncGenerator[Any, None]:
        while (item := await self._inputs.get()) is not None:
            yield item
            await self._ack.get()

    async def _abandon(self) -> None:
        self._closed = True
        await self._outputs.aclose()

    async def append(self, prompt: Any, num_positions: int, pooling_params: Any) -> Any:
        """Append ``prompt`` (``num_positions`` session positions once expanded) and return the engine's output for
        it. ``pooling_params`` carries this input's offsets (absolute session positions inside this input); offsets
        outside it are refused before the engine sees them and leave the session usable."""
        async with self._lock:
            if self._closed:
                raise PoolingSessionError("pooling session is closed")
            if num_positions < 1:
                raise ValueError("an append needs at least one position")
            start, stop = self._num_tokens, self._num_tokens + num_positions
            offsets = offsets_of(pooling_params)
            if offsets and not all(start <= o < stop for o in offsets):
                raise ValueError(f"readout offsets must lie in this input's positions [{start}, {stop})")
            self._started = True
            await self._inputs.put(self._make_input(prompt, pooling_params))
            try:
                output = await anext(self._outputs)
            except StopAsyncIteration as exc:
                self._closed = True
                raise PoolingSessionError("pooling session ended before producing an append output") from exc
            except BaseException:
                await self._abandon()
                raise
            if output.finished:
                await self._abandon()
                raise PoolingSessionError("pooling session append unexpectedly finished the request")
            processed = output.prompt_token_ids
            if isinstance(processed, list) and len(processed) != stop:
                # The caller's positions no longer match the engine's (for an image: its expansion).
                await self._abandon()
                raise PoolingSessionError(f"vLLM holds {len(processed)} session tokens, the appends give {stop}")
            self._num_tokens = stop
            await self._ack.put(None)
            return output

    async def close(self) -> None:
        """Finish the request (frees its blocks). Idempotent."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if not self._started:
                await self._outputs.aclose()
                return
            await self._inputs.put(None)
            try:
                await anext(self._outputs)
            except StopAsyncIteration:
                return
            finally:
                await self._outputs.aclose()
            raise PoolingSessionError("pooling session emitted an output after its input ended")
