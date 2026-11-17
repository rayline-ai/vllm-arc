# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``GatherPooler`` (stock vLLM 0.33.0): the hidden state at each requested absolute offset.

A request asks for rows with ``PoolingParams(task="token_embed", extra_kwargs={"arc_gather": {"v": 1,
"offsets": [...]}})``; the finished request's output is ``[len(offsets), hidden]`` in request order, in the model
dtype. Per step the runner hands the pooler the step's flattened hidden states and a cursor: request ``i``'s chunk
is ``num_scheduled_tokens[i]`` rows covering absolute positions ``[seq_lens[i] - n, seq_lens[i])``, which also holds
after a prefix-cache hit (the first chunk then starts at the hit length). Only the requested rows are kept between
chunks, never the whole prompt.

It never raises, because it runs in the engine core: an offset a prefix hit already covered (below the first
computed position) gets a NaN row, which the server reads as an over-hit; a request without a valid ``arc_gather``
(the warm-up dummy run) gets an empty ``[0, hidden]`` output.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

#: The ``extra_kwargs`` key and its schema version.
KEY = "arc_gather"
VERSION = 1


class _Gather:
    """One request's offsets, sorted, and how far its chunks have consumed them (kept on its ``PoolingStates``)."""

    __slots__ = ("sorted", "inverse", "next", "pieces", "end")

    def __init__(self, offsets: torch.Tensor) -> None:
        order = torch.argsort(offsets, stable=True)
        self.sorted = offsets[order]
        self.inverse = torch.argsort(order)  # sorted row k is request row order[k]; row j is sorted row inverse[j]
        self.next = 0  # the first sorted offset no chunk has reached yet
        self.pieces: list[torch.Tensor | int] = []  # tensors of selected rows, or a count of NaN rows
        self.end = 0  # the absolute end of the last chunk seen

    def restart(self) -> None:
        """The request's computation started over (V1 keeps its pooling state across a preemption)."""
        self.next, self.pieces, self.end = 0, [], 0


_INVALID = object()


def parse(extra_kwargs: Any) -> _Gather | object:
    """The request's gather spec, or ``_INVALID`` (never raises)."""
    try:
        spec = extra_kwargs[KEY]
        if spec["v"] != VERSION:
            return _INVALID
        offsets = spec["offsets"]
        if not isinstance(offsets, (list, tuple)) or not all(type(o) is int and o >= 0 for o in offsets):
            return _INVALID
        return _Gather(torch.tensor(offsets, dtype=torch.int64))
    except (KeyError, TypeError, IndexError, ValueError, RuntimeError):
        return _INVALID


def _h2d(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device, non_blocking=True)


def gather(hidden_states: torch.Tensor, num_scheduled: Sequence[int], seq_lens: Sequence[int],
           finished: Sequence[bool], extra_kwargs: Sequence[Any], states: Sequence[Any],
           h2d: Callable[[torch.Tensor, torch.device], torch.Tensor] = _h2d) -> list[torch.Tensor | None]:
    """One step's outputs: a ``[len(offsets), hidden]`` tensor per finished request, ``None`` per unfinished one.

    ``states`` are the requests' per-request state objects (vLLM's ``PoolingStates``); the gather spec is attached
    to one as ``arc_gather`` on its first chunk and removed when it finishes. One host-to-device copy and one
    ``index_select`` serve the whole step; the selected rows own their storage, as async scheduling requires.
    """
    width = hidden_states.shape[-1]
    local: list[torch.Tensor] = []  # per request: row indices into hidden_states, in sorted-offset order
    start = 0
    specs = []
    for n, end, kwargs, state in zip(num_scheduled, seq_lens, extra_kwargs, states):
        n, end = int(n), int(end)
        spec = getattr(state, KEY, None)
        if spec is None:
            spec = parse(kwargs)
            setattr(state, KEY, spec)
        specs.append(spec)
        if spec is _INVALID:
            local.append(torch.empty(0, dtype=torch.int64))
        else:
            lo = end - n
            if lo < spec.end:  # a chunk before the last one's end: recomputation from a preemption
                spec.restart()
            spec.end = end
            bounds = torch.searchsorted(spec.sorted, torch.tensor([lo, end], dtype=torch.int64))
            first, last = int(bounds[0]), int(bounds[1])
            if first > spec.next:  # offsets below this chunk that no chunk computed: covered by a prefix hit
                spec.pieces.append(first - spec.next)
            spec.next = max(spec.next, last)
            local.append(spec.sorted[first:last] - lo + start)
        start += n
    counts = [len(x) for x in local]
    selected: list[torch.Tensor] = []
    if sum(counts):
        index = h2d(torch.cat(local), hidden_states.device)
        selected = list(torch.split(hidden_states.index_select(0, index), counts))
    else:
        selected = [hidden_states.new_empty((0, width)) for _ in counts]

    outputs: list[torch.Tensor | None] = []
    for spec, rows, done, state in zip(specs, selected, finished, states):
        if spec is not _INVALID and len(rows):
            spec.pieces.append(rows)
        if not done:
            outputs.append(None)
            continue
        delattr(state, KEY)
        if spec is _INVALID:
            outputs.append(hidden_states.new_empty((0, width)))
            continue
        if spec.next < len(spec.sorted):  # offsets at or past the prompt's end: never computed
            spec.pieces.append(len(spec.sorted) - spec.next)
        parts = [p if isinstance(p, torch.Tensor) else hidden_states.new_full((p, width), float("nan"))
                 for p in spec.pieces]
        ordered = torch.cat(parts) if parts else hidden_states.new_empty((0, width))
        outputs.append(ordered.index_select(0, h2d(spec.inverse, ordered.device)))
    return outputs


def _pooler_base():
    from vllm.model_executor.layers.pooler import Pooler

    return Pooler


class GatherPooler(_pooler_base()):  # type: ignore[misc]
    """The model's own pooler (the pooling adapter keeps a pooler the model already set)."""

    def get_supported_tasks(self):
        return {"token_embed"}

    def get_pooling_updates(self, task):
        from vllm.model_executor.layers.pooler import PoolingParamsUpdate

        return PoolingParamsUpdate()

    def forward(self, hidden_states: torch.Tensor, pooling_metadata) -> list[torch.Tensor | None]:
        from vllm.utils.torch_utils import async_tensor_h2d

        cursor = pooling_metadata.get_pooling_cursor()
        return gather(hidden_states, cursor.num_scheduled_tokens_cpu.tolist(), cursor.seq_lens_cpu.tolist(),
                      cursor.get_finished_mask(), [p.extra_kwargs for p in pooling_metadata.pooling_params],
                      pooling_metadata.pooling_states, h2d=async_tensor_h2d)
