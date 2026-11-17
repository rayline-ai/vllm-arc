# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``GatherPooler`` on CPU, against stand-in pooling metadata (vLLM is not installed in this suite).

Requests are run through chunked steps exactly as the runner hands them over: per step the flattened hidden states of
every scheduled request, and a cursor of scheduled tokens, sequence lengths after the step and finished flags. The
rows must equal direct indexing of each request's whole hidden-state sequence, in request order.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PLUGIN = Path(__file__).parents[1]
WIDTH = 6


@pytest.fixture
def pooling(monkeypatch):
    pooler_mod = types.ModuleType("pooler")
    pooler_mod.Pooler = type("Pooler", (torch.nn.Module,), {})
    pooler_mod.PoolingParamsUpdate = lambda **kw: SimpleNamespace(**kw)
    torch_utils = types.ModuleType("torch_utils")
    torch_utils.async_tensor_h2d = lambda t, device: t.to(device)
    for name, module in {"vllm": types.ModuleType("vllm"), "vllm.model_executor": types.ModuleType("me"),
                         "vllm.model_executor.layers": types.ModuleType("layers"),
                         "vllm.model_executor.layers.pooler": pooler_mod, "vllm.utils": types.ModuleType("utils"),
                         "vllm.utils.torch_utils": torch_utils}.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in [n for n in sys.modules if n == "vllm_arc" or n.startswith("vllm_arc.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(PLUGIN))
    import vllm_arc.engine.pooling as pooling

    yield pooling
    for name in [n for n in sys.modules if n == "vllm_arc" or n.startswith("vllm_arc.")]:
        sys.modules.pop(name)


class _Request:
    """One request: its whole hidden-state sequence, the prefix-hit length, its chunk sizes and its offsets."""

    def __init__(self, length: int, chunks: list[int], offsets, hit: int = 0, seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.hidden = torch.randn(length, WIDTH, generator=gen, dtype=torch.float32).to(torch.bfloat16)
        assert hit + sum(chunks) == length
        self.chunks, self.hit, self.length = chunks, hit, length
        self.params = SimpleNamespace(extra_kwargs=None if offsets is None else
                                      ({"arc_gather": {"v": 1, "offsets": offsets}} if isinstance(offsets, list)
                                       else offsets))
        self.state = SimpleNamespace(hidden_states_cache=[])
        self.done = hit


def _run(pooler, schedule: list[list[_Request]]) -> dict[int, torch.Tensor]:
    """Run ``schedule`` (per step, the requests that get their next chunk); the finished outputs by request id."""
    outputs = {}
    for step in schedule:
        rows, n, ends = [], [], []
        for r in step:
            size = r.chunks.pop(0)
            rows.append(r.hidden[r.done:r.done + size])
            r.done += size
            n.append(size)
            ends.append(r.done)
        hidden = torch.cat(rows)
        cursor = SimpleNamespace(num_scheduled_tokens_cpu=torch.tensor(n), seq_lens_cpu=torch.tensor(ends),
                                 get_finished_mask=lambda ends=ends, step=step: [e == r.length
                                                                                for e, r in zip(ends, step)])
        meta = SimpleNamespace(get_pooling_cursor=lambda c=cursor: c, pooling_params=[r.params for r in step],
                               pooling_states=[r.state for r in step])
        out = pooler(hidden, meta)
        assert len(out) == len(step)
        for r, o in zip(step, out):
            if r.done < r.length:
                assert o is None
            else:
                outputs[id(r)] = o
        hidden.fill_(0)  # the runner reuses its buffer: finished rows must own their storage
    return outputs


def test_rows_equal_direct_indexing_in_request_order_across_chunks_and_a_prefix_hit(pooling):
    """Two requests, three chunks each, batched together; B starts after a 5-token prefix hit, so its offset 2 is
    covered (a NaN row); unsorted and duplicate offsets come back in request order."""
    a = _Request(30, [10, 12, 8], [29, 0, 9, 10, 10, 21, 22], seed=1)
    b = _Request(25, [7, 7, 6], [24, 2, 5, 6, 11, 12, 18], hit=5, seed=2)
    out = _run(pooling.GatherPooler(), [[a, b], [a, b], [a, b]])
    assert torch.equal(out[id(a)], a.hidden[[29, 0, 9, 10, 10, 21, 22]])
    rows_b = out[id(b)]
    assert rows_b.dtype == torch.bfloat16 and rows_b.shape == (7, WIDTH)
    assert torch.isnan(rows_b[1]).all() and torch.isfinite(rows_b[[0, 2, 3, 4, 5, 6]]).all()
    assert torch.equal(rows_b[[0, 2, 3, 4, 5, 6]], b.hidden[[24, 5, 6, 11, 12, 18]])
    assert not hasattr(a.state, "arc_gather") and not hasattr(b.state, "arc_gather")


def test_requests_that_finish_at_different_steps_and_one_shot_requests(pooling):
    """A one-chunk request finishing in the first step beside a three-chunk one; a request whose every offset sits
    in its last chunk; an empty offset list."""
    long = _Request(40, [16, 16, 8], [39, 15, 16, 31, 32], seed=3)
    short = _Request(9, [9], [8, 0, 4], seed=4)
    tail = _Request(20, [8, 8, 4], [19, 16], seed=5)
    empty = _Request(5, [5], [], seed=6)
    out = _run(pooling.GatherPooler(), [[long, short, tail, empty], [long, tail], [tail, long]])
    assert torch.equal(out[id(long)], long.hidden[[39, 15, 16, 31, 32]])
    assert torch.equal(out[id(short)], short.hidden[[8, 0, 4]])
    assert torch.equal(out[id(tail)], tail.hidden[[19, 16]])
    assert out[id(empty)].shape == (0, WIDTH)


@pytest.mark.parametrize("extra", [None, {}, {"arc_gather": {"v": 2, "offsets": [0]}},
                                   {"arc_gather": {"v": 1, "offsets": [0, -1]}},
                                   {"arc_gather": {"v": 1, "offsets": [0.5]}}, {"arc_gather": "x"}])
def test_no_valid_gather_spec_gives_an_empty_output_and_never_raises(pooling, extra):
    """The warm-up dummy run (``extra_kwargs=None``) and any malformed spec: ``[0, hidden]`` at the end, ``None``
    before; a valid request in the same batch is unaffected."""
    bad = _Request(12, [6, 6], extra if extra is not None else None, seed=7)
    good = _Request(12, [6, 6], [11, 0], seed=8)
    out = _run(pooling.GatherPooler(), [[bad, good], [bad, good]])
    assert out[id(bad)].shape == (0, WIDTH)
    assert torch.equal(out[id(good)], good.hidden[[11, 0]])


def test_an_offset_past_the_prompt_gets_a_nan_row(pooling):
    """The server refuses such offsets; the pooler still never raises."""
    r = _Request(8, [8], [3, 8, 100], seed=9)
    rows = _run(pooling.GatherPooler(), [[r]])[id(r)]
    assert torch.equal(rows[0], r.hidden[3]) and torch.isnan(rows[1:]).all()


def test_a_request_recomputed_after_a_preemption_starts_its_rows_over(pooling):
    """V1 keeps a request's pooling state across a preemption and recomputes it from 0 (or from a hit): the rows
    gathered before the restart are dropped, not duplicated."""
    r = _Request(20, [8, 8, 4], [0, 3, 12, 19], seed=10)
    pooler = pooling.GatherPooler()
    _run(pooler, [[r]])  # first chunk, then preempted
    r.done, r.chunks = 0, [10, 10]
    rows = _run(pooler, [[r], [r]])[id(r)]
    assert torch.equal(rows, r.hidden[[0, 3, 12, 19]])


def test_supported_task_is_token_embed_only(pooling):
    assert pooling.GatherPooler().get_supported_tasks() == {"token_embed"}
