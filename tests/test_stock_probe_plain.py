# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""Every ``ArcStockProbe`` method's return crosses ``collective_rpc``, whose msgspec encoder refuses non-JSON types
(``torch.__version__`` is a ``TorchVersion``) unless insecure pickle serialization is on. The probe is driven on CPU
over fake worker state carrying the types the worker really holds (torch/numpy scalars, dtypes, a TorchVersion),
and its return must survive a strict JSON round trip unchanged. ``arc_check_pinned_memory`` returns None."""
from __future__ import annotations

import enum
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from vllm_arc.engine import probe as stock_probe  # noqa: E402
from vllm_arc.engine.images import PIXEL_TABLE_SHA256  # noqa: E402

Ns = types.SimpleNamespace


def _strict_json(value):
    """JSON with no fallback, and every node's type exactly a JSON type: ``json`` accepts a ``str`` subclass such
    as ``TorchVersion``, vLLM's msgspec encoder does not."""
    def exact(v):
        assert type(v) in (dict, list, str, int, float, bool, type(None)), (type(v), v)
        for child in (v.values() if type(v) is dict else v if type(v) is list else ()):
            exact(child)
    exact(value)
    return json.loads(json.dumps(value, allow_nan=False))


class ChunkGatedDeltaRule(torch.nn.Module):
    gdn_prefill_backend = "flashinfer"

    def __init__(self):
        super().__init__()
        self._forward_method = self.forward_cuda

    def forward_cuda(self):
        pass


class _Backend(enum.Enum):  # vLLM's AttentionBackendEnum shape
    TORCH_SDPA = "torch_sdpa"


class IdentityInputNorm(torch.nn.Module):
    pass


def test_every_probe_return_round_trips_as_plain_json(monkeypatch: pytest.MonkeyPatch) -> None:
    rope = Ns(scaling_factor=torch.tensor(4.0), max_position_embeddings=262144, base=np.float64(1e7), rotary_dim=64,
              head_size=256, mrope_section=(11, 11, 10), mrope_interleaved=np.bool_(True), mscale=1.1386,
              cos_sin_cache=torch.zeros(8, 4), _compute_inv_freq=lambda f: torch.tensor([1.0, 0.5]))
    attn = Ns(rotary_emb=rope, use_fused_qk_norm_rope_gate=False)
    monkeypatch.setitem(sys.modules, "vllm", Ns(__version__=torch.torch_version.TorchVersion("0.33.0")))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", Ns(async_llm=Ns(ARC_PATCH="0001-retained-pooling-session")))
    monkeypatch.setitem(sys.modules, "vllm_arc.engine.models", Ns(fused_qk_layers=lambda m: [attn] * 3,
                                                                  multimodal_pruning=lambda c: np.bool_(False)))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA RTX PRO 6000")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))

    model = torch.nn.Module()
    model.language_model = Ns(model=Ns(arc_readout_depths=(12,), do_not_compile=True))
    model.rules = torch.nn.ModuleList([ChunkGatedDeltaRule() for _ in range(3)])
    model.arc_fused_qk_stock = (True, True, True)
    model.visual = torch.nn.Linear(2, 2)
    model.visual.attn_backend, model.visual.input_norm = _Backend.TORCH_SDPA, IdentityInputNorm()
    model.arc_visual_calls = 2
    model.arc_pixel_table = (torch.arange(256, dtype=torch.float32) - 127.5) / 127.5
    spec = Ns(block_size=576, dtype=torch.bfloat16, dtypes=(torch.float32, torch.bfloat16))
    kv = Ns(kv_cache_groups=[Ns(kv_cache_spec=spec)], hash_block_size=192, cache_hit_alignment_tokens=np.int64(192),
            num_blocks=torch.tensor(23391))
    worker = stock_probe.ArcStockProbe()
    worker.model_runner = Ns(get_model=lambda: model, kv_cache_config=kv)
    worker.vllm_config = Ns(
        cache_config=Ns(enable_prefix_caching=True, block_size=576, mamba_block_size=576, mamba_cache_mode="align",
                        prefix_match_unit=192, prefix_caching_hash_algo="sha256", prefix_cache_retention_interval=0,
                        mamba_ssm_cache_dtype=torch.float32),
        scheduler_config=Ns(enable_chunked_prefill=True, async_scheduling=False, max_num_batched_tokens=8192,
                            max_num_seqs=68),
        compilation_config=Ns(cudagraph_mm_encoder=False, mode=Ns(name="NONE"), backend="inductor",
                              custom_ops=["all"], cudagraph_mode=Ns(name="NONE")),
        kernel_config=Ns(ir_op_priority=Ns(rms_norm=["vllm_c", "native"], fused_add_rms_norm=["vllm_c", "native"])),
        model_config=Ns(hf_text_config=Ns(rope_parameters={"rope_type": "yarn", "factor": np.float32(4.0)},
                                          max_position_embeddings=1048576), enforce_eager=True,
                        multimodal_config=Ns(mm_device_do_normalize=False, mm_encoder_tp_mode="weights",
                                             get_limit_per_prompt={"image": 32, "video": 0}.get)))

    out = worker.arc_stock_probe()
    assert _strict_json(out) == out
    assert out["vision"] == {"param_dtypes": ["float32"], "attn_backend": "TORCH_SDPA",
                             "input_norm": "IdentityInputNorm", "mm_device_do_normalize": False,
                             "mm_encoder_tp_mode": "weights", "image_limit": 32, "video_limit": 0, "visual_calls": 2,
                             "pixel_table": {"device": "cpu", "dtype": "float32", "sha256": PIXEL_TABLE_SHA256},
                             "cudagraph_mm_encoder": False}
    assert worker.arc_stock_visual_calls() == 2
    assert out["compiled_forward"] is None  # eager: no compile wrapper on the text model
    assert out["compilation"] == {"mode": "NONE", "backend": "inductor", "custom_ops": ["all"], "cudagraph_mode": "NONE",
                                  "ir_op_priority": {"rms_norm": ["vllm_c", "native"],
                                                     "fused_add_rms_norm": ["vllm_c", "native"]}}
    assert type(out["torch_version"]) is str and out["torch_version"] == str(torch.__version__)
    assert out["versions"]["vllm"] == "0.33.0" and type(out["versions"]["vllm"]) is str
    assert out["rope_effective"][0]["scaling_factor"] == 4.0 and out["rope_effective"][0]["mrope_interleaved"] is True
    assert out["cache"]["num_blocks"] == 23391 and out["cache"]["cache_hit_alignment_tokens"] == 192
    assert out["gdn"] == [{"backend": "flashinfer", "forward": "forward_cuda"}] * 3
    assert out["cache"]["mamba_ssm_cache_dtype"] == "float32"
    # Ns runner: no patch-0001 hook on its class, and no extensions counted
    assert out["arc_patch"] == {"id": "0001-retained-pooling-session", "runner_extend": False, "extends": 0}
