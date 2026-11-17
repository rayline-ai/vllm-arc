# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The stock engine's ``worker_extension_cls``: its methods run in the worker, over ``collective_rpc``.

``arc_stock_probe`` reports what actually runs (plain types only): versions, the model runner, the model and pooler
classes, every GDN prefill op's backend and bound forward, the fused-QK switch per full-attention layer (stock's
choice and the running value), the cache grid as the worker's KV cache config resolved it, and the scheduler
settings. ``startup.assertions`` judges it. ``arc_check_pinned_memory`` is today's wake-up check.
"""
from __future__ import annotations

import json
import os


def _model(worker):
    model = worker.model_runner.get_model()
    for _ in range(4):  # compile / CUDA-graph wrappers keep the module as ``runnable`` or ``model``
        if hasattr(model, "language_model"):
            return model
        model = getattr(model, "runnable", None) or getattr(model, "model", None) or model
    return model


def _dtype(value) -> str | None:
    return None if value is None else str(value).removeprefix("torch.")


def plain(value):
    """``value`` as JSON types only. ``collective_rpc`` returns go through vLLM's msgspec encoder, which refuses
    anything else (``torch.__version__`` is a ``TorchVersion``, a ``str`` subclass) unless insecure pickle
    serialization is allowed, which the server never sets. Numbers and strings keep their value; any other object
    becomes its ``str``."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(v) for v in value]
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():  # torch / numpy 0-d scalars
        return plain(item())
    return str(value)


class ArcStockProbe:
    def arc_patch_extends(self) -> int:
        """How many streaming updates patch 0001's V2 hook extended in place (0 without the patch)."""
        return int(getattr(self.model_runner, "arc_extends", 0))

    def arc_stock_probe(self) -> dict:
        import torch
        import vllm

        from .models import fused_qk_layers, multimodal_pruning

        model = _model(self)
        cache = self.vllm_config.cache_config
        sched = self.vllm_config.scheduler_config
        kv = getattr(self.model_runner, "kv_cache_config", None)
        groups = []
        for group in getattr(kv, "kv_cache_groups", None) or []:
            spec = group.kv_cache_spec
            dtypes = getattr(spec, "dtypes", None)
            groups.append({"spec": type(spec).__name__, "block_size": getattr(spec, "block_size", None),
                           "dtype": _dtype(getattr(spec, "dtype", None)),
                           "dtypes": [_dtype(d) for d in dtypes] if dtypes else None})
        gdn = [{"backend": getattr(m, "gdn_prefill_backend", None),
                "forward": getattr(getattr(m, "_forward_method", None), "__name__", None)}
               for m in model.modules() if type(m).__name__ == "ChunkGatedDeltaRule"]
        inner = model.language_model.model
        runner = type(self.model_runner)
        text = self.vllm_config.model_config.hf_text_config
        try:
            import arc_triton_pins

            tune = arc_triton_pins.tune_id() or "timed"
        except ImportError:
            tune = None
        from importlib import metadata

        def version(dist):
            try:
                return metadata.version(dist)
            except metadata.PackageNotFoundError:
                return None

        # uv checks the wheel's sha256 at install but records only its URL (direct_url.json, PEP 610); the server
        # names the hash when the URL is the recipe's pinned one.
        try:
            wheel_url = json.loads(metadata.distribution("vllm").read_text("direct_url.json") or "{}").get("url")
        except (metadata.PackageNotFoundError, ValueError):
            wheel_url = None
        from vllm.v1.engine import async_llm

        return plain({
            # Patch 0001 as it names itself, its V2 runner hook (the delta write), and the hook's extensions so far.
            "arc_patch": {"id": getattr(async_llm, "ARC_PATCH", None),
                          "runner_extend": hasattr(runner, "_arc_extend_request"),
                          "extends": self.arc_patch_extends()},
            "versions": {"vllm": vllm.__version__, "vllm_wheel_url": wheel_url,
                         "torch": torch.__version__, "flashinfer": version("flashinfer-python"),
                         "triton": version("triton")},
            "vllm_version": vllm.__version__, "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "runner": f"{runner.__module__}.{runner.__name__}",
            "model_mro": [c.__name__ for c in type(model).__mro__],
            "pooler": type(getattr(model, "pooler", None)).__name__,
            "gdn": gdn,
            "fused_qk_stock": list(getattr(model, "arc_fused_qk_stock", [])),
            "fused_qk_running": [bool(a.use_fused_qk_norm_rope_gate) for a in fused_qk_layers(model)],
            "inner_model": type(inner).__name__, "do_not_compile": getattr(inner, "do_not_compile", None),
            "compiled_forward": compiled_forward(inner),
            # EQ's engine only (``models.ArcEqModel``); an E12 probe has no such key.
            **({"readout_depths": list(inner.arc_readout_depths)} if getattr(inner, "arc_readout_depths", None)
               else {}),
            "rope_parameters": dict(getattr(text, "rope_parameters", None) or {}),
            "text_max_position_embeddings": getattr(text, "max_position_embeddings", None),
            "rope_effective": rope_effective(model),
            "multimodal_pruning": multimodal_pruning(self.vllm_config),
            "vision": vision(model, self.vllm_config),
            "cache": {
                "enable_prefix_caching": cache.enable_prefix_caching, "block_size": cache.block_size,
                "mamba_block_size": cache.mamba_block_size, "mamba_cache_mode": cache.mamba_cache_mode,
                "prefix_match_unit": cache.prefix_match_unit, "prefix_caching_hash_algo": cache.prefix_caching_hash_algo,
                "prefix_cache_retention_interval": cache.prefix_cache_retention_interval,
                "mamba_ssm_cache_dtype": _dtype(cache.mamba_ssm_cache_dtype),
                "hash_block_size": getattr(kv, "hash_block_size", None),
                "cache_hit_alignment_tokens": getattr(kv, "cache_hit_alignment_tokens", None),
                "num_blocks": getattr(kv, "num_blocks", None), "groups": groups,
            },
            "scheduler": {"enable_chunked_prefill": sched.enable_chunked_prefill,
                          "async_scheduling": sched.async_scheduling,
                          "max_num_batched_tokens": sched.max_num_batched_tokens, "max_num_seqs": sched.max_num_seqs},
            "enforce_eager": self.vllm_config.model_config.enforce_eager,
            "compilation": compilation(self.vllm_config),
            "env": {k: os.environ.get(k) for k in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_TRITON_FORCE_FIRST_CONFIG",
                                                   "ARC_TRITON_PINS", "CUDA_HOME")},
            "autotune": tune,
        })

    def arc_stock_visual_calls(self) -> int:
        """ViT calls since the model was built (``ArcBaseForPooling.arc_visual_calls``)."""
        return int(getattr(_model(self), "arc_visual_calls", -1))

    def arc_stock_vit_inputs(self) -> list | None:
        """The ViT inputs recorded since the last call (``ArcBaseForPooling.arc_vit_inputs``), cleared; ``None``
        when recording is off."""
        model = _model(self)
        recorded = getattr(model, "arc_vit_inputs", None)
        if recorded is None:
            return None
        drained, model.arc_vit_inputs = list(recorded), []
        return plain(drained)

    def arc_check_pinned_memory(self) -> None:
        import torch

        empty_cache = getattr(torch._C, "_host_emptyCache", None)
        if empty_cache is not None:
            empty_cache()
        host = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)
        host.to("cuda", non_blocking=True)
        torch.cuda.synchronize()


def compiled_forward(inner) -> str | None:
    """The forward vLLM's compile wrapper handed ``torch.compile`` (``TorchCompileWithNoGuardsWrapper``'s
    ``_compiled_callable``, its original callable) by qualified name; ``None`` when the text model runs eager."""
    fn = getattr(getattr(inner, "_compiled_callable", None), "_torchdynamo_orig_callable", None)
    return getattr(getattr(fn, "__func__", fn), "__qualname__", None)


def compilation(vllm_config) -> dict:
    """The compilation settings as resolved (``startup.COMPILATION``'s keys, enums by name) and the norms' IR op
    priority."""
    cc = vllm_config.compilation_config
    priority = getattr(getattr(vllm_config, "kernel_config", None), "ir_op_priority", None)
    return {"mode": getattr(cc.mode, "name", cc.mode), "backend": cc.backend, "custom_ops": list(cc.custom_ops),
            "cudagraph_mode": getattr(cc.cudagraph_mode, "name", cc.cudagraph_mode),
            "ir_op_priority": {op: list(getattr(priority, op, None) or []) for op in ("rms_norm", "fused_add_rms_norm")}}


def vision(model, vllm_config) -> dict:
    """The vision tower as it runs: its parameter dtypes, attention backend and input normalisation, the engine's
    image limits, the ViT calls so far, the model's pixel table, and whether vLLM captures the encoder in CUDA graphs
    (which would call the ViT directly, past ``_process_image_input``)."""
    visual = getattr(model, "visual", None)
    mm = vllm_config.model_config.multimodal_config
    backend = getattr(visual, "attn_backend", None)
    return {"param_dtypes": sorted({_dtype(p.dtype) for p in visual.parameters()}) if visual is not None else [],
            "attn_backend": getattr(backend, "name", backend),
            "input_norm": type(getattr(visual, "input_norm", None)).__name__,
            "mm_device_do_normalize": getattr(mm, "mm_device_do_normalize", None),
            "mm_encoder_tp_mode": getattr(mm, "mm_encoder_tp_mode", None),
            "image_limit": mm.get_limit_per_prompt("image") if mm is not None else None,
            "video_limit": mm.get_limit_per_prompt("video") if mm is not None else None,
            "visual_calls": getattr(model, "arc_visual_calls", None),
            "pixel_table": pixel_table(getattr(model, "arc_pixel_table", None)),
            "cudagraph_mm_encoder": getattr(getattr(vllm_config, "compilation_config", None), "cudagraph_mm_encoder",
                                            None)}


def pixel_table(table) -> dict | None:
    """The uint8-level table the model normalises pixels with (``models.pixel_table``): where it was built, its dtype,
    and the sha256 of its bytes."""
    import hashlib

    if table is None:
        return None
    return {"device": table.device.type, "dtype": _dtype(table.dtype),
            "sha256": hashlib.sha256(table.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}


def rope_effective(model) -> list[dict]:
    """The rotary module each full-attention layer actually runs, as constructed (not as configured): scaling
    factor, base, rotary dim, M-RoPE section, the cache length YaRN's correction range reads
    (``max_position_embeddings``; for M-RoPE that is ``original x positions factor``), the cos/sin cache rows, and the
    inverse frequencies the module computes (float32 on CPU, as hex floats plus their sha256)."""
    from .models import fused_qk_layers

    out = []
    for attn in fused_qk_layers(model):
        rope = getattr(attn, "rotary_emb", None)
        section = getattr(rope, "mrope_section", None)
        cache = getattr(rope, "cos_sin_cache", None)
        out.append({"class": type(rope).__name__, "scaling_factor": getattr(rope, "scaling_factor", None),
                    "cache_max_position": getattr(rope, "max_position_embeddings", None),
                    "cos_sin_cache_rows": int(cache.shape[0]) if cache is not None else None,
                    "base": getattr(rope, "base", None), "rotary_dim": getattr(rope, "rotary_dim", None),
                    "head_size": getattr(rope, "head_size", None),
                    "mrope_section": list(section) if section is not None else None,
                    "mrope_interleaved": getattr(rope, "mrope_interleaved", None),
                    "mscale": getattr(rope, "mscale", None), **inv_freq(rope)})
    return out


def inv_freq(rope) -> dict:
    """The module's own ``_compute_inv_freq`` on CPU in float32: hex floats and the sha256 of their bytes."""
    import hashlib

    import torch

    compute = getattr(rope, "_compute_inv_freq", None)
    if compute is None:
        return {"inv_freq_hex": None, "inv_freq_sha256": None}
    with torch.device("cpu"):
        arg = getattr(rope, "scaling_factor", None)
        freq = (compute(arg) if arg is not None else compute(rope.base)).to("cpu", torch.float32).contiguous()
    return {"inv_freq_hex": [float(x).hex() for x in freq.tolist()],
            "inv_freq_sha256": hashlib.sha256(freq.numpy().tobytes()).hexdigest()}
