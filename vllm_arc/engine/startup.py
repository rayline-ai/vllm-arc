# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""Start-up assertions: each a hard stop, because each is a silent arithmetic change.

``Engine`` is what the server claims to run (its settings, the session route and the prefix hit grid, from env);
``assertions(probe, engine)`` judges what the worker probe (``probe.ArcStockProbe.arc_stock_probe``) says actually
runs. Pure functions, so they are tested on CPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

VLLM_VERSION = "0.33.0"
VLLM_COMMIT = "4ba787ec9959e566d3b587a796dff07673324da5"
V2_RUNNER = "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
V1_RUNNER = "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
#: Prefix hit grids (``prefix_match_unit``; 0 leaves it unset). 256 does not divide the 576-token block.
UNITS = (0, 64, 192)
#: The GDN prefill kernel the engine requests (``gdn_prefill_backend``) and the forward it binds.
GDN_BACKEND, GDN_FORWARD = "flashinfer", "forward_cuda"
ARCHITECTURE = "ArcBaseForPooling"
HASH_ALGO = "sha256"
#: The env setting that picks the prefix hit grid (the prefix route only).
UNIT_ENV = "ARC_STOCK_PREFIX_UNIT"
#: The env setting that picks the session route: ``retained`` (patch 0001: one live streaming pooling request per
#: session, prefix caching off, as the fork ran) or ``prefix`` (sessions as cache salts over prefix caching).
SESSIONS_ENV = "ARC_STOCK_SESSIONS"
SESSIONS = ("retained", "prefix")
#: Patch 0001 as it names itself (``vllm.v1.engine.async_llm.ARC_PATCH``); the retained route needs it.
ARC_PATCH = "0001-retained-pooling-session"
#: A compiled engine's ``compilation_config`` (EQ's, ``Engine.compiled``): the fork EQ's as its engine log printed it
#: (vLLM's defaults with eager off; a pooling model gets piecewise CUDA graphs only). Every other engine runs eager.
COMPILATION = {"mode": "VLLM_COMPILE", "backend": "inductor", "custom_ops": ["none"], "cudagraph_mode": "PIECEWISE"}
#: The norms' IR op priority a compiled engine resolves to (vLLM's CUDA default under Inductor), as the fork EQ's.
COMPILED_NORMS = {"rms_norm": ["native"], "fused_add_rms_norm": ["native"]}


@dataclass(frozen=True)
class Engine:
    unit: int = 0  # prefix_match_unit; 0 = unset
    sessions: str = "retained"
    compiled: bool = False  # torch.compile and piecewise CUDA graphs (``COMPILATION``): EQ only; E12 runs eager

    @classmethod
    def from_env(cls, env=os.environ) -> "Engine":
        try:
            unit = int(env.get(UNIT_ENV) or 0)
        except ValueError:
            raise ValueError(f"{UNIT_ENV} is one of {UNITS}, got {env.get(UNIT_ENV)!r}") from None
        engine = cls(unit=unit, sessions=env.get(SESSIONS_ENV) or "retained")
        if engine.sessions not in SESSIONS:
            raise ValueError(f"{SESSIONS_ENV} is one of {SESSIONS}, got {engine.sessions!r}")
        if engine.unit not in UNITS:
            raise ValueError(f"{UNIT_ENV} is one of {UNITS}, got {engine.unit}")
        if engine.retained and engine.unit:
            raise ValueError(f"{UNIT_ENV} sets the prefix hit grid; the retained route runs without prefix caching")
        return engine

    @property
    def retained(self) -> bool:
        return self.sessions == "retained"

    @property
    def label(self) -> str:
        """The session route's part of the engine id: ``retained``, or the prefix hit grid (``u0``, ``u192``)."""
        return "retained" if self.retained else f"u{self.unit}"


def _check(name: str, observed, expected, ok: bool) -> dict:
    return {"name": name, "observed": observed, "expected": expected, "ok": bool(ok)}


def assertions(probe: dict, engine: Engine, rope_parameters: dict | None) -> list[dict]:
    """Every start-up assertion over one probe; the server starts only if all are ``ok``."""
    cache = probe["cache"]
    out = [
        _check("vllm_version", probe["vllm_version"], VLLM_VERSION, probe["vllm_version"] == VLLM_VERSION),
        _check("cuda_runtime_major", probe["torch_cuda"], "13.x",
               str(probe["torch_cuda"] or "").split(".")[0] == "13"),
    ]
    gdn = probe["gdn"]
    kinds = sorted({(g["backend"], g["forward"]) for g in gdn})
    out.append(_check("gdn_prefill_backend", [list(k) for k in kinds] + [len(gdn)],
                      f"{GDN_BACKEND}/{GDN_FORWARD} in every GDN layer",
                      bool(gdn) and kinds == [(GDN_BACKEND, GDN_FORWARD)]))
    out.append(_check("model_runner", probe["runner"], V2_RUNNER, probe["runner"] == V2_RUNNER))
    out.append(_check("model_class", probe["model_mro"][:3], ARCHITECTURE, ARCHITECTURE in probe["model_mro"]))
    out.append(_check("pooler", probe["pooler"], "GatherPooler", probe["pooler"] == "GatherPooler"))
    running = probe["fused_qk_running"]
    out.append(_check("fused_qk_norm_rope_gate", {"stock": probe["fused_qk_stock"], "running": running},
                      "False on every full-attention layer", bool(running) and not any(running)))
    if engine.retained:
        out.extend(retained_checks(probe))
    else:
        out.extend(prefix_cache_checks(cache, engine))
    out.append(_check("gdn_state_dtype", cache["mamba_ssm_cache_dtype"], "float32",
                      cache["mamba_ssm_cache_dtype"] == "float32"))
    out.append(_check("chunked_prefill", probe["scheduler"]["enable_chunked_prefill"], True,
                      probe["scheduler"]["enable_chunked_prefill"] is True))
    out.append(compiled_check(probe) if engine.compiled
               else _check("enforce_eager", probe["enforce_eager"], True, probe["enforce_eager"] is True))
    out.append(rope_check(probe["rope_parameters"], rope_parameters))
    out.append(rope_effective_check(probe.get("rope_effective"), rope_parameters))
    out.append(_check("multimodal_pruning", probe.get("multimodal_pruning"), False,
                      probe.get("multimodal_pruning") is False))
    return out


def compiled_check(probe: dict) -> dict:
    """A compiled engine (``Engine.compiled``): eager off, ``COMPILATION`` as resolved, the norms' IR ops ``native``
    (``COMPILED_NORMS``), and the text model compiling (its ``do_not_compile`` False)."""
    observed = {"enforce_eager": probe["enforce_eager"], **(probe.get("compilation") or {}),
                "do_not_compile": probe.get("do_not_compile")}
    expected = {"enforce_eager": False, **COMPILATION, "ir_op_priority": COMPILED_NORMS, "do_not_compile": False}
    return _check("compiled", observed, expected, observed == expected)


def retained_checks(probe: dict) -> list[dict]:
    """The retained route: patch 0001 installed with its V2 runner hook, and no prefix caching (mamba mode
    ``none``: no block-aligned split, so a chunk is the whole token budget and an append fits one engine step)."""
    cache, patch = probe["cache"], probe.get("arc_patch") or {}
    return [
        _check("prefix_caching", [cache["enable_prefix_caching"], cache["mamba_cache_mode"]], [False, "none"],
               cache["enable_prefix_caching"] is False and cache["mamba_cache_mode"] == "none"),
        _check("arc_patch", patch, {"id": ARC_PATCH, "runner_extend": True},
               patch.get("id") == ARC_PATCH and patch.get("runner_extend") is True),
    ]


def prefix_cache_checks(cache: dict, engine: Engine) -> list[dict]:
    """The prefix route: prefix caching in ``align`` mode on the configured hit grid, sha256, no retention."""
    out = [_check("prefix_caching", [cache["enable_prefix_caching"], cache["mamba_cache_mode"]], [True, "align"],
                  cache["enable_prefix_caching"] is True and cache["mamba_cache_mode"] == "align")]
    unit = cache["prefix_match_unit"] or 0
    grid = [cache["block_size"], cache["hash_block_size"], cache["cache_hit_alignment_tokens"], unit]
    if engine.unit:
        grid_ok = (unit == engine.unit and cache["hash_block_size"] == engine.unit
                   and cache["cache_hit_alignment_tokens"] == engine.unit and cache["block_size"] % engine.unit == 0)
        expected_grid = f"hash and hit grid {engine.unit}, block a multiple"
    else:
        grid_ok = unit == 0 and cache["hash_block_size"] is not None and (
            cache["cache_hit_alignment_tokens"] == cache["hash_block_size"])
        expected_grid = "no prefix_match_unit; hits on one resolved grid"
    out.append(_check("hit_grid [block, hash, hit_alignment, unit]", grid, expected_grid, grid_ok))
    out.append(_check("hash_algo", cache["prefix_caching_hash_algo"], HASH_ALGO,
                      cache["prefix_caching_hash_algo"] == HASH_ALGO))
    out.append(_check("retention_interval", cache["prefix_cache_retention_interval"], 0,
                      cache["prefix_cache_retention_interval"] == 0))
    return out


def rope_check(loaded: dict, expected: dict | None) -> dict:
    """RoPE as loaded equals the configured variant on every key the variant names (a newer transformers may add
    defaulted keys; they are reported, not judged); the native RoPE (``None``) loads no extended ``rope_type``."""
    loaded = dict(loaded or {})
    extended = loaded.get("rope_type") not in (None, "default")
    if expected is None:
        return _check("rope_parameters", loaded, "native", not extended)
    ok = extended and all(loaded.get(k) == v for k, v in expected.items())
    extra = sorted(set(loaded) - set(expected))
    return _check("rope_parameters", {"loaded": loaded, "extra_keys": extra}, expected, ok)


#: The fork's (98e91a960 / 66029e2) YaRN M-RoPE inverse frequencies for the ``yarn-f4`` variant (rotary dim 64,
#: base 1e7, factor 4, beta 32/1, truncate, cache length 262,144 x 4: correction range (16, 24)), from the fork's own
#: ``_compute_inv_freq`` on CPU in float32. Key: the arithmetic inputs; value: (sha256 of the float32 bytes, hex floats).
FORK_YARN_INV_FREQ = {
    (64, 10000000.0, 4.0, 262144): (
        "07e55d7ebefc4c83b73378caaf1d76c53526a88000a2264d97d0942677f62539",
        ("0x1.0000000000000p+0", "0x1.3566560000000p-1", "0x1.75f0360000000p-2", "0x1.c3f06a0000000p-3",
         "0x1.111aee0000000p-3", "0x1.4a12ac0000000p-4", "0x1.8eec7e0000000p-5", "0x1.e222ee0000000p-6",
         "0x1.235a720000000p-6", "0x1.6020a40000000p-7", "0x1.a9942a0000000p-8", "0x1.012cfa0000000p-8",
         "0x1.36d21a0000000p-9", "0x1.77a7da0000000p-10", "0x1.c603c40000000p-11", "0x1.125c040000000p-11",
         "0x1.4b96c00000000p-12", "0x1.6b2f5e0000000p-13", "0x1.8988f00000000p-14", "0x1.a4be6c0000000p-15",
         "0x1.ba2e4c0000000p-16", "0x1.c6410c0000000p-17", "0x1.c41fd60000000p-18", "0x1.ad575c0000000p-19",
         "0x1.7961820000000p-20", "0x1.c8198c0000000p-21", "0x1.139e940000000p-21", "0x1.4d1c980000000p-22",
         "0x1.9298ae0000000p-23", "0x1.e6933a0000000p-24", "0x1.26091c0000000p-24", "0x1.635e880000000p-25")),
}
#: The fork's M-RoPE cache length multiplier (mirrors ``models.FORK_MROPE_POSITIONS_FACTOR``; this module stays
#: importable without vLLM).
FORK_MROPE_POSITIONS_FACTOR = 4
#: Rows of the cos/sin cache the fork's code builds, measured (its own ``_compute_cos_sin_cache`` on CPU): YaRN
#: tabulates ``cache length x factor`` = 1,048,576 x 4 positions. Key: (original positions, factor).
FORK_YARN_COS_SIN_ROWS = {(262144, 4.0): 4194304}


def rope_effective_check(effective: list[dict] | None, expected: dict | None) -> dict:
    """The rotary module every full-attention layer runs is the fork's YaRN: its inverse frequencies equal the fork's
    reference exactly (sha256 of the float32 bytes), its M-RoPE cache length is ``original x 4`` (the length YaRN's
    correction range reads) and its cos/sin table has the fork's measured rows, and factor, base, rotary dim (head size x partial factor) and M-RoPE section match.
    Judges the module, not the config. Native: no layer is scaled. On an inverse-frequency mismatch the report names
    the differing pairs and the largest relative difference."""
    layers = list(effective or [])
    if not layers:
        return _check("rope_effective", layers, "a rotary module per full-attention layer", False)
    if expected is None:
        return _check("rope_effective", layers, "native", all(r.get("scaling_factor") in (None, 1.0) for r in layers))
    original = int(expected["original_max_position_embeddings"])
    want = {"scaling_factor": float(expected["factor"]), "base": float(expected["rope_theta"]),
            "cache_max_position": original * FORK_MROPE_POSITIONS_FACTOR,
            "cos_sin_cache_rows": FORK_YARN_COS_SIN_ROWS.get((original, float(expected["factor"]))),
            "mrope_section": list(expected["mrope_section"]),
            "mrope_interleaved": bool(expected.get("mrope_interleaved", False))}
    got = {"layers": len(layers), "first": {k: v for k, v in layers[0].items() if k != "inv_freq_hex"}}
    head = layers[0].get("head_size")
    dim = int(head * expected.get("partial_rotary_factor", 1.0)) if head is not None else None
    reference = FORK_YARN_INV_FREQ.get((dim, want["base"], want["scaling_factor"], original))
    if reference is None:
        return _check("rope_effective", got, "a fork inverse-frequency reference for this variant", False)
    want["inv_freq_sha256"] = reference[0]

    def matches(r: dict) -> bool:
        dim_ok = (r.get("head_size") is not None and r.get("rotary_dim")
                  == int(r["head_size"] * expected.get("partial_rotary_factor", 1.0)))
        return dim_ok and all(r.get(k) is not None and (float(r[k]) if k in ("scaling_factor", "base") else r[k]) == v
                              for k, v in want.items())

    if layers[0].get("inv_freq_sha256") != reference[0]:
        got["inv_freq_vs_fork"] = _freq_diff(layers[0].get("inv_freq_hex"), reference[1])
    return _check("rope_effective", got, want, all(matches(r) for r in layers))


def _freq_diff(got_hex, ref_hex) -> dict:
    if not got_hex or len(got_hex) != len(ref_hex):
        return {"pairs": len(got_hex or ()), "reference_pairs": len(ref_hex)}
    got = [float.fromhex(x) for x in got_hex]
    ref = [float.fromhex(x) for x in ref_hex]
    differ = [i for i, (a, b) in enumerate(zip(got, ref)) if a != b]
    return {"differing_pairs": differ,
            "max_rel_diff": max((abs(got[i] - ref[i]) / abs(ref[i]) for i in differ), default=0.0)}


def failed(checks: list[dict]) -> list[dict]:
    return [c for c in checks if not c["ok"]]


def vision_assertions(probe: dict, images: dict | None) -> list[dict]:
    """The vision tower runs the ARC profile's compute (``images.VISION``) on every stock engine, since vLLM runs it
    at start even when no image is served: float32 parameters, the SDPA backend, no fused device norm (the model
    normalises the processor's uint8 patches itself, ``models.vit_pixels``, with its CPU-built table), no encoder
    CUDA graphs, the ViT weights replicated. With
    ``images`` served, at most ``max_images`` images per input and no video."""
    from .images import PIXEL_TABLE_SHA256, VISION_ATTENTION

    v = probe.get("vision") or {}
    out = [
        _check("vit_dtype", v.get("param_dtypes"), ["float32"], v.get("param_dtypes") == ["float32"]),
        _check("vit_attn_backend", v.get("attn_backend"), VISION_ATTENTION, v.get("attn_backend") == VISION_ATTENTION),
        _check("vit_input_norm", [v.get("input_norm"), v.get("mm_device_do_normalize")], ["IdentityInputNorm", False],
               v.get("input_norm") == "IdentityInputNorm" and v.get("mm_device_do_normalize") is False),
        _check("vit_tp_mode", v.get("mm_encoder_tp_mode"), "weights", v.get("mm_encoder_tp_mode") == "weights"),
        # The table built on the CPU (under vLLM's target-device context it would be CUDA arithmetic), equal to the
        # HF processor's levels.
        _check("vit_pixel_table", v.get("pixel_table"), {"device": "cpu", "dtype": "float32",
                                                          "sha256": PIXEL_TABLE_SHA256},
               v.get("pixel_table") == {"device": "cpu", "dtype": "float32", "sha256": PIXEL_TABLE_SHA256}),
        # Encoder CUDA graphs call the ViT directly on the uint8 patches, past the model's normalisation.
        _check("cudagraph_mm_encoder", v.get("cudagraph_mm_encoder"), False, v.get("cudagraph_mm_encoder") is False),
    ]
    if images:
        limits = [v.get("image_limit"), v.get("video_limit")]
        out.append(_check("image_limits [image, video]", limits, [images["max_images"], 0],
                          limits == [images["max_images"], 0]))
    return out
