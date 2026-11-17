# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The ARC encoder server on stock vLLM 0.33.0 + this plugin: the E12 ``/encode`` contract, text and, with
``--images arc-image-v1`` (``ARC_VLLM_IMAGES``), images (``images``).

Sessions are ``RetainedSessions`` by default (patch 0001: one live streaming pooling request per session, prefix
caching off, as the fork ran) or, with ``ARC_STOCK_SESSIONS=prefix``, ``PrefixCacheSessions`` (sessions as cache
salts); one-shots are always a plain ``encode``. Readouts come from ``GatherPooler``; the engine refuses to serve
unless every start-up assertion holds (``startup``) and the GPU self-test passes. Wire contract, env and engine args:
``README.md`` beside this file.

    vllm-arc-server --port 8765   (or python -m vllm_arc.server)
"""
# No ``from __future__ import annotations``: FastAPI resolves string annotations in module globals, and the route
# functions' ``Request`` is imported inside ``build`` (fastapi is not needed to import this module).
import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from ..engine import compat
from ..engine import images as img
from ..engine.recipe import FREEZE_ENV, VLLM_WHEEL, VLLM_WHEEL_SHA256
from ..engine.pooling_session import PoolingSession, PoolingSessionError
from ..engine.retained import RetainedSessions
from ..engine.sessions import KvPool, Pass, PrefixCacheSessions, Refused, Result
from ..engine.startup import (ARCHITECTURE, COMPILATION, GDN_BACKEND, HASH_ALGO, SESSIONS_ENV, VLLM_COMMIT, Engine, assertions,
                      failed, vision_assertions)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
#: The RoPE variants a release may declare; must equal the release format's (copied, not imported).
ROPE_VARIANTS = {
    "yarn-f4": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000,
                "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144},
}
ROPE_CAPACITY = {"": 262144, "yarn-f4": 1010000}
#: The served depth: E12, the base checkpoint cut to 12 decoder layers.
DEPTH = 12
WORKER_EXTENSION = "vllm_arc.engine.probe.ArcStockProbe"
#: The env a deployment names the plugin's source commit in (reported on ``/health``).
PLUGIN_COMMIT_ENV = "ARC_VLLM_ARC_COMMIT"


def plugin_digest(root: Path | None = None) -> str:
    """The engine id's ``arc``: 12 hex of ``digest.engine_digest`` over the engine's runtime files (``vllm_arc``'s
    ``__init__``, ``engine/``, ``eq/``, ``server/`` and the lock), with comments and docstrings ignored."""
    from ..engine.digest import engine_digest

    return engine_digest(root or Path(__file__).resolve().parents[1])[:12]


def engine_id(engine: Engine, backend: str, rope: str, images: dict | None = None) -> str:
    """vLLM's commit, the plugin's source digest, the readout, depth, RoPE, the prefix hit grid and the image settings:
    each is an arithmetic input, so reference outputs keyed by this id never match another build. A
    text-only engine's id has no image part. Under the load guard's override (``compat.unlocked``) it ends in
    ``+unqualified``."""
    return (f"vllm@{VLLM_COMMIT[:12]}+arc-{plugin_digest()}+gather-{backend}+depth{DEPTH}"
            + (f"+rope-{rope}" if rope else "") + f"+stock-{engine.label}"
            + (img.image_engine_suffix(images) if images else "")
            + ("+unqualified" if compat.unlocked() else ""))


def stock_config(config: dict, rope: str, arch: str) -> None:
    """The stock engine's edits to a checkpoint's ``config.json`` (in place): ``architectures`` naming ``arch`` (so no
    in-tree architecture is overridden) and ``rope`` (a ``ROPE_VARIANTS`` name) in the text config, with YaRN's
    positions as stock vLLM reads them."""
    config["architectures"] = [arch]
    text = config.get("text_config", config)
    if rope:
        text["rope_parameters"] = dict(ROPE_VARIANTS[rope])
        if ROPE_VARIANTS[rope]["rope_type"] == "yarn":
            # Stock vLLM (0.33, config/model.py _get_and_verify_max_len) reads a YaRN config's max_position_embeddings
            # as already scaled (Transformers' convention) and no longer multiplies it by the factor, as the fork did.
            # The YaRN arithmetic reads only factor and original_max_position_embeddings (get_rope), unchanged; the
            # start-up check rope_effective judges the module built from it.
            p = ROPE_VARIANTS[rope]
            text["max_position_embeddings"] = int(p["original_max_position_embeddings"] * p["factor"])


def truncated_checkpoint(root: str, rope: str = "", *, depth: int = DEPTH, arch: str = ARCHITECTURE,
                         readout_depths: tuple | None = None) -> str:
    """The pinned base cut to ``depth`` (``DEPTH``), with ``rope`` in its text config and ``architectures`` naming
    ``arch`` (``ArcBaseForPooling``), built once under ``root``: the fork server's ``truncated_checkpoint`` plus the
    architecture (`stock_config`). ``readout_depths`` (EQ's, with ``depth`` 24 and EQ's architecture) is written to
    the text config and names the directory; E12 never sets it."""
    import re
    import shutil

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    target = Path(root) / (f"{MODEL_ID.replace('/', '--')}-{MODEL_REVISION[:12]}-layers{depth}"
                           + (f"-{rope}-mpe" if rope else "") + f"-{arch}"  # -mpe: scaled positions
                           + ("-readouts" + "-".join(map(str, readout_depths)) if readout_depths else ""))
    if (target / "config.json").exists():
        return str(target)
    source = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))
    staging = target.with_name(target.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    pattern = re.compile(r"language_model\.layers\.(\d+)\.")
    index: dict = {"metadata": {}, "weight_map": {}}
    for shard in sorted(source.glob("*.safetensors")):
        tensors = {k: v for k, v in load_file(shard).items()
                   if not ((m := pattern.search(k)) and int(m.group(1)) >= depth) and not k.startswith("mtp.")}
        if tensors:
            save_file(tensors, staging / shard.name, metadata={"format": "pt"})
            index["weight_map"].update({k: shard.name for k in tensors})
    (staging / "model.safetensors.index.json").write_text(json.dumps(index))
    for path in source.iterdir():
        if path.suffix != ".safetensors" and path.name not in ("model.safetensors.index.json", "config.json"):
            shutil.copy(path, staging / path.name)
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    text["num_hidden_layers"] = depth
    text["layer_types"] = text["layer_types"][:depth]
    text.pop("mtp_num_hidden_layers", None)
    stock_config(config, rope, arch)
    if readout_depths:
        text["readout_depths"] = list(readout_depths)
    (staging / "config.json").write_text(json.dumps(config, indent=1))
    staging.rename(target)
    return str(target)


def engine_args(engine: Engine, checkpoint: str, *, budget: int, max_model_len: int, max_sessions: int,
                scratch_slots: int, gpu_memory: float, images: dict | None = None) -> dict:
    """The exact ``AsyncEngineArgs`` keywords (``pooler_config`` as a dict; ``build`` makes it a ``PoolerConfig``),
    with the vision tower's settings and, when ``images`` are served, the image limits and processor settings. Eager,
    except a compiled engine (``Engine.compiled``, EQ's): eager off and ``startup.COMPILATION``."""
    return dict(
        model=checkpoint, tokenizer=MODEL_ID, tokenizer_revision=MODEL_REVISION,
        runner="pooling", convert="embed", dtype="bfloat16",
        max_model_len=max_model_len, max_num_batched_tokens=budget, max_num_seqs=max_sessions + scratch_slots,
        enable_chunked_prefill=True, gpu_memory_utilization=gpu_memory,
        **(dict(enforce_eager=False, compilation_config=dict(COMPILATION)) if engine.compiled
           else dict(enforce_eager=True)),
        gdn_prefill_backend=GDN_BACKEND,
        pooler_config={"seq_pooling_type": "MEAN", "tok_pooling_type": "ALL", "use_activation": False},
        enable_log_requests=False, enable_sleep_mode=True, worker_extension_cls=WORKER_EXTENSION,
        # Retained sessions keep each session's KV and GDN state live, so they run without prefix caching, as the
        # fork does: vLLM then sets mamba mode "none", which needs no block-aligned split, so a chunk is the whole
        # budget (8192, not 8064) and an append fits one engine step.
        **(dict(enable_prefix_caching=False) if engine.retained else dict(
            enable_prefix_caching=True, mamba_cache_mode="align", prefix_match_unit=engine.unit or None,
            prefix_caching_hash_algo=HASH_ALGO, prefix_cache_retention_interval=0)),
        **img.vision_engine_args(images), **img.image_engine_args(images),
    )


def gather_params(offsets: list[int], skip: bool):
    from vllm.pooling_params import PoolingParams

    return PoolingParams(task="token_embed", use_activation=False, skip_reading_prefix_cache=skip,
                         extra_kwargs={"arc_gather": {"v": 1, "offsets": list(offsets)}})


def engine_prompt(tokens: list[int], images: img.ImageInput | None) -> dict:
    """One input's engine prompt: its tokens, or with ``images`` their collapsed ids and pixels."""
    return img.prompt(tokens, images)


def check_images(out: Pass, tokens: list[int], images: img.ImageInput | None) -> Pass:
    """With images, the engine expanded this input exactly as the caller did (``images.require_expanded`` over its
    span of the processed ids); ``ImageExpansionError`` (422 ``image_input_refused``) otherwise."""
    if images is not None:
        img.require_expanded(list(out.processed or ())[out.start:out.start + out.encoded], tokens)
    return out


def engine_pass(engine):
    """``PrefixCacheSessions``' ``run``: one ``encode`` of the full sequence under ``salt``. With ``images`` (an
    ``images.ImageInput``), the prompt is its collapsed ids and pixels; the engine refusing it is 422
    ``image_input_refused``, and so is an expansion other than ``tokens`` (``images.require_expanded``)."""

    async def run(tokens: list[int], offsets: list[int], salt: str, skip: bool,
                  images: img.ImageInput | None = None) -> Pass:
        final = None
        try:
            async for out in engine.encode({**engine_prompt(tokens, images), "cache_salt": salt},
                                           gather_params(offsets, skip), request_id=uuid.uuid4().hex):
                final = out
        except ValueError as exc:  # vLLM's input processing refuses a request with ValueError
            if images is None:
                raise
            raise Refused(422, "image_input_refused", reason=str(exc)[:500]) from exc
        if final is None:
            raise RuntimeError("the engine returned no output")
        rows = final.outputs.data.float().cpu()
        if rows.shape[0] != len(offsets):
            raise RuntimeError(f"GatherPooler returned {rows.shape[0]} rows for {len(offsets)} offsets")
        return check_images(Pass(rows, int(final.num_cached_tokens or 0), len(tokens), final.prompt_token_ids, 0),
                            tokens, images)

    return run


def open_session(engine):
    """``RetainedSessions``' ``open_session``: a live streaming pooling request (patch 0001)."""

    def start() -> PoolingSession:
        return PoolingSession(engine, pooling_params=gather_params([], True), request_id=uuid.uuid4().hex)

    return start


async def session_append(session: PoolingSession, tokens: list[int], offsets: list[int],
                         images: img.ImageInput | None = None) -> Pass:
    """``RetainedSessions``' ``append``: one input through a live session; only this input reaches the engine.
    ``len(tokens)`` is its session positions (an image run at its expanded length). With images, vLLM refusing the
    input, or a session length other than the expanded one, is 422 ``image_input_refused``, as is an expansion other
    than the caller's (``check_images``); ``RetainedSessions`` closes the session on any of them."""
    start = session.num_tokens
    try:
        out = await session.append(engine_prompt(tokens, images), len(tokens), gather_params(offsets, True))
    except (ValueError, PoolingSessionError) as exc:
        if images is None:
            raise
        raise Refused(422, "image_input_refused", reason=str(exc)[:500]) from exc
    rows = out.outputs.data.float().cpu()
    if rows.shape[0] != len(offsets):
        raise RuntimeError(f"GatherPooler returned {rows.shape[0]} rows for {len(offsets)} offsets")
    return check_images(Pass(rows, int(out.num_cached_tokens or 0), len(tokens), out.prompt_token_ids, start),
                        tokens, images)


async def patch_extends(engine) -> int:
    """Patch 0001's in-place extension count, summed over workers."""
    return sum(await engine.collective_rpc("arc_patch_extends"))


async def selftest(run, vocab: int, max_model_len: int, start_session=None, extends=None) -> dict:
    """S0 on the GPU: sparse readouts equal the all-rows readout of the same input, bit for bit, in request order
    (1k tokens: one chunk; 20k tokens: three chunks at budget 8192), all finite. Then, reported against a one-shot of
    the same 1,064 tokens (equality is not expected), one append of 64 tokens after 1,000:
    - retained (``start_session``): two appends through one live session, finite and in shape, and the second must
      have taken patch 0001's in-place extension (``extends()`` counts them), not stock's full re-add;
    - prefix: an append under the first pass's salt, finite and hitting no more than its first 1,000 tokens.
    Every pass writes cache blocks; the caller resets the prefix cache after."""
    import torch

    rng = random.Random(20261111)
    report: dict[str, Any] = {"ok": True, "cases": []}
    for n, sparse in ((1000, [999, 0, 3, 500, 500, 998, 255, 1]),
                      (20000, [8191, 8192, 0, 19999, 16384, 16383, 12345, 8191])):
        if n > max_model_len:
            report["cases"].append({"tokens": n, "skipped": "max_model_len"})
            continue
        tokens = [rng.randrange(1000, min(vocab, 150000)) for _ in range(n)]
        part = await run(tokens, sparse, uuid.uuid4().hex, True)
        full = await run(tokens, list(range(n)), uuid.uuid4().hex, True)
        expect = full.rows[torch.tensor(sparse)]
        equal = part.rows.shape == expect.shape and torch.equal(part.rows, expect)
        finite = bool(torch.isfinite(full.rows).all())
        diff = float((part.rows - expect).abs().max()) if part.rows.shape == expect.shape else None
        case = {"tokens": n, "rows": list(part.rows.shape), "bit_equal": equal, "finite": finite, "max_abs": diff}
        report["cases"].append(case)
        report["ok"] &= equal and finite
    tokens = [rng.randrange(1000, min(vocab, 150000)) for _ in range(1064)]
    tail = list(range(1000, 1064, 7))
    once = await run(tokens, tail, uuid.uuid4().hex, True)
    if start_session is not None:
        before = await extends()
        session = start_session()
        try:
            await session_append(session, tokens[:1000], [999, 0])
            kept = await session_append(session, tokens[1000:], tail)
        finally:
            await session.close()
        cos = torch.nn.functional.cosine_similarity(kept.rows, once.rows, dim=-1)
        report["retained"] = {"rows": list(kept.rows.shape), "finite": bool(torch.isfinite(kept.rows).all()),
                              "max_abs_vs_oneshot": float((kept.rows - once.rows).abs().max()),
                              "cos_min_vs_oneshot": float(cos.min()), "extends": await extends() - before}
        report["ok"] &= (report["retained"]["finite"] and kept.rows.shape == once.rows.shape
                         and report["retained"]["extends"] > 0)
        return report
    salt = uuid.uuid4().hex
    await run(tokens[:1000], [999], salt, True)
    appended = await run(tokens, tail, salt, False)
    cos = torch.nn.functional.cosine_similarity(appended.rows, once.rows, dim=-1)
    report["append"] = {"cached": appended.cached, "finite": bool(torch.isfinite(appended.rows).all()),
                        "max_abs_vs_oneshot": float((appended.rows - once.rows).abs().max()),
                        "cos_min_vs_oneshot": float(cos.min())}
    report["ok"] &= report["append"]["finite"] and appended.cached <= 1000
    return report


def _selftest_images(ids: img.VisionIds, settings: dict, sizes, base: int = 0, text: int = 16):
    """Synthetic images (no file is read) between text, from absolute ``base``: ``(tokens, declarations)``."""
    import base64
    import hashlib

    factor = settings["rule"]["factor"]
    declared, tokens = [], [1000 + i for i in range(text)]
    for h, w in sizes:
        raw = bytes((x * 7 + y * 3 + c * 50) % 256 for y in range(h) for x in range(w) for c in range(3))
        n = (h // factor) * (w // factor)
        digest = hashlib.sha256(raw).hexdigest()
        declared.append({"offset": base + len(tokens), "n": n, "height": h, "width": w, "sha256": digest,
                         "rgb_sha256": digest, "bytes": base64.b64encode(raw).decode()})
        tokens += [ids.start] + [ids.pad] * n + [ids.end] + [2000 + i for i in range(8)]
    return tokens, declared


async def image_selftest(run, ids: img.VisionIds, settings: dict, visual_calls, start_session=None,
                         extends=None) -> dict:
    """Images at start, each case checked for the declared expansion (``run`` and ``session_append`` refuse any
    other) and finite rows:
    - one-shot: one input with two images of different sizes between text runs the ViT exactly twice;
    - retained (``start_session``): text, then an input carrying one image (one ViT call), then a text append, which
      must take patch 0001's in-place extension (``extends()``), not the full re-add, and run no ViT; reported, the
      image's and the tail's rows against a one-shot of the same sequence (``cos_vs_oneshot``).
    A refusal on any case fails the self-test (it is reported, not raised)."""
    import torch

    report: dict[str, Any] = {"ok": True}

    def finite(p: Pass) -> bool:
        return bool(torch.isfinite(p.rows).all())

    try:
        tokens, declared = _selftest_images(ids, settings, ((256, 256), (256, 384)))  # 64 and 96 pads
        offsets = [d["offset"] + d["n"] for d in declared] + [len(tokens) - 1]
        before = await visual_calls()
        out = await run(tokens, offsets, uuid.uuid4().hex, True, img.image_input(tokens, 0, declared, ids, settings))
        report["one_shot"] = {"tokens": len(tokens), "images": len(declared), "finite": finite(out),
                              "visual_calls": await visual_calls() - before, "rows": list(out.rows.shape)}
        report["ok"] &= (report["one_shot"]["visual_calls"] == len(declared) and report["one_shot"]["finite"]
                         and out.rows.shape[0] == len(offsets))
        if start_session is not None:
            session = start_session()
            try:
                head = [1000 + i for i in range(500)]
                await session_append(session, head, [len(head) - 1])
                chunk, [image] = _selftest_images(ids, settings, ((256, 320),), base=len(head), text=4)
                calls, ext = await visual_calls(), await extends()
                with_image = await session_append(session, chunk, [image["offset"] + image["n"]],
                                                  img.image_input(chunk, len(head), [image], ids, settings))
                image_calls = await visual_calls() - calls
                calls, ext_image = await visual_calls(), await extends() - ext
                tail_base = len(head) + len(chunk)
                tail_tokens = [3000 + i for i in range(64)]
                tail = await session_append(session, tail_tokens, [tail_base + 63])
                report["retained"] = {"image_visual_calls": image_calls, "image_extends": ext_image,
                                      "tail_visual_calls": await visual_calls() - calls,
                                      "tail_extends": await extends() - ext - ext_image,
                                      "finite": finite(with_image) and finite(tail)}
            finally:
                await session.close()
            # Reported, not judged: the same sequence as one input. Positions after an image in a session (the full
            # re-add's M-RoPE, then the in-place extension's delta) that differ from a one-shot's show here as a
            # cosine far from 1; chunking differs, so bit equality is not expected.
            full = head + chunk + tail_tokens
            once = await run(full, [image["offset"] + image["n"], tail_base + 63], uuid.uuid4().hex, True,
                             img.image_input(full, 0, [image], ids, settings))
            retained_rows = torch.cat([with_image.rows, tail.rows])
            cos = torch.nn.functional.cosine_similarity(retained_rows, once.rows, dim=-1)
            report["retained"].update(cos_vs_oneshot=[float(c) for c in cos],
                                      max_abs_vs_oneshot=float((retained_rows - once.rows).abs().max()))
            r = report["retained"]
            report["ok"] &= (r["image_visual_calls"] == 1 and r["tail_visual_calls"] == 0 and r["tail_extends"] > 0
                             and r["finite"])
    except Refused as exc:
        report.update(ok=False, refused={"code": exc.code, **exc.detail})
    return report


def engine_health(probe: dict, engine: Engine, counters: dict, **detail) -> dict:
    """``/health``'s ``stock`` block: what runs (backend, runner, fused QK, cache grid, versions, autotune), the
    session counters, and the full probe, assertions, self-test and engine args under ``detail``."""
    cache_keys = ("block_size", "hash_block_size", "cache_hit_alignment_tokens", "prefix_match_unit",
                  "mamba_block_size", "mamba_cache_mode", "mamba_ssm_cache_dtype", "enable_prefix_caching")
    backends = sorted({g["backend"] for g in probe["gdn"]})
    env = probe.get("env") or {}
    autotune = ("force_first" if env.get("VLLM_TRITON_FORCE_FIRST_CONFIG") == "1"
                else "timed" if probe.get("autotune") in (None, "timed") else "pinned")
    return {
        "settings": engine.label, "sessions": engine.sessions, "arc_patch": probe.get("arc_patch"),
        "gdn_prefill_backend": backends[0] if len(backends) == 1 else "mixed:" + ",".join(map(str, backends)),
        "runner": "v1" if probe["runner"].startswith("vllm.v1.worker.gpu_model_runner.") else (
            "v2" if probe["runner"].startswith("vllm.v1.worker.gpu.model_runner.") else probe["runner"]),
        "fused_qk": {"enabled_layers": sum(probe["fused_qk_running"]),
                     "full_attention_layers": len(probe["fused_qk_running"]),
                     "stock_choice": any(probe["fused_qk_stock"])},
        "cache": {k: probe["cache"].get(k) for k in cache_keys},
        "counters": {"over_hit": counters["over_hits"], "rebuild": counters["rebuilds"],
                     "miss": counters["zero_hit_appends"] + counters["short_hit_appends"]},
        "versions": {**probe.get("versions", {}),
                     "vllm_wheel_sha256": VLLM_WHEEL_SHA256 if probe.get("versions", {}).get("vllm_wheel_url") == VLLM_WHEEL
                     else None,
                     "plugin_commit": os.environ.get(PLUGIN_COMMIT_ENV),
                     "plugin_digest": plugin_digest(), "freeze_sha256": _freeze_sha256()},
        "autotune": autotune,
        "startup_ok": all(c["ok"] for c in detail.get("checks") or [{"ok": False}]),
        "selftest": detail.get("selftest"),
        "detail": {"probe": probe, "counters": counters, **detail},
    }


def _freeze_sha256() -> str | None:
    path = os.environ.get(FREEZE_ENV)
    if not path:
        return None
    with contextlib.suppress(OSError):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return None


def _engine_dead(engine) -> bool:
    return getattr(engine, "errored", False) is True


def build(engine_settings: Engine, *, budget: int, max_model_len: int, max_sessions: int, idle_seconds: float,
          scratch_slots: int, checkpoint_root: str, rope: str, images: str = "", eq_addon: str = ""):
    """The app. ``eq_addon`` (``ARC_EQ_ADDON``) serves EQ instead of E12 (``eq``): the 24-layer engine on the prefix
    route, ``/encode`` one-shots and ``/branches``; empty (the default) leaves every E12 path as it is."""
    import numpy as np
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse

    eq = None
    if eq_addon:
        from ..eq import engine as eq

        if engine_settings.retained or images:
            raise ValueError("EQ serves the prefix route (ARC_STOCK_SESSIONS=prefix) and text only (no --images)")
        if not engine_settings.compiled:
            raise ValueError("EQ runs compiled (eq.engine)")
        eq_addon = eq.addon_dir(eq_addon)
    elif engine_settings.compiled:
        raise ValueError("only EQ runs compiled; E12 and images run eager")
    if rope not in ROPE_CAPACITY:
        raise ValueError(f"unknown rope variant {rope!r}")
    if max_model_len > ROPE_CAPACITY[rope]:
        raise ValueError(f"max_model_len {max_model_len} exceeds the {rope or 'native'} RoPE capacity")
    served_images = img.image_settings(images)  # None: text only
    # The engine's processes read the route from here: models' load guard checks the patched files on the retained
    # route (compat.retained_route), whoever builds the app.
    os.environ[SESSIONS_ENV] = engine_settings.sessions
    state: dict[str, Any] = {"images": served_images}
    if served_images and os.environ.get(img.VIT_INPUTS_ENV) == "1":  # the worker records them under the same setting
        state["vit_lock"] = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        from vllm.config import PoolerConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        if eq is not None:
            checkpoint, addon_status = eq.checkpoint(eq_addon, checkpoint_root, rope)
        else:
            checkpoint = truncated_checkpoint(checkpoint_root, rope)
        args = engine_args(engine_settings, checkpoint, budget=budget, max_model_len=max_model_len, max_sessions=max_sessions,
                           scratch_slots=scratch_slots, gpu_memory=float(os.environ.get("ARC_VLLM_GPU_MEMORY", "0.80")),
                           images=served_images)
        state["engine_args"] = args
        engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**{**args, "pooler_config": PoolerConfig(**args["pooler_config"])}))
        try:
            [probe] = (await engine.collective_rpc("arc_stock_probe"))[:1]
            checks = (assertions(probe, engine_settings, ROPE_VARIANTS.get(rope))
                      + vision_assertions(probe, served_images) + (eq.assertions(probe) if eq is not None else []))
            state.update(engine=engine, probe=probe, checks=checks)
            backend = sorted({g["backend"] for g in probe["gdn"]} or {"none"})
            state["identity"] = {"engine": engine_id(engine_settings, "-".join(backend), rope, served_images),
                                 "model": MODEL_ID, "revision": MODEL_REVISION, "depth": DEPTH, "budget": budget,
                                 "max_model_len": max_model_len, "rope": rope or None,
                                 "rope_parameters": ROPE_VARIANTS.get(rope)}
            if eq is not None:
                state["identity"] = eq.identity(state["identity"], eq_addon, addon_status, probe["cache"])
            if failed(checks):
                raise StartupFailed("assertions", failed(checks))
            if served_images:  # a text-only engine's /health has no images key
                state["vision_ids"] = img.VisionIds.from_engine(engine)
                state["identity"]["images"] = {**served_images,
                                               "processor": img.processor_identity(MODEL_ID, MODEL_REVISION)}
            run = engine_pass(engine)
            vocab = engine.vllm_config.model_config.get_vocab_size()
            retained = engine_settings.retained
            state["selftest"] = await selftest(run, vocab, max_model_len, open_session(engine) if retained else None,
                                               (lambda: patch_extends(engine)) if retained else None)
            if served_images:
                async def visual_calls() -> int:
                    return int((await engine.collective_rpc("arc_stock_visual_calls"))[0])

                state["selftest"]["images"] = await image_selftest(
                    run, state["vision_ids"], served_images, visual_calls,
                    open_session(engine) if retained else None, (lambda: patch_extends(engine)) if retained else None)
                state["selftest"]["ok"] &= state["selftest"]["images"]["ok"]
            await engine.reset_prefix_cache()
            if not state["selftest"]["ok"]:
                raise StartupFailed("selftest", state["selftest"])
            hit_alignment = probe["cache"]["cache_hit_alignment_tokens"] or probe["cache"]["block_size"]
            pool = KvPool.from_engine(engine)
            state["sessions"] = (
                RetainedSessions(open_session(engine), session_append, run, pool, max_sessions, idle_seconds,
                                 scratch_slots, hit_alignment) if retained
                else PrefixCacheSessions(run, pool, max_sessions, idle_seconds, scratch_slots, hit_alignment))
            if eq is not None:  # EQ's encodes; the sessions above serve nothing on EQ (/close answers 0)
                state["eq"] = eq.encodes(engine, pool, addon_status, hit_alignment, prompt=engine_prompt,
                                         params=gather_params)
            print(json.dumps({"event": "arc_stock_ready", "engine": state["identity"]["engine"],
                              "cache": probe["cache"], "selftest": state["selftest"]}, default=str), flush=True)
        except StartupFailed as exc:
            engine.shutdown()
            print(json.dumps({"event": "arc_stock_startup_failed", "stage": exc.stage, "detail": exc.detail},
                             default=str), flush=True)
            os._exit(3)  # S0 fails: the process exits non-zero, nothing is served
        except BaseException:
            engine.shutdown()
            raise
        watchdog = asyncio.create_task(_watch(state))
        yield
        watchdog.cancel()
        engine.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> Any:
        if _engine_dead(state.get("engine")):
            return JSONResponse({"status": "engine_dead"}, status_code=503)
        sessions = state["sessions"]
        stock = engine_health(state["probe"], engine_settings, sessions.counters, checks=state["checks"],
                              selftest=state["selftest"], engine_args=state["engine_args"])
        if engine_settings.retained:
            # Live, not the start-up probe's snapshot: the runner's in-place extensions since start (self-test
            # included), read from the workers now.
            stock["arc_patch"] = {**(stock["arc_patch"] or {}), "extends": await patch_extends(state["engine"])}
        return {"status": "ok", "sessions": len(sessions.entries), "kv": sessions.usage(), **state["identity"],
                "stock": stock, **({"eq": {**state["identity"]["eq"], **state["eq"].usage()}} if "eq" in state else {})}

    def rows_response(result: Result) -> Response:
        data = result.rows.numpy().astype(np.float32)
        vit = getattr(result, "vit_inputs", None)
        return Response(data.tobytes(), media_type="application/octet-stream",
                        headers={**({VIT_INPUTS_HEADER: json.dumps(vit, separators=(",", ":"))} if vit is not None
                                    else {}),"x-rows": str(data.shape[0]), "x-dim": str(data.shape[1] if data.ndim == 2 else 0),
                                 "x-cached": str(result.cached), "x-cached-tokens": str(result.cached),
                                 "x-over-hit": str(int(result.over_hit)),
                                 "x-encoded": str(result.encoded)})

    @app.post("/encode")
    async def encode(request: Request) -> Response:
        try:
            body = await request.json()
            if "eq" in state:  # EQ: norm(h_12) one-shots, the fork EQ server's response
                status, out = await eq.encode_request(state, body)
                if status != 200:
                    return JSONResponse(out, status_code=status)
                data = out.astype(np.float32)
                return Response(data.tobytes(), media_type="application/octet-stream",
                                headers={"x-rows": str(data.shape[0]), "x-dim": str(data.shape[1])})
            images_body = isinstance(body, dict) and bool(body.get("images"))
            status, out = await (vit_inputs_request if images_body and state.get("vit_lock") else encode_request)(
                state, body)
            return rows_response(out) if status == 200 else JSONResponse(out, status_code=status)
        except Exception as exc:  # noqa: BLE001 -- answered and logged by type alone
            return _internal_error("encode", exc, state)

    async def branches(request: Request) -> Response:
        """EQ's question branches (the fork EQ server's ``/branches``): rows ``norm(h_24)`` per suffix, concatenated;
        headers ``x-counts``, ``x-dim``, ``x-cached``, ``x-misses``, ``x-timing``."""
        try:
            started = time.perf_counter()
            body = await request.json()
            timing = {"parse": (time.perf_counter() - started) * 1000}
            status, out = await eq.branches_request(state, body, timing=timing)
            if status != 200:
                return JSONResponse(out, status_code=status)
            rows, cached, misses = out
            data = np.concatenate(rows).astype(np.float32)
            timing["server"] = (time.perf_counter() - started) * 1000
            return Response(data.tobytes(), media_type="application/octet-stream",
                            headers={"x-counts": ",".join(str(len(r)) for r in rows), "x-dim": str(data.shape[1]),
                                     "x-cached": ",".join(map(str, cached)), "x-misses": str(misses),
                                     "x-timing": ",".join(f"{k}={v:.1f}" for k, v in timing.items())})
        except Exception as exc:  # noqa: BLE001 -- answered and logged by type alone
            return _internal_error("branches", exc, state)

    if eq is not None:  # E12's routes stay exactly as they are
        app.post("/branches")(branches)

    @app.post("/close")
    async def close(request: Request) -> Response:
        body = await request.json()
        owner = body.get("owner") if isinstance(body, dict) else None
        if not isinstance(owner, str) or not owner:
            return JSONResponse({"error": "invalid_owner"}, status_code=422)
        closed, busy = await state["sessions"].close_owner(owner)
        if busy:
            return JSONResponse({"error": "session_busy", "busy": busy}, status_code=409)
        return JSONResponse({"closed": closed})

    @app.post("/sleep")
    async def sleep() -> Any:
        sessions = state["sessions"]
        async with sessions.lock:
            await sessions.close_all()  # a retained session's live request ends (its blocks are freed)
            await state["engine"].reset_prefix_cache()
            await state["engine"].sleep(level=1)
        return {"sleeping": True}

    @app.post("/wake_up")
    async def wake_up() -> Any:
        await state["engine"].wake_up()
        await state["engine"].collective_rpc("arc_check_pinned_memory")
        return {"sleeping": False}

    return app


#: The response header that carries an image request's ViT inputs when ``ARC_STOCK_VIT_INPUTS=1``: a JSON list, one
#: object per ViT call in call order (the request's images in order), each ``{"sha256", "dtype", "shape",
#: "grid_thw"}`` (``models.vit_input_digest``). Absent when recording is off or the request carries no images.
VIT_INPUTS_HEADER = "x-arc-vit-inputs"


async def vit_inputs_request(state: dict, body: Any) -> tuple[int, Any]:
    """``encode_request`` for an image request with ViT input recording on: image requests run one at a time, so the
    calls recorded between draining before and after are this request's, and the result carries them
    (``VIT_INPUTS_HEADER``). A debug path: it serializes image requests."""
    engine = state["engine"]
    async with state["vit_lock"]:
        await engine.collective_rpc("arc_stock_vit_inputs")  # whatever an earlier request left
        status, out = await encode_request(state, body)
        recorded = (await engine.collective_rpc("arc_stock_vit_inputs"))[0]
    if status == 200:
        out.vit_inputs = list(recorded or [])
    return status, out


async def encode_request(state: dict, body: Any) -> tuple[int, Any]:
    """One ``/encode``: ``(200, Result)`` or ``(status, error body)``; today's validation and refusals."""
    if not isinstance(body, dict):
        return 422, {"error": "invalid_body"}
    tokens, offsets = body.get("tokens"), body.get("offsets")
    # Token ids are checked here: a non-int one would pass vLLM's range check and fail in the engine core's request
    # decoder, outside its per-request error handling.
    if (not isinstance(tokens, list) or not isinstance(offsets, list) or not all(type(o) is int for o in offsets)
            or not all(type(t) is int for t in tokens)):
        return 422, {"error": "invalid_body"}
    try:
        base = int(body.get("base", 0))
    except (TypeError, ValueError):
        return 422, {"error": "invalid_body"}
    retain = body.get("retain", True)
    if not isinstance(retain, bool) or (not retain and base != 0):
        return 422, {"error": "invalid_retain"}
    if not tokens or any(not base <= o < base + len(tokens) for o in offsets):
        return 422, {"error": "offsets_out_of_range"}
    declared = body.get("images")
    sessions: RetainedSessions | PrefixCacheSessions = state["sessions"]
    try:
        images = None
        if declared:
            if not state.get("images"):
                return 422, {"error": "images_not_served"}
            # A retained encode with images on the prefix-cache route is refused there (images_need_retained).
            images = img.image_input(tokens, base, declared, state["vision_ids"], state["images"])
        if not retain:
            result = await sessions.encode_once(tokens, offsets, images=images)
        else:
            owner = body.get("owner")
            if owner is not None and (not isinstance(owner, str) or not owner):
                return 422, {"error": "invalid_owner"}
            session = body.get("session")
            if session is None:  # a retained encode names its session; absence is never a key
                return 422, {"error": "invalid_body"}
            result = await sessions.encode(str(session), base, tokens, offsets, owner, images=images)
    except img.InvalidImages as exc:
        return 422, {"error": "invalid_images", "reason": str(exc)}
    except Refused as exc:
        return exc.status, {"error": exc.code, **exc.detail}
    if result is None:
        return 409, {"error": "session_unknown"}
    return 200, result


class StartupFailed(Exception):
    def __init__(self, stage: str, detail: Any) -> None:
        super().__init__(stage)
        self.stage, self.detail = stage, detail


def _internal_error(route: str, exc: Exception, state: dict):
    from fastapi import Response

    print(json.dumps({"event": "arc_stock_internal_error", "route": route, "error": type(exc).__name__}), flush=True)
    if _engine_dead(state.get("engine")) or type(exc).__name__ == "EngineDeadError":
        os._exit(1)
    return Response("Internal Server Error", status_code=500, media_type="text/plain")


async def _watch(state: dict, interval: float = 1.0) -> None:
    while True:
        await asyncio.sleep(interval)
        if _engine_dead(state.get("engine")):
            print(json.dumps({"event": "arc_stock_engine_dead"}), flush=True)
            os._exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    # Accepted so a launcher's `--backend` line works; the engine always requests flashinfer and the
    # start-up assertions judge what actually runs.
    parser.add_argument("--backend", default=os.environ.get("ARC_VLLM_BACKEND", "flashinfer"))
    parser.add_argument("--budget", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("ARC_ENCODER_MAX_MODEL_LEN", 262144)))
    parser.add_argument("--rope", default=os.environ.get("ARC_ENCODER_ROPE", "yarn-f4"))
    parser.add_argument("--max-sessions", type=int, default=int(os.environ.get("ARC_VLLM_MAX_SESSIONS", 64)))
    parser.add_argument("--idle-seconds", type=float, default=float(os.environ.get("ARC_VLLM_IDLE_SECONDS") or 900))
    parser.add_argument("--scratch-slots", type=int, default=4)
    parser.add_argument("--images", default=os.environ.get("ARC_VLLM_IMAGES", ""),
                        help=f"an image rule ({', '.join(img.IMAGE_RULES)}); empty serves text only")
    parser.add_argument("--checkpoint-root", default=os.environ.get("ARC_ENCODER_CHECKPOINTS",
                                                                  "/tmp/arc-encoder-checkpoints"))
    parser.add_argument("--eq-addon", default=os.environ.get("ARC_EQ_ADDON", ""),
                        help="EQ: serve the 24-layer engine with this question-head add-on (prefix route, text only)")
    args = parser.parse_args()
    # EQ runs the prefix route: ARC_STOCK_SESSIONS defaults to prefix there (retained is refused by build).
    settings = Engine.from_env({SESSIONS_ENV: "prefix", **os.environ} if args.eq_addon else os.environ)
    # The locked vLLM release (and, on the retained route, its patch set), before any engine or model is built.
    compat.require(patched=settings.retained)
    if args.backend != "flashinfer":
        raise SystemExit(f"the stock engine requests the flashinfer GDN prefill backend, not {args.backend!r}")
    # The venv's own tools (FlashInfer's JIT runs ``ninja`` by name) when the server is started by its absolute path.
    venv_bin = os.path.dirname(sys.executable)
    if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}" + os.environ.get("PATH", "")
    if os.environ.get("ARC_VLLM_CUDA_HOME") and not os.environ.get("CUDA_HOME"):
        os.environ["CUDA_HOME"] = os.environ["ARC_VLLM_CUDA_HOME"]
        os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get("PATH", "")
    if args.eq_addon:  # EQ runs compiled (eq.engine)
        settings = dataclasses.replace(settings, compiled=True)
    app = build(settings, budget=args.budget, max_model_len=args.max_model_len, max_sessions=args.max_sessions,
                idle_seconds=args.idle_seconds, scratch_slots=args.scratch_slots,
                checkpoint_root=args.checkpoint_root, rope=args.rope, images=args.images, eq_addon=args.eq_addon)
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
