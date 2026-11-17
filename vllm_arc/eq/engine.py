# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""EQ on the stock engine: ``ARC_EQ_ADDON`` (``--eq-addon``) serves the 24-layer ``ArcEqForPooling`` engine with a
question-head add-on, ``/encode`` (``norm(h_12)``) and ``/branches`` (``norm(h_24)`` at suffix positions), as
``vllm_arc.server`` does on the fork.

The encodes are the fork server's own (``vllm_arc.server.eq.EqEncodes``: admission, the prefix-first branches, the
counters) over this engine; only the prompt and the pooling params are the stock engine's (``cache_salt`` and
``GatherPooler``'s ``arc_gather`` offsets). The add-on is verified as on the fork (``vllm_arc.eq.addon``) and
its checkpoint written with the stock engine's architecture and positions (``server.stock_config``); an add-on that
refuses serves the base at 24 layers, every question ``question_head_unavailable``.

The engine runs the prefix route (``Engine(sessions="prefix")``): prefix caching in ``align`` mode, sha256 block
hashes, no retention. The fork capped a prefix hit at a request's smallest readout offset in an overlay
(``kv_cache_manager.py``); stock vLLM has no such cap, so a hit past a readout (``EqEncodes``' ``hit_over_readout``)
is answered by encoding that request again under a fresh salt (prefix caching cannot serve it then), which is
exact. Text only: EQ refuses ``--images``.

EQ runs compiled (``Engine.compiled``: torch.compile with Inductor, custom ops off, piecewise CUDA graphs), as the
fork EQ did; E12 and the image engine run eager. Compiled, the norms are Inductor's native (fused) arithmetic rather
than the eager CUDA kernels, so EQ's ``h_12`` is not E12's readout bit for bit; the EQ parity gate measures it.
"""
from __future__ import annotations

import os
from typing import Any

from vllm_arc import EQ_ARCHITECTURE

from ..engine.startup import Engine, _check

#: EQ's shared settings and encodes (no vLLM import).
from . import encodes as fork_eq  # noqa: E402

ARCHITECTURE = EQ_ARCHITECTURE
DEPTH = 24
READOUT_DEPTHS = tuple(fork_eq.READOUT_DEPTHS)  # (12, 24)
HIDDEN = fork_eq.HIDDEN


def engine(unit: int = 0) -> Engine:
    """EQ's engine settings: the prefix route on hit grid ``unit``, compiled (torch.compile and piecewise CUDA
    graphs, ``startup.COMPILATION``, as the fork EQ ran)."""
    return Engine(unit=unit, sessions="prefix", compiled=True)


def checkpoint(addon: str, root: str, rope: str) -> tuple[str, dict]:
    """``(checkpoint, add-on status)``: the verified add-on's checkpoint, or, when it refuses, the base at 24 layers
    with the same ``readout_depths`` (as the fork server's ``eq_checkpoint``)."""
    import json

    from vllm_arc.server import app as server

    from . import addon as eq_addon

    try:
        path, manifest = eq_addon.checkpoint(
            eq_addon.resolve(addon), root, rope_parameters=server.ROPE_VARIANTS.get(rope), rope=rope,
            readout_depths=list(READOUT_DEPTHS), config_hook=lambda c: server.stock_config(c, rope, ARCHITECTURE),
            name_suffix=f"-{ARCHITECTURE}")
        return path, {"name": manifest["name"], "verified": True, "reason": None}
    except eq_addon.AddonRefused as exc:
        print(json.dumps({"event": "arc_eq_addon_refused", "reason": exc.reason, "detail": exc.detail[:300]}),
              flush=True)
        return (server.truncated_checkpoint(root, rope, depth=DEPTH, arch=ARCHITECTURE, readout_depths=READOUT_DEPTHS),
                {"name": None, "verified": False, "reason": exc.reason})


def addon_dir(addon: str) -> str:
    """``ARC_EQ_ADDON`` as a local directory (a release ``<repo>@<tag>`` resolved), as the identity names it."""
    from . import addon as eq_addon

    try:
        return eq_addon.resolve(addon)
    except eq_addon.AddonRefused:
        return addon


def assertions(probe: dict) -> list[dict]:
    """EQ's start-up assertions beside the prefix route's: the model reads every readout depth through the pair path
    (``ArcEqModel``), and the forward vLLM compiled is that pair path (a model built as ``Qwen3_5Model`` and given
    ``ArcEqModel``'s class afterwards would compile ``Qwen3_5Model.forward``: ``h_24`` alone)."""
    return [_check("eq_readout_depths", [probe.get("inner_model"), probe.get("readout_depths")],
                   ["ArcEqModel", list(READOUT_DEPTHS)],
                   probe.get("inner_model") == "ArcEqModel" and probe.get("readout_depths") == list(READOUT_DEPTHS)),
            _check("eq_compiled_forward", probe.get("compiled_forward"), "ArcEqModel.forward",
                   probe.get("compiled_forward") == "ArcEqModel.forward")]


def identity(identity: dict, addon: str, addon_status: dict, cache: dict) -> dict:
    """EQ's identity: ``/encode`` answers E12's readout (``norm(h_12)``), so it is the stock id plus ``+eq-<add-on
    directory>`` (a distinct host), with an ``eq`` block (as the fork server's ``eq_identity``)."""
    name = os.path.basename(os.path.normpath(addon))
    return {**identity, "engine": identity["engine"] + f"+eq-{name}",
            "eq": {"addon_dir": name, "readout_depths": list(READOUT_DEPTHS), "engine_depth": DEPTH,
                   "addon": addon_status, "block_size": cache.get("block_size"),
                   "prefix_match_unit": cache.get("cache_hit_alignment_tokens") or cache.get("block_size"),
                   "mamba_cache_mode": cache.get("mamba_cache_mode"),
                   "mamba_block_size": cache.get("mamba_block_size")}}


def encodes(engine_client: Any, pool: Any, addon_status: dict, unit: int, *, prompt, params) -> Any:
    """``EqEncodes`` over a started engine; ``prompt(tokens, images)`` and ``params(offsets, skip)`` are the stock
    server's. Settings ``ARC_EQ_INFLIGHT`` (8) and ``ARC_EQ_ADMIT_WAIT_S`` (0.25), as on the fork."""

    def eq_prompt(tokens, images, salt):
        return {**prompt(tokens, images), "cache_salt": salt}

    def eq_params(offsets, images=None):
        return params(offsets, False)

    return fork_eq.EqEncodes(
        engine_client, prompt=eq_prompt, params=eq_params, max_model_len=pool.max_model_len, unit=unit,
        budget_tokens=int(pool.blocks * pool.block_size * 0.9),
        max_inflight=int(os.environ.get("ARC_EQ_INFLIGHT") or 8),
        admit_wait_s=float(os.environ.get("ARC_EQ_ADMIT_WAIT_S") or 0.25), addon=addon_status)


async def encode_request(state: dict, body: Any) -> tuple[int, Any]:
    """One EQ ``/encode``: a one-shot (``retain: false``), optionally under the caller's ``cache_salt``; ``(200,
    rows)`` (``[len(offsets), HIDDEN]`` float32, ``norm(h_12)``) or ``(status, error body)``."""
    if not isinstance(body, dict):
        return 422, {"error": "invalid_body"}
    tokens, offsets = body.get("tokens"), body.get("offsets")
    if (not isinstance(tokens, list) or not isinstance(offsets, list) or not all(type(o) is int for o in offsets)
            or not all(type(t) is int for t in tokens)):
        return 422, {"error": "invalid_body"}
    if body.get("retain", True) is not False or body.get("base", 0) != 0:
        return 422, {"error": "retain_not_served"}  # EQ is stateless: every encode is a one-shot
    if body.get("images"):
        return 422, {"error": "images_not_served"}
    if not tokens or not offsets or any(not 0 <= o < len(tokens) for o in offsets):
        return 422, {"error": "offsets_out_of_range"}
    if len(tokens) > state["eq"].max_model_len:  # as the E12 one-shot refuses it (sessions.py), not vLLM's 500
        return 422, {"error": "exceeds_kv_budget", "tokens": len(tokens), "budget_tokens": state["eq"].max_model_len}
    salt = body.get("cache_salt")
    if salt is not None and (not isinstance(salt, str) or not salt):
        return 422, {"error": "invalid_cache_salt"}
    try:
        rows = await state["eq"].encode(tokens, offsets, salt=salt)
    except fork_eq.Refused as exc:
        return exc.status, {"error": exc.code, **exc.detail}
    return 200, rows


async def branches_request(state: dict, body: Any, *, timing: dict | None = None) -> tuple[int, Any]:
    """One ``/branches``: ``(200, (rows per suffix, cached tokens per branch, misses))`` or ``(status, error body)``;
    the fork server's validation and refusals."""
    if not isinstance(body, dict):
        return 422, {"error": "invalid_body"}
    prefix, suffixes, salt = body.get("tokens"), body.get("suffixes"), body.get("cache_salt")
    if (not isinstance(prefix, list) or not isinstance(suffixes, list) or not suffixes
            or not all(isinstance(x, dict) and isinstance(x.get("tokens"), list) and isinstance(x.get("offsets"), list)
                       for x in suffixes)
            or not all(type(t) is int for t in prefix)
            or not all(type(v) is int for x in suffixes for v in [*x["tokens"], *x["offsets"]])):
        return 422, {"error": "invalid_body"}
    if salt is not None and (not isinstance(salt, str) or not salt):
        return 422, {"error": "invalid_cache_salt"}
    try:
        out = await state["eq"].branches(prefix, [(x["tokens"], x["offsets"]) for x in suffixes], salt=salt,
                                         timing=timing)
    except fork_eq.Refused as exc:
        return exc.status, {"error": exc.code, **exc.detail}
    return 200, out
