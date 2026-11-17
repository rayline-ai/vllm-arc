# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The EQ engine's checkpoint: the pinned base at 24 layers with a question-head add-on's layers 12-23, verified first.

Runs in the vLLM environment (torch, safetensors, transformers), beside ``vllm_arc.server``.
An add-on is a directory holding:

- ``upper.safetensors``: exactly the ``layers.<cut>..<depth-1>.*`` tensors, keyed ``layers.<i>.<name>``;
- ``pointer.safetensors``: the pointer head, used by the client, not by the engine (only its hashes are checked here);
- ``manifest.json``: schema ``rayline.arc-question-head-addon.v1``, every tensor's shape, dtype and sha256, the base
  model, revision, RoPE and cut, and the trunk ``lower_digest`` the add-on was trained against.

`verify` refuses (``AddonRefused``, one named reason) unless every manifest refusal clears; `checkpoint` then writes
the served checkpoint once, complete or absent. The trunk check (`trunk_digest`) recomputes ``lower_digest`` over the
checkpoint's token embedding, layers 0..cut-1 and final norm as the manifest defines it, so a trunk that differs from
the one the add-on was trained on by a single byte refuses.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

SCHEMA = "rayline.arc-question-head-addon.v1"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DEPTH = 24
#: The served model's tensor prefix for the text stack's layers (the base's safetensors keys).
LAYER_KEY = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
#: The add-on's own key form (``layers.<i>.<name>``).
ADDON_KEY = re.compile(r"^layers\.(\d+)\.(.+)$")


#: ``ARC_EQ_ADDON`` naming a published ARC release that declares a question head (``arc_release.json``
#: ``question_head``), by an immutable tag: a dated tag on a line repo
#: (``<org>/arc-1.0-private@20261108``, release ``arc-1.0-20261108``; ``<org>/arc-1.0`` and
#: ``<org>/ARC-1.0`` read the same: the hub ignores case) or a
#: release-named tag on any repo (``<org>/<repo>@arc-1.0-20261108``).
RELEASE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*/(?:(?i:arc)-(?P<line>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))(?i:-private)?"
                         r"@(?P<day>\d{8})"
                         r"|[A-Za-z0-9][A-Za-z0-9._-]*@(?P<name>arc-(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)-\d{8}))$")


class AddonRefused(ValueError):
    """The add-on does not load: ``reason`` is one stable code, ``detail`` says what differed (no tensor data)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


def resolve(addon: str) -> str:
    """The local add-on directory ``ARC_EQ_ADDON`` names: a directory as given, or, for a release ``<repo>@<tag>``
    (`RELEASE_REF`), that release's declared ``question_head`` directory, downloaded once into the HF cache (only
    ``arc_release.json`` and ``question_head/``). The tag must hold the release it names. Its tensors are then
    verified as any add-on's are (`verify`), so nothing here hashes them. Refuses ``release_unreadable`` or
    ``release_has_no_question_head``."""
    found = RELEASE_REF.match(addon)
    if found is None:
        return addon
    repo, _, tag = addon.partition("@")
    expected = found["name"] or f"arc-{found['line']}-{found['day']}"
    try:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(repo, revision=tag, repo_type="model",
                                      allow_patterns=["arc_release.json", "question_head/*", "question_head/**"]))
        release = json.loads((root / "arc_release.json").read_text())
    except Exception as exc:  # noqa: BLE001 -- any hub or file failure is one refusal, named by type
        raise AddonRefused("release_unreadable", f"{addon}: {type(exc).__name__}") from None
    if release.get("name") != expected:
        raise AddonRefused("release_unreadable", f"{addon} holds release {release.get('name')!r}")
    declared = release.get("question_head")
    if not declared:
        raise AddonRefused("release_has_no_question_head", addon)
    return str(root / declared["path"])


def tensor_sha256(tensor: Any) -> str:
    """sha256 of a tensor's bytes, as the add-on builder computes it."""
    import torch

    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def read_manifest(addon: str | Path) -> dict:
    path = Path(addon) / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise AddonRefused("manifest_unreadable", type(exc).__name__) from None
    if manifest.get("schema") != SCHEMA:
        raise AddonRefused("schema_differs", str(manifest.get("schema")))
    return manifest


def _check_tensors(tensors: dict, declared: dict, what: str) -> None:
    missing, extra = sorted(set(declared) - set(tensors)), sorted(set(tensors) - set(declared))
    if missing or extra:
        raise AddonRefused(f"{what}_tensor_set_differs", f"missing {missing[:4]}, extra {extra[:4]}")
    for name, spec in declared.items():
        t = tensors[name]
        if list(t.shape) != list(spec["shape"]) or str(t.dtype) != spec["dtype"]:
            raise AddonRefused(f"{what}_tensor_misshapen", name)
        if tensor_sha256(t) != spec["sha256"]:
            raise AddonRefused(f"{what}_tensor_sha256_differs", name)


def verify(addon: str | Path, *, rope: str, cut: int = 12, depth: int = DEPTH) -> dict:
    """The add-on's manifest once its files match it and it is the served base, RoPE, cut and depth.

    Refuses ``base_differs``, ``rope_differs``, ``cut_differs``, ``depth_differs``, ``upper_below_cut`` /
    ``upper_outside_layers``, ``upper_tensor_set_differs`` / ``_misshapen`` / ``_sha256_differs`` and the same for
    ``pointer``. The trunk is checked against the built checkpoint (`checkpoint`), not here."""
    from safetensors.torch import load_file

    manifest = read_manifest(addon)
    base = manifest.get("base_model") or {}
    if (base.get("model_id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
        raise AddonRefused("base_differs", f"{base.get('model_id')}@{str(base.get('revision'))[:12]}")
    if base.get("rope") != rope:
        raise AddonRefused("rope_differs", f"add-on {base.get('rope')!r}, served {rope!r}")
    if manifest.get("cut") != cut:
        raise AddonRefused("cut_differs", str(manifest.get("cut")))
    if manifest.get("depth") != depth:
        raise AddonRefused("depth_differs", str(manifest.get("depth")))
    for name in manifest["upper_tensors"]:
        m = ADDON_KEY.match(name)
        if m is None or not cut <= int(m.group(1)) < depth:
            raise AddonRefused("upper_below_cut" if m and int(m.group(1)) < cut else "upper_outside_layers", name)
    try:
        upper = load_file(str(Path(addon) / "upper.safetensors"))
        pointer = load_file(str(Path(addon) / "pointer.safetensors"))
    except (OSError, ValueError, RuntimeError) as exc:
        raise AddonRefused("tensors_unreadable", type(exc).__name__) from None
    _check_tensors(upper, manifest["upper_tensors"], "upper")
    _check_tensors(pointer, (manifest.get("pointer_head") or {}).get("tensors") or {}, "pointer")
    return manifest


def trunk_digest(model_dir: str | Path, cut: int = 12, dtype: str = "bfloat16") -> str:
    """The manifest's ``lower_digest`` over the text model: loaded with ``AutoModel`` in float32 (its
    ``language_model``), then every floating parameter and every floating buffer but ``inv_freq`` cast to ``dtype``
    (bf16, the checkpoint's own precision; the trunk is all frozen). The token embedding, layers ``[0, cut)`` and the final norm, parameters and
    buffers by sorted name, each as ``name:shape:dtype`` then its bytes."""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32)
    model = getattr(model, "language_model", model)
    cast = getattr(torch, dtype)
    for module in model.modules():
        for _, p in module.named_parameters(recurse=False):
            if p.dtype.is_floating_point:
                p.data = p.data.to(cast)
        for name, b in list(module.named_buffers(recurse=False)):
            if b.dtype.is_floating_point and name != "inv_freq":
                module._buffers[name] = b.to(cast)
    h = hashlib.sha256()
    named = [("embed_tokens", model.embed_tokens), ("norm", model.norm)]
    named += [(f"layers.{i}", model.layers[i]) for i in range(cut)]
    for prefix, module in named:
        tensors = dict(module.named_parameters()) | dict(module.named_buffers())
        for name in sorted(tensors):
            stem = name.rsplit(".", 1)[0]
            if name.endswith((".a", ".b")) and f"{stem}.base.weight" in tensors:
                continue
            t = tensors[name].detach().contiguous().cpu()
            h.update(f"{prefix}.{name.replace('.base.', '.')}:{tuple(t.shape)}:{t.dtype}".encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    del model
    return h.hexdigest()


def _base_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))


def checkpoint(addon: str | Path, root: str | Path, *, rope_parameters: dict | None, rope: str,
               readout_depths: list[int], cut: int = 12, digest_dtype: str = "bfloat16",
               config_hook: Any = None, name_suffix: str = "") -> tuple[str, dict]:
    """``(path, manifest)``: the served EQ checkpoint for a verified add-on, built once under ``root``.

    The base's text stack at all 24 layers with ``layers.<cut>..23`` replaced by the add-on's tensors, cast to the
    base tensor's dtype (the engine serves bf16; G2 of the plan bounds the cast). The text config gets the served
    ``rope_parameters`` and ``readout_depths`` (the fork overlay's multi-depth readout). Before the directory is
    published, `trunk_digest` over it must equal the manifest's ``trunk_lower_digest`` (``trunk_differs``); the
    directory name carries the add-on's name and the upper tensors' digest, so two add-ons never share it.

    ``config_hook`` (another engine's config edits: the stock engine's architecture and positions) is called on the
    config last, and ``name_suffix`` keeps that checkpoint's directory apart; both unset build the fork engine's."""
    from safetensors.torch import load_file, save_file

    manifest = verify(addon, rope=rope, cut=cut)
    upper_digest = hashlib.sha256(json.dumps(
        {k: v["sha256"] for k, v in sorted(manifest["upper_tensors"].items())}).encode()).hexdigest()
    target = Path(root) / f"eq-{manifest['name']}-{upper_digest[:12]}-{rope or 'native'}{name_suffix}"
    expected = {"addon": manifest["name"], "trunk_lower_digest": manifest["base_model"]["trunk_lower_digest"],
                "upper_digest": upper_digest, "digest_dtype": digest_dtype, "readout_depths": list(readout_depths)}
    if (target / "config.json").exists():
        try:  # reused only when it was verified against exactly this manifest and these build parameters
            if json.loads((target / "eq_verified.json").read_text()) == expected:
                return str(target), manifest
        except (OSError, ValueError):
            pass
        shutil.rmtree(target, ignore_errors=True)
    source = _base_snapshot()
    staging = target.with_name(target.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    upper = load_file(str(Path(addon) / "upper.safetensors"))
    replaced: set[str] = set()
    index: dict = {"metadata": {}, "weight_map": {}}
    for shard in sorted(source.glob("*.safetensors")):
        tensors = {}
        for key, base_value in load_file(shard).items():
            if key.startswith("mtp."):
                continue
            m = LAYER_KEY.match(key)
            if m and int(m.group(1)) >= cut:
                name = f"layers.{m.group(1)}.{m.group(2)}"
                if name not in upper:
                    raise AddonRefused("upper_tensor_set_differs", f"the base has {name}, the add-on does not")
                tensors[key] = upper[name].to(base_value.dtype).contiguous()
                replaced.add(name)
            else:
                tensors[key] = base_value
        if tensors:
            save_file(tensors, staging / shard.name, metadata={"format": "pt"})
            index["weight_map"].update({k: shard.name for k in tensors})
    if replaced != set(upper):
        raise AddonRefused("upper_tensor_set_differs", f"not in the base: {sorted(set(upper) - replaced)[:4]}")
    (staging / "model.safetensors.index.json").write_text(json.dumps(index))
    for path in source.iterdir():
        if path.suffix != ".safetensors" and path.name not in ("model.safetensors.index.json", "config.json"):
            shutil.copy(path, staging / path.name)
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", config)
    if text["num_hidden_layers"] != DEPTH:
        raise AddonRefused("depth_differs", f"base has {text['num_hidden_layers']} layers")
    text.pop("mtp_num_hidden_layers", None)
    if rope_parameters:
        text["rope_parameters"] = dict(rope_parameters)
    text["readout_depths"] = list(readout_depths)
    if config_hook is not None:
        config_hook(config)
    (staging / "config.json").write_text(json.dumps(config, indent=1))
    got = trunk_digest(staging, cut, digest_dtype)
    if got != manifest["base_model"]["trunk_lower_digest"]:
        shutil.rmtree(staging, ignore_errors=True)
        raise AddonRefused("trunk_differs", f"{got[:12]} != {manifest['base_model']['trunk_lower_digest'][:12]}")
    (staging / "eq_verified.json").write_text(json.dumps(expected))
    staging.rename(target)
    return str(target), manifest
