"""An ARC release's Stage-1 head, loaded from the release artifact itself and run in the vLLM process.

The release is the only input: ``arc_release.json`` (the manifest, which pins every file's sha256), ``config.json``
(``arc.stage_one``: the compiled head's candidates, regimes and selector id) and the safetensors weights, whose
``stage_one.*`` tensors are the head. No pickle, NumPy selector or training-SDK import is read here: the head runs in
the vLLM process's own environment. The head's code is ``arc_heads.py``, the same torch-only module the ARC training
SDK (not yet public) uses to compile and verify a release, so training-side verification and serving run one
implementation.

    head = load_stage_one(release_dir, device="cuda")
    p, usage = head(readout_rows, "fixed", context=request_features)   # float64, [M, n, C] and [M, n, C, 4]
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .arc_heads import stage_one_module

MANIFEST = "arc_release.json"
CONFIG = "config.json"
WEIGHTS = "model.safetensors"
PREFIX = "stage_one."


class ReleaseRefused(ValueError):
    """The release directory is not one consistent ARC release, or carries no Stage-1 head."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class StageOneHead:
    """The compiled head and what it scores: row 0 of each output is the point (an ensemble's calibrated member
    mean), the rest its draws, in ``candidates`` order; NaN is unknown, as the selector's ``predict`` returns it."""

    module: torch.nn.Module
    release: str
    selector_id: str
    candidates: list[dict]
    regimes: list[str]
    reads_context: bool
    device: torch.device
    #: The usage head reads warm-cache features (``w_cache``): ``cache`` [n, models, 6] is then required.
    reads_cache: bool = False

    def __call__(self, x: torch.Tensor, regimes: str | Sequence[str], context: torch.Tensor | None = None,
                 cache: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        names = [regimes] * len(x) if isinstance(regimes, str) else list(regimes)
        unknown = sorted(set(names) - set(self.regimes))
        if unknown:
            raise ValueError(f"regimes {unknown} are not this head's ({self.regimes})")
        if (context is not None) != self.reads_context:
            raise ValueError("this head reads request context features: pass context" if self.reads_context
                             else "this head reads no request context features")
        if (cache is not None) != self.reads_cache:
            raise ValueError("this head reads warm-cache features: pass cache" if self.reads_cache
                             else "this head reads no warm-cache features")
        index = torch.tensor([self.regimes.index(r) for r in names], dtype=torch.int64, device=self.device)
        with torch.no_grad():
            return self.module(x.to(self.device, torch.float64), index,
                               None if context is None else context.to(self.device, torch.float64),
                               None if cache is None else cache.to(self.device, torch.float64))


def load_stage_one(release_dir: str | Path, *, device: str | torch.device = "cpu") -> StageOneHead:
    """The release's Stage-1 head on ``device``: refused unless the manifest's sha256 for ``config.json`` and the
    weights match the files, the config names the manifest's release, and the weights carry a ``stage_one.*`` head."""
    from safetensors.torch import load_file

    release_dir = Path(release_dir)
    manifest = json.loads((release_dir / MANIFEST).read_text())
    for name in (CONFIG, WEIGHTS):
        pinned = manifest.get("files", {}).get(name)
        if pinned is None or _sha256(release_dir / name) != pinned:
            raise ReleaseRefused(f"{name} is not the file {MANIFEST} pins")
    arc = json.loads((release_dir / CONFIG).read_text()).get("arc") or {}
    if arc.get("release") != manifest.get("name") or arc.get("package_sha256") != manifest.get("package_sha256"):
        raise ReleaseRefused(f"{CONFIG} and {MANIFEST} name different releases")
    meta = arc.get("stage_one")
    tensors = {k.removeprefix(PREFIX): v for k, v in load_file(str(release_dir / WEIGHTS)).items()
               if k.startswith(PREFIX)}
    if meta is None or not tensors:
        raise ReleaseRefused("the release carries no Stage-1 head")
    device = torch.device(device)
    module = stage_one_module({k: v.to(device) for k, v in tensors.items() if k != "auxiliary"})
    return StageOneHead(module=module, release=manifest["name"], selector_id=meta["selector_id"],
                        candidates=list(meta["candidates"]), regimes=list(meta["regimes"]),
                        reads_context="ctx_mean" in tensors, device=device, reads_cache="w_cache" in tensors)


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m vllm_arc.heads RELEASE IN.npz OUT.npz [--device cuda]``: run the release's Stage-1 head on recorded
    inputs (``x_<k>``, ``regime_<k>``, optional ``context_<k>`` and ``cache_<k>``) in this process's torch, writing ``p_<k>`` /
    ``usage_<k>``; prints the device and torch version as JSON. The parity harness drives it in the vLLM env."""
    import argparse
    import sys

    import numpy as np

    parser = argparse.ArgumentParser(prog="vllm_arc.heads")
    parser.add_argument("release")
    parser.add_argument("inputs")
    parser.add_argument("outputs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace", action="store_true",
                        help="also run the first call on cpu and --device with the head's stages instrumented, and "
                             "print where their unknown (NaN) counts diverge")
    args = parser.parse_args(argv)
    head = load_stage_one(args.release, device=args.device)
    calls = np.load(args.inputs, allow_pickle=False)
    out = {}
    for name in sorted(n for n in calls.files if n.startswith("x_")):
        k = name.removeprefix("x_")
        context = calls[f"context_{k}"] if f"context_{k}" in calls.files else None
        cache = calls[f"cache_{k}"] if f"cache_{k}" in calls.files else None
        p, usage = head(torch.from_numpy(calls[name]), [str(r) for r in calls[f"regime_{k}"]],
                        None if context is None else torch.from_numpy(context),
                        None if cache is None else torch.from_numpy(cache))
        out[f"p_{k}"], out[f"usage_{k}"] = p.cpu().numpy(), usage.cpu().numpy()
    np.savez(args.outputs, **out)
    if args.trace and out:
        first = sorted(n for n in calls.files if n.startswith("x_"))[0].removeprefix("x_")
        context = calls[f"context_{first}"] if f"context_{first}" in calls.files else None
        cache = calls[f"cache_{first}"] if f"cache_{first}" in calls.files else None
        print(json.dumps({"trace": {device: _trace(args.release, device, calls[f"x_{first}"],
                                                     [str(r) for r in calls[f"regime_{first}"]], context, cache)
                                    for device in ("cpu", args.device)}}))
    device = torch.device(args.device)
    print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "device": str(device),
                      "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                      "calls": len(out) // 2, "release": head.release, "selector_id": head.selector_id}))


def _trace(release_dir: str, device: str, x, regimes: list[str], context, cache=None) -> list:  # noqa: ANN001
    """The head's stages on ``device``, each stage's input and output unknown counts and their sum of finite values
    (``arc_heads``' stage functions, wrapped for the one call)."""
    from . import arc_heads

    log: list = []
    originals = {name: getattr(arc_heads, name) for name in ("_platt", "apply_shrink")}

    def stats(t: torch.Tensor) -> dict:
        finite = torch.isfinite(t)
        return {"shape": list(t.shape), "nan": int(torch.isnan(t).sum()), "inf": int(torch.isinf(t).sum()),
                "sum": float(torch.where(finite, t, torch.zeros_like(t)).sum())}

    def wrap(name: str):  # noqa: ANN202
        def run(*args):  # noqa: ANN002, ANN202
            out = originals[name](*args)
            log.append({"stage": name, "in": [stats(a) for a in args if isinstance(a, torch.Tensor)], "out": stats(out)})
            return out
        return run

    try:
        for name in originals:
            setattr(arc_heads, name, wrap(name))
        head = load_stage_one(release_dir, device=device)
        p, usage = head(torch.from_numpy(x), regimes, None if context is None else torch.from_numpy(context),
                        None if cache is None else torch.from_numpy(cache))
        log.append({"stage": "output", "p": stats(p), "usage": stats(usage)})
    finally:
        for name, value in originals.items():
            setattr(arc_heads, name, value)
    return log


if __name__ == "__main__":
    main()
