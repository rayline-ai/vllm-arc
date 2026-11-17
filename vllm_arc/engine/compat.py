# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The engine this plugin is built for (``engine.lock`` beside this file) and the guard that refuses any other.

``require()`` runs before vLLM builds a model: at the import of ``models`` (where vLLM's registry loads
``ArcBaseForPooling``) and at the top of the server's ``main``. It imports no vLLM module. It has two levels:

- always, the installed ``vllm`` distribution's version must be the lock's ``vllm_version``;
- with ``patched=True`` (the retained session route, which needs the lock's patches), the bytes of every file the
  lock's patches change must also be the locked ones, so a missing patch, or an edited or extra patch on those files,
  is refused before any weight loads. Files the patches do not touch are not read (hashing the whole wheel at every
  load is not worth its time): the image installs the wheel by sha256, and the start-up probe
  (``startup.assertions``) still judges the running engine after it is built.

``models`` checks the patched files when the session route env (``startup.SESSIONS_ENV``) is ``retained``; the server
exports its route there before any engine is built. Any other route, or vLLM loading the architectures outside the
server, runs on unpatched vLLM of the locked release.

``VLLM_ARC_UNLOCKED=1`` (``UNLOCKED_ENV``) turns a refusal into a ``RuntimeWarning`` and marks the server's engine id
``+unqualified`` (``unlocked``). The start-up assertions are not affected by it.

``engine.lock`` holds one ``key value...`` entry per line (``#`` starts a comment):

- ``python``, ``uv``, ``vllm_version``, ``vllm_commit``, ``vllm_wheel_url``, ``vllm_wheel_sha256``, ``exclude_newer``;
- ``patch_source <repository> <release tag> <commit>``: where the vLLM patches are published;
- ``patch <sha256> <file>``: each vLLM patch, a file of the ``patch_source`` release, in the order it is applied;
- ``patched <sha256> <path under site-packages>``: each file the patches change, as it is after all of them;
- ``pin <name>==<version>``: every other distribution in the engine's venv (the full transitive set; the plugin
  itself is installed from this repository).

It is plain text so ``install.sh`` and the image recipe (``recipe.py``) read the same pins.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

LOCK_PATH = Path(__file__).with_name("engine.lock")
#: Set to ``1`` to load on a vLLM the lock refuses, with a warning; the engine id is then marked ``+unqualified``.
UNLOCKED_ENV = "VLLM_ARC_UNLOCKED"
_SINGLE = ("python", "uv", "vllm_version", "vllm_commit", "vllm_wheel_url", "vllm_wheel_sha256", "exclude_newer")


class IncompatibleEngine(RuntimeError):
    """The installed vLLM is not the release and patch set ``engine.lock`` names."""


@dataclass(frozen=True)
class EngineLock:
    values: dict
    patches: tuple  # ((sha256, file name in the patch_source release), ...) in apply order
    patched: tuple  # ((sha256, path under site-packages), ...)
    pins: tuple  # ("name==version", ...)

    def __getitem__(self, key: str):
        return self.values[key]


def read_lock(path: Path = LOCK_PATH) -> EngineLock:
    values: dict = {}
    patches, patched, pins = [], [], []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].split()
        if not line:
            continue
        key, args = line[0], line[1:]
        if key in _SINGLE and len(args) == 1 and key not in values:
            values[key] = args[0]
        elif key == "patch_source" and len(args) == 3 and key not in values:
            values[key] = tuple(args)  # (repository, release tag, commit)
        elif key in ("patch", "patched") and len(args) == 2 and len(args[0]) == 64:
            (patches if key == "patch" else patched).append((args[0], args[1]))
        elif key == "pin" and len(args) == 1 and "==" in args[0]:
            pins.append(args[0])
        else:
            raise ValueError(f"{path}:{number}: not a lock entry: {raw!r}")
    missing = [k for k in (*_SINGLE, "patch_source") if k not in values]
    if missing or not patches or not patched or not pins:
        raise ValueError(f"{path}: incomplete lock (missing {missing or 'patch/patched/pin entries'})")
    return EngineLock(values, tuple(patches), tuple(patched), tuple(pins))


def installed_vllm() -> tuple:
    """The installed ``vllm`` distribution's version and the directory that holds its package (site-packages),
    found without importing it; ``None`` for either when absent."""
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = None
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):  # ValueError: a module in sys.modules without a spec
        spec = None
    site = Path(spec.origin).parent.parent if spec is not None and spec.origin else None
    return version, site


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def problems(lock: EngineLock, version: str | None, site: Path | None, *, patched: bool = True) -> list:
    """Each way the installed vLLM differs from the lock; empty when it is the locked release (and, with
    ``patched``, the locked patch set)."""
    out = []
    if version != lock["vllm_version"]:
        out.append(f"vllm {version or 'not installed'} (locked {lock['vllm_version']})")
    if not patched:
        return out
    if site is None:
        return out + ["no importable vllm package"]
    for sha, rel in lock.patched:
        got = _sha256(site / rel)
        if got != sha:
            out.append(f"{rel}: sha256 {got or 'missing'} (locked {sha[:12]}..., after the locked patches)")
    return out


def unlocked(env=os.environ) -> bool:
    """Whether the override (``UNLOCKED_ENV=1``) is set: refusals warn instead, and the engine id is unqualified."""
    return env.get(UNLOCKED_ENV) == "1"


def require(lock: EngineLock | None = None, *, patched: bool = True, env=os.environ) -> list:
    """Raise ``IncompatibleEngine`` unless the installed vLLM is the locked release and, with ``patched``, carries
    exactly the locked patches. Under the override (``unlocked``) the refusal is a ``RuntimeWarning`` instead; the
    problems it waived are returned (empty when there are none)."""
    lock = lock or read_lock()
    found = problems(lock, *installed_vllm(), patched=patched)
    if not found:
        return []
    names = ", ".join(Path(p).name for _, p in lock.patches)
    built = f"vllm_arc.engine is built for vLLM {lock['vllm_version']}" + (f" with {names}" if patched else "")
    detail = "; ".join(found)
    if not unlocked(env):
        raise IncompatibleEngine(f"{built} (vllm_arc/engine/engine.lock), refusing: {detail} "
                                 f"({UNLOCKED_ENV}=1 loads it anyway, as an unqualified engine)")
    warnings.warn(f"{built} (vllm_arc/engine/engine.lock); {UNLOCKED_ENV}=1, loading an unqualified engine: {detail}",
                  RuntimeWarning, stacklevel=2)
    return found


def retained_route(env=os.environ) -> bool:
    """Whether the session route env selects the retained route, the one that needs the lock's patches. Unset is
    not retained: the server always exports its route before an engine is built."""
    from .startup import SESSIONS_ENV

    return env.get(SESSIONS_ENV) == "retained"
