# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""Compose the plugin with vLLM and the fork's patch on CPU: ``python ci/compose_check.py [--latest] WORKDIR``.

Default (the locked engine): download engine.lock's vLLM wheel (checked by sha256) and the patch release
(``deploy/fetch_patches.py``), unpack the wheel with no dependencies, run the recipe's own patch steps (sha256 check,
``git apply``) on it, and run ``compat.require()`` against the result: the patch applies to the locked wheel and
leaves exactly the files the lock names. vLLM is not imported (its dependencies are not installed), so this needs no
GPU and no torch.

``--latest`` (the weekly upstream canary): the newest vLLM release on PyPI instead, with ``git apply --check`` only. A
failure means the patch stack needs rebasing onto that release (a new ``arc/v<version>`` branch in the fork); it is a
signal, not a defect of this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _download(url: str, dest: Path, sha256: str | None = None) -> Path:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "vllm-arc-ci"}), timeout=600) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if sha256 and got != sha256:
        raise SystemExit(f"{url}: sha256 {got}, expected {sha256}")
    dest.write_bytes(data)
    return dest


def _latest_wheel() -> tuple[str, str, str]:
    with urllib.request.urlopen("https://pypi.org/pypi/vllm/json", timeout=60) as r:
        meta = json.load(r)
    version = meta["info"]["version"]
    for f in meta["urls"]:
        if f["packagetype"] == "bdist_wheel" and "manylinux" in f["filename"] and "x86_64" in f["filename"]:
            return version, f["url"], f["digests"]["sha256"]
    raise SystemExit(f"vllm {version}: no manylinux x86_64 wheel on PyPI")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "deploy"))
    recipe = importlib.import_module("vllm_arc.engine.recipe")
    fetch_patches = importlib.import_module("fetch_patches")

    work = args.workdir.resolve()
    site, patch_dir = work / "site", work / "patches"
    site.mkdir(parents=True)
    fetch_patches.fetch(patch_dir)
    lock = recipe.LOCK
    if args.latest:
        version, url, sha = _latest_wheel()
    else:
        version, url, sha = lock["vllm_version"], lock["vllm_wheel_url"], lock["vllm_wheel_sha256"]
    print(f"vllm {version}: {url}")
    with zipfile.ZipFile(_download(url, work / Path(url).name, sha)) as z:
        z.extractall(site)

    # site is its own repository, so git apply never resolves paths against an enclosing checkout (a WORKDIR inside
    # one would otherwise make it skip every path silently and succeed).
    subprocess.run(["git", "init", "-q", str(site)], check=True)
    git_apply = ["git", "-C", str(site), "apply", "-p1"] + (["--check"] if args.latest else [])
    for _sha, name in lock.patches:
        done = subprocess.run(git_apply + [str(patch_dir / name)], capture_output=True, text=True)
        if done.returncode:
            print(done.stderr, file=sys.stderr)
            print(f"{name} does not apply to vllm {version}", file=sys.stderr)
            return 1
        print(f"{name}: applies to vllm {version}")
    if args.latest:
        return 0

    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    compat = importlib.import_module("vllm_arc.engine.compat")
    compat.require()
    print("compat.require: ok (the patched wheel is the locked engine)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
