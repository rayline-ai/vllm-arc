# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""Download the engine's vLLM patch files into a directory: ``python3 deploy/fetch_patches.py DEST``.

``engine.lock`` names the patch stack's release (``patch_source <repository> <release tag> <commit>``) and each patch
file with its sha256 (``patch``). Each file is downloaded from that GitHub release and written to ``DEST/<file>`` only
when its sha256 is the lock's; anything else exits 1 and writes nothing. A private repository needs a token:
``GH_TOKEN`` or ``GITHUB_TOKEN`` in the environment, else the GitHub CLI's (``gh auth token``) when ``gh`` is on PATH.
Standard library only, so it runs before any venv exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "vllm_arc" / "engine" / "engine.lock"


def lock_patches(lock: Path = LOCK) -> tuple[tuple[str, str, str], list[tuple[str, str]]]:
    """``((repository, release tag, commit), [(sha256, file name), ...])`` from the lock, without importing the
    package (it imports nothing from vLLM, but this script runs on hosts with no venv at all)."""
    source, patches = None, []
    for raw in lock.read_text().splitlines():
        line = raw.split("#", 1)[0].split()
        if line[:1] == ["patch_source"] and len(line) == 4:
            source = tuple(line[1:])
        elif line[:1] == ["patch"] and len(line) == 3:
            patches.append((line[1], line[2]))
    if source is None or not patches:
        raise SystemExit(f"{lock}: no patch_source or patch entries")
    return source, patches


def _token() -> str | None:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token or not shutil.which("gh"):
        return token
    out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    return out.stdout.strip() or None


def _get(url: str, token: str | None, accept: str) -> bytes:
    headers = {"Accept": accept, "User-Agent": "vllm-arc-fetch-patches"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        return response.read()


def fetch(dest: Path, lock: Path = LOCK) -> list[Path]:
    (repo, tag, _commit), patches = lock_patches(lock)
    token = _token()
    release = json.loads(_get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", token,
                              "application/vnd.github+json"))
    assets = {a["name"]: a["url"] for a in release.get("assets", [])}
    blobs = []
    for sha, name in patches:
        if name not in assets:
            raise SystemExit(f"{repo} release {tag} has no asset {name}")
        data = _get(assets[name], token, "application/octet-stream")
        got = hashlib.sha256(data).hexdigest()
        if got != sha:
            raise SystemExit(f"{name}: sha256 {got}, engine.lock names {sha}")
        blobs.append((name, data))
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in blobs:
        (dest / name).write_bytes(data)
        written.append(dest / name)
    return written


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        return 2
    for path in fetch(Path(argv[1])):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
