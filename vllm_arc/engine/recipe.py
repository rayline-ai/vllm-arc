# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The stock-engine image as plain data: stock ``vllm==0.33.0`` (PyPI; its ``torch==2.14.0`` is the CUDA 13 build)
plus ``PATCHES`` applied to the installed wheel plus this plugin. No fork, no overlay. A deployment builds its image
from ``recipe()`` in the order ``README.md`` gives; ``install.sh`` runs the same commands on a host. Every version is
``engine.lock``'s (``compat.read_lock``): the vLLM wheel by URL and sha256 and every other distribution of the venv
by exact version, installed without resolution (``--no-deps``). Nothing here imports a cloud SDK."""
from __future__ import annotations

from .compat import read_lock
from .startup import VLLM_COMMIT

LOCK = read_lock()
#: The encoder base image and uv, as the fork image's, so only vLLM and the plugin differ from it.
BASE_IMAGE = "python:3.13.14-slim-bookworm@sha256:67a1e1f215ccda113cfc024e8639049257e88f273898f595b61476d128d387e8"
UV_VERSION = LOCK["uv"]
PYTHON = "3.13"
VENV = "/opt/vllm-venv"
SITE = f"{VENV}/lib/python{PYTHON}/site-packages"
PLUGIN_DIR = "/opt/vllm-arc"
#: ``uv pip freeze`` of the built venv; ``/health`` reports its sha256 (``FREEZE_ENV`` names it in the container).
FREEZE = "/opt/vllm-arc-freeze.txt"
FREEZE_ENV = "ARC_STOCK_FREEZE"
HF_CACHE = "/cache/huggingface"
#: The release's own x86_64 wheel on PyPI, by URL and sha256 (uv verifies the hash; ``/health`` reports it).
VLLM_WHEEL = LOCK["vllm_wheel_url"]
VLLM_WHEEL_SHA256 = LOCK["vllm_wheel_sha256"]
#: Release-age floor (dependency policy: >= 7 days), the date the locked set was resolved under (2026-11-11). With
#: every version pinned it also keeps each pin to the files PyPI held then: a file uploaded later is never chosen.
EXCLUDE_NEWER = LOCK["exclude_newer"]
#: Every other distribution of the venv, exact (``engine.lock`` ``pin``): torch 2.14.0, FlashInfer, Triton, FlashInfer's
#: JIT toolchain (the CUDA 13.0 nvcc and cccl torch 2.14.0 depends on) and ninja included.
PINS = LOCK.pins
#: vLLM patch files applied to the installed wheel, in order (file names in ``patch_source``'s release; ``a/vllm/...``
#: ``b/vllm/...``). Not in this repository: engine.lock pins each by sha256, so the engine id's digest covers them.
PATCHES = tuple(name for _, name in LOCK.patches)
#: Where the image holds the patch files; a deployment copies each there (``recipe()["patches"]``) before
#: ``plugin_commands`` run. ``deploy/fetch_patches.py`` downloads them from ``engine.lock``'s ``patch_source``.
PATCH_DIR = "/opt/vllm-arc-patches"
PATCH_SOURCE = LOCK["patch_source"]


def recipe(hf_cache: str = HF_CACHE, *, venv: str = VENV, plugin_dir: str = PLUGIN_DIR, freeze: str = FREEZE,
           patch_dir: str = PATCH_DIR) -> dict:
    """The image's steps; ``venv``, ``plugin_dir``, ``freeze`` and ``patch_dir`` move them for ``install.sh``
    (defaults: the image)."""
    py = f"{venv}/bin/python"
    site = f"{venv}/lib/python{PYTHON}/site-packages"
    return {
        "base_image": BASE_IMAGE,
        # Triton builds its launcher with a C compiler at engine start.
        "apt": ["git", "gcc", "libc6-dev"],
        "commands": [
            f"pip install --no-cache-dir uv=={UV_VERSION}",
            # --clear: a base image that already has a venv here gets a fresh one, .pth files included.
            f"uv venv --clear {venv} --python {PYTHON}",
            # PyPI only, no extra index: the release's own wheel (published 2026-11-02), hash-checked, and the locked
            # set exactly, without resolution.
            f"uv pip install --python {py} --no-config --no-deps --exclude-newer {EXCLUDE_NEWER} "
            f"'vllm @ {VLLM_WHEEL}#sha256={VLLM_WHEEL_SHA256}' {' '.join(PINS)}",
            # FlashInfer's JIT reads CUDA_HOME from ARC_VLLM_CUDA_HOME; fail the build, not a GPU start, if absent.
            f"test -x {site}/nvidia/cu13/bin/nvcc",
        ],
        "venv": venv,
        "freeze_path": freeze,
        "plugin_dir": plugin_dir,
        # The vLLM patch files to place before plugin_commands: (file name, sha256, image path), from PATCH_SOURCE.
        "patches": [{"file": name, "sha256": sha, "path": f"{patch_dir}/{name}"} for sha, name in LOCK.patches],
        "patch_source": {"repository": PATCH_SOURCE[0], "release": PATCH_SOURCE[1], "commit": PATCH_SOURCE[2]},
        "plugin_commands": [
            # The vLLM patches onto the installed wheel (git apply works outside a repository; git is in the image).
            *(f"echo '{sha}  {patch_dir}/{name}' | sha256sum -c -" for sha, name in LOCK.patches),
            *(f"git -C {site} apply -p1 {patch_dir}/{name}" for name in PATCHES),
            f"uv pip install --python {py} --no-config --no-deps --exclude-newer {EXCLUDE_NEWER} {plugin_dir}",
            f"uv pip freeze --python {py} > {freeze}",
            # The server is the installed package's: an image whose venv cannot import it fails its build here.
            f"{py} -c 'import vllm_arc.server.app'",
        ],
        # Optional: a deployment's Triton autotune pins module copied here, then the .pth line.
        "triton_pins_target": f"{site}/arc_triton_pins.py",
        "triton_pins_commands": [f"echo 'import arc_triton_pins; arc_triton_pins.activate()' > {site}/arc_triton_pins.pth"],
        "triton_pins_env": {"ARC_TRITON_PINS": "1"},
        "env": {"VLLM_COMMIT": VLLM_COMMIT, "HF_HUB_CACHE": f"{hf_cache}/hub",
                "ARC_ENCODER_CHECKPOINTS": f"{hf_cache}/arc-truncated",
                "ARC_VLLM_CUDA_HOME": f"{site}/nvidia/cu13", FREEZE_ENV: freeze},
        "python": py,
        "server_command": [py, "-m", "vllm_arc.server"],
    }
