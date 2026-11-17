# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The public guard (``ci/public_guard.py``) flags what it exists to flag and passes what the repository needs.

Denylisted samples are assembled from parts, so this file does not trip the guard it tests.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("public_guard", ROOT / "ci" / "public_guard.py")
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


@pytest.mark.parametrize("source", [
    "import rayline" + "_router.serving\n",
    "from package_store import load\n",
    "import os, service\n",
    "from path" + "finder.x import y\n",
    "importlib.import_module('rayline" + "_router.core')\n",
    "__import__('service')\n",
    "importlib.import_module(name='package_store')\n",
    "import os, modal as m\n",
])
def test_private_imports_are_flagged(source):
    assert guard.private_imports(source)


@pytest.mark.parametrize("source", [
    "import vllm_arc.heads\nfrom . import service\nfrom .service import x\n",
    "import services\nimport torch\n",
])
def test_public_and_relative_imports_pass(source):
    assert guard.private_imports(source) == []


@pytest.mark.parametrize("text", [
    "see " + "path" + "finder" + "#3233",
    "path" + "finder docs/plans/x.md",
    "path" + "finder==1.0",
    "see memex" + "-desktop#7178",
    "atlas" + "futures/router" + "-infra",
    "V" + "SR's deadline",
    "per ADR" + " 0122",
    "rayline" + "-ai/arc-staging",
    "rayline" + "-ai/arc-1.0",
    "rayline" + "-ai/vllm-arc-staging",
    "rayline" + "-ai/ARC-1.0-private",
    "rayline" + "-ai/vllm-arc.extra",
    "atlas" + "futures/path" + "finder",
    "atlas" + "futures/vllm-private",
    "app = modal" + ".App('x')",
    "https://x--y" + ".modal.run/health",
    "db" + ".internal:5432",
    "10" + ".0.0.12",
    "host=10" + ".4.0.35",
    "/home" + "/someone/code",
])
def test_internal_references_are_flagged(text):
    assert guard.denylisted(text)


@pytest.mark.parametrize("text", [
    "https://huggingface.co/rayline-ai/ARC-1.0",
    "https://github.com/rayline-ai/vllm-arc",
    "https://github.com/atlasfutures/vllm.git",
    "https://github.com/atlasfutures/vllm/commit/66029e2d",
    "git+https://github.com/rayline-ai/vllm-arc.git@main",
    "https://huggingface.co/rayline-ai/ARC-1.0.git",
    "Weights: rayline-ai/ARC-1.0.",
    "atlasfutures/vllm@66029e2d",
    "CUDA 13.0, vLLM 0.26.1, torch 2.10.0",
    "pin nvidia-curand==10.4.0.35",
    "pin cuda-path" + "finder==1.8.5",
    "cuda-path" + "finder (==1.8.5)",
    "cuda-path" + "finder[extra]>=1.8",
    "upstream RFC #66134",
])
def test_public_references_pass(text):
    assert guard.denylisted(text) == []


def test_spdx_header_is_required_near_the_top():
    assert not guard.lacks_spdx("# SPDX-License-Identifier: Apache-2.0\nx = 1\n")
    assert guard.lacks_spdx("x = 1\n" * 10 + "# SPDX-License-Identifier: Apache-2.0\n")


def test_this_repository_passes():
    if subprocess.run(["git", "-C", str(ROOT), "rev-parse"], capture_output=True).returncode:
        pytest.skip("not a git checkout")
    assert guard.check(ROOT) == []



@pytest.mark.parametrize("text", [
    "the P2 sp" + "ike's server",
    "a parity can" + "didate ran",
    "seen in the dev sm" + "oke",
    "PATH" + "FINDER's ROPE_VARIANTS",
    "a rayline" + "_router-based dispatch",
    "the memex" + "-desktop-side client",
])
def test_internal_vocabulary_is_flagged(text):
    assert guard.denylisted(text)


def test_this_file_names_no_internal_term():
    """The guard's own source is exempt from the denylist; it must still name no internal term in plain text."""
    assert guard.denylisted((ROOT / "ci" / "public_guard.py").read_text()) == []


def test_hash_cli_matches_term_hash(capsys):
    assert guard.main(["public_guard.py", "--hash", "Some Term"]) == 0
    assert capsys.readouterr().out.strip() == guard.term_hash("some term")
