# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``vllm_arc/arc_heads.py`` must stay the same Stage-1 head implementation that releases are verified with when they
are built (it may differ from that module only in comments, docstrings and one error message).

This pin makes every change to it deliberate: change the reference module first, port the change here, then record
the new digest below in the same commit.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ARC_HEADS = Path(__file__).parents[1] / "vllm_arc" / "arc_heads.py"
ARC_HEADS_SHA256 = "f2636c9879f4505469648a91f1c4fa98415a2987672e908876387d605ec8824a"


def test_arc_heads_is_a_regular_file_with_the_recorded_digest():
    assert ARC_HEADS.is_file() and not ARC_HEADS.is_symlink()
    assert hashlib.sha256(ARC_HEADS.read_bytes()).hexdigest() == ARC_HEADS_SHA256
