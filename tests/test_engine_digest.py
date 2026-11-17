# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The engine digest's normalisation (``vllm_arc.engine.digest``): what moves the engine id and what does not."""
from __future__ import annotations

import pytest

from vllm_arc.engine.digest import source_tokens as tokens


@pytest.mark.parametrize("a, b", [
    ('def f():\n    "doc a"\n    return 1\n', 'def f():\n    """doc b,\n    two lines"""\n    return 1\n'),
    ('def f():\n    ("doc a")\n    return 1\n', 'def f():\n    ("doc b")\n    return 1\n'),
    ('def f():\n    "a" "b"\n    return 1\n', 'def f():\n    "c" "d"\n    return 1\n'),
    ('x = 1\n', '# a comment\n\nx = 1  # another\n'),
])
def test_comments_and_docstrings_do_not_count(a, b):
    assert tokens(a) == tokens(b)


@pytest.mark.parametrize("a, b", [
    ('x = "a"\n', 'x = "b"\n'),
    ('x=1\nprint(f"{x=}")\n', 'x=1\nprint(f"{ x =}")\n'),  # debug f-strings print their own spacing
    ('f"{{a}} {b!r:>{w}}"\n', 'f"{{a}} {c!r:>{w}}"\n'),
    ('def f():\n    return 1\n', 'def f():\n    x = "doc"\n    return 1\n'),  # not a docstring
])
def test_code_changes_count(a, b):
    assert tokens(a) != tokens(b)


def test_token_names_not_numbers():
    # Token numbers differ between Python versions; the stream names them.
    assert tokens('x = 1\n').split(b"\n")[0] == b"NAME\tx"
