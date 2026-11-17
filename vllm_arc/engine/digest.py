# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The engine digest: what the engine id's ``arc-<12 hex>`` covers.

Only the files the engine runs: ``vllm_arc/__init__.py`` (registration), ``engine/``, ``eq/``, ``server/`` and
``engine/engine.lock``. Python sources are hashed as their token stream with comments, blank lines and docstrings
removed, so a comment or docstring edit leaves the id unchanged while any code change moves it. The lock is hashed as
its entries (comments removed). ``heads``, the ``stock`` compatibility aliases, tests and docs are not part of it.
Needs Python 3.12 or later (one tokenizer for f-strings).
"""
from __future__ import annotations

import ast
import hashlib
import io
import tokenize
from pathlib import Path

#: Package-relative directories and files the digest covers.
COVERED_DIRS = ("engine", "eq", "server")
COVERED_FILES = ("__init__.py",)


def _docstring_spans(tree: ast.AST) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """``(start, end)`` of every module, class and function docstring expression (all of its string parts)."""
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                v = first.value
                spans.append(((v.lineno, v.col_offset), (v.end_lineno, v.end_col_offset)))
    return spans


def source_tokens(source: str) -> bytes:
    """``source`` as its code tokens, one per line, ``<token name>\t<string>``: comments, blank lines, docstrings
    (every token inside a docstring's span) and the NEWLINE that ends a docstring statement dropped. Token names, not
    numbers, so the stream is the same on every Python the digest supports. Inside an f-string's replacement fields
    the whitespace between tokens is kept too: it is output there (``f"{x=}"`` and ``f"{ x =}"`` print differently)."""
    spans = _docstring_spans(ast.parse(source))
    lines = source.splitlines(keepends=True)
    out, skip_newline, fdepth, prev_end, prev_name = [], False, 0, (1, 0), ""
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        name = tokenize.tok_name[tok.type]
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING):
            prev_end = tok.end
            continue
        if any(s <= tok.start and tok.end <= e for s, e in spans):
            skip_newline = True
            prev_end = tok.end
            continue
        if skip_newline and tok.type == tokenize.NEWLINE:
            skip_newline = False
            prev_end = tok.end
            continue
        skip_newline = False
        # Only between two tokens of a replacement field: literal text positions (FSTRING_MIDDLE, ``{{``) vary.
        if fdepth and name not in ("FSTRING_MIDDLE", "FSTRING_END") and prev_name not in ("FSTRING_START",
                                                                                       "FSTRING_MIDDLE"):
            gap = lines[tok.start[0] - 1][prev_end[1]:tok.start[1]] if prev_end[0] == tok.start[0] else "\n"
            if gap:
                out.append(f"GAP\t{gap!r}")
        if name == "FSTRING_START":
            fdepth += 1
        elif name == "FSTRING_END":
            fdepth -= 1
        out.append(f"{name}\t{tok.string}")
        prev_end, prev_name = tok.end, name
    return "\n".join(out).encode()


def lock_entries(text: str) -> bytes:
    lines = (" ".join(raw.split("#", 1)[0].split()) for raw in text.splitlines())
    return "\n".join(line for line in lines if line).encode()


def covered(root: Path) -> list[Path]:
    files = [root / f for f in COVERED_FILES]
    for d in COVERED_DIRS:
        files += [*(root / d).rglob("*.py"), *(root / d).rglob("*.lock")]
    return sorted(p for p in files if p.is_file() and "__pycache__" not in p.parts)


def engine_digest(root: Path) -> str:
    """sha256 (hex) over each covered file's package-relative path and normalised content."""
    h = hashlib.sha256()
    for path in covered(root):
        text = path.read_text()
        body = source_tokens(text) if path.suffix == ".py" else lock_entries(text)
        h.update(str(path.relative_to(root)).encode() + b"\0" + body + b"\0")
    return h.hexdigest()
