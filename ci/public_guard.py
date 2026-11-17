# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""Checks that every tracked file is fit for the public repository: ``python ci/public_guard.py [ROOT]``.

- imports: no Python file imports a private package (``PRIVATE_MODULES``), directly or through
  ``importlib.import_module`` / ``__import__`` with a literal name.
- denylist: no file names an internal host, cloud app or volume, private model or source repository, internal service
  or milestone, or internal decision record (``DENYLIST``, ``INTERNAL_TERM_HASHES``).
- spdx: every file carries an Apache-2.0 SPDX header in its first lines, except the licence files themselves, empty
  files, and formats that cannot hold a comment (``SPDX_EXEMPT_SUFFIXES``).

Exits 1 and prints one line per finding when any check fails.
"""
from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
from pathlib import Path

#: Modal, which only private deployment code uses. Private packages are matched by hash (``PRIVATE_MODULE_HASHES``).
PRIVATE_MODULES = frozenset({"modal"})

#: (name, pattern). Patterns are generic on purpose, so this file names no private resource itself.
DENYLIST: tuple[tuple[str, re.Pattern[str]], ...] = tuple((name, re.compile(pattern)) for name, pattern in (
    ("decision record reference", r"\bADR[ -]?0\d"),
    ("Modal resource", r"\bmodal\.(?:App|Volume|Secret|Image|Dict|Queue|NetworkFileSystem)\b"),
    ("Modal import", r"(?m)^\s*(?:import|from)\s+modal\b"),
    ("Modal endpoint host", r"(?i)[\w-]+\.modal\.run\b"),
    ("internal host", r"(?i)\b[\w-]+\.(?:internal|ts\.net|corp|lan)\b"),
    # Not after "==": a pinned version (nvidia-curand==10.4.0.35) has an address's shape.
    ("private IPv4 address", r"(?<!==)\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"),
    ("private model or dataset repository", r"\brayline-ai/(?!(?:ARC-1\.0|vllm|vllm-arc)(?:\.git)?(?![\w-]|\.[\w.-]))[\w.-]+"),
    ("private source repository", r"\batlasfutures/(?!vllm(?:\.git)?(?![\w-]|\.[\w.-]))[\w.-]+"),
    ("home-directory path", r"(?:/home|/Users)/[\w.-]+/"),
))

LICENCE_FILES = frozenset({"LICENSE", "NOTICE"})
SPDX_EXEMPT_SUFFIXES = frozenset({".json", ".safetensors", ".npz", ".npy", ".png", ".jpg", ".bin", ".patch"})
SPDX_LINE = "SPDX-License-Identifier: Apache-2.0"
SPDX_WITHIN_LINES = 5
#: This file holds the denylist's own patterns.
DENYLIST_EXEMPT = frozenset({"ci/public_guard.py"})
#: Internal names (private repositories and services, internal milestone vocabulary), as the first 16 hex digits of
#: the sha256 of each lower-cased term, so this public file does not name them. A term is one word or two adjacent
#: words; a word is ``[a-z0-9_]`` runs joined by single hyphens, and each part of a hyphenated word is checked too,
#: except in a version-pinned distribution name (``cuda-<name>==1.0``).
#: Add one with ``python ci/public_guard.py --hash "<term>"``.
INTERNAL_TERM_HASHES = frozenset({
    "0e43650b148e1557", "06f2da652f7bf676", "27f024e4bc25011d", "4365fe0531734250",  # private repos, service
    "8513d97e91242edc", "356407a7813470ae", "4af695c0fe952b24", "18c4fd68ca527f23",  # milestone vocabulary
    "69f75462c8a5d180", "7ae8a5b72309a211", "98cf5385d7ebc3b0", "e0dee0bb2ea0bedb",  # private code names
})
#: Private top-level packages, hashed as ``INTERNAL_TERM_HASHES`` is.
PRIVATE_MODULE_HASHES = frozenset({"e0dee0bb2ea0bedb", "0e43650b148e1557", "9df6b026a8c6c26e", "1ed2a1a6ce365d36"})
WORD = re.compile(r"[a-z0-9_]+(?:-[a-z0-9_]+)*")
#: What follows a distribution name in a requirement: optional extras, an optional parenthesis, a version operator.
VERSION_PIN = re.compile(r"\s*(?:\[[^\]]*\])?\s*\(?\s*(?:===|==|>=|<=|~=|!=|<|>)")


def private_imports(source: str, path: str = "<source>") -> list[str]:
    """Each import of a private package in a Python source, as ``path:line: module``."""
    found = []
    for node in ast.walk(ast.parse(source, path)):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            args = [*node.args[:1], *(k.value for k in node.keywords if k.arg == "name")]
            if called in ("import_module", "__import__"):
                names = [a.value for a in args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        found += [f"{path}:{node.lineno}: imports {name}" for name in names
                  if name.split(".")[0] in PRIVATE_MODULES or term_hash(name.split(".")[0]) in PRIVATE_MODULE_HASHES]
    return found


def term_hash(term: str) -> str:
    return hashlib.sha256(term.lower().encode()).hexdigest()[:16]


def internal_terms(line: str) -> list[str]:
    """Each one- or two-word term of a line whose hash is in ``INTERNAL_TERM_HASHES``."""
    lowered = line.lower()
    words, parts = [], []
    for match in WORD.finditer(lowered):
        words.append(match.group(0))
        # A hyphenated word's parts count too ("<name>-based"), except in a pinned distribution name
        # ("cuda-<name>==1.0"): a public package may share a part with an internal name.
        if "-" in match.group(0) and not VERSION_PIN.match(lowered, match.end()):
            split = match.group(0).split("-")
            parts += ["-".join(split[i:j]) for i in range(len(split)) for j in range(i + 1, len(split) + 1)]
    terms = words + parts + [f"{a} {b}" for a, b in zip(words, words[1:])]
    return [term for term in terms if term_hash(term) in INTERNAL_TERM_HASHES]


def denylisted(text: str, path: str = "<text>") -> list[str]:
    """Each denylisted reference in a text, as ``path:line: kind: match``."""
    found = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for name, pattern in DENYLIST:
            found += [f"{path}:{lineno}: {name}: {m.group(0)}" for m in pattern.finditer(line)]
        found += [f"{path}:{lineno}: internal name: {term}" for term in internal_terms(line)]
    return found


def lacks_spdx(text: str) -> bool:
    return SPDX_LINE not in "\n".join(text.splitlines()[:SPDX_WITHIN_LINES])


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout
    return sorted(name for name in out.decode().split("\0") if name)


def check(root: Path) -> list[str]:
    findings = []
    for name in tracked_files(root):
        path = root / name
        if not path.is_file():  # a deletion not yet committed
            continue
        data = path.read_bytes()
        try:
            text = data.decode()
        except UnicodeDecodeError:
            if name.endswith(".py"):  # fail closed: the import and SPDX checks need the source
                findings.append(f"{name}:1: Python source is not UTF-8")
            continue  # binary: covered by the secret scan and review, not by these text checks
        if name.endswith(".py"):
            findings += private_imports(text, name)
        if name not in DENYLIST_EXEMPT:
            findings += denylisted(text, name)
        if (name not in LICENCE_FILES and data.strip() and Path(name).suffix not in SPDX_EXEMPT_SUFFIXES
                and lacks_spdx(text)):
            findings.append(f"{name}:1: no '{SPDX_LINE}' in the first {SPDX_WITHIN_LINES} lines")
    return findings


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--hash":
        print(term_hash(argv[2]))
        return 0
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    findings = check(root)
    for finding in findings:
        print(finding)
    print(f"public_guard: {len(findings)} finding(s) in {len(tracked_files(root))} tracked files", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
