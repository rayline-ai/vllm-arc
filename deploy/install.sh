#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
#
# Builds the stock engine's venv on this host from this checkout: the image recipe's own commands
# (vllm_arc.engine.recipe.recipe(), which reads engine.lock), with the venv at $VLLM_VENV and this checkout as the
# plugin directory:
#   1. a fresh venv (Python 3.13; engine.lock names the patch release the image runs);
#   2. the vLLM release wheel by URL and sha256 plus every engine.lock pin, without resolution (--no-deps);
#   3. each locked vLLM patch, downloaded from engine.lock's patch_source release (deploy/fetch_patches.py) into
#      $VLLM_VENV.arc-patches (outside the venv, which uv venv --clear wipes) and checked by sha256, applied to the installed wheel with git apply;
#   4. this plugin, without dependencies; the venv's freeze to $VLLM_VENV/vllm-arc-freeze.txt; an import of the server.
# The recipe's uv bootstrap (pip install uv) is not run: uv at engine.lock's version must be on PATH. Triton pins are
# a deployment's own and are not installed.
#
# Usage: VLLM_VENV=/opt/vllm-venv deploy/install.sh
# VLLM_VENV is an absolute path; it and this checkout's path hold no spaces or shell syntax (the recipe's commands
# carry paths unquoted). The venv's Python must be engine.lock's patch release (uv venv --python 3.13 picks one).
# Needs bash, git, python3 and uv on PATH, gcc with the C library headers (Triton builds its launcher at engine start)
# and network access to PyPI and GitHub (a private patch_source needs GH_TOKEN or a logged-in gh). The last lines printed are the environment the server process needs.
set -euo pipefail

: "${VLLM_VENV:?install.sh: set VLLM_VENV to the venv directory to create}"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for path in "$VLLM_VENV" "$ROOT"; do
  [[ "$path" =~ ^/[A-Za-z0-9_./+-]+$ ]] || { echo "install.sh: '$path' is not an absolute path of [A-Za-z0-9_./+-]" >&2; exit 1; }
done
# git apply runs in the venv's site-packages: stop git from finding a repository above it (a venv inside a checkout),
# which would skip every patch path and still exit 0.
export GIT_CEILING_DIRECTORIES="$VLLM_VENV"
for tool in git python3 uv; do
  command -v "$tool" >/dev/null || { echo "install.sh: $tool is not on PATH" >&2; exit 1; }
done
WANT_UV=$(awk '$1 == "uv" { print $2; exit }' "$ROOT/vllm_arc/engine/engine.lock")
WANT_PY=$(awk '$1 == "python" { print $2; exit }' "$ROOT/vllm_arc/engine/engine.lock")
HAVE_UV=$(uv --version | awk '{ print $2 }')
[ "$HAVE_UV" = "$WANT_UV" ] || { echo "install.sh: uv $HAVE_UV on PATH; engine.lock builds with uv $WANT_UV" >&2; exit 1; }

python3 -I "$ROOT/deploy/fetch_patches.py" "$VLLM_VENV.arc-patches" >&2

STEPS=$(python3 -I - "$ROOT" "$VLLM_VENV" <<'EOF'
import json, shlex, sys
sys.path.insert(0, sys.argv[1])
from vllm_arc.engine.recipe import UV_VERSION, recipe

root, venv = sys.argv[1], sys.argv[2]
r = recipe(venv=venv, plugin_dir=root, freeze=f"{venv}/vllm-arc-freeze.txt", patch_dir=f"{venv}.arc-patches")
bootstrap = f"pip install --no-cache-dir uv=={UV_VERSION}"
assert r["commands"][0] == bootstrap, r["commands"][0]
for command in r["commands"][1:] + r["plugin_commands"]:
    print(command)
print("# env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in r["env"].items()))
EOF
)
while IFS= read -r step; do
  case "$step" in
    "# env "*) ENV_LINE=${step#"# env "} ;;
    *) echo "install.sh: $step" >&2; bash -euo pipefail -c "$step" ;;
  esac
  if [[ "$step" == "uv venv "* ]]; then
    HAVE_PY=$("$VLLM_VENV/bin/python" -c 'import platform; print(platform.python_version())')
    [ "$HAVE_PY" = "$WANT_PY" ] || { echo "install.sh: the venv has Python $HAVE_PY; engine.lock runs $WANT_PY" \
      "(make it the one uv finds first, e.g. UV_PYTHON=$WANT_PY)" >&2; exit 1; }
  fi
done <<< "$STEPS"

echo "install.sh: stock engine in $VLLM_VENV (engine.lock)"
echo "install.sh: run $VLLM_VENV/bin/vllm-arc-server with $ENV_LINE ARC_VLLM_ARC_COMMIT=<this checkout's commit>"
