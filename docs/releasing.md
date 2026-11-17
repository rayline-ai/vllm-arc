<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc. -->

# Releasing

## Versions

The plugin has its own version, `X.Y.Z` in `pyproject.toml` (PEP 440), and each release line supports exactly one
vLLM release: the one `vllm_arc/engine/engine.lock` names, which the load guard (`vllm_arc.engine.compat`) enforces.
This follows the practice of other out-of-tree vLLM platform plugins such as vllm-ascend and vllm-gaudi, whose
releases each target one vLLM release rather than a range.

- `main` is the line for the vLLM release in `engine.lock` today. Between releases its version is the next one with
  a `.devN` suffix; at a release it is the released version (now `0.1.1`).
- A minor version (`0.Y.0`) starts a line, usually because the supported vLLM release changed. Patch versions
  (`0.Y.Z`) fix a line without changing its vLLM release.
- When more than one line is maintained (an older vLLM release still deployed), the older line lives on a branch
  `releases/v<vllm-version>` (for example `releases/v0.33.0`) and takes fixes by cherry-pick; `main` moves on.
- The README's compatibility table lists each plugin line with its vLLM release and patch; update it with every
  minor release.

Tags are `vX.Y.Z`, the `v` plus `pyproject.toml`'s version exactly (`v0.1.0`, `v0.2.0rc1`); the release workflow
refuses a tag that differs. Tags are never moved or reused.

## The vLLM fork and its patch releases

The engine's vLLM patch lives in a public GitHub fork of vLLM,
[`rayline-ai/vllm`](https://github.com/rayline-ai/vllm). Each supported vLLM release has a branch `arc/v<vllm-version>` there (for example `arc/v0.33.0`): the
upstream release tag plus the patch commits. A patch release of the fork is tagged `arc-v<vllm-version>-<n>` (for
example `arc-v0.33.0-1`) and carries the patch file as its release asset. `engine.lock` names that release on its
`patch_source` line and pins the patch file by sha256, so the fork release is always published before the plugin
release that pins it.

## Engine id

The engine id the server reports (`docs/engine.md`) includes a digest of the plugin's runtime code: `vllm_arc/__init__.py`,
`vllm_arc/engine/`, `vllm_arc/eq/`, `vllm_arc/server/` and `engine.lock`, with comments and docstrings ignored
(`tests/test_engine_digest.py`). Any change to that code or to the lock changes the engine id, and outputs recorded
under the old id do not verify under the new one. Changes to comments, documentation or tests do not. A change that
moves the engine id starts a new release, qualified again before it is published, and its release notes say so.

## How releases are cut

Development happens in a private staging repository. Each public release is one curated commit on top of the
previous public release tag, published after the public guard (`ci/public_guard.py`), a secret scan and the CPU
tests pass on it. The public history is therefore one commit per release.

## Steps

1. Check `engine.lock`'s `patch_source` line: it must name a published patch release of the public fork (above).
   If the release pins a new vLLM patch, publish the fork's patch release first. A lock that names any other
   repository must not be released: `deploy/install.sh` downloads the patch from the repository the lock names,
   so the install would fail for everyone without access to it.
2. Prepare the release commit: set `version` in `pyproject.toml` to the release (drop `.devN`), add the release's
   notes (what changed, the supported vLLM release and fork patch release, whether the engine id changed), and
   update the README's compatibility table. CI must be green on it.
3. Tag the release commit and push the tag:

   ```sh
   git tag -a v0.1.0 -m "vllm-arc 0.1.0"
   git push origin v0.1.0
   ```

4. The tag push runs `.github/workflows/release.yml`: it checks the tag against the version, installs the build backend from
   `ci/build-constraints.txt` (exact versions, hash-checked), sets `SOURCE_DATE_EPOCH` to the tagged commit's time,
   builds the sdist with `python -m build --no-isolation`, normalizes it (sorted entries, fixed times, owner and
   modes, fixed gzip header), builds the wheel from that sdist, writes
   `SHA256SUMS`, and creates the GitHub release with the sdist, the wheel and `SHA256SUMS` attached.
5. Bump the version to the next `.dev0`.

A manual run of the workflow (`workflow_dispatch`) builds and uploads the artifacts to the run without creating a
release, as a dry run.

## Reproducing a build

The same commit builds to the same bytes. To check a release's artifacts, check out its tag, install the pinned
backend into a Python 3.12 venv (`pip install --require-hashes -r ci/build-constraints.txt`), run the `build` step of
`.github/workflows/release.yml` with that venv's Python first on `PATH` and `RUNNER_TEMP` set to an empty directory,
and compare `dist/` with the release's `SHA256SUMS` (`cd dist && sha256sum -c SHA256SUMS`).

To update the build backend, change the pins and their hashes in `ci/build-constraints.txt` (both the wheel's and
the sdist's sha256, from PyPI) in a change of its own.

## Moving to a new vLLM release

The weekly upstream canary (`.github/workflows/upstream-canary.yml`) is the trigger: when it reports a new vLLM
release the plugin should follow,

1. open a fork branch `arc/v<new-version>` from the upstream release tag and rebase the patch onto it;
2. publish the fork's patch release `arc-v<new-version>-1`;
3. re-pin `engine.lock` (vLLM version, commit, wheel and sha256, the patch source and sha256s, the venv pins) and
   requalify the engine;
4. release a new plugin line (a new minor version); if the previous line is still deployed, branch it to
   `releases/v<old-version>` first.

## Contributions

Commits must carry a DCO sign-off (`git commit -s`). Accepted pull requests are imported into the release process and
appear in the next release commit, with `Co-authored-by` credit for their authors. Changes to vLLM itself belong
upstream in vLLM; the fork carries only the patch the plugin needs. Report security issues privately to
team@rayline.ai rather than in a public issue.

## PyPI

Releases are GitHub releases only: install a release from its attached wheel, checked against `SHA256SUMS`. Whether
to publish to PyPI, and whether to reserve the project name `vllm-arc` there before the first release, is an open
maintainer decision. If PyPI publishing is adopted, it would be a further job in the release workflow using trusted
publishing (OIDC, no stored token) that uploads the same artifacts the GitHub release carries.
