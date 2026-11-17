<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc. -->

# CI

## Jobs

| Workflow | Job | Runs on | Needs |
|---|---|---|---|
| `ci.yml` | public guard (imports, denylist, SPDX) | every PR, push to main | nothing |
| `ci.yml` | secret scan (gitleaks) | every PR, push to main | nothing |
| `ci.yml` | CPU tests (Python 3.12, 3.13) | every PR, push to main | nothing (vLLM is not installed; its tests skip) |
| `ci.yml` | engine compose (vLLM 0.33.0 + fork patch + plugin) | every PR, push to main | the public fork release |
| `engine-vllm.yml` | patched vLLM 0.33.0 tests (CPU) | PRs touching `vllm_arc/engine/**` (the lock included), those tests, `ci/compose_check.py`, `deploy/fetch_patches.py` or the workflow; push to main; manual dispatch | the public fork release |
| `upstream-canary.yml` | patch applies to the newest vLLM | weekly, manual dispatch | the public fork release |
| `upstream-canary.yml` | CPU tests | weekly, manual dispatch | nothing |

- **Engine compose:** downloads the locked vLLM wheel (checked by sha256) and the patch file from the fork release (checked by sha256). It applies the patch the way the image does, then runs `compat.require()`. vLLM is not imported.
- **Patched vLLM tests:** the same compose, then CPU torch and the lock's other pins minus the GPU-only distributions. It runs `tests/test_patch_0001.py` and `tests/test_stock_eq_model.py` with the patched `vllm` importable. A skipped test fails the job, because a skip there means vLLM did not import.
- **Recorded EQ rows** (`tests/fixtures/stock_eq_rows.npz`): the CPU tests compare the EQ forward with recorded
  bfloat16 rows bit for bit, at the ATen CPU level the jobs pin (`ATEN_CPU_CAPABILITY=avx2`). Float32 bits differ
  between GitHub runners, so float32 is checked only against a reference forward on the same machine. The bfloat16
  rows were recorded on one x86 host; if they ever differ on another runner, drop to the same-machine check for
  bfloat16 too rather than adding a tolerance.
- **Canary:** checks that the patch also applies to the newest vLLM release on PyPI. A failure means the patch stack needs rebasing; it is not a defect of this repository.

## Fork release

The patch file is the asset `arc-v0.33.0.patch` of the public release `arc-v0.33.0-1` of `rayline-ai/vllm`, named on
`engine.lock`'s `patch_source` line. The jobs that fetch it need no secret: they use the workflow's own token only
for the GitHub API's rate limit, so they run the same on pull requests from forks.
