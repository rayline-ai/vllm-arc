<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc. -->

## What and why

<!-- What this change does, and why it is needed. Link the issue it addresses, if any. -->

## Engine id

- [ ] This change does **not** move the engine id (no code change in `vllm_arc/__init__.py`, `vllm_arc/engine/`,
      `vllm_arc/eq/` or `vllm_arc/server/`, and no `engine.lock` entry change).
- [ ] This change **does** move the engine id. Reason, and evidence that outputs are unchanged or changed as
      intended:

## Checklist

- [ ] Every commit is signed off (`git commit -s`, see CONTRIBUTING.md).
- [ ] Tests added or updated for changed behaviour; `python -m pytest -q` passes.
- [ ] `python ci/public_guard.py .` passes.
- [ ] `CHANGELOG.md` updated under "Unreleased" for user-visible changes.
