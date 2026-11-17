<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc. -->

# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project intends to
adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from its first release.

## [Unreleased]

## [0.1.1]

### Fixed

- Retained sessions no longer wait for the engine while holding the session lock. A session's close runs after the
  lock is released and is bounded by `ARC_VLLM_ADMISSION_WAIT_S` (default 300 s; past it the engine request is
  aborted). An encode that waits for the lock or a scratch slot past that bound gets 503 `admission_timeout`, and an
  append starts only after the closes before it have ended. `/sleep` on the retained route refuses new retained
  encodes (503 `sleeping`) until `/wake_up`, lets appends in flight finish, and closes every session outside the lock.
  This fixes the known issue in 0.1.0.

## [0.1.0]

### Added

- The encoder engine on stock vLLM 0.33.0 (`vllm_arc/engine/`): one vLLM patch, published by the vLLM fork's release
  and pinned by sha256 in `vllm_arc/engine/engine.lock`, applied to the stock wheel; no fork build or overlay.
  `docs/engine.md` describes it and `deploy/install.sh` builds it.
- The load guard (`vllm_arc/engine/compat.py`): refuses a vLLM other than the locked release, or one whose patched
  files do not match the lock.
- The encoder HTTP server (`vllm_arc/server/`, `vllm-arc-server`), with retained pooling sessions by default and
  prefix-cache sessions as an option.
- Image-reading releases (`arc-image-v1`) on the engine, with float32 ViT pixels (dev testing only).
- The EQ model (`vllm_arc/eq/`) and its `/branches` route; EQ runs use piecewise CUDA graphs.
- An engine id whose plugin digest (`vllm_arc/engine/digest.py`) covers the engine's code and lock, ignoring comments
  and docstrings, and is the same on Python 3.12 and 3.13.
- CI: the public guard (SPDX headers, private imports, and a denylist of internal hosts, cloud resources, private
  repositories and internal names, the names matched by hash so the guard itself names none), a secret scan, and the CPU test suite on
  Python 3.12 and 3.13.
- Contributor documentation: contributing guide, security policy, code of conduct, and issue and pull request
  templates.

### Known issues

- Retained sessions: an engine close that never returns can stall retained encodes, because the session store waits
  for the close while holding its lock, with no time limit. The fix is planned for 0.1.1. Prefix-cache sessions and
  one-shot encodes are not affected.
