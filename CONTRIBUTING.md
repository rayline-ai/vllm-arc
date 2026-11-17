<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc. -->

# Contributing to vllm-arc

Thanks for your interest. vllm-arc is a work in progress (see the README), so interfaces may change, and larger
changes are best discussed in an issue before you open a pull request.

## Development setup

You need Python 3.12 or newer (CI runs 3.12 and 3.13). The CPU test suite needs neither a GPU nor vLLM.

CI installs exact pins from `ci/requirements-test.txt`, with torch from the PyTorch CPU index. To match it:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    -r ci/requirements-test.txt
pip install --no-deps -e .
```

Running the encoder server needs the qualified engine: vLLM 0.33.0 with the patch that `vllm_arc/engine/engine.lock`
pins, at the lock's exact versions. `docs/engine.md` describes it and `deploy/install.sh` builds it.

## Running the tests

```bash
python -m pytest -q
```

This runs the CPU suite: everything under `tests/`. It uses a stand-in for vLLM, so it needs no GPU, vLLM or
released weights; the few tests that import vLLM or transformers skip when those are not installed. CI runs the same
command on Python 3.12 and 3.13.

## The public guard

Every pull request runs `ci/public_guard.py`. Run it before you push:

```bash
python ci/public_guard.py .
```

It checks that every tracked file:

- carries `SPDX-License-Identifier: Apache-2.0` in its first five lines (the licence files, empty files and formats
  that cannot hold a comment, such as `.json` and images, are exempt). By convention an `SPDX-FileCopyrightText`
  line follows it; copy the comment form from an existing file of the same type;
- imports no private package;
- names no internal host, private address, private repository or internal tracker reference.

If the guard flags your change, change the text rather than the guard. CI also runs a secret scan (gitleaks) over
the pull request's commits.

## Sign your commits (DCO)

Contributions are accepted under the [Developer Certificate of Origin](https://developercertificate.org/) (DCO).
Every commit must carry a `Signed-off-by` line with your real name and an email address you can be reached at. It
certifies that you wrote the change or otherwise have the right to submit it under the Apache License 2.0:

```bash
git commit -s
```

To sign off commits you already made on a branch: `git rebase --signoff origin/main`.

## Pull requests

- Keep each pull request to one change, and describe what it changes and why.
- Add or update tests for behaviour you change, and make sure the tests and the public guard pass.
- Say whether the change moves the **engine id**. Its plugin digest (`vllm_arc/engine/digest.py`) covers the code of
  `vllm_arc/__init__.py`, `vllm_arc/engine/`, `vllm_arc/eq/` and `vllm_arc/server/`, and the entries of
  `vllm_arc/engine/engine.lock`. Comments, blank lines and docstrings do not count; any code or lock change does.
  A change that moves it needs a stated reason and, for model or engine behaviour, evidence that outputs are
  unchanged or changed as intended.
- `engine.lock` pins the engine that was qualified. Change a pin only together with re-qualifying the engine.
- Update `CHANGELOG.md` under "Unreleased" for user-visible changes.

## Reporting security issues

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
