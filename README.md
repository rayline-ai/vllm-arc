<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc. -->

# vllm-arc

A [vLLM](https://github.com/vllm-project/vllm) plugin that serves the encoder of **Rayline ARC** (Adaptive Routing Controller, described in the [Rayline report](https://rayline.ai/research/report)) on stock vLLM
`0.33.0`. ARC reads the encoder's hidden state at chosen token offsets ("readouts"), one row per offset. The plugin
adds the encoder architectures, a pooler that returns those rows, an HTTP server with an `/encode` endpoint, and a
loader for an ARC release's Stage-1 head.

The plugin contains no vLLM code. One small vLLM patch, published as a release asset of the vLLM fork
[`rayline-ai/vllm`](https://github.com/rayline-ai/vllm) (branch `arc/v0.33.0`), is needed only
for **retained sessions**, where the engine keeps a session's state between appends. One-shot encodes and the
prefix-cache session route run on unpatched vLLM `0.33.0`.

> **Pre-alpha.** Interfaces may change without notice.
>
> **Known issue (0.1.0):** on the retained-session route, an engine close that never returns can stall retained
> encodes. The fix is planned for 0.1.1; see the [changelog](CHANGELOG.md).

Weights: [huggingface.co/rayline-ai/ARC-1.0](https://huggingface.co/rayline-ai/ARC-1.0).

The encoder is the release's base model (`Qwen/Qwen3.5-0.8B` at the revision its `arc_release.json` pins), which the
server cuts to the release's depth and registers as `ArcBaseForPooling` (`ArcEqForPooling` with a question head). The
release's own `config.json` names `ArcImagePolicyForPooling`, the reference implementation's class; vLLM does not load
that file. From the release the plugin reads the Stage-1 head (`vllm_arc.heads.load_stage_one`) and the question head
(`--eq-addon rayline-ai/ARC-1.0@20261111`).

## Compatibility

| vllm-arc | vLLM | vLLM patch (retained sessions only) |
|---|---|---|
| 0.1.0 | 0.33.0 (`4ba787ec9959`) | `arc-v0.33.0.patch` from [`rayline-ai/vllm`](https://github.com/rayline-ai/vllm) release [`arc-v0.33.0-1`](https://github.com/rayline-ai/vllm/releases/tag/arc-v0.33.0-1) |

The exact environment, including the vLLM wheel's sha256, the patch file's sha256 and every other distribution's
version, is pinned in [`vllm_arc/engine/engine.lock`](vllm_arc/engine/engine.lock). Its `patch_source` line names the fork
release the patch comes from.

## What's in the package

| Module | What it does |
|---|---|
| `vllm_arc/__init__.py` | `register()`, the `vllm.general_plugins` entry point. It registers `ArcBaseForPooling` (the encoder) and `ArcEqForPooling` (EQ: readouts at several depths). Both load lazily from `vllm_arc.engine.models`. |
| `vllm_arc/engine/` | The vLLM-facing engine: model classes, `GatherPooler`, sessions, start-up assertions, the load guard (`compat.py`), the install recipe and `engine.lock`. |
| `vllm_arc/server/` | The HTTP server: `vllm-arc-server` or `python -m vllm_arc.server`. |
| `vllm_arc/eq/` | EQ, the 24-layer engine with a question-head add-on (`--eq-addon`). |
| `vllm_arc/heads.py`, `vllm_arc/arc_heads.py` | The Stage-1 head (see below). |
| `deploy/` | `install.sh` (the locked environment) and `fetch_patches.py` (downloads the patch). |

## Install

### Locked environment (all routes, including retained sessions)

`deploy/install.sh` builds a venv with exactly what `engine.lock` pins: Python 3.13, the vLLM `0.33.0` wheel by URL
and sha256, every other pin with `--no-deps`, the vLLM patch (downloaded and checked by sha256, then applied with
`git apply`), and this plugin.

```bash
git clone https://github.com/rayline-ai/vllm-arc && cd vllm-arc
VLLM_VENV=/opt/vllm-venv deploy/install.sh
```

Requirements: Linux x86_64, `bash`, `git`, `python3`, and `uv` at the lock's version on `PATH`; `gcc` with the C
library headers (Triton builds its launcher at engine start); network access to PyPI and GitHub. The patch is
downloaded with `GH_TOKEN` or `GITHUB_TOKEN`, or `gh`'s login, when one is set. `VLLM_VENV` must be an absolute path.
The last lines the script prints are the server command (`$VLLM_VENV/bin/vllm-arc-server`) and the environment
its process needs; set that environment before starting the server.

### Stock vLLM (no patch)

For one-shot encodes, the offline example and the prefix-cache session route:

```bash
pip install ".[engine]"        # from a checkout of this repository; the extra pins vllm==0.33.0
```

On unpatched vLLM `0.33.0` the plugin loads by default for the non-retained routes (the prefix route, EQ and offline
`LLM.encode`); no setting is needed. The retained route needs the patch and refuses to start without it.
`VLLM_ARC_UNLOCKED=1` turns the remaining refusals (another vLLM version, or the retained route without the patch)
into warnings and marks the engine id as unqualified. Note that `pip install` resolves its own dependency versions, which can differ
from the locked set.

## Quickstart: the server

Start the server on a GPU host, from the environment the plugin is installed in (activate the venv, or call
`<venv>/bin/vllm-arc-server`). Without the patch, select the prefix-cache route:

```bash
ARC_STOCK_SESSIONS=prefix vllm-arc-server --port 8765
```

On the first start the server downloads the pinned `Qwen/Qwen3.5-0.8B` base from the Hugging Face Hub and builds
the encoder checkpoint once under `--checkpoint-root` (default `/tmp/arc-encoder-checkpoints`). The encoder is
that base cut to 12 decoder layers, with YaRN RoPE (`--rope yarn-f4`) by default. The server then runs start-up
assertions and a GPU self-test, and exits non-zero without serving if any fails. It listens on `127.0.0.1` only.

Encode one input, as a one-shot that keeps no session (`"retain": false`). Token ids are Qwen3.5 tokenizer ids,
and offsets are absolute positions in `tokens`:

```bash
curl -s -D - -o rows.bin http://127.0.0.1:8765/encode \
  -H 'content-type: application/json' \
  -d '{"tokens": [1000, 1001, 1002, 1003], "offsets": [3, 0], "retain": false}'
```

The 200 body is float32 rows, row-major, `[x-rows, x-dim]`, one row per offset in request order. A refusal is a
JSON body `{"error": "<code>", ...}`. `GET /health` reports the engine id, model, depth, RoPE and the start-up
checks.

Useful flags: `--port`, `--budget` (max tokens per engine step, default 8192), `--max-model-len`, `--rope`,
`--max-sessions`, `--idle-seconds`, `--checkpoint-root`, `--images arc-image-v1` (image inputs), and `--eq-addon`.
Environment settings such as `ARC_STOCK_SESSIONS` (`retained`, the default, or `prefix`) and
`ARC_VLLM_GPU_MEMORY` are listed in [`docs/engine.md`](docs/engine.md), together with the full wire contract
(sessions, `/close`, `/sleep`, `/wake_up`, images, refusal codes).

[`examples/encode_client.py`](examples/encode_client.py) is a standard-library client for the same endpoint.

## Offline: `LLM.encode`

No patch is needed. Build the encoder checkpoint the way the server does, then ask `GatherPooler` for rows at
chosen offsets:

```python
from vllm import LLM
from vllm.config import PoolerConfig

from vllm_arc.engine.startup import Engine
from vllm_arc.server.app import engine_args, engine_prompt, gather_params, truncated_checkpoint

checkpoint = truncated_checkpoint("/tmp/arc-encoder-checkpoints", "yarn-f4")
kwargs = engine_args(Engine(sessions="retained"), checkpoint, budget=8192, max_model_len=32768,
                     max_sessions=8, scratch_slots=0, gpu_memory=0.80)   # one-shot engine: prefix caching off
kwargs.pop("enable_log_requests", None)                                  # an AsyncLLM-only argument
kwargs["pooler_config"] = PoolerConfig(**kwargs["pooler_config"])
llm = LLM(**kwargs)

tokens = llm.get_tokenizer().encode("The ARC encoder reads one hidden state per offset.")
offsets = [0, len(tokens) - 1]
[out] = llm.encode([engine_prompt(tokens, None)], pooling_params=[gather_params(offsets, True)],
                   pooling_task="token_embed")
rows = out.outputs.data.float()   # [len(offsets), hidden], in request order
```

The plugin registers its architectures through the entry point when vLLM starts, so no import of `vllm_arc` is
needed to load the model. A request names its offsets in
`PoolingParams(task="token_embed", extra_kwargs={"arc_gather": {"v": 1, "offsets": [...]}})`, which is what
`gather_params` builds. A complete script is [`examples/offline_encode.py`](examples/offline_encode.py); unlike the
server, it does not run the start-up assertions or the self-test.

## Hardware

The engine requires CUDA 13 and the FlashInfer GatedDeltaNet prefill kernel; a start-up assertion refuses any other
GDN prefill backend. FlashInfer builds the kernel at first use, so `CUDA_HOME` must point at a CUDA 13 toolkit (the
server also reads `ARC_VLLM_CUDA_HOME`).

- Verified on: NVIDIA RTX PRO 6000 (Blackwell).
- Known to fail on: NVIDIA L4.

Other GPUs are untested.

## The Stage-1 head

An ARC release carries a Stage-1 head: a small torch module that, given encoder readout rows, scores each candidate
in the release for a regime. It returns a success probability and per-candidate token usage. The head runs in the
vLLM process and needs no training code:

```python
from vllm_arc.heads import load_stage_one

head = load_stage_one("path/to/arc-release", device="cuda")
p, usage = head(rows, "fixed")   # float64: [M, n, C] and [M, n, C, 4]
```

`load_stage_one` refuses a release whose `config.json` or weights do not match the sha256 pins in its
`arc_release.json`. Some heads also need request `context` or warm-cache `cache` features; the head says which
(`head.reads_context`, `head.reads_cache`) and refuses a call that passes the wrong set. Its candidates and regimes
are `head.candidates` and `head.regimes`. `python -m vllm_arc.heads RELEASE IN.npz OUT.npz` runs it on recorded
inputs. Install the `heads` extra (`torch`, `safetensors`, `numpy`) to use it outside a vLLM environment.

## Development

```bash
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
  -r ci/requirements-test.txt
python -m pytest -q          # CPU only; no GPU or vLLM install needed
python ci/public_guard.py .  # licence headers and repository hygiene
```

The CI jobs and what each needs are described in [`docs/ci.md`](docs/ci.md).

## Documentation

- [`docs/engine.md`](docs/engine.md): the engine, settings, endpoints and the install recipe.
- [`docs/engine-design.md`](docs/engine-design.md): why the engine is built the way it is.
- [`docs/vllm-internals.md`](docs/vllm-internals.md): the vLLM internals the plugin depends on.
- [`docs/releasing.md`](docs/releasing.md): how versions and the vLLM patch are released.
- [`examples/README.md`](examples/README.md), [`examples/offline_encode.py`](examples/offline_encode.py),
  [`examples/encode_client.py`](examples/encode_client.py).
- [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md). Report security issues privately to
  team@rayline.ai rather than in a public issue.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The ARC encoder is derived from
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache-2.0).
