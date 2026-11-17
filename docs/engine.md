<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc. -->

# Running the engine

How to install, start and call the ARC encoder server. Why it is built this way (sessions, the vLLM patch, the lock,
the engine arguments and the start-up checks in full) is in [engine-design.md](engine-design.md).

The engine is vLLM `0.33.0` (`4ba787ec9959e566d3b587a796dff07673324da5`) plus one vLLM patch plus this plugin.
The patch is not in this repository: it is the one commit of the vLLM fork's `arc/v0.33.0` branch, published
as the asset `arc-v0.33.0.patch` of that fork's release named on `engine.lock`'s `patch_source` line
(`rayline-ai/vllm`), and pinned there by sha256. Only retained sessions need the patch: the `prefix` session route,
one-shots and EQ run on the unpatched vLLM `0.33.0` release. The server encodes token sequences with the E12 encoder (the pinned Qwen3.5-0.8B base cut to 12 decoder
layers) and returns the hidden rows at the offsets a request names: text, and optionally images. With a question-head
add-on it serves EQ instead (below).

## 1. Install

You need a Linux host with an NVIDIA GPU and the CUDA 13 runtime, `bash`, `git`, `python3`, gcc with the C library
headers, uv at the version `engine.lock` names, and network access to PyPI and GitHub (a private patch source
also needs `GH_TOKEN` or a logged-in `gh`). The venv runs the exact Python patch release `engine.lock` names
(`python 3.13.14`); `install.sh` stops if uv picks another, so make it the one uv finds first, for example with
`UV_PYTHON=3.13.14`.

```bash
git clone https://github.com/rayline-ai/vllm-arc && cd vllm-arc
VLLM_VENV=/opt/vllm-venv deploy/install.sh
```

`install.sh` builds a fresh venv at `$VLLM_VENV` exactly as `engine.lock` describes it: the vLLM release wheel by URL
and sha256, every other distribution at its locked version (nothing is resolved), the patch downloaded and checked
by sha256 and applied to the installed wheel, then this checkout as the plugin. `VLLM_VENV` must be an absolute path
without spaces. The last lines it prints are the environment variables the server process needs.

To build a container image instead, apply `vllm_arc.engine.recipe.recipe()`, which returns the same steps as plain
data: see [engine-design.md](engine-design.md#image-recipe).

## 2. Start the server

```bash
<the env line install.sh printed> ARC_VLLM_ARC_COMMIT=$(git rev-parse HEAD) \
  /opt/vllm-venv/bin/vllm-arc-server --port 8765
```

`python -m vllm_arc.server` is the same program. It listens on `127.0.0.1`. The process needs `CUDA_HOME` (taken
from `ARC_VLLM_CUDA_HOME` when unset).

The server checks the engine before it serves anything, and refuses to start if a check fails: it exits with status
3 and prints one `arc_stock_startup_failed` JSON line naming the failed check. What is checked is listed under
[Start-up checks](engine-design.md#start-up-checks). When it is up, `GET /health` answers `{"status": "ok", ...}`.

Command-line flags: `--port` (8765), `--budget` (8192, the tokens of one prefill chunk), `--scratch-slots` (4),
`--max-sessions`, `--idle-seconds`, `--max-model-len`, `--rope`, `--checkpoint-root`, `--images`, `--eq-addon`,
and `--backend` (`flashinfer` only).

## 3. Encode

`POST /encode` takes token ids and the offsets to read, and answers with float32 rows, row-major, as raw bytes.
`x-rows` and `x-dim` give the shape.

```bash
curl -s localhost:8765/encode -H 'content-type: application/json' \
  -d '{"tokens": [151644, 872, 198, 9707, 151645], "offsets": [4], "retain": false}' \
  -D - -o rows.f32
```

A one-shot (`"retain": false`) encodes the tokens and keeps nothing. A session keeps its state in the engine, so
each append sends and encodes only its new tokens:

```json
{"session": "s1", "owner": "client-a", "base": 0,    "tokens": [/* 1000 ids */], "offsets": [999]}
{"session": "s1", "owner": "client-a", "base": 1000, "tokens": [/* 64 ids */],   "offsets": [1063]}
```

`base` is the session's length before this input (`0` starts or restarts the session) and `offsets` are absolute
positions inside this input. `POST /close {"owner": "client-a"}` ends every session of that owner.

## Settings

| Env | Values | Default |
|---|---|---|
| `ARC_STOCK_SESSIONS` | how sessions are kept: `retained` (needs the patch; prefix caching off) or `prefix` | `retained` (with EQ: `prefix`, and `retained` is refused) |
| `ARC_STOCK_PREFIX_UNIT` | the prefix route's hit grid: `0` (no `prefix_match_unit`), `64`, `192`; refused on `retained` | `0` |
| `ARC_ENCODER_ROPE` | `yarn-f4` or empty (the checkpoint's own RoPE) | `yarn-f4` |
| `ARC_ENCODER_MAX_MODEL_LEN` | token capacity, at most the RoPE variant's | `262144` |
| `ARC_VLLM_IMAGES` | an image rule (`arc-image-v1`), or empty for text only | empty |
| `ARC_ENCODER_CHECKPOINTS` | where the truncated checkpoint is built once | `/tmp/arc-encoder-checkpoints` (the recipe's env sets its own) |
| `ARC_VLLM_GPU_MEMORY` | the fraction of GPU memory vLLM may use | `0.80` |
| `ARC_VLLM_MAX_SESSIONS` | sessions kept at once; idle ones are dropped least recently used first | `64` |
| `ARC_VLLM_IDLE_SECONDS` | a session idle this long is closed at the next session encode | `900` |
| `ARC_EQ_ADDON` | EQ (below): a question-head add-on directory or release `<repo>@<tag>` | empty (E12) |
| `VLLM_ARC_UNLOCKED` | `1` loads on a vLLM the load guard refuses, with a warning; the engine id then ends `+unqualified` | unset |
| `ARC_EQ_INFLIGHT`, `ARC_EQ_ADMIT_WAIT_S` | EQ's admission: requests in flight, and how long one waits for a slot | `8`, `0.25` |

The two session routes differ in cost, not in the wire contract. On `retained` an append sends only its new tokens.
On `prefix` the server re-sends the session's whole sequence on every append and relies on vLLM's prefix cache; it
needs no patch, so the load guard checks only the vLLM version there, and it is the route EQ uses.

## Endpoints

### `POST /encode`

Body: `{"session", "base", "tokens", "offsets", "retain"?, "owner"?, "images"?}`.

- `tokens`: the input's token ids. `offsets`: the positions to read, absolute, each inside this input.
- `retain` (default `true`): `false` is a one-shot and needs `base` 0; `session` and `owner` are ignored.
- `session`, `base`, `owner`: as in [Encode](#3-encode). `owner` groups sessions for `/close`.
- `images`: only on a server started with an image rule. Per image inside this input: `offset` (its
  `<|vision_start|>`, absolute), `n` pads, `sha256`, `rgb_sha256`, `height`, `width` and the packed RGB `bytes`
  (base64). The tokens carry every image already expanded to its pads.

200: float32 rows `[len(offsets), dim]`. Headers: `x-rows`, `x-dim`; `x-cached` / `x-cached-tokens`, the tokens the
final pass did not recompute (on `retained`, the session length before the append); `x-encoded`, the tokens sent on
the final pass (on `retained`, the append's own); `x-over-hit`, `1` when the first pass hit the cache past a
requested offset, or returned a non-finite row, and was redone.

| Status | `error` | Meaning |
|---|---|---|
| 409 | `session_unknown` | no such session at this `base`, or it is busy: start again from `base` 0 |
| 422 | `invalid_body`, `invalid_retain`, `invalid_owner` | a malformed request |
| 422 | `offsets_out_of_range` | an offset outside this input, or no tokens |
| 422 | `exceeds_kv_budget` | the sequence is longer than the engine can hold |
| 503 | `kv_exhausted`, `sessions_exhausted` | on `retained`: only busy sessions stand in the way of the KV budget or the session count; retry |
| 422 | `images_not_served` | images sent to a text-only server |
| 422 | `invalid_images` | a malformed or undeclared run of image tokens, or pixels that do not match their hash or the rule |
| 422 | `too_many_images` | more than 32 images in one input |
| 422 | `image_input_refused` | vLLM refused the input, or processed it to other ids than the request's expanded tokens; a session that took it is closed and answers 409 until the client rebuilds it |
| 422 | `images_need_retained` | a session input with images on `prefix` (one-shots with images are served on both routes) |

### `POST /close`, `POST /sleep`, `POST /wake_up`

- `/close {"owner"}`: ends that owner's sessions and answers `{"closed": n}`; 422 `invalid_owner`. Idle sessions
  are closed first; if any is mid-encode the answer is 409 `{"error": "session_busy", "busy": k}` with the idle
  ones already closed, so retry for the rest.
- `/sleep`: closes every session, clears the prefix cache and puts the engine to sleep (vLLM sleep level 1).
  `/wake_up` reverses it. Sessions do not survive a sleep.

Closing, evicting or sleeping frees a retained session's KV blocks.

### `GET /health`

`status`, `engine` (the engine id), `model`, `revision`, `depth`, `budget`, `max_model_len`, `rope`,
`rope_parameters`, `sessions`, `kv`, and a `stock` block with what the running engine actually is: the session
route, the patch (`arc_patch`: its id, whether the runner hook is present, and `extends`, the in-place extensions
since start, read live on each request), GDN backend, model runner, fused QK, autotune mode, cache grid, session counters, versions (vLLM, its wheel URL and sha256,
torch, FlashInfer, Triton, plugin commit and digest, freeze sha256), the start-up result and self-test, and the
full probe under `detail`. A server with images adds an `images` block (`rule`, `vision`, `max_images`,
`mm_processor_kwargs`, `processor`); a text-only server has no `images` key. 503 `{"status": "engine_dead"}` when
the engine has died.

The engine id names everything that can change a row's arithmetic, so compare it before comparing rows from two
servers: [Engine id](engine-design.md#engine-id).

## EQ

Started with `ARC_EQ_ADDON` (`--eq-addon`), the server serves EQ instead of E12: the base at all 24 layers with the
add-on's layers 12-23, reading at depths 12 and 24. EQ runs the `prefix` route, text only, compiled.

- `POST /encode`: one-shots only (`"retain": false`, else 422 `retain_not_served`), with an optional `cache_salt`.
  Rows are `norm(h_12)`; headers `x-rows`, `x-dim`.
- `POST /branches {"tokens", "suffixes": [{"tokens", "offsets"}], "cache_salt"?}`: encodes the prefix once, then
  each `prefix + suffix` under the same salt. Rows are `norm(h_24)` at the suffix positions, concatenated; headers
  `x-counts` (rows per suffix), `x-dim`, `x-cached`, `x-misses`, `x-timing`.
- `/health` adds an `eq` block: the add-on's status, the readout depths, the cache grid and the counters.

EQ admits `ARC_EQ_INFLIGHT` requests at once, within a token budget of 90% of the KV pool; one that waits longer
than `ARC_EQ_ADMIT_WAIT_S` for room is answered 429 `capacity`. An add-on that fails verification is not loaded: the server serves the base and answers
every question request 503 `question_head_unavailable`. The details are under [EQ](engine-design.md#eq).
