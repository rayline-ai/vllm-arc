<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc. -->

# Engine design notes

Why the engine is built the way it is. To install, start and call it, read [engine.md](engine.md) first.

## Layout

| Path | Holds |
|---|---|
| `vllm_arc/engine/` | the vLLM-facing engine: models, pooling, sessions, start-up, the load guard (`compat`), the image recipe, `engine.lock` |
| `vllm_arc/eq/` | EQ: the add-on's verification, its encodes and admission |
| `vllm_arc/server/` | the HTTP server (`python -m vllm_arc.server`, `vllm-arc-server`) |
| `deploy/` | `install.sh`, `fetch_patches.py` |

## Sessions

A session is a token sequence a client extends over time, reading rows from each new input. There are two routes
(`ARC_STOCK_SESSIONS`).

**`retained`** (the default). Each session is one live streaming pooling request (`retained.RetainedSessions` over
`pooling_session.PoolingSession`), so an append sends only its new tokens and the engine keeps the session's KV
blocks and GatedDeltaNet state. This needs the vLLM patch (next section). The route runs without prefix caching.
Admission is by the engine's KV blocks: when only busy sessions stand in the way of the budget or the session
count, a request is answered 503 rather than queued.

**`prefix`**. vLLM keeps nothing per session: the server holds each session's token sequence and a cache salt
(`sessions.PrefixCacheSessions`), re-sends the whole sequence on every append, and vLLM's prefix cache supplies what
was already computed. A cache hit that covers a requested offset would return a row the engine did not compute in
this pass, so such a pass (`over_hit`) is redone with cache reads off. An input with images cannot be appended on
this route (`images_need_retained`): every append would re-send and re-process every earlier image.

## The vLLM patch

vLLM 0.33.0 has no way to hold a pooling request open and feed it more input. Patch 0001 adds it, in four files of
the engine core:

- a pooling request accepts a streaming input, with the offsets to read for each input in
  `PoolingParams.extra_kwargs["arc_gather"]`;
- a paused pooling session resumes with the update's params;
- on the V2 model runner, a resumed request is extended in place: only the new token ids, M-RoPE positions and
  block ids are written. Without this, vLLM re-adds the whole request on every append, which costs O(session
  length).

An update that brings an image takes vLLM's full re-add; a text update after it extends in place again.

The patch is not in this repository: it is the one commit of the vLLM fork's `arc/v0.33.0` branch, published
as the asset `arc-v0.33.0.patch` of that fork's release named on `engine.lock`'s `patch_source` line, and pinned there
by sha256. Only the `retained` route uses it.

The plugin also touches three vLLM internals that have no public hook yet. They are listed, with the upstream change
that would remove each, in [vllm-internals.md](vllm-internals.md).

## Lock (`engine.lock`)

The engine's environment, exactly: vLLM's version, commit and wheel (URL, sha256), the release-age floor, each vLLM
patch's source release (`patch_source <repository> <tag> <commit>`) and each patch file with its sha256, the
sha256 of every file the patches leave in the wheel,
and a `pin` for every other distribution of the venv. The pins are one frozen venv: installed by the recipe and
frozen with the plugin's own line, they give a single freeze sha256, which `/health` reports. The format is in
`engine/compat.py`; `recipe()`, `install.sh` and `fetch_patches.py` read it, and the engine id's plugin digest
covers it. Changing any entry changes the engine id (comments and spacing do not).

## Image recipe

`vllm_arc.engine.recipe.recipe()` returns the build as plain data (this repository holds no cloud SDK code). A
deployment applies it in this order:

1. On a fresh image: `r["base_image"]`, then `apt_install(*r["apt"])`. On an image built from the same base with uv
   and these apt packages, skip both.
2. Run `r["commands"]`: uv, a fresh venv at `r["venv"]` (`uv venv --clear` wipes one already there, `.pth` files
   included), then one install, `--no-deps` under `--exclude-newer` (the release-age floor): the vLLM release wheel
   by PyPI URL and sha256 (uv refuses a mismatch) and every other distribution at `engine.lock`'s exact version
   (torch, FlashInfer, Triton, the CUDA 13.0 JIT toolchain FlashInfer needs, `ninja`, and the rest of the transitive
   set). Nothing is resolved, so nothing floats.
3. Put each of `r["patches"]` at its `path` (`python3 deploy/fetch_patches.py <dir>` downloads them from
   `r["patch_source"]`'s release and checks each sha256; a private fork needs `GH_TOKEN`). Copy this repository at
   the deployed commit to `r["plugin_dir"]`, then run `r["plugin_commands"]`: a sha256 check and `git apply` of each
   patch onto the installed wheel, the plugin with `--no-deps`, the venv's freeze to `r["freeze_path"]`, and an
   import of the server.
4. Optional Triton autotune pins: copy the deployment's pins module to `r["triton_pins_target"]`, run
   `r["triton_pins_commands"]`, add `r["triton_pins_env"]`.
5. Set `r["env"]`, plus `ARC_VLLM_ARC_COMMIT=<the commit copied in step 3>` (reported on `/health`).

Start the server with `r["server_command"] + ["--port", "<port>"]`. `deploy/install.sh` runs the same steps on a
host.

## Load guard

Before any engine or model is built, `compat.require()` (at the top of the server's `main`, and at the import of
`engine/models.py`, where vLLM's registry loads `ArcBaseForPooling`) refuses a vLLM other than the lock's, without
importing vLLM. It has two levels:

- Always: the installed `vllm` distribution's version must be the lock's.
- On the `retained` route, which needs the patch: the sha256 of every file the locked patches change must also be
  the locked one, so a missing, edited or extra patch on those files is refused.

A refusal raises `compat.IncompatibleEngine`. Other files of the wheel are not read: the image installs the wheel
by sha256. On any other route, and when vLLM loads the architectures outside the server, the plugin runs on the
unpatched vLLM release.

`VLLM_ARC_UNLOCKED=1` turns a refusal into a warning, for trying a vLLM build the lock does not name. The engine id
then ends in `+unqualified`, and the start-up checks still apply.

## Engine arguments

`server.engine_args` builds every engine with the pooling runner, bfloat16, eager execution (EQ: compiled), chunked
prefill, FlashInfer's GatedDeltaNet prefill, and the probe as `worker_extension_cls`.

- On `retained`, prefix caching is off. vLLM then sets mamba mode `none`, so there is no block-aligned split and a
  prefill chunk is the whole 8192-token budget.
- On `prefix`, prefix caching runs in `align` mode on the hit grid, with sha256 block hashes and no retention.

**The vision tower.** vLLM runs the vision tower at start even when no image is served (it profiles the ViT on a
dummy image), so every engine configures it:

- `mm_encoder_attn_backend=TORCH_SDPA`: vLLM's default, FLASH_ATTN, has no float32 kernel.
- `mm_device_do_normalize=False`: vLLM's default moves normalisation into a fused `x * w + b` kernel on the device,
  which rounds differently from the image processor.
- `ArcBaseForPooling` runs the ViT in float32, one call per image, and casts each image's embeddings to bfloat16.
- `mm_processor_kwargs {do_rescale: false, do_normalize: false}`. The processor emits uint8 patches and
  `models.vit_pixels` maps each level to the Hugging Face processor's float32 value. vLLM casts any floating
  processor output to bfloat16 on the host, so normalising in the processor would round the ViT's input.

With images served, also `limit_mm_per_prompt {image: 32, video: 0}`, `mm_processor_kwargs {do_resize: false, size:
the rule's band}` and `mm_processor_cache_gb 0`.

**Image inputs.** The client expands each image to its pad tokens and declares it; the server checks the pixels
against their hash and the image rule before the pass, and checks after the pass that vLLM processed the input to
exactly the client's expanded ids (`image_input_refused` otherwise).

## EQ

With a question-head add-on the server builds `ArcEqForPooling` (`engine/models.py`): the base at all 24 layers with
the add-on's layers 12-23. `vllm_arc.eq.addon` verifies the add-on against its manifest and against the served
base before anything loads; an add-on that is refused leaves the server serving the base.

- **Rows.** `norm(h_12) | norm(h_24)`. Each readout depth's `(hidden, residual)` pair goes through the final norm,
  the fused add-and-norm, so `h_12` is the 12-layer engine's readout and not `norm(hidden + residual)`. Checked on
  CPU in `tests/test_stock_eq_model.py`.
- **Route.** `prefix`, text only; `retained` is refused. Encodes and admission are `vllm_arc.eq.encodes.EqEncodes`.
- **Hits past a readout.** vLLM's prefix cache has no cap below a request's first readout, and the patch adds none:
  a hit past a readout (`hit_over_readout`) re-encodes that request under a fresh salt.
- **Compiled.** torch.compile (Inductor, custom ops off) with piecewise CUDA graphs (`startup.COMPILATION`); E12
  and images stay eager. `ArcEqModel` is built in `Qwen3_5Model`'s place, because vLLM's compile wrapper binds the
  forward when the model is built. The norms run Inductor's native arithmetic, so EQ's `h_12` does not equal E12's
  eager readout bit for bit.
- **Start-up.** The assertion `compiled` replaces `enforce_eager`, and `eq_readout_depths` requires `ArcEqModel`
  with readout depths `[12, 24]`.

## Engine id

```
vllm@4ba787ec9959+arc-<plugin digest>+gather-flashinfer+depth12+rope-<rope>+stock-<route>
```

`<route>` is `retained` or the prefix hit grid `u<unit>`; with an empty RoPE setting the id has no `+rope-` part. With images served the id adds
`+images-<rule>-vitfloat32-<settings digest>`; EQ's is the prefix route's plus `+eq-<add-on directory>`. An engine
loaded under `VLLM_ARC_UNLOCKED=1` ends in `+unqualified`.

The plugin digest (`engine/digest.py`) covers the code of `vllm_arc/__init__.py`, `engine/`, `eq/` and `server/`
with comments and docstrings dropped, and `engine.lock`, which names the patch by sha256. Every token of the id is
an input to the arithmetic, so reference rows recorded on one engine id never verify on another.

## Start-up checks

The server refuses to start (exit 3, one `arc_stock_startup_failed` JSON line) unless every assertion in
`startup.assertions` holds:

- vLLM 0.33.0, CUDA runtime 13, the V2 model runner;
- `ArcBaseForPooling` with `GatherPooler`;
- FlashInfer GatedDeltaNet prefill in every layer, float32 GatedDeltaNet state, chunked prefill;
- fused QK off on every full-attention layer;
- eager mode (EQ: compiled, native norms);
- on `retained`: patch 0001 with its V2 runner hook, and prefix caching off (mamba mode `none`);
- on `prefix`: prefix caching in `align` mode on the configured hit grid, sha256 hashes, no retention;
- the configured RoPE as loaded, and the YaRN rotary module as built (its inverse frequencies by sha256);
- video pruning off.

And every one in `startup.vision_assertions`: ViT parameters float32, ViT attention TORCH_SDPA, no fused device
norm, the ViT weights replicated, and with images served the 32-image, no-video limits.

Then the GPU self-test:

1. Sparse readouts equal the all-rows readout of the same input bit for bit (1k and 20k tokens).
2. Against a one-shot of the same 1,064 tokens: on `retained`, two appends through one live session (1,000 then 64
   tokens) are finite and in shape, and the second took the patch's in-place extension
   (`selftest.retained.extends` > 0, counted by the runner); on `prefix`, an append is finite and hits no further
   than its stored prefix.
3. With images served (`selftest.images`): one input with two images of different sizes expands to the declared
   pads, runs the ViT exactly twice and reads finite rows; on `retained`, a session of text, then an input with one
   image (one ViT call), then a text append that takes the in-place extension and runs no ViT.

The result of every check and of the self-test is on `/health`.
