<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc. -->

# vLLM internals the plugin touches

`vllm_arc` is an out-of-tree plugin for vLLM 0.33.0. Apart from public entry points (`ModelRegistry`, subclassing
`Qwen3_5ForConditionalGeneration` and `Qwen3_5Model`), it touches three vLLM internals. Each goes through
`vllm_arc/engine/internals.py`, which checks that the symbol exists first and raises `MissingVllmInternal` naming
the supported vLLM release when it does not. The two temporary replacements are made under one module-level
`threading.Lock` and restored in a `finally`, so a failed build leaves vLLM's module unchanged.

The notes follow the patch-note layout used by vLLM hardware plugins: what is touched, why, how, and the upstream
status with the plan to remove it.

## 1. `vllm.model_executor.models.qwen3_5.mrope_positions_factor`

- **Where:** `vllm_arc/engine/models.py`, `ArcBaseForPooling.__init__` (`qwen3_5_replacements`).
- **Why:** YaRN's correction range reads the rotary module's `max_position_embeddings`, which for M-RoPE is the
  cache length `original x positions factor`. The ARC encoder was trained and qualified with a positions factor of
  4 (correction range (16, 24)). vLLM 0.33.0 uses 1 unless video pruning is on (range (14, 22)), which moves 9 of
  32 inverse frequencies.
- **How:** while the model is constructed, the module-level function is replaced with one returning 4, then
  restored. Video pruning is refused for this architecture, so positions never exceed the token count. The start-up
  probe compares the built inverse frequencies with the expected ones.
- **Upstream status / plan to remove:** not upstream. Proposal: let a `Qwen3_5ForConditionalGeneration` subclass
  override the M-RoPE positions factor (a class attribute or method read where the rotary embedding is built). Once
  a vLLM release has that hook, the plugin overrides it and this replacement is deleted.

## 2. `vllm.model_executor.models.qwen3_5.Qwen3_5Model`

- **Where:** `vllm_arc/engine/models.py`, `ArcEqForPooling` (`qwen3_5_replacements`).
- **Why:** the EQ engine needs a text model whose `forward` returns the normed readout at several depths
  (`ArcEqModel`). It must be built as that class, not swapped in after construction: `support_torch_compile` binds
  `forward` when the model is built, so only a model built as `ArcEqModel` compiles its forward.
- **How:** while the model is constructed, the module's `Qwen3_5Model` name is replaced with `ArcEqModel` (a
  subclass), then restored. The constructor checks that the inner model it got is an `ArcEqModel`.
- **Upstream status / plan to remove:** not upstream. Proposal: an overridable inner-model class on
  `Qwen3_5ForConditionalGeneration` (for example a class attribute naming the text model class it builds). With
  that hook the plugin sets the attribute on its subclass and this replacement is deleted.

## 3. `vllm.model_executor.models.config.MODELS_CONFIG_MAP`

- **Where:** `vllm_arc/__init__.py`, `register()`.
- **Why:** vLLM's per-architecture config hook for Qwen3.5 (`Qwen3_5ForConditionalGenerationConfig`) keeps the
  GatedDeltaNet recurrent state in float32 (`mamba_ssm_dtype`) between chunked-prefill steps. The map is keyed by
  architecture name, so the plugin's architectures get no hook unless they are added.
- **How:** at plugin registration, `MODELS_CONFIG_MAP[<architecture>]` is set to
  `Qwen3_5ForConditionalGenerationConfig` unless an entry already exists (under the same lock).
- **Upstream status / plan to remove:** not upstream. Proposal: a public registration path for a model's config
  hook, or a lookup that follows the model class's bases. Until then the entry is a permanent addition.

## Load guard

`vllm_arc/engine/compat.py` checks the installed vLLM against `vllm_arc/engine/engine.lock` before a model is
built: the release always, and the patched files' sha256 only on the retained session route, which needs the
lock's vLLM patch. `VLLM_ARC_UNLOCKED=1` turns a refusal into a warning and marks the server's engine id
`+unqualified`.
