# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""Offline encode with ``vllm.LLM``: the hidden state at chosen token offsets, from the ARC encoder.

    python examples/offline_encode.py --text "The quick brown fox" --offsets=-1,0

Prerequisites:

- A CUDA GPU, and the engine environment ``docs/engine.md`` describes: vLLM 0.33.0 at ``engine.lock``'s pins, with
  this plugin installed into it (``pip install --no-deps .``). FlashInfer's GDN prefill kernel is built at first use,
  so ``CUDA_HOME`` must point at a CUDA 13 toolkit and the venv's ``ninja`` must be on ``PATH``.
- The locked vLLM patch, today. This example runs one-shot encodes only, which stock 0.33.0 serves (the patch is
  needed for retained sessions, which this example does not use), but the plugin's load guard
  (``vllm_arc.engine.compat.require``, run when vLLM imports the model) currently refuses a vLLM without the locked
  patch. ``docs/engine.md`` ("Start-up") states what the guard checks.
- Network access to the Hugging Face Hub on the first run: the pinned base checkpoint is downloaded and cut to the
  served depth once, under ``--checkpoint-root``, exactly as the server builds it.

The engine is built with the server's own engine arguments (``vllm_arc.server.app.engine_args``) for a one-shot
engine: pooling runner, bfloat16, eager, prefix caching off. Each prompt asks ``GatherPooler`` for the rows at its
offsets (``vllm_arc.server.app.gather_params``), so the output is ``[len(offsets), hidden]`` in request order. Unlike
the server, this script does not run the start-up assertions or the GPU self-test.
"""
from __future__ import annotations

import argparse
import os

from vllm_arc.engine import compat
from vllm_arc.engine.startup import Engine
from vllm_arc.server.app import (DEPTH, MODEL_ID, ROPE_CAPACITY, engine_args, engine_prompt, gather_params,
                                 truncated_checkpoint)


def parse_offsets(spec: str, length: int) -> list[int]:
    """Comma-separated offsets; a negative one counts from the end, as a Python index."""
    offsets = []
    for part in spec.split(","):
        offset = int(part)
        offset = offset + length if offset < 0 else offset
        if not 0 <= offset < length:
            raise SystemExit(f"offset {part} is outside the {length}-token input")
        offsets.append(offset)
    return offsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--text", action="append",
                        help="an input text, tokenized with the model's tokenizer (repeat for a batch)")
    parser.add_argument("--offsets", default="-1",
                        help="comma-separated token offsets to read, negative from the end (default: -1, the last)")
    parser.add_argument("--rope", default="yarn-f4", choices=sorted(ROPE_CAPACITY),
                        help="RoPE variant; empty string for the checkpoint's own (default: yarn-f4, as the server)")
    parser.add_argument("--max-model-len", type=int, default=32768,
                        help="engine context in tokens (the server's default is 262144)")
    parser.add_argument("--budget", type=int, default=8192, help="max tokens per engine step (chunked prefill)")
    parser.add_argument("--max-num-seqs", type=int, default=8, help="max requests in one engine step")
    parser.add_argument("--gpu-memory", type=float, default=0.80, help="gpu_memory_utilization")
    parser.add_argument("--checkpoint-root", default=os.environ.get("ARC_ENCODER_CHECKPOINTS",
                                                                  "/tmp/arc-encoder-checkpoints"),
                        help="where the truncated checkpoint is built once")
    args = parser.parse_args()
    texts = args.text or ["The ARC encoder reads one hidden state per requested offset."]
    # The same guard vLLM's model import runs, before anything is downloaded or built.
    compat.require()

    from vllm import LLM
    from vllm.config import PoolerConfig

    # The pinned base cut to the served depth, its config naming ArcBaseForPooling (the plugin's entry point
    # registers it when vLLM starts).
    checkpoint = truncated_checkpoint(args.checkpoint_root, args.rope)
    kwargs = engine_args(Engine(sessions="retained"), checkpoint, budget=args.budget,
                         max_model_len=args.max_model_len, max_sessions=args.max_num_seqs, scratch_slots=0,
                         gpu_memory=args.gpu_memory)
    # AsyncLLM-only arguments: the server's request logging switch.
    kwargs.pop("enable_log_requests", None)
    kwargs["pooler_config"] = PoolerConfig(**kwargs["pooler_config"])
    llm = LLM(**kwargs)

    tokenizer = llm.get_tokenizer()
    prompts, params = [], []
    for text in texts:
        tokens = tokenizer.encode(text)
        offsets = parse_offsets(args.offsets, len(tokens))
        prompts.append(engine_prompt(tokens, None))
        # skip_reading_prefix_cache: a one-shot reads no cached prefix (prefix caching is off here anyway).
        params.append(gather_params(offsets, True))

    outputs = llm.encode(prompts, pooling_params=params, pooling_task="token_embed", use_tqdm=False)
    print(f"model {MODEL_ID} cut to {DEPTH} layers, rope {args.rope or 'native'}")
    for text, prompt, param, output in zip(texts, prompts, params, outputs):
        rows = output.outputs.data.float().cpu()
        offsets = param.extra_kwargs["arc_gather"]["offsets"]
        print(f"{len(prompt['prompt_token_ids'])} tokens, offsets {offsets}: rows {tuple(rows.shape)} "
              f"(norms {[round(float(n), 3) for n in rows.norm(dim=-1)]}) <- {text[:60]!r}")


if __name__ == "__main__":
    main()
