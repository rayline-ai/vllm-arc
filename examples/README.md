<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc. -->

# Examples

Two ways to read ARC encoder rows: the hidden state at chosen token offsets, one row per offset.

| Script | Runs where | Needs |
|---|---|---|
| [`offline_encode.py`](offline_encode.py) | the GPU host, in-process `vllm.LLM` | a CUDA GPU and the engine environment of [`docs/engine.md`](../docs/engine.md) with this plugin installed |
| [`encode_client.py`](encode_client.py) | anywhere that reaches the server | Python 3.10+ only (numpy optional) |

## Offline: `vllm.LLM.encode`

```sh
python examples/offline_encode.py --text "The quick brown fox" --offsets=-1,0
```

It builds the truncated checkpoint the way the server does (the pinned base cut to 12 layers, under
`--checkpoint-root`, downloaded on the first run), constructs `vllm.LLM` with the server's engine arguments for a
one-shot engine (pooling runner, bfloat16, eager, prefix caching off) and calls `LLM.encode(...,
pooling_task="token_embed")` with the plugin's gather parameters. It prints each input's row shape,
`(len(offsets), hidden)`.

One-shot encodes use stock vLLM 0.33.0's pooling path; the locked vLLM patch is needed only for the server's
retained sessions. The plugin's load guard currently still refuses a vLLM without that patch, so today the example
needs the full engine environment; see the docstring and `docs/engine.md` ("Start-up").

## Server: `POST /encode`

```sh
vllm-arc-server --port 8765          # on the GPU host
python examples/encode_client.py --url http://127.0.0.1:8765 --tokens 1000,1001,1002,1003 --offsets 3,0
```

The client checks `GET /health`, then sends one one-shot `POST /encode` (`"retain": false`) with token ids and
offsets, and decodes the float32 response body using the `x-rows` and `x-dim` headers. Import it to use
`health()`, `encode()` and `decode_rows()` from your own code. The full wire contract, retained sessions and the
refusal codes are in [`docs/engine.md`](../docs/engine.md) ("Endpoints").
