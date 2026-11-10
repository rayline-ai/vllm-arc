# vllm-arc

A [vLLM](https://github.com/vllm-project/vllm) plugin for **Rayline ARC** releases. It registers the release
encoders with vLLM and runs the release's Stage-1 head next to them.

> **Work in progress: end-to-end serving is not yet supported.**
> This repository has the engine side only: the encoder architectures and the Stage-1 head loader. The
> decision/policy API is **not included yet**. That covers Stage 2, the caller-facing signals (PII and topic), the
> question head, and the decide routes. Stage-1 scoring runs through `vllm_arc.heads`. Interfaces may change without
> notice.

Weights: [huggingface.co/rayline-ai/ARC-1.0](https://huggingface.co/rayline-ai/ARC-1.0) (release `20261111`).

## What's here

| Module | What it does |
|---|---|
| `vllm_arc/__init__.py` | `register()`, the `vllm.general_plugins` entry point. It registers `ArcPolicyForPooling` (text) and `ArcImagePolicyForPooling` (image-reading releases), along with their config hooks. |
| `vllm_arc/model.py` | The two architectures. The release's truncated Qwen3.5 stack runs as a pooling model, and head tensors are skipped at load. |
| `vllm_arc/config.py` | The image release's engine settings: a float32 TORCH_SDPA ViT and one ViT call per image. |
| `vllm_arc/heads.py` | `load_stage_one(release_dir)` checks the release manifest's sha256 pins and returns the Stage-1 head as a torch module. |
| `vllm_arc/arc_heads.py` | The Stage-1 head itself. It imports torch only. |
| `examples/readouts.py` | Block readouts (token GATHER) from one vLLM pooling engine, with optional per-module diagnostics. Requires the fork. |

## Installing vLLM (pinned build + fork overlay)

ARC readouts are checked for bit-level parity against one specific vLLM build. They come from
[`atlasfutures/vllm@66029e2d`](https://github.com/atlasfutures/vllm/commit/66029e2de173524d192e4b41578fbcd4a1146c98),
which sits 23 commits on top of upstream
[`98e91a96`](https://github.com/vllm-project/vllm/commit/98e91a9600eb75b2de14ef27f13b10088d1a1279). Install the
upstream nightly wheel for that commit, then copy the fork's changed Python files over it. This was tested on
Linux with CUDA 13.0 and Python 3.12.

```bash
BASE=98e91a9600eb75b2de14ef27f13b10088d1a1279
FORK=66029e2de173524d192e4b41578fbcd4a1146c98
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install "vllm==0.26.1rc1.dev36+g98e91a960" \
  --extra-index-url "https://wheels.vllm.ai/$BASE/cu130" --index-strategy unsafe-best-match

git init vllm-fork && git -C vllm-fork remote add origin https://github.com/atlasfutures/vllm.git
git -C vllm-fork fetch --depth 1 origin "$FORK" "$BASE" && git -C vllm-fork checkout --detach "$FORK"
SITE=$(python -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')
(cd vllm-fork && git diff --name-only --diff-filter=AM "$BASE..$FORK" -- vllm/ \
  | while read -r f; do cp "$f" "$SITE/$f"; done)

uv pip install "git+https://github.com/rayline-ai/vllm-arc"
```

What needs the fork:

- **GATHER readouts at chosen offsets** (`PoolingParams(readout_offsets=...)` with `tok_pooling_type="GATHER"`).
  ARC reads the encoder's hidden state at block boundaries, and stock vLLM has no equivalent. `examples/readouts.py`
  depends on it.
- Pooling sessions (incremental encodes that keep their state), the GatedDeltaNet `torch_reference` prefill backend,
  and `mm_encoder_dtype`. The reference serving setup uses these. The plugin's model classes don't depend on them
  directly.

The model registration and the Stage-1 head don't need the fork, but readout parity has only been verified on
the build above.

## Usage

```python
from vllm import LLM
from vllm.config import PoolerConfig
from vllm.pooling_params import PoolingParams

# The plugin registers itself through its entry point when vLLM starts.
llm = LLM(model="path/to/arc-release", runner="pooling", dtype="bfloat16", enforce_eager=True,
          enable_prefix_caching=False,
          pooler_config=PoolerConfig(task="token_embed", tok_pooling_type="GATHER", use_activation=False))
[out] = llm.encode([{"prompt_token_ids": token_ids}], pooling_task="token_embed",
                   pooling_params=PoolingParams(task="token_embed", use_activation=False,
                                                readout_offsets=offsets))          # fork only
rows = out.outputs.data.float()

from vllm_arc.heads import load_stage_one
head = load_stage_one("path/to/arc-release", device="cuda")
p, usage = head(rows, "fixed")   # success probability [M, n, C] and per-leg token usage [M, n, C, 4]
```

The arguments a head needs, such as `context` and `cache` features, depend on the release. `load_stage_one` refuses
a release whose files don't match its `arc_release.json` pins.

## Development

```bash
pip install -e ".[test]"
pytest tests/           # CPU only, uses a stand-in for vLLM
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). ARC encoders are derived from
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache-2.0).
