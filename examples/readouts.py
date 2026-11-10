"""Block readouts from one vLLM pooling engine (runs in the vLLM environment): ``readouts.py MODEL OUT.npz``.

Reads ``cases.json`` beside OUT ({"cases": [{"tokens": [...], "offsets": [...]}]}) and writes each case's GATHER
rows, the reference encode (token GATHER, no activation, bfloat16, eager). Requires the vLLM fork (``readout_offsets``).
``ARC_MAX_MODEL_LEN`` (default 65,536) is the engine's context; the probe raises it for a long case.

A case may carry ``images`` (paths of PNGs already sized by the ARC image rule, in prompt order): its
``tokens`` then hold one ``<|image_pad|>`` per image, which vLLM expands, and ``offsets`` are in the expanded
sequence. ``ARC_VIT_FP32=1`` runs a stock multimodal engine under the image profile's vision compute (float32
TORCH_SDPA ViT, one call per image; what ``ArcImagePolicyForPooling`` does itself), as the reference for a release.

With ``ARC_DIAG=<path>`` (and ``VLLM_ENABLE_V1_MULTIPROCESSING=0``, so the model is in this process) it also
writes the engine's resolved config and every decoder sub-module's output over the first case's forward to
``<path>`` (torch.save): the evidence that localizes a readout drift between two engines to one module.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    from vllm import LLM
    from vllm.config import PoolerConfig
    from vllm.pooling_params import PoolingParams

    model, out = sys.argv[1], Path(sys.argv[2])
    cases = json.loads((out.parent / "cases.json").read_text())["cases"]
    most_images = max((len(case.get("images", ())) for case in cases), default=0)
    vit_fp32 = os.environ.get("ARC_VIT_FP32") == "1"
    llm = LLM(model=model, tokenizer=os.environ.get("ARC_TOKENIZER", model), runner="pooling", dtype="bfloat16",
              max_model_len=int(os.environ.get("ARC_MAX_MODEL_LEN", 65536)), enforce_eager=True, enable_prefix_caching=False,
              gdn_prefill_backend=os.environ.get("ARC_VLLM_BACKEND", "flashinfer"),
              gpu_memory_utilization=0.6,
              **({"mamba_ssm_cache_dtype": os.environ["ARC_SSM_DTYPE"]} if os.environ.get("ARC_SSM_DTYPE") else {}),
              **({"block_size": int(os.environ["ARC_BLOCK_SIZE"])} if os.environ.get("ARC_BLOCK_SIZE") else {}),
              **({"limit_mm_per_prompt": {"image": most_images, "video": 0}, "mm_processor_cache_gb": 0}
                 if most_images else {}),
              **({"mm_encoder_attn_backend": "TORCH_SDPA"} if vit_fp32 else {}),
              pooler_config=PoolerConfig(task="token_embed", seq_pooling_type="MEAN", tok_pooling_type="GATHER", use_activation=False))
    if vit_fp32:
        llm.apply_model(_vision_compute)
    vision = llm.apply_model(lambda m: {"vit_dtype": str(getattr(getattr(m, "visual", None), "dtype", None)),
                                        "vit_attn_backend": str(getattr(getattr(m, "visual", None), "attn_backend",
                                                                        None))})[0]
    if os.environ.get("ARC_DIAG"):
        _diagnose(llm, cases[0]["tokens"], os.environ["ARC_DIAG"])
    rows = {}
    for i, case in enumerate(cases):
        [result] = llm.encode([_prompt(case)], pooling_task="token_embed",
                              pooling_params=PoolingParams(task="token_embed", use_activation=False,
                                                           readout_offsets=case["offsets"]))
        rows[f"case{i}"] = result.outputs.data.float().cpu().numpy()
    np.savez(out, **rows)
    cache = llm.llm_engine.vllm_config.cache_config
    print(json.dumps({"model": model, "architecture": llm.llm_engine.model_config.architectures,
                      "rotary": llm.apply_model(_rotary_facts)[0],
                      "mamba_ssm_cache_dtype": str(cache.mamba_ssm_cache_dtype),
                      "mamba_cache_dtype": str(cache.mamba_cache_dtype), **vision,
                      "expanded_lengths": {f"case{i}": len(case["tokens"]) - len(case.get("images", ()))
                                           + sum(case.get("image_tokens", ())) for i, case in enumerate(cases)},
                      "shapes": {k: list(v.shape) for k, v in rows.items()}}), flush=True)


def _prompt(case: dict) -> dict:
    prompt = {"prompt_token_ids": case["tokens"]}
    if case.get("images"):
        from PIL import Image

        prompt["multi_modal_data"] = {"image": [Image.open(path).convert("RGB") for path in case["images"]]}
        prompt["mm_processor_kwargs"] = {"do_resize": False}  # already sized by the ARC image rule; never resized again
    return prompt


def _vision_compute(model) -> None:
    """ARC_VIT_FP32: a stock Qwen3.5 engine under the image profile's vision compute, the reference ``ArcImagePolicyForPooling`` meets."""
    import torch

    model.visual.float()

    def per_image(image_input):
        grid = image_input["image_grid_thw"]
        pixels = image_input["pixel_values"].type(model.visual.dtype)
        return tuple(model.visual(chunk, grid_thw=grid[i:i + 1])
                     for i, chunk in enumerate(pixels.split(grid.prod(-1).tolist())))

    model._process_image_input = per_image
    assert model.visual.dtype == torch.float32


def _rotary_facts(model) -> dict:
    """Each attention layer's constructed rotary embedding: the evidence that a YaRN config reached the arithmetic
    (class, scaling factor, mscale) rather than only the reported ``rope_type``."""
    facts = {}
    for name, module in model.named_modules():
        rotary = getattr(module, "rotary_emb", None)
        if rotary is not None:
            facts[name] = {"class": type(rotary).__name__,
                           "scaling_factor": getattr(rotary, "scaling_factor", None),
                           "mscale": float(getattr(rotary, "mscale", 1.0) or 1.0),
                           "cache_positions": int(rotary.cos_sin_cache.shape[0])
                           if hasattr(rotary, "cos_sin_cache") else None}
    return facts


def _diagnose(llm, tokens: list[int], path: str) -> None:
    """One forward over ``tokens`` with hooks on every decoder sub-module; save their outputs and the config."""
    import torch
    from vllm.pooling_params import PoolingParams

    captured: dict = {}
    calls: list[dict] = []
    facts: dict = {}
    hooked = ("embed_tokens", "input_layernorm", "linear_attn", "self_attn", "post_attention_layernorm", "mlp",
              "norm", "rotary_emb")

    def install(model):
        handles = []
        facts["model_class"] = [c.__name__ for c in type(model).__mro__][:12]
        for name, module in model.named_modules():
            if hasattr(module, "rotary_emb"):
                facts[f"{name}.rotary_emb"] = type(module.rotary_emb).__name__
            if hasattr(module, "gdn_prefill_backend"):
                facts[f"{name}.gdn_prefill_backend"] = str(module.gdn_prefill_backend)
            if hasattr(module, "attn") and hasattr(module.attn, "attn_backend"):
                facts[f"{name}.attn_backend"] = str(module.attn.attn_backend)
            if name.rsplit(".", 1)[-1] in hooked or type(module).__name__.endswith("DecoderLayer"):
                def hook(mod, args, kwargs, out, name=name):
                    if name in captured:
                        return
                    outs = out if isinstance(out, tuple) else (out,)
                    captured[name] = [o.detach().clone().cpu() for o in outs if isinstance(o, torch.Tensor)]
                    if name.endswith("rotary_emb"):
                        pos = kwargs.get("positions", args[0] if args else None)
                        captured[name + ".positions"] = [pos.detach().clone().cpu()]
                handles.append(module.register_forward_hook(hook, with_kwargs=True))

        def top(mod, args, kwargs):
            if calls:
                return
            names = ("input_ids", "positions", "intermediate_tensors", "inputs_embeds")
            seen = dict(zip(names, args)) | {k: v for k, v in kwargs.items() if k in names}
            calls.append({k: (None if v is None else [str(getattr(v, "dtype", "")), list(getattr(v, "shape", []))])
                          for k, v in seen.items()})
            for k, v in seen.items():
                if isinstance(v, torch.Tensor):
                    captured[f"forward.{k}"] = [v.detach().clone().cpu()]
        handles.append(model.register_forward_pre_hook(top, with_kwargs=True))
        return handles

    [handles] = llm.apply_model(install)
    llm.encode([{"prompt_token_ids": tokens}], pooling_task="token_embed",
               pooling_params=PoolingParams(task="token_embed", use_activation=False,
                                            readout_offsets=[len(tokens) - 1]))
    for handle in handles:
        handle.remove()
    config = llm.llm_engine.vllm_config
    model_config = config.model_config
    facts.update({
        "forward_call": calls[0] if calls else None,
        "model_config": {k: str(getattr(model_config, k, None)) for k in (
            "architecture", "architectures", "is_multimodal_model", "uses_mrope", "is_hybrid", "dtype",
            "max_model_len", "attn_type", "is_chunked_prefill_supported")},
        "hf_text_config": model_config.hf_text_config.to_dict(),
        "cache_config": {k: str(v) for k, v in vars(config.cache_config).items()},
        "scheduler_config": {k: str(v) for k, v in vars(config.scheduler_config).items()},
        "compilation_config.mode": str(config.compilation_config.mode),
        "additional_config": str(config.additional_config),
    })
    torch.save({"facts": facts, "captured": captured}, path)


def compare_diagnoses(left_path: str, right_path: str) -> dict:
    """``readouts.py --compare A.pt B.pt``: config facts that differ, and each hooked output, in execution order."""
    import torch

    def load(path):
        saved = torch.load(path, weights_only=False)
        # The multimodal class nests the text model under ``language_model.``; align the two module trees.
        return saved["facts"], {k.removeprefix("language_model."): v for k, v in saved["captured"].items()}

    (left_facts, left), (right_facts, right) = load(left_path), load(right_path)

    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
        return out

    lf, rf = flat({k.removeprefix("language_model."): v for k, v in left_facts.items()}), \
        flat({k.removeprefix("language_model."): v for k, v in right_facts.items()})
    facts_diff = {k: [str(lf.get(k)), str(rf.get(k))] for k in sorted(lf.keys() | rf.keys())
                  if str(lf.get(k)) != str(rf.get(k))}
    modules = []
    for name in [*left, *(k for k in right if k not in left)]:
        if name not in left or name not in right:
            modules.append({"module": name, "only_in": "left" if name in left else "right"})
            continue
        row = {"module": name, "outputs": []}
        for x, y in zip(left[name], right[name]):
            if x.shape != y.shape or x.dtype != y.dtype:
                row["outputs"].append({"shape": [list(x.shape), list(y.shape)], "dtype": [str(x.dtype), str(y.dtype)]})
                continue
            xf, yf = x.float(), y.float()
            row["outputs"].append({"shape": list(x.shape), "dtype": str(x.dtype), "equal": bool(torch.equal(x, y)),
                                   "max_abs_diff": float((xf - yf).abs().max()) if x.numel() else 0.0,
                                   "n_diff": int((x != y).sum())})
        modules.append(row)
    return {"facts_diff": facts_diff, "left_facts": left_facts, "right_facts": right_facts, "modules": modules}


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        print(json.dumps(compare_diagnoses(sys.argv[2], sys.argv[3]), default=str), flush=True)
    else:
        main()
