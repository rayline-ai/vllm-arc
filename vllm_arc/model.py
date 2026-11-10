"""``ArcPolicyForPooling``: the release's truncated Qwen3.5 text stack in vLLM; ``ArcImagePolicyForPooling``: the
same stack with the vision tower, for a release whose encoding profile reads images (below).

The release ships a text-only checkpoint (``model.*``: embeddings, the first ``depth`` decoder layers, the final
norm), which is exactly vLLM's ``Qwen3_5ForCausalLM`` layout; the model is that class plus the hybrid-state
interface vLLM needs to size the GatedDeltaNet cache (Qwen3.5's multimodal top-level class carries it; the text
class does not, because upstream never serves it as a top-level model). The pooling runner wraps it with the
configured pooler (token GATHER for block readouts), as it wraps the multimodal reference model. The heads
(``stage_one.*``, ``stage_two.*``, ``signals.*``) are read from the same file by the decision layer, not by the
engine, so loading skips them. Qwen3.5's config declares M-RoPE, which the multimodal class positions; for text the
three position streams are equal, and this class supplies them.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration

from . import VISION_ATTENTION

#: The release's head tensors, which the decision layer reads and the engine never loads: Stage 1, Stage 2 and the
#: caller-facing signal heads (``signals.*``).
HEAD_PREFIXES = ("stage_one.", "stage_two.", "signals.")


class ArcPolicyForPooling(Qwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    # The multimodal class's classmethods read only the text config, which is this model's whole config.
    get_mamba_state_dtype_from_config = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config.__func__)
    get_mamba_state_shape_from_config = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config.__func__)
    get_mamba_state_copy_func = classmethod(Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func.__func__)

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # vLLM fuses the full-attention layers' gate split + QK-RMSNorm + RoPE into one Triton kernel only when the
        # model has no multimodal config ("text only"), which this class never has and the multimodal reference
        # class always has. The fused kernel rounds differently from the eager split/norm/M-RoPE path, which moves
        # the readouts off the reference encoder's (cosine 0.9998 on L4). Take the reference model's path.
        for layer in self.model.layers:
            if hasattr(getattr(layer, "self_attn", None), "use_fused_qk_norm_rope_gate"):
                layer.self_attn.use_fused_qk_norm_rope_gate = False

    def get_mrope_input_positions(self, input_tokens: list[int], mm_features: list) -> tuple[torch.Tensor, int]:
        """Text only: the T, H and W streams are each the token index (what Qwen3-VL computes with no media)."""
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1).clone(), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights((name, w) for name, w in weights if not name.startswith(HEAD_PREFIXES))


class ArcImagePolicyForPooling(Qwen3_5ForConditionalGeneration):
    """An image-reading release: ``ArcPolicyForPooling`` plus Qwen3.5's own vision tower.

    The release keeps the base model's multimodal layout (``model.visual.*``, ``model.language_model.*`` truncated
    to ``depth`` layers), so this is the multimodal Qwen3.5 class, which also brings Qwen3-VL's
    M-RoPE rule over image spans (``get_mrope_input_positions``; the encoding profile's ``qwen3_vl_mrope``) and the
    eager attention path (a model with a multimodal config never takes the fused one). On top of it, the profile's
    ``vision`` compute: the ViT runs in float32 with the TORCH_SDPA backend
    (``config.ArcImagePolicyConfig`` sets it), one ViT call per image, and its embeddings are cast to the language
    model's bfloat16 once, where they replace the image pads (``_merge_multimodal_embeddings``). A bfloat16 ViT, or
    images batched through one call, moves readouts off the training encoder's.
    """

    def __init__(self, *, vllm_config, prefix: str = "model") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        backend = getattr(self.visual.attn_backend, "name", self.visual.attn_backend)
        if backend != VISION_ATTENTION:
            raise ValueError(f"{type(self).__name__} runs its vision tower with {VISION_ATTENTION}, not {backend}: "
                             "start vLLM through the vllm-arc plugin, which sets mm_encoder_attn_backend")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights((name, w) for name, w in weights if not name.startswith(HEAD_PREFIXES))
        self.visual.float()  # the profile's ViT dtype; the checkpoint's bfloat16 values are exact in float32
        return loaded

    def _process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
        """One float32 ViT call per image; upstream runs every image in one call."""
        if image_input["type"] != "pixel_values":
            raise ValueError("an ARC release encodes images from their pixels, never precomputed embeddings")
        grid_thw = image_input["image_grid_thw"]
        pixels = image_input["pixel_values"].type(self.visual.dtype)
        patches = grid_thw.prod(-1).tolist()
        if pixels.shape[0] != sum(patches):
            raise ValueError(f"{pixels.shape[0]} pixel patches for image grids totalling {sum(patches)}")
        return tuple(self.visual(chunk, grid_thw=grid_thw[i:i + 1])
                     for i, chunk in enumerate(pixels.split(patches)))
