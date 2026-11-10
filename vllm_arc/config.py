"""vLLM config hooks for ARC releases, keyed by release architecture (``MODELS_CONFIG_MAP``)."""
from __future__ import annotations

from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from . import VISION_ATTENTION


class ArcImagePolicyConfig(Qwen3_5ForConditionalGenerationConfig):
    """Qwen3.5's hook (the GatedDeltaNet state in float32), plus the engine settings the image profile's
    ``vision`` compute needs: the TORCH_SDPA ViT backend, which runs float32, and no encoder CUDA
    graph or data-parallel ViT, both of which run several images through one ViT call.

    An unset backend is set; any other choice is refused rather than overridden.
    """

    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        super().verify_and_update_config(vllm_config)
        multimodal = vllm_config.model_config.multimodal_config
        if multimodal is None:
            raise ValueError("an ARC image release needs vLLM's multimodal config")
        if multimodal.mm_encoder_attn_backend is None:
            multimodal.mm_encoder_attn_backend = AttentionBackendEnum[VISION_ATTENTION]
        elif multimodal.mm_encoder_attn_backend.name != VISION_ATTENTION:
            raise ValueError(f"an ARC image release runs its vision tower with {VISION_ATTENTION}, "
                             f"not mm_encoder_attn_backend={multimodal.mm_encoder_attn_backend.name}")
        if multimodal.mm_encoder_tp_mode == "data":
            raise ValueError("an ARC image release runs one ViT call per image; mm_encoder_tp_mode=data batches them")
        if getattr(vllm_config.compilation_config, "cudagraph_mm_encoder", False):
            raise ValueError("an ARC image release runs one ViT call per image; cudagraph_mm_encoder batches them")
