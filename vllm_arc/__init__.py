"""vLLM plugin for Rayline ARC policy releases.

Installed into the vLLM environment, its ``vllm.general_plugins`` entry point registers the architectures an ARC
release's ``config.json`` names, so ``vllm serve <release> --runner pooling`` loads it: ``ArcPolicyForPooling``
(text) and ``ArcImagePolicyForPooling`` (a release whose encoding profile reads images).
"""

ARCHITECTURE = "ArcPolicyForPooling"
IMAGE_ARCHITECTURE = "ArcImagePolicyForPooling"
#: The image profile's ViT attention backend: it runs float32 (flash attention has no float32 kernel).
VISION_ATTENTION = "TORCH_SDPA"


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Qwen3_5ForConditionalGenerationConfig

    from .config import ArcImagePolicyConfig

    for architecture in (ARCHITECTURE, IMAGE_ARCHITECTURE):
        if architecture not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(architecture, f"vllm_arc.model:{architecture}")
    # vLLM's Qwen3.5 config hook is keyed by architecture: it sets the GatedDeltaNet recurrent-state cache to the
    # config's mamba_ssm_dtype (float32), as for the multimodal Qwen3.5 class. Without it the state is cached in the
    # model dtype (bfloat16) between chunked-prefill steps, and readouts past the first chunk (> 8192 tokens on L4)
    # drift from the multimodal reference encoder's.
    MODELS_CONFIG_MAP.setdefault(ARCHITECTURE, Qwen3_5ForConditionalGenerationConfig)
    MODELS_CONFIG_MAP.setdefault(IMAGE_ARCHITECTURE, ArcImagePolicyConfig)
