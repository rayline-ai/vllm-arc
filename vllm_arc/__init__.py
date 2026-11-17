# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""vLLM plugin for Rayline ARC releases.

Installed into a vLLM environment, the ``vllm.general_plugins`` entry point (``register``) registers the ARC encoder
architectures: ``ArcBaseForPooling`` (the encoder) and ``ArcEqForPooling`` (EQ, readouts at several depths). Both are
loaded lazily from ``vllm_arc.engine.models``, so a vLLM process that never loads them never imports that module.
"""

__all__ = ["ARCHITECTURE", "EQ_ARCHITECTURE", "register"]

#: The encoder's architecture, named by the checkpoint the server builds.
ARCHITECTURE = "ArcBaseForPooling"
#: EQ's engine: the base at 24 layers reading ``readout_depths``.
EQ_ARCHITECTURE = "ArcEqForPooling"


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models import config as vllm_model_config

    from .engine.internals import register_config

    for architecture in (ARCHITECTURE, EQ_ARCHITECTURE):
        if architecture not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(architecture, f"vllm_arc.engine.models:{architecture}")
        # vLLM's Qwen3.5 config hook is keyed by architecture: it keeps the GatedDeltaNet recurrent state in the
        # config's mamba_ssm_dtype (float32) between chunked-prefill steps, as for the multimodal Qwen3.5 class.
        register_config(vllm_model_config, architecture, "Qwen3_5ForConditionalGenerationConfig")
