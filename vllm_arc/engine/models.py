# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The ARC encoder architecture on stock vLLM 0.33.0.

``ArcBaseForPooling``: Qwen3.5's multimodal class (the E12 shape: the base checkpoint cut to depth 12, the vision
tower kept) with ``GatherPooler`` as its pooler. The server's own checkpoint names it in ``architectures``, so no
in-tree architecture is overridden.

It forces the eager QK-norm/RoPE/gate path on every full-attention layer (stock takes the fused Triton kernel for
this class, which moved readouts on the fork). What stock chose is kept on the model (``arc_fused_qk_stock``) for the
start-up probe.

It builds M-RoPE with the fork's positions factor 4 (``FORK_MROPE_POSITIONS_FACTOR``). YaRN's correction range reads
the rotary module's ``max_position_embeddings``, which for M-RoPE is the cache length ``original x positions factor``;
the fork's ``mrope.py`` fixed that factor at 4 (correction range (16, 24)), while stock 0.33 uses 1 unless video
pruning is on (range (14, 22): 9 of 32 inverse frequencies move, up to 3.2 rad of angle at 53k tokens). The factor is
patched only for the model's construction; the start-up check compares the built inverse frequencies with the fork's.

Its vision tower runs the ARC encoding profile's ``vision`` compute, which the fork ran through two fork-only engine
args (``mm_encoder_dtype``, ``mm_encoder_per_item``): the ViT in float32 (``load_weights``), one ViT call per image,
and each image's embeddings cast to the model dtype (``_process_image_input``, as the fork-era
image architecture did). The ViT's float32 needs the SDPA attention backend (``images.vision_engine_args``).

Its pixels are normalised here, in float32 (``vit_pixels``). Stock casts every floating processor output to the model
dtype on the host (``InputProcessingContext._postprocess_output``), so a normalised float32 tensor would reach the ViT
already rounded to bfloat16. The processor therefore emits raw uint8 patches (``images.image_settings``: no rescale, no
normalise), which stock leaves alone, and each 0-255 level maps to the value the HF processor computes for it.

``arc_visual_calls`` counts ViT calls, for the start-up image self-test. With ``ARC_STOCK_VIT_INPUTS=1`` (a debug
setting, off by default: it copies each image's pixels to the host) ``arc_vit_inputs`` records, per ViT call, the
sha256 of the float32 tensor the ViT receives and its ``grid_thw`` (``vit_input_digest``), which the server returns.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from itertools import islice

import torch

from . import compat

# Before vLLM builds the model: a vLLM release other than engine.lock's refuses here, and on the retained route a
# patch set other than its own (compat).
compat.require(patched=compat.retained_route())

from vllm.model_executor.models import qwen3_5  # noqa: E402
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5Model  # noqa: E402

from .images import VIT_INPUTS_ENV  # noqa: E402
from .internals import replaced  # noqa: E402
from .pooling import GatherPooler  # noqa: E402


def fused_qk_layers(model) -> list:
    """The full-attention modules that carry the fused QK-norm/RoPE/gate switch."""
    return [layer.self_attn for layer in model.language_model.model.layers
            if hasattr(getattr(layer, "self_attn", None), "use_fused_qk_norm_rope_gate")]


def vit_input_digest(pixels: torch.Tensor, grid_thw: torch.Tensor) -> dict:
    """One ViT call's input: the sha256 of its pixel tensor's bytes (row-major, as the ViT receives it), its dtype,
    shape and ``grid_thw``."""
    host = pixels.detach().contiguous().cpu()
    return {"sha256": hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest(),
            "dtype": str(host.dtype).removeprefix("torch."), "shape": list(host.shape),
            "grid_thw": [int(v) for v in grid_thw.reshape(-1).tolist()]}


def pixel_table() -> torch.Tensor:
    """The float32 ViT input for each uint8 level, computed on CPU exactly as the HF Qwen2-VL processor normalises:
    rescale 1/255 then mean 0.5 / std 0.5, which transformers fuses into one ``(x - 127.5) / 127.5`` in float32
    (``_fuse_mean_std_and_rescale_factor``, then torchvision's ``normalize``). A table, not arithmetic on the device:
    CUDA divides by a scalar as a multiply by its reciprocal, which rounds differently. Built on the CPU explicitly:
    vLLM constructs the model under its target device (and default dtype) context."""
    return (torch.arange(256, dtype=torch.float32, device="cpu") - 127.5) / 127.5


def require_uint8(pixels: torch.Tensor) -> None:
    """Any pixel dtype but uint8 is refused: a floating tensor has been normalised (and cast to the model dtype)
    before it got here."""
    if pixels.dtype != torch.uint8:
        raise ValueError(f"the ViT takes the processor's uint8 patches, got {pixels.dtype}: the processor normalised "
                         "them (mm_processor_kwargs must set do_rescale and do_normalize to False)")


def vit_pixels(pixels: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """The ViT's float32 input from one image's uint8 patches, ``table`` already on their device (an index lookup
    takes integer indices, so this holds one image's int32 copy at a time)."""
    require_uint8(pixels)
    return table[pixels.int()]


#: The fork's (98e91a960) M-RoPE cache length multiplier, which YaRN's correction range reads.
FORK_MROPE_POSITIONS_FACTOR = 4


def multimodal_pruning(vllm_config) -> bool:
    """Stock's own pruning test (``qwen3_vl.mrope_positions_factor``): video pruning on the model config."""
    mm_config = vllm_config.model_config.multimodal_config
    return bool(mm_config is not None and mm_config.is_multimodal_pruning_enabled())


class ArcBaseForPooling(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "model") -> None:
        if multimodal_pruning(vllm_config):
            raise ValueError("ArcBaseForPooling runs with video pruning off: the positions factor patch "
                             "assumes positions never exceed the token count")
        with replaced(qwen3_5, **self.qwen3_5_replacements()):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.pooler = GatherPooler()
        attns = fused_qk_layers(self)
        self.arc_fused_qk_stock = [bool(a.use_fused_qk_norm_rope_gate) for a in attns]
        for attn in attns:
            attn.use_fused_qk_norm_rope_gate = False
        self.arc_lm_dtype = vllm_config.model_config.dtype
        self.arc_visual_calls = 0
        self.arc_pixel_table = pixel_table()  # on the CPU; the start-up probe hashes it
        self.arc_pixel_tables: dict[torch.device, torch.Tensor] = {}  # its copy per device, made once
        self.arc_vit_inputs: list[dict] | None = [] if os.environ.get(VIT_INPUTS_ENV) == "1" else None

    @classmethod
    def qwen3_5_replacements(cls) -> dict:
        """The ``qwen3_5`` module attributes replaced while the model is built (``internals.replaced``)."""
        return {"mrope_positions_factor": lambda _cfg: FORK_MROPE_POSITIONS_FACTOR}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        self.visual.float()  # the profile's ViT dtype; the checkpoint's bfloat16 values are exact in float32
        return loaded

    def _process_image_input(self, image_input) -> tuple[torch.Tensor, ...]:
        """One float32 ViT call per image on its float32-normalised pixels (``vit_pixels``), each image's embeddings
        cast to the model dtype; upstream runs every image in one call and leaves the cast to the merge."""
        if image_input["type"] != "pixel_values":
            raise ValueError("the ARC encoder encodes images from their pixels, never precomputed embeddings")
        grid_thw = image_input["image_grid_thw"]
        pixels = image_input["pixel_values"]
        require_uint8(pixels)
        patches = grid_thw.prod(-1).tolist()
        if pixels.shape[0] != sum(patches):
            raise ValueError(f"{pixels.shape[0]} pixel patches for image grids totalling {sum(patches)}")
        table = self.arc_pixel_tables.get(pixels.device)
        if table is None:
            table = self.arc_pixel_tables[pixels.device] = self.arc_pixel_table.to(pixels.device)
        out = []
        for i, image in enumerate(pixels.split(patches)):
            chunk = vit_pixels(image, table)
            if self.arc_vit_inputs is not None:
                self.arc_vit_inputs.append(vit_input_digest(chunk, grid_thw[i]))
            out.append(self.visual(chunk, grid_thw=grid_thw[i:i + 1]).to(self.arc_lm_dtype))
            self.arc_visual_calls += 1
        return tuple(out)


def readout_depths(config) -> tuple:
    """The text config's ``readout_depths`` (written by ``eq.checkpoint``): increasing, from 1, ending at the layer
    count; ``ValueError`` otherwise."""
    depths = tuple(int(d) for d in (getattr(config, "readout_depths", None) or ()))
    layers = int(config.num_hidden_layers)
    if len(depths) < 2 or list(depths) != sorted(set(depths)) or depths[0] < 1 or depths[-1] != layers:
        raise ValueError(f"ArcEqForPooling needs increasing readout_depths ending at {layers}, got {depths}")
    return depths


class ArcEqModel(Qwen3_5Model):
    """``Qwen3_5Model`` with only ``forward`` replaced, for EQ (``ArcEqForPooling``): every readout depth's
    ``(hidden, residual)`` pair goes through the final norm, the fused add-and-norm, and the results are concatenated
    in depth order. A depth's readout is then the depth-``d`` engine's final readout, not ``norm(hidden + residual)``:
    the fork's multi-depth pair path. With one depth, the layer count, it is stock's
    forward on one pipeline rank without sequence parallelism (no auxiliary hidden states: EQ serves no Eagle).

    It is built in ``Qwen3_5Model``'s place (``ArcEqForPooling``), never swapped in after: ``Qwen3_5Model``'s
    ``support_torch_compile`` binds ``self.forward`` when the model is built (``TorchCompileWithNoGuardsWrapper``), and
    its bytecode hook rewrites ``type(self).forward``, so only a model built as this class compiles this forward. Its
    readout depths are read here, before the first call, when Dynamo traces them in as constants."""

    arc_readout_depths: tuple = ()

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.arc_readout_depths = readout_depths(self.config)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None
        marks = set(self.arc_readout_depths[:-1])
        kept = []
        for layer_idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer),
                                          start=self.start_layer):
            hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, residual=residual)
            if layer_idx + 1 in marks:
                # Eager, the next layer's fused add-and-norm writes its inputs in place: keep copies. Compiled, the
                # norms are functional (custom ops off) and the copies change no value.
                kept.append((hidden_states.clone(), residual.clone()))
        final, _ = self.norm(hidden_states, residual)
        return torch.cat([*(self.norm(h, r)[0] for h, r in kept), final], dim=-1)


class ArcEqForPooling(ArcBaseForPooling):
    """EQ's engine: ``ArcBaseForPooling`` at all 24 layers (the base, or the base with a question-head add-on's layers
    12-23), its rows ``cat(norm(h_d) for d in readout_depths)`` (``[12, 24]``: ``2 x hidden`` wide) from the text
    config's ``readout_depths`` (written by ``eq.checkpoint``). Everything else, the vision tower and its float32
    compute included, is ``ArcBaseForPooling``'s. Pipeline and sequence parallelism are refused.

    Its text model is built as ``ArcEqModel`` (``qwen3_5.Qwen3_5Model`` names it while the model is built, as the
    positions factor is patched), so it runs eager or compiled; the EQ engine runs it compiled (``Engine.compiled``)."""

    def __init__(self, *, vllm_config, prefix: str = "model") -> None:
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("ArcEqForPooling runs without pipeline or sequence parallelism")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        inner = self.language_model.model
        if type(inner) is not ArcEqModel:
            raise TypeError(f"ArcEqForPooling built {type(inner).__name__} as its text model, not ArcEqModel")
        if getattr(inner, "use_sequence_parallel", False):
            raise ValueError("ArcEqForPooling runs without pipeline or sequence parallelism")

    @classmethod
    def qwen3_5_replacements(cls) -> dict:
        return {**super().qwen3_5_replacements(), "Qwen3_5Model": ArcEqModel}
