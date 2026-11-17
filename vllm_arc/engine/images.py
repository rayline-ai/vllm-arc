# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""Images on the stock engine: the fork-era server's ``arc-image-v1`` request handling,
ported to stock vLLM 0.33.0.

A request may carry ``"images": [{"offset", "n", "sha256", "rgb_sha256", "height", "width", "bytes"}]``: per image,
the absolute position of its ``<|vision_start|>`` in the expanded ``tokens``, its ``<|image_pad|>`` count, its source
sha256, and its pixels as the client prepared them (packed RGB in base64, already sized by the rule, so the
engine never resizes). Only the images inside this input's ``[base, base + len(tokens))`` are sent. ``tokens``,
``offsets``, ``base`` and every length stay in expanded positions; only the engine prompt collapses each run to one
pad, which vLLM's processor expands again.

What the fork did inside vLLM, and where it is here:

- ``mm_encoder_dtype=float32`` and ``mm_encoder_per_item``: ``models.ArcBaseForPooling`` (the ViT in float32, one
  call per image, each embedding cast to the model dtype).
- ``PoolingParams.image_token_counts`` (refused before the pass): `require_expanded` after the pass, over the ids
  the engine processed, which must be the caller's expanded tokens exactly. A mismatch is ``ImageExpansionError``
  (422 ``image_input_refused``); a retained session that took it is dropped by its owner.
- Normalisation: not in the processor, which would hand stock a float32 tensor that stock casts to the model dtype
  (bfloat16) on the host; not in stock's default device norm (``mm_device_do_normalize``), a fused ``x * w + b``
  kernel that rounds differently. The processor emits uint8 patches (``do_rescale`` and ``do_normalize`` off) and
  ``models.vit_pixels`` maps each level to the HF processor's float32 value.

No vLLM import at module level: everything but `prompt` and `processor_identity` runs on CPU in tests.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass

from .sessions import Refused

#: The image rules this server can serve, as the client's image rule identity reports them (the
#: ``arc-image-v1`` rule; a client should refuse an engine whose rule or vision differs from its own).
IMAGE_RULES = {"arc-image-v1": {"rule": "arc-image-v1", "factor": 32, "patch_size": 16, "merge_size": 2,
                                "min_pixels": 65_536, "max_pixels": 1_048_576, "max_long_edge": 2_048,
                                "max_aspect_ratio": 200, "resample": "pillow_bicubic",
                                "decode": "first_frame_exif_transpose_rgb_on_white"}}
#: The vision compute the image rule requires: the ViT in float32, one call per image, embeddings cast to bfloat16.
VISION = {"dtype": "float32", "embed_cast": "bfloat16", "per_image_calls": True}
MAX_IMAGES = 32  # per engine input (``limit_mm_per_prompt``); an input with more is refused, never truncated
#: The debug setting (off by default) under which the worker records each ViT call's input digest and the server
#: returns an image request's digests (``server.VIT_INPUTS_HEADER``): ``1`` on.
VIT_INPUTS_ENV = "ARC_STOCK_VIT_INPUTS"
#: What every stock engine's processor runs with: raw uint8 patches, no rescale or normalise (``models.vit_pixels``
#: normalises). A text-only engine needs them too: vLLM profiles the ViT at start with a dummy image.
PIXEL_KWARGS = {"do_rescale": False, "do_normalize": False}
#: sha256 of ``models.pixel_table()``'s float32 bytes as built on the CPU: the HF processor's value for each uint8
#: level (tests pin it against the processor's own output). The start-up check compares the running table with it.
PIXEL_TABLE_SHA256 = "b4925e7ab450610f0996fb5700e2b9cda8e3a8d98983df9659a9d9d6b952397b"
#: The ViT attention backend: stock picks FLASH_ATTN for this model, which has no float32 kernel.
VISION_ATTENTION = "TORCH_SDPA"


def image_settings(rule: str) -> dict | None:
    """The image settings an engine started with ``rule`` serves; ``None`` for a text-only engine."""
    if not rule:
        return None
    if rule not in IMAGE_RULES:
        raise ValueError(f"unknown image rule {rule!r}; this server serves {sorted(IMAGE_RULES)}")
    pixels = IMAGE_RULES[rule]
    return {"rule": pixels, "vision": VISION, "max_images": MAX_IMAGES,
            # Images arrive sized by the rule, so the processor must not resize; its size band is the rule's, which
            # is also what the engine profiles its encoder cache with. It must not rescale or normalise either: it
            # emits uint8 patches, which the model normalises in float32 (``models.vit_pixels``).
            "mm_processor_kwargs": {"do_resize": False, **PIXEL_KWARGS,
                                    "size": {"shortest_edge": pixels["min_pixels"],
                                             "longest_edge": pixels["max_pixels"]}}}


def vision_engine_args(settings: dict | None = None) -> dict:
    """What every stock engine passes, images served or not: the ViT runs (at least when vLLM profiles it at start,
    on a dummy image) in float32, which needs the SDPA backend, stock's fused device norm stays off, and the processor
    emits uint8 patches (``models.vit_pixels`` normalises; it refuses anything else). With ``settings``, the
    processor kwargs are the served images' (which include ``PIXEL_KWARGS``)."""
    return {"mm_encoder_attn_backend": VISION_ATTENTION, "mm_device_do_normalize": False,
            "mm_processor_kwargs": dict(settings["mm_processor_kwargs"]) if settings else dict(PIXEL_KWARGS)}


def image_engine_args(settings: dict | None) -> dict:
    """The ``AsyncEngineArgs`` an image-reading engine adds (its processor kwargs come with ``vision_engine_args``);
    a text-only engine adds none."""
    if not settings:
        return {}
    return {"limit_mm_per_prompt": {"image": settings["max_images"], "video": 0}, "mm_processor_cache_gb": 0}


def image_engine_suffix(settings: dict) -> str:
    """The engine id's image part: the rule, the ViT dtype, and a digest of every image setting (the fork's)."""
    tag = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    return f"+images-{settings['rule']['rule']}-vit{settings['vision']['dtype']}-{tag}"


@dataclass(frozen=True)
class VisionIds:
    start: int
    pad: int
    end: int

    @classmethod
    def from_engine(cls, engine) -> "VisionIds":
        hf = engine.vllm_config.model_config.hf_config
        return cls(hf.vision_start_token_id, hf.image_token_id, hf.vision_end_token_id)


@dataclass(frozen=True)
class ImageInput:
    """One engine input's images: the prompt with each run collapsed, the declared counts, and the pixels."""

    prompt_ids: list[int]
    counts: list[int]
    pixels: list


class InvalidImages(ValueError):
    """A request whose image runs or pixels the server will not encode (422 ``invalid_images``)."""


class ImageExpansionError(Refused):
    """The engine processed an image input to other ids than the caller's expanded tokens (422
    ``image_input_refused``). Its readouts are discarded; a retained session that took the input is dropped."""

    def __init__(self, reason: str) -> None:
        super().__init__(422, "image_input_refused", reason=reason[:500])


def collapse(tokens: list[int], base: int, spans: list[tuple[int, int]], ids: VisionIds) -> tuple[list[int], list[int]]:
    """``tokens`` with each declared ``(offset, n)`` run collapsed to ``<|vision_start|><|image_pad|><|vision_end|>``.

    ``offset`` is the run's ``<|vision_start|>`` in absolute expanded positions. Every run must lie inside this input,
    in order and without overlap, as exactly a start, ``n`` pads and an end. No vision token may stand outside a
    declared run, so the engine never sees an image placeholder the caller did not declare.
    """
    out: list[int] = []
    cursor = 0
    for offset, n in spans:
        local = offset - base
        if type(n) is not int or n < 1 or local < cursor or local + n + 2 > len(tokens):
            raise InvalidImages(f"image run at {offset} (n={n}) is out of order or outside this input")
        if (tokens[local] != ids.start or tokens[local + n + 1] != ids.end
                or any(t != ids.pad for t in tokens[local + 1:local + n + 1])):
            raise InvalidImages(f"tokens at {offset} are not <|vision_start|>, {n} <|image_pad|>, <|vision_end|>")
        out.extend(tokens[cursor:local])
        out.extend((ids.start, ids.pad, ids.end))
        cursor = local + n + 2
    out.extend(tokens[cursor:])
    reserved = {ids.start, ids.pad, ids.end}
    if sum(t in reserved for t in out) != 3 * len(spans):
        raise InvalidImages("a vision token stands outside the declared image runs")
    return out, [n for _, n in spans]


def decode_pixels(image: dict, rule: dict):
    """The declared image as a PIL RGB image, after checking its bytes against their hash and the rule's grid."""
    from PIL import Image

    height, width, n = image.get("height"), image.get("width"), image.get("n")
    factor = rule["factor"]
    if not all(type(v) is int and v > 0 for v in (height, width, n)):
        raise InvalidImages("an image needs positive integer height, width and n")
    if (height % factor or width % factor or not rule["min_pixels"] <= height * width <= rule["max_pixels"]
            or max(height, width) > rule["max_long_edge"]):
        raise InvalidImages(f"{height}x{width} is not a size the {rule['rule']} rule produces")
    if n != (height // factor) * (width // factor):
        raise InvalidImages(f"{height}x{width} expands to {(height // factor) * (width // factor)} tokens, not {n}")
    try:
        raw = base64.b64decode(image["bytes"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidImages("image bytes are not base64") from exc
    if len(raw) != height * width * 3 or hashlib.sha256(raw).hexdigest() != image.get("rgb_sha256"):
        raise InvalidImages("image bytes are not the declared packed RGB pixels")
    return Image.frombytes("RGB", (width, height), raw)


def image_input(tokens: list[int], base: int, images, ids: VisionIds, settings: dict) -> ImageInput:
    """Validate one input's declared images and build its collapsed engine prompt (``InvalidImages`` if not)."""
    if not isinstance(images, list) or not all(isinstance(image, dict) for image in images):
        raise InvalidImages("images is a list of objects")
    if len(images) > settings["max_images"]:
        raise Refused(422, "too_many_images", images=len(images), max_images=settings["max_images"])
    spans = [(image.get("offset"), image.get("n")) for image in images]
    if any(type(offset) is not int for offset, _ in spans):
        raise InvalidImages("each image names its integer offset")
    prompt_ids, counts = collapse(tokens, base, spans, ids)
    return ImageInput(prompt_ids, counts, [decode_pixels(image, settings["rule"]) for image in images])


def prompt(tokens: list[int], images: ImageInput | None, **extra) -> dict:
    """The engine prompt for one input: a copy of the tokens, or with images of the collapsed ids, and the pixels.
    ``extra`` (e.g. ``cache_salt``) is added as is. A copy, because vLLM keeps a streaming session's first input's
    list as the session's ``prompt_token_ids`` and ``extend``-s later inputs onto it in place."""
    if images is None:
        return {"prompt_token_ids": list(tokens), **extra}
    return {"prompt_token_ids": list(images.prompt_ids), "multi_modal_data": {"image": images.pixels}, **extra}


def require_expanded(processed, tokens: list[int]) -> None:
    """vLLM expanded this input's images exactly as the caller did: the ids it processed are ``tokens`` (each run a
    start, the declared pads, an end, at its declared position). ``ImageExpansionError`` otherwise."""
    processed = list(processed or ())
    if processed == tokens:
        return
    if len(processed) != len(tokens):
        raise ImageExpansionError(f"vLLM processed {len(processed)} positions for {len(tokens)} expanded tokens")
    first = next(i for i, (a, b) in enumerate(zip(processed, tokens)) if a != b)
    raise ImageExpansionError(f"vLLM's processed ids differ from the expanded tokens first at local position {first}")


def processor_identity(model: str, revision: str | None) -> dict:
    """The image processor vLLM runs on this model, and the libraries that set its pixels and the ViT's input."""
    import PIL
    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model, revision=revision, use_fast=True)  # as vLLM's Qwen3-VL info
    return {"class": type(processor.image_processor).__name__, "transformers": transformers.__version__,
            "pillow": PIL.__version__}
