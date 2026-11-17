# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The stock engine's image request handling on CPU (``vllm_arc.engine.images``): declared runs collapse to one pad
each and every malformed run, pixel buffer or count is refused before the engine; the expansion check after the
pass; the engine args and id an image engine adds; and the vision start-up assertions."""
from __future__ import annotations

import base64
import hashlib
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
sys.path.insert(0, str(Path(__file__).parents[1]))

from vllm_arc.engine import images as img  # noqa: E402
from vllm_arc.server import app as server  # noqa: E402
from vllm_arc.engine.sessions import Refused  # noqa: E402
from vllm_arc.engine.startup import Engine, failed, vision_assertions  # noqa: E402

IDS = img.VisionIds(start=9001, pad=9002, end=9003)
SETTINGS = img.image_settings("arc-image-v1")


def _image(offset: int, h: int = 256, w: int = 256, **change) -> dict:
    raw = bytes((x + y + c) % 256 for y in range(h) for x in range(w) for c in range(3))
    digest = hashlib.sha256(raw).hexdigest()
    return {"offset": offset, "n": (h // 32) * (w // 32), "height": h, "width": w, "sha256": digest,
            "rgb_sha256": digest, "bytes": base64.b64encode(raw).decode(), **change}


def _tokens(base: int, *images: dict) -> list[int]:
    """Text ids around each declared run, starting at absolute position ``base``."""
    out: list[int] = []
    for image in images:
        out += [5] * (image["offset"] - base - len(out)) + [IDS.start] + [IDS.pad] * image["n"] + [IDS.end]
    return out + [6, 7]


def test_declared_runs_collapse_to_one_pad_each_with_their_pixels():
    first, second = _image(102), _image(170, 256, 384)
    tokens = _tokens(100, first, second)
    out = img.image_input(tokens, 100, [first, second], IDS, SETTINGS)
    assert out.counts == [64, 96]
    assert out.prompt_ids == [5, 5, IDS.start, IDS.pad, IDS.end, 5, 5, IDS.start, IDS.pad, IDS.end, 6, 7]
    assert [p.size for p in out.pixels] == [(256, 256), (384, 256)] and out.pixels[0].mode == "RGB"
    assert img.prompt(tokens, out, cache_salt="s") == {"prompt_token_ids": out.prompt_ids, "cache_salt": "s",
                                                       "multi_modal_data": {"image": out.pixels}}
    assert img.prompt(tokens, None, cache_salt="s") == {"prompt_token_ids": tokens, "cache_salt": "s"}


def _stray(tokens):
    return tokens[:1] + [IDS.pad] + tokens[2:]


@pytest.mark.parametrize("declared, edit, reason", [
    ([_image(2)], lambda t: t, None),
    ([_image(3)], lambda t: t[1:] + [6], "are not <|vision_start|>"),  # the run sits one position earlier
    ([_image(2, n=63)], lambda t: t, "expands to 64 tokens, not 63"),
    ([_image(2)], _stray, "outside the declared"),
    ([_image(2), _image(2)], lambda t: t, "out of order"),
    ([_image(2, h=224)], lambda t: t, "not a size"),
    ([_image(2, n=64, height=256, width=288)], lambda t: t, "expands to 72 tokens, not 64"),
    ([_image(2, rgb_sha256="0" * 64)], lambda t: t, "not the declared packed RGB"),
    ([_image(2, bytes="***")], lambda t: t, "not base64"),
    ([{**_image(2), "offset": "2"}], lambda t: t, "integer offset"),
    ([["not", "an", "object"]], lambda t: t, "list of objects"),
])
def test_each_malformed_image_input_is_refused_before_the_engine(declared, edit, reason):
    tokens = edit(_tokens(0, *[d for d in declared if isinstance(d, dict) and isinstance(d["offset"], int)][:1]))
    if reason is None:
        assert img.image_input(tokens, 0, declared, IDS, SETTINGS).counts == [64]
        return
    with pytest.raises(img.InvalidImages, match=reason.replace("|", r"\|")):
        img.image_input(tokens, 0, declared, IDS, SETTINGS)


def test_more_images_than_one_input_takes_are_refused_not_truncated():
    images = [_image(2 + 66 * k) for k in range(img.MAX_IMAGES + 1)]
    with pytest.raises(Refused) as exc:
        img.image_input(_tokens(0, *images), 0, images, IDS, SETTINGS)
    assert (exc.value.status, exc.value.code, exc.value.detail) == (
        422, "too_many_images", {"images": 33, "max_images": 32})


@pytest.mark.parametrize("processed, message", [
    ([1, 2, 3], None), ([1, 2], "processed 2 positions for 3"), ([1, 9, 3], "first at local position 1"),
])
def test_the_expansion_check_requires_the_processed_ids_to_be_the_callers_tokens(processed, message):
    if message is None:
        img.require_expanded(processed, [1, 2, 3])
        return
    with pytest.raises(img.ImageExpansionError) as exc:
        img.require_expanded(processed, [1, 2, 3])
    assert (exc.value.status, exc.value.code) == (422, "image_input_refused") and message in exc.value.detail["reason"]


def test_engine_args_and_id_add_the_image_settings_only_when_images_are_served():
    """Every engine runs the ViT in float32 on SDPA, no fused device norm, and a processor that emits uint8 patches
    (vLLM profiles the ViT on a dummy image even on a text-only engine, and ``models.vit_pixels`` refuses anything
    else); an image engine adds the limits and the rule's processor settings, and its id an image part a text-only
    engine's id never has."""
    common = dict(budget=8192, max_model_len=262144, max_sessions=64, scratch_slots=4, gpu_memory=0.8)
    text = server.engine_args(Engine(), "/ckpt", **common)
    image = server.engine_args(Engine(), "/ckpt", **common, images=SETTINGS)
    assert (text["mm_encoder_attn_backend"], text["mm_device_do_normalize"]) == ("TORCH_SDPA", False)
    assert text["mm_processor_kwargs"] == {"do_rescale": False, "do_normalize": False}
    assert "limit_mm_per_prompt" not in text
    assert {k: v for k, v in image.items() if k not in text} == {
        "limit_mm_per_prompt": {"image": 32, "video": 0}, "mm_processor_cache_gb": 0}
    assert image["mm_processor_kwargs"] == {"do_resize": False, "do_rescale": False, "do_normalize": False,
                                            "size": {"shortest_edge": 65536, "longest_edge": 1048576}}
    plain = server.engine_id(Engine(), "flashinfer", "yarn-f4")
    served = server.engine_id(Engine(), "flashinfer", "yarn-f4", SETTINGS)
    assert re.fullmatch(re.escape(plain) + r"\+images-arc-image-v1-vitfloat32-[0-9a-f]{12}", served)
    with pytest.raises(ValueError, match="unknown image rule"):
        img.image_settings("arc-image-v2")


VISION = {"param_dtypes": ["float32"], "attn_backend": "TORCH_SDPA", "input_norm": "IdentityInputNorm",
          "mm_device_do_normalize": False, "mm_encoder_tp_mode": "weights", "image_limit": 32, "video_limit": 0,
          "visual_calls": 0, "pixel_table": {"device": "cpu", "dtype": "float32", "sha256": img.PIXEL_TABLE_SHA256},
          "cudagraph_mm_encoder": False}


@pytest.mark.parametrize("change, name", [
    ({}, None),
    ({"param_dtypes": ["bfloat16"]}, "vit_dtype"),
    ({"param_dtypes": ["bfloat16", "float32"]}, "vit_dtype"),
    ({"attn_backend": "FLASH_ATTN"}, "vit_attn_backend"),
    ({"input_norm": "FusedMMInputNorm", "mm_device_do_normalize": True}, "vit_input_norm"),
    ({"mm_encoder_tp_mode": "data"}, "vit_tp_mode"),
    ({"video_limit": 999}, "image_limits [image, video]"),
    ({"pixel_table": {**VISION["pixel_table"], "device": "cuda"}}, "vit_pixel_table"),
    ({"pixel_table": {**VISION["pixel_table"], "sha256": "0" * 64}}, "vit_pixel_table"),
    ({"pixel_table": None}, "vit_pixel_table"),
    ({"cudagraph_mm_encoder": True}, "cudagraph_mm_encoder"),
    ({"cudagraph_mm_encoder": None}, "cudagraph_mm_encoder"),
])
def test_each_vision_change_fails_its_own_assertion(change, name):
    probe = {"vision": {**VISION, **change}}
    assert [c["name"] for c in failed(vision_assertions(probe, SETTINGS))] == ([name] if name else [])


def test_a_text_only_engine_does_not_judge_image_limits():
    probe = {"vision": {**VISION, "image_limit": 999, "video_limit": 999}}
    assert failed(vision_assertions(probe, None)) == []
