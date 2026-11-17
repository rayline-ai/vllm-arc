# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``ArcBaseForPooling`` on CPU against a stand-in Qwen3.5: the gather pooler, the forced eager QK path, the fork's
M-RoPE positions factor during construction only, the refusal of video pruning, and the profile's vision compute
(float32 ViT, one call per image, each image's embeddings in the model dtype)."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PLUGIN = Path(__file__).parents[1]


class _Attn:
    def __init__(self):
        self.use_fused_qk_norm_rope_gate = True


class _Layer(torch.nn.Module):
    def __init__(self, attention: bool):
        super().__init__()
        if attention:
            self.self_attn = _Attn()


class _Visual(torch.nn.Module):
    """A ViT stand-in: records each call's pixel dtype and image count; 4 patches merge into one embedding."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
        self.calls = []

    @property
    def dtype(self):
        return self.proj.weight.dtype

    def forward(self, pixels, grid_thw):
        self.calls.append((pixels.dtype, grid_thw.shape[0]))
        return self.proj(pixels).reshape(-1, 4, 2).sum(1)


@pytest.fixture
def models(monkeypatch):
    seen = {}

    class Qwen3_5Model(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.layers = torch.nn.ModuleList(_Layer(attention=k % 3 == 0) for k in range(1, config.num_hidden_layers + 1))

    class Qwen3_5ForConditionalGeneration(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix="model"):
            super().__init__()
            seen["positions_factor"] = qwen.mrope_positions_factor(vllm_config)  # read during construction
            self.language_model = SimpleNamespace(model=Qwen3_5Model(vllm_config.text_config))
            self.visual = _Visual()

        def load_weights(self, weights):
            return {name for name, _ in weights}

    qwen = types.ModuleType("qwen3_5")
    qwen.Qwen3_5Model, qwen.Qwen3_5ForConditionalGeneration = Qwen3_5Model, Qwen3_5ForConditionalGeneration
    qwen.mrope_positions_factor = lambda _cfg: 1  # stock 0.33 without video pruning
    models_pkg = types.ModuleType("models")
    models_pkg.qwen3_5 = qwen
    pooler_mod = types.ModuleType("pooler")
    pooler_mod.Pooler = type("Pooler", (torch.nn.Module,), {})
    for name, module in {"vllm": types.ModuleType("vllm"), "vllm.model_executor": types.ModuleType("me"),
                         "vllm.model_executor.layers": types.ModuleType("layers"),
                         "vllm.model_executor.layers.pooler": pooler_mod,
                         "vllm.model_executor.models": models_pkg,
                         "vllm.model_executor.models.qwen3_5": qwen}.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in [n for n in sys.modules if n == "vllm_arc" or n.startswith("vllm_arc.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(PLUGIN))
    monkeypatch.delenv("ARC_STOCK_SESSIONS", raising=False)  # outside the server: no route, so no patch check
    import vllm_arc.engine.compat as compat

    guarded = []
    monkeypatch.setattr(compat, "require", lambda *a, **k: guarded.append(k))  # a stand-in, not the locked vLLM
    import vllm_arc.engine.models as models

    assert guarded == [{"patched": False}]  # the load guard ran before vLLM's model module was read

    models.seen, models.qwen = seen, qwen
    yield models
    for name in [n for n in sys.modules if n == "vllm_arc" or n.startswith("vllm_arc.")]:
        sys.modules.pop(name)


class _MultimodalConfig:
    def __init__(self, pruning):
        self.pruning = pruning

    def is_multimodal_pruning_enabled(self):
        return self.pruning


def _config(layers=6, pruning=False):
    return SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=layers),
                           model_config=SimpleNamespace(multimodal_config=_MultimodalConfig(pruning),
                                                        dtype=torch.bfloat16))


def test_base_installs_the_gather_pooler_and_forces_the_eager_qk_path(models):
    model = models.ArcBaseForPooling(vllm_config=_config())
    assert type(model.pooler).__name__ == "GatherPooler"
    assert model.arc_fused_qk_stock == [True, True]
    assert [a.use_fused_qk_norm_rope_gate for a in models.fused_qk_layers(model)] == [False, False]


def test_the_fork_positions_factor_applies_during_construction_only(models):
    models.ArcBaseForPooling(vllm_config=_config())
    assert models.seen["positions_factor"] == models.FORK_MROPE_POSITIONS_FACTOR == 4
    assert models.qwen.mrope_positions_factor(None) == 1  # restored after construction


def test_video_pruning_is_refused(models):
    with pytest.raises(ValueError, match="video pruning"):
        models.ArcBaseForPooling(vllm_config=_config(pruning=True))


def test_the_vit_runs_in_float32_once_per_image_and_returns_model_dtype_embeddings(models):
    """After loading, the ViT's parameters are float32; two images run as two ViT calls on float32 pixels, each
    returning that image's embeddings (patches / 4) in the model dtype, and the call counter counts both."""
    model = models.ArcBaseForPooling(vllm_config=_config())
    assert model.load_weights([("a", None)]) == {"a"}
    assert model.visual.dtype == torch.float32
    grid = torch.tensor([[1, 4, 4], [1, 2, 4]])  # 16 and 8 patches
    out = model._process_image_input({"type": "pixel_values", "image_grid_thw": grid,
                                      "pixel_values": torch.ones(24, 3, dtype=torch.uint8)})
    assert [tuple(e.shape) for e in out] == [(4, 2), (2, 2)]
    assert all(e.dtype == torch.bfloat16 for e in out)
    assert model.visual.calls == [(torch.float32, 1), (torch.float32, 1)]
    assert model.arc_visual_calls == 2


@pytest.mark.parametrize("image_input, match", [
    ({"type": "image_embeds", "image_grid_thw": torch.tensor([[1, 2, 2]]), "image_embeds": torch.ones(1, 2)},
     "pixels"),
    ({"type": "pixel_values", "image_grid_thw": torch.tensor([[1, 2, 2]]),
      "pixel_values": torch.ones(5, 3, dtype=torch.uint8)}, "5 pixel patches"),
    # A normalised (floating) tensor: stock has already cast it to the model dtype on the host.
    ({"type": "pixel_values", "image_grid_thw": torch.tensor([[1, 2, 2]]),
      "pixel_values": torch.ones(4, 3, dtype=torch.bfloat16)}, "uint8 patches, got torch.bfloat16"),
    ({"type": "pixel_values", "image_grid_thw": torch.tensor([[1, 2, 2]]),
      "pixel_values": torch.ones(4, 3, dtype=torch.float32)}, "uint8 patches, got torch.float32"),
])
def test_precomputed_embeddings_and_mismatched_patches_are_refused(models, image_input, match):
    model = models.ArcBaseForPooling(vllm_config=_config())
    with pytest.raises(ValueError, match=match):
        model._process_image_input(image_input)
    assert model.visual.calls == []


def test_with_vit_input_recording_each_call_records_its_float32_pixels_digest_and_grid(models, monkeypatch):
    """``ARC_STOCK_VIT_INPUTS=1``: per ViT call, the sha256 of the float32 bytes the ViT receives (here hashed
    independently from the same uint8 levels, normalised) and that image's grid; off by default."""
    import hashlib

    assert models.ArcBaseForPooling(vllm_config=_config()).arc_vit_inputs is None
    monkeypatch.setenv("ARC_STOCK_VIT_INPUTS", "1")
    model = models.ArcBaseForPooling(vllm_config=_config())
    model.load_weights([])
    pixels = (torch.arange(24 * 3) * 7 % 256).to(torch.uint8).reshape(24, 3)
    model._process_image_input({"type": "pixel_values", "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 4]]),
                                "pixel_values": pixels})
    expect = (pixels.to(torch.float32) - 127.5) / 127.5
    assert model.arc_vit_inputs == [
        {"sha256": hashlib.sha256(expect[:16].numpy().tobytes()).hexdigest(), "dtype": "float32", "shape": [16, 3],
         "grid_thw": [1, 4, 4]},
        {"sha256": hashlib.sha256(expect[16:].numpy().tobytes()).hexdigest(), "dtype": "float32", "shape": [8, 3],
         "grid_thw": [1, 2, 4]}]


# The HF Qwen2-VL image processor's float32 ``pixel_values`` (the release's preprocessor config: patch 16, merge 2,
# temporal patch 2, mean and std 0.5; transformers 5.20.2, torchvision backend, CPU) for ``_synthetic(h, w)``, whose
# channels hold every uint8 level, so the pair pins the processor's value for each of the 256 levels.
HF_PIXEL_SHA256 = {(256, 256): "a68dddc9d7450cbff1b9be8268134a8d9fc0eba8b1e92c1dd37c90bda5f31620",
                   (256, 384): "0df72134d02776e310fedc41ccdd4e3b73991d5f562aeb8c02f64a57737647c7"}


def _synthetic(h: int, w: int) -> bytes:
    """Packed RGB whose every channel takes each of the 256 levels."""
    return bytes((y * w + x + 85 * c) % 256 for y in range(h) for x in range(w) for c in range(3))


def _patches(raw: bytes, h: int, w: int, ps: int = 16, ms: int = 2, tps: int = 2) -> torch.Tensor:
    """The processor's uint8 patches for packed RGB (no rescale, no normalise): the frame repeated ``tps`` times and
    cut into ``ps`` patches in merge-window order, as Qwen2-VL's processor lays them out (checked against it below)."""
    x = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(h, w, 3).permute(2, 0, 1)
    x = x.unsqueeze(0).expand(tps, -1, -1, -1)
    gh, gw = h // ps, w // ps
    x = x.reshape(1, tps, 3, gh // ms, ms, ps, gw // ms, ms, ps).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(gh * gw, 3 * tps * ps * ps)


class _RecordingVisual(torch.nn.Module):
    """A float32 ViT stand-in that keeps the exact tensor each call receives."""

    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.received = []

    def forward(self, pixels, grid_thw):
        self.received.append(pixels)
        return torch.zeros(pixels.shape[0] // 4, 2)


def _vit_inputs(models, monkeypatch, pixels: torch.Tensor, grid: list[list[int]]):
    """Run the plugin's own image path on ``pixels``; what the ViT received and what the debug header records."""
    monkeypatch.setenv("ARC_STOCK_VIT_INPUTS", "1")
    model = models.ArcBaseForPooling(vllm_config=_config())
    model.load_weights([])
    model.visual = _RecordingVisual()
    model._process_image_input({"type": "pixel_values", "image_grid_thw": torch.tensor(grid), "pixel_values": pixels})
    return model.visual.received, model.arc_vit_inputs


@pytest.mark.parametrize("h, w", sorted(HF_PIXEL_SHA256))
def test_the_vit_receives_the_hf_processors_float32_pixels_bit_for_bit(models, monkeypatch, h, w):
    """The processor's uint8 patches through ``_process_image_input``: the ViT receives float32 bytes equal to the HF
    processor's float32 ``pixel_values`` (pinned above), with no bfloat16 step, and the ``x-arc-vit-inputs`` digest
    is the sha256 of exactly that tensor."""
    import hashlib

    grid = [[1, h // 16, w // 16]]
    [received], recorded = _vit_inputs(models, monkeypatch, _patches(_synthetic(h, w), h, w), grid)
    sha = hashlib.sha256(received.numpy().tobytes()).hexdigest()
    assert received.dtype == torch.float32 and sha == HF_PIXEL_SHA256[(h, w)]
    assert recorded == [{"sha256": sha, "dtype": "float32", "shape": list(received.shape), "grid_thw": grid[0]}]


def test_the_pixel_table_is_the_processors_value_for_every_level(models):
    """Each uint8 level maps to ``(x - 127.5) / 127.5`` in float32, none of them rounded through bfloat16 (stock's
    cast, which moves 254 of the 256 levels)."""
    table = models.pixel_table()
    assert table.dtype == torch.float32 and table.shape == (256,)
    assert torch.equal(table, (torch.arange(256, dtype=torch.float32) - 127.5) / 127.5)
    assert int((table.to(torch.bfloat16).to(torch.float32) != table).sum()) == 254


def test_against_the_live_hf_processor(models, monkeypatch):
    """Where transformers and torchvision are installed: the HF Qwen2-VL processor run with the engine's
    ``mm_processor_kwargs`` emits uint8 patches equal to ``_patches``, and the plugin's path turns them into exactly
    the processor's own float32 ``pixel_values`` (which must still match the pinned sha)."""
    import hashlib

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    from PIL import Image

    from vllm_arc.engine import images as img

    processor = transformers.Qwen2VLImageProcessor(image_mean=[0.5] * 3, image_std=[0.5] * 3, patch_size=16,
                                                   merge_size=2, temporal_patch_size=2)
    kwargs = img.image_settings("arc-image-v1")["mm_processor_kwargs"]
    for (h, w), pinned in HF_PIXEL_SHA256.items():
        image = Image.frombytes("RGB", (w, h), _synthetic(h, w))
        raw = processor(images=[image], **kwargs, return_tensors="pt")
        hf = processor(images=[image], **{**kwargs, "do_rescale": True, "do_normalize": True},
                       return_tensors="pt")["pixel_values"]
        assert raw["pixel_values"].dtype == torch.uint8 and torch.equal(raw["pixel_values"], _patches(
            _synthetic(h, w), h, w))
        [received], _ = _vit_inputs(models, monkeypatch, raw["pixel_values"], raw["image_grid_thw"].tolist())
        assert hf.dtype == torch.float32 and torch.equal(received, hf)
        assert hashlib.sha256(hf.numpy().tobytes()).hexdigest() == pinned


#: sha256 of the 256-level float32 table built on the CPU (the levels the pinned HF shas above cover).
PIXEL_TABLE_SHA256 = "b4925e7ab450610f0996fb5700e2b9cda8e3a8d98983df9659a9d9d6b952397b"


@pytest.mark.parametrize("device", ["meta", "cpu"])
def test_the_pixel_table_is_built_on_the_cpu_under_vllms_construction_context(models, device):
    """vLLM builds the model under ``with torch.device(target)`` and a bfloat16 default dtype; the table is still the
    CPU float32 one (on CUDA the scalar divide rounds 126 of the 256 levels differently)."""
    import hashlib

    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = models.ArcBaseForPooling(vllm_config=_config())
    finally:
        torch.set_default_dtype(before)
    table = model.arc_pixel_table
    assert (table.device.type, table.dtype) == ("cpu", torch.float32)
    assert hashlib.sha256(table.numpy().tobytes()).hexdigest() == PIXEL_TABLE_SHA256
    from vllm_arc.engine import images

    assert images.PIXEL_TABLE_SHA256 == PIXEL_TABLE_SHA256  # what the start-up check compares the running table with


def test_the_device_copy_of_the_table_is_made_once(models, monkeypatch):
    """Each image normalises against one cached copy of the table on the pixels' device, not a fresh ``.to``."""
    model = models.ArcBaseForPooling(vllm_config=_config())
    model.load_weights([])
    copies = []
    real_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: copies.append(a) or real_to(self, *a, **k))
    for _ in range(3):
        model._process_image_input({"type": "pixel_values", "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 4]]),
                                    "pixel_values": torch.ones(24, 3, dtype=torch.uint8)})
    assert list(model.arc_pixel_tables) == [torch.device("cpu")]
    assert sum(1 for a in copies if a and isinstance(a[0], torch.device)) == 1
