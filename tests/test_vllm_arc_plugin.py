"""The vllm-arc plugin's image release path, on CPU against a stand-in vLLM.

vLLM is not installed in this suite, so these tests replace the few
vLLM names the plugin imports with minimal stand-ins and check what the plugin itself adds: both release
architectures and their config hooks are registered; the image config hook sets the float32-capable ViT attention
backend and refuses engine settings that batch images through one ViT call; the image model drops the heads, runs
its ViT in float32, and calls it once per image. Loading a real release needs a GPU and the vLLM build in the
README.
"""
from __future__ import annotations

import enum
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PLUGIN = Path(__file__).parents[1]
Backend = enum.Enum("AttentionBackendEnum", ["TORCH_SDPA", "FLASH_ATTN"])


class _Visual(torch.nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.patch_embed = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.attn_backend = backend
        self.calls = []

    @property
    def dtype(self):
        return self.patch_embed.weight.dtype

    def forward(self, pixels, grid_thw):
        self.calls.append((pixels.clone(), grid_thw.clone()))
        return self.patch_embed(pixels)


class _Multimodal(torch.nn.Module):
    """Qwen3_5ForConditionalGeneration's surface the plugin relies on: ``visual`` and ``load_weights``."""

    backend = Backend.TORCH_SDPA

    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__()
        self.visual = _Visual(self.backend)
        self.loaded = []

    def load_weights(self, weights):
        self.loaded = [name for name, _ in weights]
        return set(self.loaded)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config): ...

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config): ...

    @classmethod
    def get_mamba_state_copy_func(cls): ...


class _BaseConfig:
    calls = []

    @staticmethod
    def verify_and_update_config(vllm_config):
        _BaseConfig.calls.append(vllm_config)


@pytest.fixture
def plugin(monkeypatch):
    registry = SimpleNamespace(archs={}, get_supported_archs=lambda: list(registry.archs))
    registry.register_model = lambda arch, target: registry.archs.setdefault(arch, target)
    config_map = {}
    modules = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor.models.config": types.ModuleType("config"),
        "vllm.model_executor.models.interfaces": types.ModuleType("interfaces"),
        "vllm.model_executor.models.qwen3_5": types.ModuleType("qwen3_5"),
        "vllm.v1.attention.backends.registry": types.ModuleType("registry"),
    }
    modules["vllm"].ModelRegistry = registry
    modules["vllm.model_executor.models.config"].__dict__.update(
        MODELS_CONFIG_MAP=config_map, Qwen3_5ForConditionalGenerationConfig=_BaseConfig)
    modules["vllm.model_executor.models.interfaces"].__dict__.update(IsHybrid=type("IsHybrid", (), {}),
                                                                     SupportsMRoPE=type("SupportsMRoPE", (), {}))
    modules["vllm.model_executor.models.qwen3_5"].__dict__.update(
        Qwen3_5ForCausalLM=type("Qwen3_5ForCausalLM", (), {}), Qwen3_5ForConditionalGeneration=_Multimodal)
    modules["vllm.v1.attention.backends.registry"].AttentionBackendEnum = Backend
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in [n for n in sys.modules if n == "vllm_arc" or n.startswith("vllm_arc.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(PLUGIN))
    import vllm_arc
    import vllm_arc.config
    import vllm_arc.model

    _BaseConfig.calls.clear()
    yield SimpleNamespace(package=vllm_arc, config=vllm_arc.config, model=vllm_arc.model, archs=registry.archs,
                          config_map=config_map)
    for name in [n for n in sys.modules if n == "vllm_arc" or n.startswith("vllm_arc.")]:
        sys.modules.pop(name)


def test_register_names_both_release_architectures_and_their_config_hooks(plugin):
    """A release's config.json names one of two architectures; each resolves to its class and config hook."""
    plugin.package.register()
    plugin.package.register()
    assert plugin.archs == {"ArcPolicyForPooling": "vllm_arc.model:ArcPolicyForPooling",
                            "ArcImagePolicyForPooling": "vllm_arc.model:ArcImagePolicyForPooling"}
    assert plugin.config_map == {"ArcPolicyForPooling": _BaseConfig,
                                 "ArcImagePolicyForPooling": plugin.config.ArcImagePolicyConfig}


def _engine(backend=None, tp_mode="weights", cudagraph=False):
    return SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=SimpleNamespace(mm_encoder_attn_backend=backend,
                                                                       mm_encoder_tp_mode=tp_mode)),
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=cudagraph))


@pytest.mark.parametrize("engine, refusal", [
    (_engine(Backend.FLASH_ATTN), "not mm_encoder_attn_backend=FLASH_ATTN"),
    (_engine(tp_mode="data"), "mm_encoder_tp_mode=data"),
    (_engine(cudagraph=True), "cudagraph_mm_encoder"),
])
def test_the_image_config_hook_sets_the_float32_vit_backend_and_refuses_batched_vit_calls(plugin, engine, refusal):
    """Unset, the ViT backend becomes TORCH_SDPA (after Qwen3.5's own hook); a chosen other backend, a data-parallel
    ViT or an encoder CUDA graph is refused, since each breaks float32 or one-call-per-image."""
    unset = _engine()
    plugin.config.ArcImagePolicyConfig.verify_and_update_config(unset)
    assert unset.model_config.multimodal_config.mm_encoder_attn_backend is Backend.TORCH_SDPA
    assert _BaseConfig.calls == [unset]
    with pytest.raises(ValueError, match=refusal):
        plugin.config.ArcImagePolicyConfig.verify_and_update_config(engine)


def test_the_image_model_loads_without_heads_and_runs_one_float32_vit_call_per_image(plugin, monkeypatch):
    """Heads are skipped, the ViT is float32 after loading, and two images make two ViT calls over their own
    patches and grid rows; precomputed embeddings and a patch count that disagrees with the grids are refused,
    and so is a ViT built with another attention backend."""
    model = plugin.model.ArcImagePolicyForPooling(vllm_config=None)
    weights = [(name, torch.zeros(1)) for name in (
        "model.visual.blocks.0.attn.qkv.weight", "model.language_model.norm.weight", "stage_one.mean",
        "stage_two.head.weight", "signals.topic.operation.net.0.weight")]
    assert model.load_weights(iter(weights)) == {"model.visual.blocks.0.attn.qkv.weight",
                                                 "model.language_model.norm.weight"}
    assert model.visual.dtype == torch.float32

    grid = torch.tensor([[1, 2, 2], [1, 2, 4]])
    pixels = torch.arange(12 * 4, dtype=torch.bfloat16).reshape(12, 4)
    embeds = model._process_image_input({"type": "pixel_values", "pixel_values": pixels, "image_grid_thw": grid})
    assert [e.shape[0] for e in embeds] == [4, 8] and all(e.dtype == torch.float32 for e in embeds)
    assert [(p.dtype, p.shape[0], g.tolist()) for p, g in model.visual.calls] == [
        (torch.float32, 4, [[1, 2, 2]]), (torch.float32, 8, [[1, 2, 4]])]
    assert torch.equal(model.visual.calls[1][0], pixels[4:].float())

    with pytest.raises(ValueError, match="never precomputed embeddings"):
        model._process_image_input({"type": "image_embeds", "image_embeds": pixels, "image_grid_thw": grid})
    with pytest.raises(ValueError, match="11 pixel patches"):
        model._process_image_input({"type": "pixel_values", "pixel_values": pixels[:11], "image_grid_thw": grid})
    monkeypatch.setattr(_Multimodal, "backend", Backend.FLASH_ATTN)
    with pytest.raises(ValueError, match="not FLASH_ATTN"):
        plugin.model.ArcImagePolicyForPooling(vllm_config=None)
