# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The checkpoints on CPU, over a stand-in base snapshot (no hub, no safetensors file): E12's config is byte for byte
what it was before EQ (the edits written out here as the previous code made them), and EQ's is the base at 24 layers
naming ``ArcEqForPooling`` with ``readout_depths``, through the fork's add-on builder or, when the add-on refuses, the
stock builder."""
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from vllm_arc.eq import addon as eq_addon  # noqa: E402
from vllm_arc.eq import engine as eq
from vllm_arc.server import app as server  # noqa: E402

BASE_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
    "text_config": {"num_hidden_layers": 24, "layer_types": [f"t{i}" for i in range(24)], "mtp_num_hidden_layers": 1,
                    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000}, "max_position_embeddings": 262144,
                    "hidden_size": 1024},
    "vision_config": {"depth": 2},
}


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    """A base snapshot: its config.json, a tokenizer file and one shard whose tensors are named per layer."""
    source = tmp_path / "base"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(BASE_CONFIG))
    (source / "tokenizer.json").write_text("{}")
    (source / "model.safetensors").write_text("")
    tensors = {f"model.language_model.layers.{i}.w": i for i in range(24)} | {"mtp.x": 0, "model.visual.w": 0}
    saved = {}
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda model, revision=None, **kw: str(source)
    st = types.ModuleType("safetensors.torch")
    st.load_file = lambda path: dict(tensors)
    st.save_file = lambda t, path, metadata=None: (saved.__setitem__(str(path), dict(t)), Path(path).write_text("x"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "safetensors", types.ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", st)
    return saved


def _previous_e12_config(rope: str) -> str:
    """E12's config.json as the stock server wrote it before EQ, edit for edit."""
    config = copy.deepcopy(BASE_CONFIG)
    config["architectures"] = ["ArcBaseForPooling"]
    text = config["text_config"]
    text["num_hidden_layers"] = 12
    text["layer_types"] = text["layer_types"][:12]
    text.pop("mtp_num_hidden_layers", None)
    if rope:
        text["rope_parameters"] = dict(server.ROPE_VARIANTS[rope])
        p = server.ROPE_VARIANTS[rope]
        text["max_position_embeddings"] = int(p["original_max_position_embeddings"] * p["factor"])
    return json.dumps(config, indent=1)


@pytest.mark.parametrize("rope", ["yarn-f4", ""])
def test_the_e12_checkpoint_is_unchanged(snapshot, tmp_path, rope):
    path = Path(server.truncated_checkpoint(str(tmp_path / "ckpt"), rope))
    assert path.name == ("Qwen--Qwen3.5-0.8B-2fc06364715b-layers12" + (f"-{rope}-mpe" if rope else "")
                         + "-ArcBaseForPooling")
    assert (path / "config.json").read_text() == _previous_e12_config(rope)
    [kept] = snapshot.values()
    assert sorted(k for k in kept if "layers" in k) == sorted(f"model.language_model.layers.{i}.w" for i in range(12))


def test_the_eq_base_checkpoint_keeps_24_layers_and_names_the_readouts(snapshot, tmp_path):
    path = Path(server.truncated_checkpoint(str(tmp_path / "ckpt"), "yarn-f4", depth=eq.DEPTH, arch=eq.ARCHITECTURE,
                                            readout_depths=eq.READOUT_DEPTHS))
    assert path.name.endswith("-layers24-yarn-f4-mpe-ArcEqForPooling-readouts12-24")
    config = json.loads((path / "config.json").read_text())
    text = config["text_config"]
    assert config["architectures"] == ["ArcEqForPooling"] and text["readout_depths"] == [12, 24]
    assert text["num_hidden_layers"] == 24 and len(text["layer_types"]) == 24 and "mtp_num_hidden_layers" not in text
    assert text["rope_parameters"] == server.ROPE_VARIANTS["yarn-f4"] and text["max_position_embeddings"] == 1048576
    [kept] = snapshot.values()
    assert sum("layers" in k for k in kept) == 24 and "mtp.x" not in kept


def test_eq_checkpoint_builds_the_addon_with_the_stock_config(monkeypatch):
    seen = {}

    def build(addon, root, **kw):
        seen.update(addon=addon, root=root, **kw)
        config = copy.deepcopy(BASE_CONFIG)
        kw["config_hook"](config)
        seen["config"] = config
        return "/ckpt/eq-qh", {"name": "qh-1"}

    monkeypatch.setattr(eq_addon, "checkpoint", build)
    monkeypatch.setattr(eq_addon, "resolve", lambda addon: f"/resolved/{addon}")
    path, status = eq.checkpoint("qh-addon/x", "/root", "yarn-f4")
    assert (path, status) == ("/ckpt/eq-qh", {"name": "qh-1", "verified": True, "reason": None})
    assert seen["addon"] == "/resolved/qh-addon/x" and seen["readout_depths"] == [12, 24]
    assert seen["name_suffix"] == "-ArcEqForPooling" and seen["rope"] == "yarn-f4"
    assert seen["config"]["architectures"] == ["ArcEqForPooling"]
    assert seen["config"]["text_config"]["max_position_embeddings"] == 1048576


def test_a_refused_addon_serves_the_base_at_24_layers(monkeypatch):
    def refuse(addon, root, **kw):
        raise eq_addon.AddonRefused("trunk_differs", "x")

    monkeypatch.setattr(eq_addon, "checkpoint", refuse)
    monkeypatch.setattr(eq_addon, "resolve", lambda addon: addon)
    calls = []
    monkeypatch.setattr(server, "truncated_checkpoint", lambda root, rope, **kw: calls.append((root, rope, kw)) or "/b")
    assert eq.checkpoint("qh-addon/x", "/root", "yarn-f4") == ("/b", {"name": None, "verified": False,
                                                                    "reason": "trunk_differs"})
    assert calls == [("/root", "yarn-f4", {"depth": 24, "arch": "ArcEqForPooling", "readout_depths": (12, 24)})]


def test_the_fork_addon_builder_is_unchanged_without_the_hook():
    import inspect

    params = inspect.signature(eq_addon.checkpoint).parameters
    assert params["config_hook"].default is None and params["name_suffix"].default == ""
