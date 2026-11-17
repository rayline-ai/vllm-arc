# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""Start-up assertions and engine args on CPU: a probe of the expected engine passes; each silent arithmetic change
the server guards against fails exactly its own assertion."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from vllm_arc.engine import recipe as image
from vllm_arc.server import app as server  # noqa: E402
from vllm_arc.engine.startup import (COMPILATION, COMPILED_NORMS, FORK_YARN_INV_FREQ, V1_RUNNER, Engine,  # noqa: E402
                                    assertions, failed)

YARN = server.ROPE_VARIANTS["yarn-f4"]
FORK_SHA, FORK_HEX = FORK_YARN_INV_FREQ[(64, 10000000.0, 4.0, 262144)]
ROPE_LAYER = {"class": "MRotaryEmbedding", "scaling_factor": 4.0, "cache_max_position": 1048576,
              "cos_sin_cache_rows": 4194304, "base": 10000000.0, "rotary_dim": 64, "head_size": 256,
              "mrope_section": [11, 11, 10], "mrope_interleaved": True, "mscale": 1.1386,
              "inv_freq_hex": list(FORK_HEX), "inv_freq_sha256": FORK_SHA}
#: Stock 0.33's inverse frequencies (positions factor 1: correction range (14, 22)), pairs 15-23 moved.
STOCK_SHA = "83543161985e6411b3b839a9f12df3c631298b10ef5b8af1bf5738089951db1c"


def _probe(**cache) -> dict:
    return {
        "vllm_version": "0.33.0", "torch_cuda": "13.0", "runner": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
        "model_mro": ["ModelForEmbedding", "ModelForPooling", "ArcBaseForPooling", "Qwen3_5ForConditionalGeneration"],
        "pooler": "GatherPooler", "gdn": [{"backend": "flashinfer", "forward": "forward_cuda"}] * 9,
        "fused_qk_stock": [True, True, True], "fused_qk_running": [False, False, False],
        "inner_model": "Qwen3_5Model", "do_not_compile": True,
        "rope_parameters": {**YARN, "beta_fast": 32},
        "rope_effective": [dict(r) for r in [ROPE_LAYER] * 3], "multimodal_pruning": False,
        "cache": {"enable_prefix_caching": True, "block_size": 1024, "mamba_block_size": 1024,
                  "mamba_cache_mode": "align", "prefix_match_unit": None, "prefix_caching_hash_algo": "sha256",
                  "prefix_cache_retention_interval": 0, "mamba_ssm_cache_dtype": "float32",
                  "hash_block_size": 1024, "cache_hit_alignment_tokens": 1024, **cache},
        "scheduler": {"enable_chunked_prefill": True}, "enforce_eager": True,
        "env": {"VLLM_TRITON_FORCE_FIRST_CONFIG": None},
    }


PREFIX = Engine(sessions="prefix")
RETAINED_CACHE = {"enable_prefix_caching": False, "mamba_cache_mode": "none", "mamba_block_size": 262144,
                  "hash_block_size": None, "cache_hit_alignment_tokens": None}
PATCH = {"id": "0001-retained-pooling-session", "runner_extend": True, "extends": 0}


def _retained_probe(**cache) -> dict:
    """The default engine's probe: patch 0001 installed and no prefix caching (vLLM resolves mamba mode "none")."""
    return {**_probe(**{**RETAINED_CACHE, **cache}), "arc_patch": dict(PATCH)}


def _failures(probe, engine=PREFIX, rope=YARN):
    return [c["name"] for c in failed(assertions(probe, engine, rope))]


def test_the_expected_engine_passes_every_assertion():
    assert _failures(_retained_probe(), Engine()) == []
    assert _failures(_probe()) == []
    assert _failures(_probe(prefix_match_unit=192, hash_block_size=192, cache_hit_alignment_tokens=192, block_size=576),
                     Engine(unit=192, sessions="prefix")) == []


@pytest.mark.parametrize("change, name", [
    (lambda p: p.update(vllm_version="0.32.1"), "vllm_version"),
    (lambda p: p.update(torch_cuda="12.9"), "cuda_runtime_major"),
    (lambda p: p["gdn"].__setitem__(3, {"backend": "triton", "forward": "forward_native"}), "gdn_prefill_backend"),
    (lambda p: p.update(gdn=[]), "gdn_prefill_backend"),
    (lambda p: p.update(runner=V1_RUNNER), "model_runner"),
    (lambda p: p.update(pooler="DispatchPooler"), "pooler"),
    (lambda p: p.update(fused_qk_running=[False, True, False]), "fused_qk_norm_rope_gate"),
    (lambda p: p.update(fused_qk_running=[]), "fused_qk_norm_rope_gate"),
    (lambda p: p.update(model_mro=["ModelForEmbedding", "Qwen3_5ForConditionalGeneration"]), "model_class"),
    (lambda p: p["cache"].update(mamba_cache_mode="all"), "prefix_caching"),
    (lambda p: p["cache"].update(prefix_caching_hash_algo="xxhash"), "hash_algo"),
    (lambda p: p["cache"].update(mamba_ssm_cache_dtype="bfloat16"), "gdn_state_dtype"),
    (lambda p: p["cache"].update(prefix_cache_retention_interval=512), "retention_interval"),
    (lambda p: p["rope_parameters"].update(factor=2.0), "rope_parameters"),
    (lambda p: p["rope_effective"][1].update(scaling_factor=1.0), "rope_effective"),
    (lambda p: p["rope_effective"][0].update(cache_max_position=262144), "rope_effective"),
    (lambda p: p["rope_effective"][1].update(cos_sin_cache_rows=1048576), "rope_effective"),
    (lambda p: p.update(multimodal_pruning=True), "multimodal_pruning"),
    (lambda p: p.pop("multimodal_pruning"), "multimodal_pruning"),
    (lambda p: p["rope_effective"][2].update(inv_freq_sha256=STOCK_SHA), "rope_effective"),
    (lambda p: p["rope_effective"][0].pop("inv_freq_sha256"), "rope_effective"),
    (lambda p: p["rope_effective"][2].update(rotary_dim=256), "rope_effective"),
    (lambda p: p["rope_effective"][0].update(mrope_section=[16, 24, 24]), "rope_effective"),
    (lambda p: p.update(rope_effective=[]), "rope_effective"),
    (lambda p: p.update(enforce_eager=False), "enforce_eager"),
])
def test_each_silent_change_fails_its_own_assertion(change, name):
    probe = copy.deepcopy(_probe())
    change(probe)
    assert _failures(probe) == [name]


#: A compiled engine's probe (EQ's: ``Engine.compiled``), as vLLM resolves ``COMPILATION`` on CUDA.
COMPILED = Engine(sessions="prefix", compiled=True)


def _compiled_probe() -> dict:
    return {**_probe(), "enforce_eager": False, "do_not_compile": False,
            "compilation": {**COMPILATION, "ir_op_priority": copy.deepcopy(COMPILED_NORMS)}}


def test_a_compiled_engine_is_judged_compiled_and_every_other_eager():
    assert _failures(_compiled_probe(), COMPILED) == []
    names = [c["name"] for c in assertions(_compiled_probe(), COMPILED, YARN)]
    assert "compiled" in names and "enforce_eager" not in names
    # An eager engine where a compiled one is claimed fails, and so does a compiled one where eager is (E12).
    assert _failures(_probe(), COMPILED) == ["compiled"]
    assert _failures(_compiled_probe(), PREFIX) == ["enforce_eager"]


@pytest.mark.parametrize("change", [
    lambda p: p["compilation"].update(cudagraph_mode="FULL_AND_PIECEWISE"),
    lambda p: p["compilation"].update(cudagraph_mode="NONE"),
    lambda p: p["compilation"].update(mode="STOCK_TORCH_COMPILE"),
    lambda p: p["compilation"].update(backend="eager"),
    lambda p: p["compilation"].update(custom_ops=["all"]),
    lambda p: p["compilation"]["ir_op_priority"].update(rms_norm=["vllm_c", "native"]),
    lambda p: p["compilation"]["ir_op_priority"].update(fused_add_rms_norm=["oink", "native"]),
    lambda p: p.update(do_not_compile=True),
    lambda p: p.update(enforce_eager=True),
    lambda p: p.pop("compilation"),
])
def test_each_compile_setting_change_fails_the_compiled_assertion(change):
    probe = _compiled_probe()
    change(probe)
    assert _failures(probe, COMPILED) == ["compiled"]


def test_only_a_compiled_engine_gets_compile_engine_args():
    common = dict(budget=8192, max_model_len=262144, max_sessions=64, scratch_slots=4, gpu_memory=0.8)
    eager = server.engine_args(PREFIX, "/ckpt", **common)
    compiled = server.engine_args(COMPILED, "/ckpt", **common)
    assert eager["enforce_eager"] is True and "compilation_config" not in eager
    assert compiled["enforce_eager"] is False and compiled["compilation_config"] == COMPILATION
    assert compiled["compilation_config"] is not COMPILATION  # vLLM may edit its copy, never the constant
    assert {k: v for k, v in compiled.items() if k not in ("enforce_eager", "compilation_config")} == {
        k: v for k, v in eager.items() if k != "enforce_eager"}
    # The settings vLLM's config parses (with vLLM installed).
    try:
        from vllm.config import CompilationConfig
    except Exception:  # noqa: BLE001 -- no vLLM (CI)
        return
    cc = CompilationConfig(**compiled["compilation_config"])
    assert (cc.mode.name, cc.backend, cc.custom_ops, cc.cudagraph_mode.name) == (
        "VLLM_COMPILE", "inductor", ["none"], "PIECEWISE")


def test_the_engine_id_covers_the_model_source(tmp_path):
    """A model change (here, ``models.py``: EQ's build and forward) changes the plugin digest, so the engine id."""
    import shutil

    package = Path(server.__file__).resolve().parents[1]
    copy_ = tmp_path / "vllm_arc"
    shutil.copytree(package, copy_, ignore=shutil.ignore_patterns("__pycache__"))
    models = copy_ / "engine" / "models.py"
    models.write_text(models.read_text().replace('"Qwen3_5Model": ArcEqModel}', '"Qwen3_5Model": Qwen3_5Model}', 1))
    assert models.read_text() != (package / "engine" / "models.py").read_text()
    assert server.plugin_digest(copy_) != server.plugin_digest()


def test_the_fork_inverse_frequency_reference_is_self_consistent():
    """The stored sha256 is of the stored values' float32 bytes, and the values are float32-exact."""
    import hashlib

    import numpy as np

    values = np.array([float.fromhex(x) for x in FORK_HEX], dtype=np.float32)
    assert [float(v).hex() for v in values] == list(FORK_HEX)
    assert hashlib.sha256(values.tobytes()).hexdigest() == FORK_SHA


def test_an_inverse_frequency_mismatch_names_the_moved_pairs():
    probe = copy.deepcopy(_probe())
    moved = [float.fromhex(x) for x in FORK_HEX]
    moved[16] *= 0.8125
    probe["rope_effective"][0].update(inv_freq_hex=[v.hex() for v in moved], inv_freq_sha256=STOCK_SHA)
    check = next(c for c in assertions(probe, PREFIX, YARN) if c["name"] == "rope_effective")
    assert not check["ok"]
    assert check["observed"]["inv_freq_vs_fork"]["differing_pairs"] == [16]
    assert check["observed"]["inv_freq_vs_fork"]["max_rel_diff"] == pytest.approx(0.1875)


def test_a_hit_grid_holds_only_if_hashes_and_hits_land_on_it():
    """A 64-unit engine whose hits fall back to the block (partial hash hits off) is not on the 64 grid."""
    engine = Engine(unit=64, sessions="prefix")
    assert _failures(_probe(prefix_match_unit=64, hash_block_size=64, cache_hit_alignment_tokens=1024), engine) == [
        "hit_grid [block, hash, hit_alignment, unit]"]
    assert _failures(_probe(), engine) == ["hit_grid [block, hash, hit_alignment, unit]"]


def test_the_default_retained_route_needs_patch_0001_and_no_prefix_caching():
    engine = Engine.from_env({})
    assert engine == Engine() and engine.retained and engine.label == "retained"
    assert Engine.from_env({"ARC_STOCK_SESSIONS": "prefix", "ARC_STOCK_PREFIX_UNIT": "192"}) == Engine(192, "prefix")
    for env in ({"ARC_STOCK_SESSIONS": "salted"}, {"ARC_STOCK_PREFIX_UNIT": "192"}):
        with pytest.raises(ValueError):
            Engine.from_env(env)
    # Without the patch, with it but without its runner hook, or with prefix caching on: each fails its own check.
    assert _failures({k: v for k, v in _retained_probe().items() if k != "arc_patch"}, engine) == ["arc_patch"]
    assert _failures({**_retained_probe(), "arc_patch": {**PATCH, "runner_extend": False}}, engine) == ["arc_patch"]
    assert _failures({**_probe(), "arc_patch": dict(PATCH)}, engine) == ["prefix_caching"]
    assert "prefix_caching" in _failures(_retained_probe())  # and the prefix route still requires it
    args = server.engine_args(engine, "/ckpt", budget=8192, max_model_len=262144, max_sessions=64, scratch_slots=4,
                              gpu_memory=0.8)
    # vLLM sets mamba mode "none" without prefix caching: no block-aligned split, so a chunk is the whole budget.
    assert args["enable_prefix_caching"] is False and args["max_num_batched_tokens"] == 8192
    assert not {"mamba_cache_mode", "prefix_match_unit", "prefix_caching_hash_algo",
                "prefix_cache_retention_interval"} & set(args)
    assert server.engine_id(engine, "flashinfer", "yarn-f4").endswith("+depth12+rope-yarn-f4+stock-retained")


def test_engine_args_are_the_interface_ones():
    args = server.engine_args(Engine(unit=64, sessions="prefix"), "/ckpt", budget=8192, max_model_len=262144,
                              max_sessions=64, scratch_slots=4, gpu_memory=0.8)
    assert args["prefix_match_unit"] == 64 and server.engine_args(
        PREFIX, "/ckpt", budget=1, max_model_len=1, max_sessions=1, scratch_slots=1, gpu_memory=0.8)[
        "prefix_match_unit"] is None
    assert {k: args[k] for k in ("runner", "convert", "enforce_eager", "gdn_prefill_backend", "enable_prefix_caching",
                                 "mamba_cache_mode", "prefix_cache_retention_interval", "max_num_seqs",
                                 "worker_extension_cls", "pooler_config")} == {
        "runner": "pooling", "convert": "embed", "enforce_eager": True, "gdn_prefill_backend": "flashinfer",
        "enable_prefix_caching": True, "mamba_cache_mode": "align", "prefix_cache_retention_interval": 0,
        "max_num_seqs": 68, "worker_extension_cls": "vllm_arc.engine.probe.ArcStockProbe",
        "pooler_config": {"seq_pooling_type": "MEAN", "tok_pooling_type": "ALL", "use_activation": False}}
    assert server.engine_id(Engine(unit=192, sessions="prefix"), "flashinfer", "yarn-f4").startswith("vllm@4ba787ec9959+arc-")
    assert server.engine_id(Engine(unit=192, sessions="prefix"), "flashinfer", "yarn-f4").endswith(
        "+gather-flashinfer+depth12+rope-yarn-f4+stock-u192")
    assert args["prefix_caching_hash_algo"] == "sha256"


def test_the_image_is_stock_pypi_vllm_hash_pinned_with_a_release_age_floor():
    r = image.recipe()
    [install] = [c for c in r["commands"] if "vllm @" in c]
    assert f"'vllm @ {image.VLLM_WHEEL}#sha256={image.VLLM_WHEEL_SHA256}'" in install
    assert image.VLLM_WHEEL.startswith("https://files.pythonhosted.org/") and "0.33.0" in image.VLLM_WHEEL
    assert all(f"--exclude-newer {image.EXCLUDE_NEWER}" in c for c in r["commands"] + r["plugin_commands"]
               if "uv pip install" in c)
    assert "nvidia-cuda-nvcc==13.0.88" in install and "nvidia-cuda-cccl==13.0.85" in install
    # The whole locked set, installed without resolution: nothing floats.
    assert " --no-deps " in install and install.endswith(" ".join(image.PINS))
    assert r["server_command"] == [f"{image.VENV}/bin/python", "-m", "vllm_arc.server"]
    assert {"venv", "freeze_path", "plugin_dir", "plugin_commands", "env", "base_image", "apt"} <= set(r)
    assert not any("index-url" in c for c in r["commands"] + r["plugin_commands"])
    # Each vLLM patch is checked against the lock's sha256, then applied to the installed wheel, before the plugin.
    checks = [c for c in r["plugin_commands"] if "sha256sum -c" in c]
    applies = [c for c in r["commands"] + r["plugin_commands"] if "git " in c]
    assert checks == [f"echo '{sha}  {image.PATCH_DIR}/{name}' | sha256sum -c -" for sha, name in image.LOCK.patches]
    assert applies == [f"git -C {image.SITE} apply -p1 {image.PATCH_DIR}/{p}" for p in image.PATCHES]
    assert r["plugin_commands"][:len(checks) + len(applies)] == checks + applies
    assert r["patches"] == [{"file": n, "sha256": s, "path": f"{image.PATCH_DIR}/{n}"} for s, n in image.LOCK.patches]
    assert r["patch_source"]["repository"] == image.PATCH_SOURCE[0]


def _package_copy(tmp_path):
    import shutil

    package = Path(server.__file__).resolve().parents[1]
    copy = tmp_path / "vllm_arc"
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert server.plugin_digest(copy) == server.plugin_digest()
    return copy


def test_the_engine_id_covers_the_vllm_patches_it_applies(tmp_path):
    """The patches are named in engine.lock by sha256 (and the files they leave by sha256): a different patch is a
    different lock, so a different engine id."""
    copy = _package_copy(tmp_path)
    lock = copy / "engine" / "engine.lock"
    [sha] = [s for s, _ in image.LOCK.patches]
    lock.write_text(lock.read_text().replace(sha, "0" * 64, 1))
    assert server.plugin_digest(copy) != server.plugin_digest()


@pytest.mark.parametrize("edit", [
    lambda t: t.replace("\nimport ", "\n# a new comment\nimport ", 1),
    lambda t: t.replace('"""The ARC encoder', '"""Reworded: the ARC encoder', 1),
    lambda t: t + "\n\n# trailing comment\n",
])
def test_comments_and_docstrings_do_not_change_the_engine_id(tmp_path, edit):
    copy = _package_copy(tmp_path)
    for rel in ("engine/models.py", "server/app.py"):
        path = copy / rel
        path.write_text(edit(path.read_text()))
    assert server.plugin_digest(copy) == server.plugin_digest()


def test_files_outside_the_engine_do_not_change_the_engine_id(tmp_path):
    copy = _package_copy(tmp_path)
    for rel in ("heads.py", "arc_heads.py"):
        path = copy / rel
        path.write_text(path.read_text() + "\nX = 1\n")
    assert server.plugin_digest(copy) == server.plugin_digest()
    eq = copy / "eq" / "engine.py"
    eq.write_text(eq.read_text() + "\nX = 1\n")
    assert server.plugin_digest(copy) != server.plugin_digest()


def test_the_engine_id_covers_the_engine_lock(tmp_path):
    """A pin is an arithmetic input: editing engine.lock changes the plugin digest, so the engine id."""
    import shutil

    package = Path(server.__file__).resolve().parents[1]
    copy = tmp_path / "vllm_arc"
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
    lock = copy / "engine" / "engine.lock"
    lock.write_text(lock.read_text().replace("pin torch==2.14.0\n", "pin torch==2.14.1\n", 1))
    assert server.plugin_digest(copy) != server.plugin_digest()


def test_the_yarn_checkpoint_carries_scaled_positions_for_stock_vllm():
    """Stock 0.33 reads a YaRN config's max_position_embeddings as already scaled: the cut checkpoint writes
    original x factor, so max_model_len 1,010,000 validates without VLLM_ALLOW_LONG_MAX_MODEL_LEN."""
    import inspect

    source = inspect.getsource(server.stock_config)
    assert 'text["max_position_embeddings"] = int(p["original_max_position_embeddings"] * p["factor"])' in source
    assert int(YARN["original_max_position_embeddings"] * YARN["factor"]) >= server.ROPE_CAPACITY["yarn-f4"]
