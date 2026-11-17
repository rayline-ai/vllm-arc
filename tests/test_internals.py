# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The plugin's touch points on vLLM internals (``vllm_arc.engine.internals``) on CPU: a missing symbol is a clear
error naming the supported vLLM, and every replaced attribute is restored, an exception included."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from vllm_arc.engine import internals  # noqa: E402


def _module() -> types.ModuleType:
    module = types.ModuleType("vllm.fake")
    module.factor = lambda _cfg: 1
    module.Model = object
    return module


def test_replaced_values_apply_in_the_body_under_the_lock_only():
    module = _module()
    original = module.factor
    with internals.replaced(module, factor=4, Model=int):
        assert (module.factor, module.Model) == (4, int)
        assert internals._LOCK.locked()
    assert (module.factor, module.Model) == (original, object)
    assert not internals._LOCK.locked()


def test_replaced_restores_on_an_exception():
    module = _module()
    original = module.factor
    with pytest.raises(KeyError):
        with internals.replaced(module, factor=4):
            raise KeyError("model build failed")
    assert module.factor is original and not internals._LOCK.locked()


def test_a_missing_symbol_is_a_clear_error_and_nothing_is_replaced():
    module = _module()
    with pytest.raises(internals.MissingVllmInternal, match=r"vllm\.fake\.absent .*supports vLLM 0\.33\.0"):
        with internals.replaced(module, factor=4, absent=1):
            pass
    assert module.factor(None) == 1 and not hasattr(module, "absent") and not internals._LOCK.locked()


def test_register_config_sets_a_missing_entry_only():
    module = types.ModuleType("vllm.model_executor.models.config")
    module.MODELS_CONFIG_MAP = {"Kept": "theirs"}
    module.Qwen3_5ForConditionalGenerationConfig = "qwen"
    internals.register_config(module, "ArcBaseForPooling", "Qwen3_5ForConditionalGenerationConfig")
    internals.register_config(module, "Kept", "Qwen3_5ForConditionalGenerationConfig")
    assert module.MODELS_CONFIG_MAP == {"Kept": "theirs", "ArcBaseForPooling": "qwen"}


@pytest.mark.parametrize("drop", ["MODELS_CONFIG_MAP", "Qwen3_5ForConditionalGenerationConfig"])
def test_register_config_names_a_missing_symbol(drop):
    module = types.ModuleType("vllm.model_executor.models.config")
    module.MODELS_CONFIG_MAP, module.Qwen3_5ForConditionalGenerationConfig = {}, "qwen"
    delattr(module, drop)
    with pytest.raises(internals.MissingVllmInternal, match=f"{drop} does not exist .*vLLM 0.33.0"):
        internals.register_config(module, "ArcBaseForPooling", "Qwen3_5ForConditionalGenerationConfig")
