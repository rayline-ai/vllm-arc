# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The plugin's touch points on vLLM internals (``docs/vllm-internals.md``), each checked to exist first.

- ``replaced(module, **values)``: module attributes replaced while a model is built, restored after it, under one
  module-level lock so two builds in one process never see each other's values.
- ``register_config(name, config)``: an entry in vLLM's ``MODELS_CONFIG_MAP``, under the same lock.

A missing attribute raises ``MissingVllmInternal`` naming the vLLM release this plugin supports, instead of an
``AttributeError`` (or a silently created attribute) deep inside vLLM. Imports no vLLM module itself.
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from types import ModuleType

#: The vLLM release whose internals these touch points name.
SUPPORTED_VLLM = "0.33.0"

_LOCK = threading.Lock()


class MissingVllmInternal(RuntimeError):
    """A vLLM internal the plugin touches is absent: the installed vLLM is not the release it supports."""


def internal(module: ModuleType, name: str):
    """``module.name``, or ``MissingVllmInternal`` when vLLM has no such attribute."""
    if not hasattr(module, name):
        raise MissingVllmInternal(f"{module.__name__}.{name} does not exist in the installed vLLM: vllm_arc supports "
                                  f"vLLM {SUPPORTED_VLLM} (see docs/vllm-internals.md)")
    return getattr(module, name)


@contextlib.contextmanager
def replaced(module: ModuleType, **values) -> Iterator[None]:
    """Replace each ``module.<name>`` with its value for the body, holding the module-level lock; every original is
    restored on exit, an exception included. Every name must already exist on ``module``."""
    with _LOCK:
        originals = {name: internal(module, name) for name in values}
        try:
            for name, value in values.items():
                setattr(module, name, value)
            yield
        finally:
            for name, value in originals.items():
                setattr(module, name, value)


def register_config(config_module: ModuleType, architecture: str, config_name: str) -> None:
    """``MODELS_CONFIG_MAP[architecture]`` set to ``config_module.<config_name>`` unless present, under the lock."""
    with _LOCK:
        table = internal(config_module, "MODELS_CONFIG_MAP")
        config = internal(config_module, config_name)
        table.setdefault(architecture, config)
