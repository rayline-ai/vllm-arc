# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""Writes ``stock_eq_rows.npz``, the recorded EQ rows ``tests/test_stock_eq_model.py`` checks ``ArcEqModel.forward``
against: ``python tests/fixtures/make_stock_eq_rows.py`` (CPU, the pinned torch of ``ci/requirements-test.txt``).

In bfloat16 (the served dtype) it runs the test's stand-in model (its seeds, layers, norm, embedding and inputs)
through ``plain_forward``, the stock readout (every layer, then the final norm), at 12 layers and at 24, and saves
the two readouts as ``<level>_<dtype>_e12`` and ``<level>_<dtype>_e24``. An EQ row is ``cat(e12, e24)``.

Float32 is not recorded: its bits are not portable. ATen's vectorised elementwise kernels round differently from
its scalar ones (``default`` and ``avx2`` differ in the last bits of most float32 entries), and even at one pinned
level (``avx2``) GitHub's x86-64 runners did not all reproduce a float32 recording. The test checks float32 only
relative to ``plain_forward`` on the same machine.

``<level>`` is the machine and ATen's CPU kernel level (the test's ``cpu_level()``, e.g. ``x86_64-avx2``). The script
records each level this host runs, from ``default`` up to its own (a level above it dies on an illegal instruction), one subprocess per level
(``ATEN_CPU_CAPABILITY``), and the test compares only at a recorded level.

Arrays hold the tensors' bits as ``int16`` (NumPy has no bfloat16), so the test compares bit patterns.

The file in the repository was recorded with this script (torch 2.15.0, levels ``x86_64-default`` and
``x86_64-avx2``; the host had no AVX-512) and checked, at the time, against the fork-era ``Qwen3NextModel.forward``
the EQ engine replaced: that forward's EQ rows (``readout_depths = [12, 24]``) and its plain readouts at 12 and 24
layers equalled the recorded rows bit for bit, in float32 and bfloat16, at each recorded level (the float32 rows
were dropped later as not portable). Rerun this only if the stand-ins or inputs change; the rows then no
longer carry that check."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
OUT = HERE / "stock_eq_rows.npz"
#: x86 levels, lowest first; elsewhere ``default`` and the host's own.
X86_LEVELS = ("default", "avx2", "avx512")
BITS = {torch.bfloat16: torch.int16}


def bits(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(BITS[t.dtype]).numpy()


def _test_module():
    sys.path.insert(0, str(HERE.parent))
    import test_stock_eq_model

    return test_stock_eq_model


def rows() -> dict[str, np.ndarray]:
    """This process's readouts, keyed ``<level>_<dtype>_<e12|e24>``."""
    t = _test_module()
    out = {}
    for dtype in BITS:
        layers, norm, embed, ids, positions = t._inputs(dtype)
        name = str(dtype).removeprefix("torch.")
        for key, depth in (("e12", t.CUT), ("e24", t.LAYERS)):
            out[f"{t.cpu_level()}_{name}_{key}"] = bits(
                t._run(t.plain_forward, t.StandIn(layers[:depth], norm, embed, ()), ids, positions))
    return out


def main() -> None:
    native = torch.backends.cpu.get_cpu_capability().lower()
    levels = X86_LEVELS[:X86_LEVELS.index(native) + 1] if native in X86_LEVELS else tuple(dict.fromkeys(
        ("default", native)))
    recorded: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for level in levels:
            part = Path(tmp) / f"{level}.npz"
            subprocess.run([sys.executable, __file__, "--level", level, str(part)], check=True,
                           env={**os.environ, "ATEN_CPU_CAPABILITY": level})
            with np.load(part) as got:
                recorded.update(got)
    np.savez(OUT, **recorded)
    print(OUT, sorted({k.rsplit("_", 2)[0] for k in recorded}))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--level"]:
        level, part = sys.argv[2], sys.argv[3]
        if not _test_module().cpu_level().endswith(f"-{level}"):
            raise SystemExit(f"ran at {_test_module().cpu_level()}, not {level}")
        np.savez(part, **rows())
    else:
        main()
