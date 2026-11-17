# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``ArcEqModel.forward`` (EQ on stock) against recorded EQ rows and against the stock readout, on stand-in layers
and inputs, on CPU.

The recorded rows (``fixtures/stock_eq_rows.npz``, written by ``fixtures/make_stock_eq_rows.py``) are the 12- and
24-layer plain readouts of these stand-ins in bfloat16 (the served dtype), stored as bits. When they were recorded
they equalled, bit for bit, the rows of the fork-era ``Qwen3NextModel.forward`` the EQ engine replaced (its EQ rows
and its plain readouts, in float32 and bfloat16), so ``ArcEqModel`` is held to that forward's output without
carrying its source.

What is not recorded: float32. Its bits depend on the CPU the test runs on, not only on ATen's kernel level: GitHub's
x86-64 runners with ``ATEN_CPU_CAPABILITY=avx2`` gave float32 rows that differed from a recording that matched on
other runners of the same image. So in float32 nothing here pins absolute values to the fork-era forward. What
still holds in float32 is relative and on the same machine: ``ArcEqModel``'s rows equal ``plain_forward``'s 12- and
24-layer readouts bit for bit, below.

The layers are stand-ins with the property the pair path exists for: like vLLM's fused add-and-norm, each layer's
input norm writes its ``(hidden, residual)`` inputs in place, so a readout that kept the pair without copies, or
read ``hidden + residual`` instead of the pair through the final norm, differs. The forwards run the same op
sequence on the same tensors, so the tolerance is zero: rows equal bit for bit (``torch.equal``), in float32 and in
bfloat16 (the served dtype). Equality within a tolerance would hide a reordered addition, which is what moves
``h_12`` off E12's readout on the GPU.

Also: ``h_12`` (the first half of an EQ row) equals the 12-layer model's readout from the same 12 layers, and the
second half equals the 24-layer model's plain readout, both computed here (``plain_forward``): the CPU mirror of
the 12-layer parity check made on the GPU. With vLLM installed, the plain readout is also stock 0.33's own
``Qwen3NextModel.forward`` (skipped otherwise)."""
from __future__ import annotations

import ast
import os
import platform
import sys
import textwrap
import types
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
ROWS = Path(__file__).parent / "fixtures" / "stock_eq_rows.npz"
HIDDEN, LAYERS, CUT = 16, 24, 12


def _pp_group():
    return SimpleNamespace(is_first_rank=True, is_last_rank=True)


def plain_forward(self, input_ids, positions):
    """The stock readout on one rank (vLLM's ``Qwen3NextModel.forward`` without pipeline or sequence parallelism):
    every layer, then the final norm; with auxiliary layers, also ``(rows, [hidden + residual after each])``."""
    hidden_states, residual = self.embed_input_ids(input_ids), None
    aux = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for layer_idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer), start=self.start_layer):
        hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, residual=residual)
        self._maybe_add_hidden_state(aux, layer_idx + 1, hidden_states, residual)
    hidden_states, _ = self.norm(hidden_states, residual)
    return (hidden_states, aux) if aux else hidden_states


def cpu_level() -> str:
    """The machine and ATen's CPU kernel level, e.g. ``x86_64-avx2``: what the recorded rows are keyed by."""
    return f"{platform.machine().lower()}-{torch.backends.cpu.get_cpu_capability().lower()}"


def recorded(dtype, key: str) -> torch.Tensor:
    """A recorded readout (``e12`` or ``e24``) from ``fixtures/stock_eq_rows.npz`` at this process's CPU level, as a
    ``dtype`` tensor. Rows are kept per level (the generator's doc), so at a level with no recording the
    comparison is skipped, by name, rather than made within a tolerance. Under CI (``CI`` set, as GitHub
    Actions sets it) a missing level fails instead: CI pins a recorded one (``ATEN_CPU_CAPABILITY=avx2`` in
    ``.github/workflows/ci.yml``), so these checks always run there."""
    name = f"{cpu_level()}_{str(dtype).removeprefix('torch.')}_{key}"
    with np.load(ROWS) as rows:
        if name not in rows:
            why = (f"no rows recorded at CPU level {cpu_level()!r} (recorded: "
                   f"{sorted({k.rsplit('_', 2)[0] for k in rows})}); see fixtures/make_stock_eq_rows.py")
            if os.environ.get("CI"):
                pytest.fail(f"{why}; pin a recorded level with ATEN_CPU_CAPABILITY")
            pytest.skip(why)
        return torch.from_numpy(rows[name].copy()).view(dtype)


def _method(source: str, cls: str, name: str, namespace: dict):
    """``cls.name`` compiled from ``source`` alone, its globals ``namespace``."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    code = textwrap.dedent(ast.get_source_segment(source, item))
                    exec(compile(code, f"<{cls}.{name}>", "exec"), namespace)  # noqa: S102 -- this repo's file
                    return namespace[name]
    raise LookupError(f"{cls}.{name}")


class FusedNorm(torch.nn.Module):
    """vLLM's RMSNorm with a residual (``forward_native``'s arithmetic), in place as the fused GPU kernel writes it:
    ``z = float(x) + float(residual)``, ``residual = z`` in the model dtype, ``x = norm(z)`` from the float ``z``."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(HIDDEN, generator=torch.Generator().manual_seed(seed)))

    def _norm(self, v: torch.Tensor, dtype) -> torch.Tensor:
        return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + self.weight.float())).to(dtype)

    def forward(self, x, residual=None):
        if residual is None:
            return self._norm(x.float(), x.dtype)
        z = x.float() + residual.float()
        residual.copy_(z.to(residual.dtype))
        x.copy_(self._norm(z, x.dtype))
        return x, residual


class Layer(torch.nn.Module):
    """A decoder layer's shape: input norm (in place, with the residual), a mix, a post norm, a second mix."""

    def __init__(self, seed: int, dtype) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(1000 + seed)
        self.input_layernorm, self.post_attention_layernorm = FusedNorm(2 * seed), FusedNorm(2 * seed + 1)
        self.a = (torch.randn(HIDDEN, HIDDEN, generator=g) / HIDDEN ** 0.5).to(dtype)
        self.b = (torch.randn(HIDDEN, HIDDEN, generator=g) / HIDDEN ** 0.5).to(dtype)
        self.use_attn_reduce_scatter_for_moe = False

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = torch.tanh(hidden_states @ self.a) + positions[0, :, None].to(hidden_states.dtype) * 1e-3
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return hidden_states @ self.b, residual


class StandIn(torch.nn.Module):
    """The attributes both forwards read: layers, the final norm, the embedding and the text config."""

    use_sequence_parallel = False

    def __init__(self, layers, norm, embed, readout_depths) -> None:
        super().__init__()
        self.layers, self.norm, self.embed = layers, norm, embed
        self.start_layer, self.end_layer = 0, len(layers)
        self.config = SimpleNamespace(readout_depths=list(readout_depths) or None)
        self.aux_hidden_state_layers: tuple = ()
        self.arc_readout_depths = tuple(readout_depths)

    def embed_input_ids(self, input_ids):
        return self.embed(input_ids)

    def _maybe_add_hidden_state(self, aux, layer_idx, hidden_states, residual):  # vLLM's EagleModelMixin, verbatim
        if layer_idx in self.aux_hidden_state_layers:
            aux.append(hidden_states + residual if residual is not None else hidden_states)
        return aux


@pytest.fixture
def eq_model(monkeypatch):
    """``vllm_arc.engine.models`` over a stand-in vLLM (``ArcEqModel`` subclasses its ``Qwen3_5Model``)."""
    qwen = types.ModuleType("qwen3_5")
    qwen.Qwen3_5Model = type("Qwen3_5Model", (torch.nn.Module,), {})
    qwen.Qwen3_5ForConditionalGeneration = type("Qwen3_5ForConditionalGeneration", (torch.nn.Module,), {})
    qwen.mrope_positions_factor = lambda _cfg: 1
    pooler = types.ModuleType("pooler")
    pooler.Pooler = type("Pooler", (torch.nn.Module,), {})
    models_pkg = types.ModuleType("models")
    models_pkg.qwen3_5 = qwen
    for name, module in {"vllm": types.ModuleType("vllm"), "vllm.model_executor": types.ModuleType("me"),
                         "vllm.model_executor.layers": types.ModuleType("layers"),
                         "vllm.model_executor.layers.pooler": pooler, "vllm.model_executor.models": models_pkg,
                         "vllm.model_executor.models.qwen3_5": qwen}.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in [n for n in sys.modules if n == "vllm_arc" or n.startswith("vllm_arc.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(ROOT))
    import vllm_arc.engine.compat as compat

    monkeypatch.setattr(compat, "require", lambda *a, **k: [])  # a stand-in, not the locked vLLM
    import vllm_arc.engine.models as models

    yield models
    for name in [n for n in sys.modules if n == "vllm_arc" or n.startswith("vllm_arc.")]:
        sys.modules.pop(name)


def _inputs(dtype, tokens: int = 37, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    layers = [Layer(i, dtype) for i in range(LAYERS)]
    norm = FusedNorm(999)
    embed = torch.nn.Embedding(500, HIDDEN).to(dtype)
    with torch.no_grad():
        embed.weight.copy_(torch.randn(500, HIDDEN, generator=g).to(dtype))
    ids = torch.randint(0, 500, (tokens,), generator=g)
    positions = torch.arange(tokens).expand(3, tokens).contiguous()  # M-RoPE's (3, tokens)
    return layers, norm, embed, ids, positions


def _run(forward, model, ids, positions):
    with torch.no_grad():
        return forward(model, ids, positions)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_arc_eq_forward_equals_the_recorded_rows_bit_for_bit(eq_model, dtype):
    """The recorded rows are the fork-era EQ forward's, so this holds ``ArcEqModel`` to it (see the module doc). In
    bfloat16 only: float32 is checked relative to ``plain_forward`` on the same machine, in the next test."""
    layers, norm, embed, ids, positions = _inputs(dtype)
    stock = _run(eq_model.ArcEqModel.forward, StandIn(layers, norm, embed, (CUT, LAYERS)), ids, positions)
    fork = torch.cat([recorded(dtype, "e12"), recorded(dtype, "e24")], dim=-1)
    assert fork.shape == stock.shape == (len(ids), 2 * HIDDEN) and fork.dtype == stock.dtype == dtype
    assert torch.equal(fork, stock)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_h12_is_the_12_layer_readout_and_h24_the_24_layer_one(eq_model, dtype):
    layers, norm, embed, ids, positions = _inputs(dtype)
    rows = _run(eq_model.ArcEqModel.forward, StandIn(layers, norm, embed, (CUT, LAYERS)), ids, positions)
    e12 = _run(plain_forward, StandIn(layers[:CUT], norm, embed, ()), ids, positions)
    e24 = _run(plain_forward, StandIn(layers, norm, embed, ()), ids, positions)
    assert torch.equal(rows[:, :HIDDEN], e12) and torch.equal(rows[:, HIDDEN:], e24)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_the_plain_readout_equals_the_recorded_rows(dtype):
    """``plain_forward`` on these stand-ins still gives the recorded readouts: the stand-ins, inputs and plain path
    the other tests use are the ones the rows were recorded from."""
    layers, norm, embed, ids, positions = _inputs(dtype)
    for key, depth in (("e12", CUT), ("e24", LAYERS)):
        assert torch.equal(_run(plain_forward, StandIn(layers[:depth], norm, embed, ()), ids, positions),
                           recorded(dtype, key))


def test_the_pair_path_is_what_holds(eq_model):
    """The two wrong readouts the pair path avoids both differ here: ``norm(hidden + residual)`` (vLLM's auxiliary
    hidden state: the sum rounds to bfloat16 before the norm) and the pair kept without copies (the next layer's norm
    overwrites it). In the served dtype, bfloat16."""
    layers, norm, embed, ids, positions = _inputs(torch.bfloat16)
    e12 = _run(plain_forward, StandIn(layers[:CUT], norm, embed, ()), ids, positions)
    model = StandIn(layers, norm, embed, ())
    model.aux_hidden_state_layers = (CUT,)
    _, [aux] = _run(plain_forward, model, ids, positions)
    assert not torch.equal(norm(aux.clone()), e12)

    import inspect

    source = textwrap.dedent(inspect.getsource(eq_model.ArcEqModel.forward))
    assert "hidden_states.clone(), residual.clone()" in source
    namespace = {"torch": torch, "islice": islice}
    exec(compile(source.replace("hidden_states.clone(), residual.clone()", "hidden_states, residual"), "<nocopy>",
                 "exec"), namespace)  # noqa: S102
    rows = _run(namespace["forward"], StandIn(layers, norm, embed, (CUT, LAYERS)), ids, positions)
    assert not torch.equal(rows[:, :HIDDEN], e12)


def test_the_stock_forward_is_the_plain_readout():
    """With vLLM installed: stock 0.33's own ``Qwen3NextModel.forward`` (compiled from its source, one rank) gives
    exactly ``ArcEqModel``'s last half, so the EQ forward changes nothing but the extra depths."""
    try:
        from vllm.model_executor.models import qwen3_next
    except Exception:  # noqa: BLE001 -- no vLLM (CI): stand-ins only
        pytest.skip("needs vLLM")
    import inspect

    source = inspect.getsource(qwen3_next)
    namespace = {**vars(qwen3_next), "get_pp_group": _pp_group}
    stock = _method(source, "Qwen3NextModel", "forward", namespace)
    sys.path.insert(0, str(ROOT))
    from vllm_arc.engine.models import ArcEqModel

    for dtype in (torch.float32, torch.bfloat16):
        layers, norm, embed, ids, positions = _inputs(dtype)
        plain = _run(stock, StandIn(layers, norm, embed, ()), ids, positions)
        rows = _run(ArcEqModel.forward, StandIn(layers, norm, embed, (CUT, LAYERS)), ids, positions)
        e12 = _run(stock, StandIn(layers[:CUT], norm, embed, ()), ids, positions)
        assert torch.equal(rows[:, HIDDEN:], plain) and torch.equal(rows[:, :HIDDEN], e12)


def _build(eq_model, monkeypatch, depths, layers=LAYERS, pp=1):
    """``ArcEqForPooling`` over stand-ins: vLLM's class (under ``ArcBaseForPooling``) builds the text model by the
    module name, as vLLM's ``Qwen3_5ForCausalLMBase`` does (``qwen3_5.Qwen3_5Model(vllm_config=..., prefix=...)``),
    and that model's ``__init__`` takes its forward as vLLM's compile wrapper does
    (``TorchCompileWithNoGuardsWrapper``)."""
    qwen = sys.modules["vllm.model_executor.models.qwen3_5"]

    def stand_in_init(self, *, vllm_config, prefix=""):
        torch.nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_text_config
        self.layers = torch.nn.ModuleList()
        self.compiled_forward = self.forward  # what the wrapper hands torch.compile

    def vllm_init(self, *, vllm_config, prefix):
        torch.nn.Module.__init__(self)
        self.language_model = SimpleNamespace(model=qwen.Qwen3_5Model(vllm_config=vllm_config, prefix=prefix))

    monkeypatch.setattr(eq_model.Qwen3_5Model, "__init__", stand_in_init)
    monkeypatch.setattr(eq_model.Qwen3_5ForConditionalGeneration, "__init__", vllm_init)
    cfg = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=pp), model_config=SimpleNamespace(
        hf_text_config=SimpleNamespace(readout_depths=depths, num_hidden_layers=layers), multimodal_config=None,
        dtype=torch.bfloat16))
    return eq_model.ArcEqForPooling(vllm_config=cfg).language_model.model


def test_the_text_model_is_built_as_arc_eq_model(eq_model, monkeypatch):
    """Built as ``ArcEqModel``, not swapped in after: the forward the compile wrapper took when the model was built is
    ``ArcEqModel.forward``, and the module name is restored once the model is built."""
    qwen = sys.modules["vllm.model_executor.models.qwen3_5"]
    inner = _build(eq_model, monkeypatch, [12, 24])
    assert type(inner) is eq_model.ArcEqModel and inner.arc_readout_depths == (12, 24)
    assert inner.compiled_forward.__func__ is eq_model.ArcEqModel.forward
    assert qwen.Qwen3_5Model is eq_model.Qwen3_5Model


def test_readout_depths_are_checked_at_construction(eq_model, monkeypatch):
    for bad in ([24], [12, 23], [24, 12], [0, 24], None):
        with pytest.raises(ValueError, match="readout_depths"):
            _build(eq_model, monkeypatch, bad)
    with pytest.raises(ValueError, match="pipeline"):
        _build(eq_model, monkeypatch, [12, 24], pp=2)


def test_a_text_model_built_as_another_class_is_refused(eq_model, monkeypatch):
    """A vLLM that builds its text model other than by the patched module name would serve without the pair path."""
    monkeypatch.setattr(eq_model.ArcBaseForPooling, "__init__", lambda self, **_: setattr(
        self, "language_model", SimpleNamespace(model=torch.nn.Module())))
    with pytest.raises(TypeError, match="not ArcEqModel"):
        eq_model.ArcEqForPooling(vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1)))


class CompiledStandIn(StandIn):
    """``StandIn`` whose forward is ``ArcEqModel.forward``, compiled as vLLM compiles it."""


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_pair_path_traces_whole_and_holds_under_torch_compile(eq_model, dtype):
    """``ArcEqModel.forward`` compiles as one graph (``fullgraph=True, dynamic=False``, vLLM's settings), with the
    stand-in norms writing their inputs in place as the eager CUDA kernel does: Dynamo's functionalisation keeps
    the copies' values, so rows equal the eager forward's bit for bit through AOTAutograd (``aot_eager``: the same
    ops, traced). Inductor (vLLM's backend) fuses and reorders arithmetic, so its rows are close, not equal: on the
    GPU, EQ compiled is not eager EQ's arithmetic, and the EQ parity gate, not equality, judges it."""
    import torch._dynamo

    layers, norm, embed, ids, positions = _inputs(dtype)
    CompiledStandIn.forward = eq_model.ArcEqModel.forward
    model = CompiledStandIn(layers, norm, embed, (CUT, LAYERS))
    eager = _run(eq_model.ArcEqModel.forward, StandIn(layers, norm, embed, (CUT, LAYERS)), ids, positions)
    torch._dynamo.reset()
    traced = torch.compile(model.forward, fullgraph=True, dynamic=False, backend="aot_eager")
    with torch.no_grad():
        rows = traced(ids, positions)
    assert rows.shape == eager.shape == (len(ids), 2 * HIDDEN) and torch.equal(rows, eager)
    torch._dynamo.reset()
    try:
        with torch.no_grad():
            fused = torch.compile(model.forward, fullgraph=True, dynamic=False, backend="inductor")(ids, positions)
    finally:
        torch._dynamo.reset()
    # Here (24 stand-in layers, CPU): float32 within 3e-6, bfloat16 cosine >= 0.9998 per row (Inductor keeps
    # intermediates in float32 between the ops eager rounds to bfloat16). Not a GPU measurement.
    cos = torch.nn.functional.cosine_similarity(fused.float(), eager.float(), dim=-1)
    assert float(cos.min()) > 0.999
    if dtype == torch.float32:
        assert torch.allclose(fused, eager, atol=1e-5, rtol=1e-5)


def test_vllm_compiles_the_class_a_model_was_built_as():
    """With vLLM installed: stock 0.33's ``support_torch_compile`` (``Qwen3_5Model``'s decorator) compiles the forward
    of the class the model was built as. A model built as the base and then given the subclass's ``__class__`` (the
    eager-only EQ build before this one) runs the base's forward compiled: on EQ, ``h_24`` alone, with no error. So
    ``ArcEqModel`` is built in ``Qwen3_5Model``'s place, and the probe's ``compiled_forward`` (asserted by EQ's
    start-up) names the forward that was compiled. ``Sub`` sets state after ``super().__init__``, as ``ArcEqModel``
    does."""
    try:
        from vllm.compilation.decorators import support_torch_compile
        from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
        from vllm.forward_context import set_forward_context
    except Exception:  # noqa: BLE001 -- no vLLM (CI)
        pytest.skip("needs vLLM")
    import torch._dynamo

    @support_torch_compile(dynamic_arg_dims={"x": 0})
    class Base(torch.nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(HIDDEN))

        def forward(self, x):
            return x * self.w

    class Sub(Base):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self.shift = 1.0  # read when Dynamo traces, after the wrapper took the forward

        def forward(self, x):
            return torch.cat([x * self.w, x * self.w + self.shift], dim=-1)

    # The CPU-safe subset of vLLM's compile: its own backend and dispatch, Dynamo's eager backend, no fusion passes.
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"), compilation_config=CompilationConfig(
        mode="VLLM_COMPILE", backend="eager", compile_sizes=[], pass_config={
            "fuse_norm_quant": False, "fuse_act_quant": False, "fuse_act_padding": False,
            "fuse_mla_dual_rms_norm": False}))
    x = torch.ones(3, HIDDEN)
    torch._dynamo.reset()
    try:
        with set_current_vllm_config(cfg), set_forward_context(None, cfg), torch.no_grad():
            built = Sub(vllm_config=cfg)
            swapped = Base(vllm_config=cfg)
            swapped.__class__ = Sub
            assert not built.do_not_compile and not swapped.do_not_compile
            assert built(x).shape == (3, 2 * HIDDEN)
            assert swapped(x).shape == (3, HIDDEN)  # Base.forward, compiled, under Sub's name
    finally:
        torch._dynamo.reset()
    sys.path.insert(0, str(ROOT))
    from vllm_arc.engine.probe import compiled_forward

    assert compiled_forward(built).endswith("Sub.forward") and compiled_forward(swapped).endswith("Base.forward")
