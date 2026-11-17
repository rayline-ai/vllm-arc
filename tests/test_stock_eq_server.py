# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""EQ on the stock server over HTTP (``ARC_EQ_ADDON``), start-up included, with the stand-in engine of
``test_stock_server_http`` answering 24-layer EQ rows (``norm(h_12) | norm(h_24)``, ``2 x HIDDEN`` wide): ``/encode``
answers ``norm(h_12)`` one-shots, ``/branches`` ``norm(h_24)`` at suffix positions with the prefix encoded first, and
an E12 server's routes and engine are unchanged."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
np = pytest.importorskip("numpy")
import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
import test_stock_server_http as http  # noqa: E402
from vllm_arc.eq import engine as eq
from vllm_arc.server import app as server  # noqa: E402
from vllm_arc.engine.startup import COMPILATION, COMPILED_NORMS  # noqa: E402

#: A compiled engine's probe fields (EQ's).
COMPILED_PROBE = {"enforce_eager": False, "do_not_compile": False, "compiled_forward": "ArcEqModel.forward",
                  "compilation": {**COMPILATION, "ir_op_priority": COMPILED_NORMS}}

H = eq.HIDDEN
fake_vllm = http.fake_vllm  # the fixture


def _eq_row(token: int, pos: int) -> list[float]:
    """A stand-in EQ row: h_12 carries the token, h_24 the position."""
    return [float(token % 97)] * H + [float(pos)] * H


class FakeEqEngine(http.FakeEngine):
    """Prefix hits against every sequence encoded under a salt (on the 16-token grid), as the engine's cache holds
    them; rows ``_eq_row``."""

    async def encode(self, prompt, params, request_id):
        from types import SimpleNamespace

        tokens, salt = list(prompt["prompt_token_ids"]), prompt["cache_salt"]
        offsets = params.extra_kwargs["arc_gather"]["offsets"]
        held = self.seen.setdefault(salt, [])
        common = max((next((i for i, (a, b) in enumerate(zip(p, tokens)) if a != b), min(len(p), len(tokens)))
                      for p in held), default=0)
        cached = 0 if params.skip_reading_prefix_cache else min(common, len(tokens) - 1) // http.GRID * http.GRID
        held.append(tokens)
        self.calls.append((len(tokens), salt, params.skip_reading_prefix_cache, cached))
        rows = torch.tensor([_eq_row(tokens[o], o) for o in offsets], dtype=torch.bfloat16).reshape(-1, 2 * H)
        yield SimpleNamespace(outputs=SimpleNamespace(data=rows), num_cached_tokens=cached, prompt_token_ids=tokens)


@pytest.fixture
def eq_vllm(fake_vllm, monkeypatch):
    original = sys.modules["vllm.v1.engine.async_llm"].AsyncLLM.from_engine_args

    def from_engine_args(args):
        original(args)
        engine = FakeEqEngine()
        engine.probe = fake_vllm.engine.probe
        fake_vllm.engine = engine
        return engine

    monkeypatch.setattr(sys.modules["vllm.v1.engine.async_llm"].AsyncLLM, "from_engine_args", from_engine_args)
    http.FakeEngine.probe = {**http.FakeEngine.probe, "inner_model": "ArcEqModel", "readout_depths": [12, 24],
                             "model_mro": ["ModelForEmbedding", "ArcEqForPooling", "ArcBaseForPooling"],
                             **COMPILED_PROBE}
    fake_vllm.checkpoints = []
    status = {"verified": True}
    fake_vllm.addon_status = status

    def checkpoint(addon, root, rope):
        fake_vllm.checkpoints.append((addon, root, rope))
        return "/ckpt/eq", {"name": "qh-test" if status["verified"] else None, "verified": status["verified"],
                            "reason": None if status["verified"] else "trunk_differs"}

    monkeypatch.setattr(eq, "checkpoint", checkpoint)
    monkeypatch.setattr(eq, "addon_dir", lambda addon: f"/addons/{addon}")
    return fake_vllm


def _eq_app(engine=None):
    # max_model_len below the self-test's 20k case: its all-rows pass is 20k stand-in rows 2 x HIDDEN wide.
    return server.build(engine or eq.engine(), budget=8192, max_model_len=16384, max_sessions=4, idle_seconds=900,
                        scratch_slots=2, checkpoint_root="/x", rope="yarn-f4", eq_addon="qh-addon/test")


def _rows(response, width):
    return np.frombuffer(response.content, dtype=np.float32).reshape(-1, width)


def test_eq_starts_on_the_prefix_route_and_encode_answers_h12_one_shots(eq_vllm):
    with TestClient(_eq_app()) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["stock"]["selftest"]["ok"]
        assert all(c["ok"] for c in health["stock"]["detail"]["checks"])
        assert "eq_readout_depths" in {c["name"] for c in health["stock"]["detail"]["checks"]}
        assert health["engine"].endswith("+gather-flashinfer+depth12+rope-yarn-f4+stock-u0+eq-test")
        assert health["eq"]["addon"]["verified"] is True and health["eq"]["readout_depths"] == [12, 24]
        assert health["eq"]["engine_depth"] == 24 and health["eq"]["prefix_match_unit"] == http.GRID
        assert eq_vllm.checkpoints == [("/addons/qh-addon/test", "/x", "yarn-f4")]
        args = eq_vllm.args
        assert args.model == "/ckpt/eq" and args.enable_prefix_caching is True and args.mamba_cache_mode == "align"
        tokens = list(range(1000, 1040))
        r = client.post("/encode", json={"base": 0, "retain": False, "tokens": tokens, "offsets": [39, 2],
                                         "cache_salt": "answer-1"})
        assert r.status_code == 200 and r.headers["x-dim"] == str(H)
        assert _rows(r, H).tolist() == [_eq_row(1039, 39)[:H], _eq_row(1002, 2)[:H]]
        assert eq_vllm.engine.calls[-1][1] == "answer-1"  # under the caller's salt, cache read
        for body, code in (({"base": 0, "tokens": tokens, "offsets": [1]}, "retain_not_served"),
                           ({"base": 0, "retain": False, "tokens": tokens, "offsets": [40]}, "offsets_out_of_range"),
                           ({"base": 0, "retain": False, "tokens": tokens, "offsets": [1], "cache_salt": ""},
                            "invalid_cache_salt"),
                           ({"base": 0, "retain": False, "tokens": tokens, "offsets": [1], "images": [{}]},
                            "images_not_served"),
                           ({"base": 0, "retain": False, "tokens": [1] * 32769, "offsets": [1]},
                            "exceeds_kv_budget")):
            r = client.post("/encode", json=body)
            assert r.status_code == 422 and r.json()["error"] == code
        assert client.get("/health").json()["eq"]["encodes"] == 1


def test_branches_read_h24_at_suffix_positions_after_the_prefix(eq_vllm):
    with TestClient(_eq_app()) as client:
        prefix = list(range(2000, 2050))
        body = {"tokens": prefix, "cache_salt": "answer-2",
                "suffixes": [{"tokens": [5, 6, 7], "offsets": [2]}, {"tokens": [8, 9], "offsets": [0, 1]}]}
        r = client.post("/branches", json=body)
        assert r.status_code == 200 and r.headers["x-counts"] == "1,2" and r.headers["x-dim"] == str(H)
        assert _rows(r, H).tolist() == [_eq_row(7, 52)[H:], _eq_row(8, 50)[H:], _eq_row(9, 51)[H:]]
        cached = [int(c) for c in r.headers["x-cached"].split(",")]
        assert cached == [48, 48] and r.headers["x-misses"] == "0"  # each branch hit the prefix's 16-token grid
        calls = eq_vllm.engine.calls[-3:]
        assert calls[0][0] == len(prefix) and all(c[1] == "answer-2" for c in calls)  # the prefix first, one salt
        usage = client.get("/health").json()["eq"]
        assert usage["branches"] == 2 and usage["hit_over_readout"] == 0
        r = client.post("/branches", json={**body, "suffixes": [{"tokens": [1], "offsets": [1]}]})
        assert r.status_code == 422 and r.json()["error"] == "offsets_out_of_range"
        assert client.post("/branches", json={"tokens": prefix, "suffixes": []}).status_code == 422


def test_a_hit_past_a_readout_is_encoded_again_under_a_fresh_salt(eq_vllm):
    """Stock has no hit cap: a repeated question under one salt hits past its own readout, and that branch alone is
    encoded again under a fresh salt (the fork server's defence, the mechanism here)."""
    with TestClient(_eq_app()) as client:
        prefix = list(range(3000, 3040))
        suffix = {"tokens": list(range(40)), "offsets": [5]}  # read at 45; the repeat's hit is 64
        r = client.post("/branches", json={"tokens": prefix, "cache_salt": "s", "suffixes": [suffix, suffix]})
        assert r.status_code == 200
        assert _rows(r, H).tolist() == [_eq_row(5, 45)[H:]] * 2
        assert client.get("/health").json()["eq"]["hit_over_readout"] == 1
        redo = eq_vllm.engine.calls[-1]
        assert redo[1] != "s" and redo[3] == 0  # a fresh salt: nothing cached


def test_an_unverified_addon_serves_encode_and_refuses_questions(eq_vllm):
    eq_vllm.addon_status["verified"] = False
    with TestClient(_eq_app()) as client:
        assert client.get("/health").json()["eq"]["addon"] == {"name": None, "verified": False,
                                                                "reason": "trunk_differs"}
        r = client.post("/encode", json={"base": 0, "retain": False, "tokens": [1, 2, 3], "offsets": [2]})
        assert r.status_code == 200
        r = client.post("/branches", json={"tokens": [1, 2], "suffixes": [{"tokens": [3], "offsets": [0]}]})
        assert r.status_code == 503 and r.json()["error"] == "question_head_unavailable"


def test_an_engine_without_the_pair_path_does_not_serve_eq(eq_vllm, monkeypatch):
    http.FakeEngine.probe = {**http.FakeEngine.probe, "inner_model": "Qwen3_5Model"}
    http.FakeEngine.probe.pop("readout_depths", None)
    exits = []
    monkeypatch.setattr(server.os, "_exit", lambda code: exits.append(code) or (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(BaseException):  # noqa: B017 -- SystemExit, wrapped by the test client's task group
        with TestClient(_eq_app()):
            pass
    assert exits == [3]


def test_eq_refuses_the_retained_route_and_images():
    with pytest.raises(ValueError, match="prefix route"):
        server.build(server.Engine(), budget=8192, max_model_len=32768, max_sessions=4, idle_seconds=900,
                     scratch_slots=2, checkpoint_root="/x", rope="yarn-f4", eq_addon="qh-addon/test")
    with pytest.raises(ValueError, match="text only"):
        server.build(eq.engine(), budget=8192, max_model_len=32768, max_sessions=4, idle_seconds=900,
                     scratch_slots=2, checkpoint_root="/x", rope="yarn-f4", images="arc-image-v1",
                     eq_addon="qh-addon/test")


@pytest.mark.parametrize("change", [
    {"enforce_eager": True, "do_not_compile": True, "compiled_forward": None},  # resolved to eager
    {"compiled_forward": "Qwen3_5Model.forward"},  # compiled, but not the pair path
])
def test_an_eq_engine_not_compiling_the_pair_path_does_not_start(eq_vllm, monkeypatch, change):
    """EQ's start-up assertions ``compiled`` and ``eq_compiled_forward``: either failing exits 3."""
    http.FakeEngine.probe = {**http.FakeEngine.probe, **change}
    exits = []
    monkeypatch.setattr(server.os, "_exit", lambda code: exits.append(code) or (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(BaseException):  # noqa: B017 -- SystemExit, wrapped by the test client's task group
        with TestClient(_eq_app()):
            pass
    assert exits == [3]


def test_only_eq_runs_compiled():
    common = dict(budget=8192, max_model_len=32768, max_sessions=4, idle_seconds=900, scratch_slots=2,
                  checkpoint_root="/x", rope="yarn-f4")
    with pytest.raises(ValueError, match="EQ runs compiled"):
        server.build(server.Engine(sessions="prefix"), **common, eq_addon="qh-addon/test")
    with pytest.raises(ValueError, match="only EQ runs compiled"):
        server.build(eq.engine(), **common)
    with pytest.raises(ValueError, match="only EQ runs compiled"):
        server.build(eq.engine(), **common, images="arc-image-v1")


def test_an_e12_server_has_no_branches_and_no_eq(fake_vllm):
    with TestClient(http._app()) as client:
        health = client.get("/health").json()
        assert "eq" not in health and "+eq-" not in health["engine"]
        assert "readout_depths" not in health["stock"]["detail"]["probe"]
        assert client.post("/branches", json={}).status_code in (404, 405)


def test_main_defaults_eq_to_the_prefix_route(monkeypatch):
    built = {}
    monkeypatch.setattr(server.compat, "require", lambda *a, **k: [])
    monkeypatch.setattr(server, "build", lambda settings, **kw: built.update(settings=settings, **kw))
    monkeypatch.setitem(sys.modules, "uvicorn", type(sys)("uvicorn"))
    sys.modules["uvicorn"].run = lambda *a, **k: None
    monkeypatch.delenv("ARC_STOCK_SESSIONS", raising=False)
    monkeypatch.setattr(sys, "argv", ["server", "--eq-addon", "qh-addon/x"])
    server.main()
    assert built["settings"] == eq.engine() and built["settings"].compiled and built["eq_addon"] == "qh-addon/x"
    monkeypatch.setattr(sys, "argv", ["server"])
    server.main()
    assert built["settings"] == server.Engine() and not built["settings"].compiled and built["eq_addon"] == ""
