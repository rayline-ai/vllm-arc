# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""The stock server over HTTP, start-up included, with a stand-in engine (CPU). Needs fastapi, httpx and numpy (CI's CPU
environment has them).

The stand-in engine answers ``encode`` as GatherPooler would over a deterministic hidden state, lands prefix hits
per salt on a 16-token grid, serves a stream of inputs as one retained request (patch 0001, counting every input
after a stream's first as an in-place extension unless it carries an image, which takes the full re-add), and
answers the start-up probe like the expected engine; a probe that says otherwise must stop the server before it
serves. Given images, it expands each collapsed run as vLLM's processor does (one pad per 32x32 pixels) and counts
ViT calls, one per image unless told to batch them.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
np = pytest.importorskip("numpy")
import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1]))
from vllm_arc.server import app as server  # noqa: E402
from vllm_arc.engine.startup import FORK_YARN_INV_FREQ  # noqa: E402

GRID = 16
START, PAD, END = 9001, 9002, 9003
FORK_SHA, FORK_HEX = FORK_YARN_INV_FREQ[(64, 10000000.0, 4.0, 262144)]


def _hidden(token: int, pos: int) -> list[float]:
    return [float(token % 97), float(pos), float((token * 31 + pos) % 13), 1.0]


class FakeEngine:
    probe: dict = {}
    batched_vit = False  # one ViT call for all of an input's images (what the model class must not do)
    extra_pads = 0  # pads the processor adds per image beyond the rule's count (an expansion the caller did not make)

    def __init__(self):
        self.seen, self.calls, self.errored, self.resets, self.visual_calls = {}, [], False, 0, 0
        self.vit_inputs = []  # as the worker records them under ARC_STOCK_VIT_INPUTS=1 (a stand-in digest)
        self.streamed, self.finished_streams = [], []
        self.extends = 0  # patch 0001's runner counter
        self.extend = True  # False: the patch is installed but never takes the in-place extension
        cache = SimpleNamespace(num_gpu_blocks=4000, block_size=GRID, kv_cache_size_tokens=64000)
        self.vllm_config = SimpleNamespace(cache_config=cache, model_config=SimpleNamespace(
            max_model_len=32768, get_vocab_size=lambda: 151000,
            hf_config=SimpleNamespace(vision_start_token_id=START, image_token_id=PAD, vision_end_token_id=END)))

    def _expand(self, ids, pixels):
        out, sizes = [], iter(pixels)
        for t in ids:
            if t == PAD:
                image = next(sizes)
                w, h = image.size
                self.vit_inputs.append({"sha256": f"fake-{hash(image.tobytes()) & 0xffff}",
                                        "grid_thw": [1, h // 16, w // 16]})
                out += [PAD] * ((w // 32) * (h // 32) + self.extra_pads)
            else:
                out.append(t)
        self.visual_calls += 1 if self.batched_vit else len(pixels)
        return out

    async def encode(self, prompt, params, request_id):
        if not isinstance(prompt, dict):  # patch 0001: a stream of inputs into one retained request
            async for out in self._stream(prompt, params, request_id):
                yield out
            return
        tokens, salt = prompt["prompt_token_ids"], prompt["cache_salt"]
        if "multi_modal_data" in prompt:
            tokens = self._expand(tokens, prompt["multi_modal_data"]["image"])
        offsets = params.extra_kwargs["arc_gather"]["offsets"]
        prior = self.seen.get(salt)
        cached = 0
        if not params.skip_reading_prefix_cache and prior is not None and tokens[:len(prior)] == prior:
            cached = (len(prior) - 1) // GRID * GRID
        self.seen[salt] = list(tokens)
        self.calls.append((len(tokens), salt, params.skip_reading_prefix_cache, cached))
        rows = torch.tensor([_hidden(tokens[o], o) if o >= cached else [float("nan")] * 4 for o in offsets],
                            dtype=torch.bfloat16).reshape(len(offsets), 4)
        yield SimpleNamespace(outputs=SimpleNamespace(data=rows), num_cached_tokens=cached,
                              prompt_token_ids=list(tokens))

    async def _stream(self, inputs, params, request_id):
        # As vLLM 0.33's output processor (``apply_streaming_update``): the first input's ``prompt_token_ids`` list is
        # kept and every later input's ids are ``extend``-ed onto it in place, and that list is the output's.
        tokens: list[int] | None = None
        async for item in inputs:
            new = item.prompt["prompt_token_ids"]
            mm = item.prompt.get("multi_modal_data")
            if mm:
                new = self._expand(new, mm["image"])
            self.streamed.append((request_id, len(new)))
            self.extends += bool(tokens) and self.extend and not mm
            if tokens is None:
                tokens = new
            else:
                tokens.extend(new)
            offsets = (item.sampling_params or params).extra_kwargs["arc_gather"]["offsets"]
            rows = torch.tensor([_hidden(tokens[o], o) for o in offsets], dtype=torch.bfloat16).reshape(len(offsets), 4)
            yield SimpleNamespace(finished=False, prompt_token_ids=tokens, outputs=SimpleNamespace(data=rows),
                                  num_cached_tokens=0)
        self.finished_streams.append(request_id)

    async def collective_rpc(self, method):
        if method == "arc_stock_vit_inputs":
            drained, self.vit_inputs = self.vit_inputs, []
            return [drained]
        return {"arc_stock_probe": [self.probe], "arc_patch_extends": [self.extends],
                "arc_stock_visual_calls": [self.visual_calls]}.get(method, [None])

    async def reset_prefix_cache(self):
        self.resets += 1
        self.seen.clear()
        return True

    async def sleep(self, level):
        pass

    async def wake_up(self):
        return True

    def shutdown(self):
        pass


def _probe():
    return {
        "versions": {"vllm": "0.33.0", "vllm_wheel_url": server.VLLM_WHEEL, "torch": "2.14.0",
                     "flashinfer": "0.7.2.post1", "triton": "3.7.0"},
        "vllm_version": "0.33.0", "torch_version": "2.14.0", "torch_cuda": "13.0", "device": "fake",
        "capability": [12, 0], "runner": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
        "model_mro": ["ModelForEmbedding", "ArcBaseForPooling"], "pooler": "GatherPooler",
        "gdn": [{"backend": "flashinfer", "forward": "forward_cuda"}], "fused_qk_stock": [True],
        "fused_qk_running": [False], "inner_model": "Qwen3_5Model", "do_not_compile": True,
        "rope_parameters": dict(server.ROPE_VARIANTS["yarn-f4"]),
        "rope_effective": [{"class": "MRotaryEmbedding", "scaling_factor": 4.0, "cache_max_position": 1048576,
                            "cos_sin_cache_rows": 4194304, "base": 10000000.0, "rotary_dim": 64, "head_size": 256,
                            "mrope_section": [11, 11, 10], "mrope_interleaved": True, "mscale": 1.1386,
                            "inv_freq_hex": list(FORK_HEX), "inv_freq_sha256": FORK_SHA}],
        "multimodal_pruning": False,
        "vision": {"param_dtypes": ["float32"], "attn_backend": "TORCH_SDPA", "input_norm": "IdentityInputNorm",
                   "mm_device_do_normalize": False, "mm_encoder_tp_mode": "weights", "image_limit": 999,
                   "video_limit": 999, "visual_calls": 0,
                   "pixel_table": {"device": "cpu", "dtype": "float32", "sha256": server.img.PIXEL_TABLE_SHA256},
                   "cudagraph_mm_encoder": False},
        "cache": {"enable_prefix_caching": True, "block_size": GRID, "mamba_block_size": GRID,
                  "mamba_cache_mode": "align", "prefix_match_unit": None, "prefix_caching_hash_algo": "sha256",
                  "prefix_cache_retention_interval": 0, "mamba_ssm_cache_dtype": "float32", "hash_block_size": GRID,
                  "cache_hit_alignment_tokens": GRID, "num_blocks": 4000, "groups": []},
        "scheduler": {"enable_chunked_prefill": True}, "enforce_eager": True, "env": {}, "autotune": None,
        "arc_patch": {"id": "0001-retained-pooling-session", "runner_extend": True, "extends": 0},
    }


@pytest.fixture
def fake_vllm(monkeypatch):
    holder = SimpleNamespace(engine=None)

    def from_engine_args(args):
        holder.engine = FakeEngine()
        holder.args = args
        if args.enable_prefix_caching is False:  # as vLLM resolves it: mamba mode "none" without prefix caching
            probe = FakeEngine.probe
            holder.engine.probe = {**probe, "cache": {**probe["cache"], "enable_prefix_caching": False,
                                                      "mamba_cache_mode": "none", "hash_block_size": None,
                                                      "cache_hit_alignment_tokens": None}}
        return holder.engine

    kw = lambda name: type(name, (), {"__init__": lambda self, **k: self.__dict__.update(k)})  # noqa: E731
    mods = {"vllm": types.ModuleType("vllm"), "vllm.config": types.ModuleType("c"),
            "vllm.engine": types.ModuleType("e"), "vllm.engine.arg_utils": types.ModuleType("a"),
            "vllm.v1": types.ModuleType("v1"), "vllm.v1.engine": types.ModuleType("v1e"),
            "vllm.v1.engine.async_llm": types.ModuleType("al"), "vllm.pooling_params": types.ModuleType("pp"),
            "vllm.engine.protocol": types.ModuleType("ep")}
    mods["vllm.engine.protocol"].StreamingInput = kw("StreamingInput")
    mods["vllm.config"].PoolerConfig = kw("PoolerConfig")
    mods["vllm.engine.arg_utils"].AsyncEngineArgs = kw("AsyncEngineArgs")
    mods["vllm.v1.engine.async_llm"].AsyncLLM = SimpleNamespace(from_engine_args=from_engine_args)
    mods["vllm.pooling_params"].PoolingParams = kw("PoolingParams")
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(server, "truncated_checkpoint", lambda root, rope: "/ckpt/base")
    monkeypatch.setattr(server.img, "processor_identity", lambda model, revision: {
        "class": "Qwen2VLImageProcessorFast", "transformers": "fake", "pillow": "fake"})
    FakeEngine.probe = _probe()
    monkeypatch.setattr(FakeEngine, "batched_vit", False)
    monkeypatch.setattr(FakeEngine, "extra_pads", 0)
    return holder


def _app(engine=server.Engine(), images=""):
    return server.build(engine, budget=8192, max_model_len=32768, max_sessions=4, idle_seconds=900, scratch_slots=2,
                        checkpoint_root="/x", rope="yarn-f4", images=images)


def _image_body(base=0, h=256, w=384, **change):
    """One image between text, at absolute ``base``: its expanded tokens and its declaration."""
    import base64
    import hashlib

    raw = bytes((x + 2 * y + c) % 256 for y in range(h) for x in range(w) for c in range(3))
    n = (h // 32) * (w // 32)
    image = {"offset": base + 3, "n": n, "height": h, "width": w, "sha256": "x",
             "rgb_sha256": hashlib.sha256(raw).hexdigest(), "bytes": base64.b64encode(raw).decode(), **change}
    return [11, 12, 13, START] + [PAD] * n + [END, 14], image


def _rows(response) -> np.ndarray:
    return np.frombuffer(response.content, dtype=np.float32).reshape(int(response.headers["x-rows"]), -1)


def test_the_default_retained_route_keeps_one_live_request_per_session_and_appends_send_only_new_tokens(fake_vllm):
    with TestClient(_app()) as client:
        health = client.get("/health").json()
        stock = health["stock"]
        assert health["status"] == "ok" and stock["selftest"]["ok"] and stock["sessions"] == "retained"
        assert all(c["ok"] for c in stock["detail"]["checks"]) and stock["arc_patch"]["runner_extend"] is True
        assert health["engine"].endswith("+gather-flashinfer+depth12+rope-yarn-f4+stock-retained")
        assert stock["selftest"]["retained"]["rows"] == [10, 4] and stock["selftest"]["retained"]["extends"] == 1
        assert stock["arc_patch"]["extends"] == 1  # live: the self-test's extension (the probe's snapshot says 0)
        assert "append" not in stock["selftest"] and stock["cache"]["enable_prefix_caching"] is False
        args = fake_vllm.args
        assert args.enable_prefix_caching is False and args.max_num_batched_tokens == 8192
        assert not {"mamba_cache_mode", "prefix_match_unit", "prefix_caching_hash_algo"} & set(vars(args))
        engine = fake_vllm.engine
        assert len(engine.finished_streams) == 1  # the self-test's session was closed
        tokens = list(range(1000, 1100))
        r = client.post("/encode", json={"session": "s", "base": 0, "tokens": tokens, "offsets": [99, 5]})
        assert r.status_code == 200 and _rows(r).tolist() == [_hidden(1099, 99), _hidden(1005, 5)]
        r = client.post("/encode", json={"session": "s", "base": 100, "tokens": [7, 8], "offsets": [101, 100]})
        assert r.status_code == 200 and _rows(r).tolist() == [_hidden(8, 101), _hidden(7, 100)]
        assert r.headers["x-cached-tokens"] == "100" and r.headers["x-encoded"] == "2"
        live = engine.streamed[-1][0]
        assert engine.streamed[-2:] == [(live, 100), (live, 2)]  # one request; the append sent 2 tokens
        assert client.get("/health").json()["stock"]["arc_patch"]["extends"] == 2  # the append's, counted live
        assert client.post("/encode", json={"session": "s", "base": 5, "tokens": [1], "offsets": [5]}).status_code == 409
        once = client.post("/encode", json={"base": 0, "retain": False, "tokens": [3, 4], "offsets": [1]})
        assert once.status_code == 200 and _rows(once).tolist() == [_hidden(4, 1)]
        assert client.get("/health").json()["kv"]["resident_tokens"] == 102
        assert client.post("/sleep").json() == {"sleeping": True}
        assert engine.finished_streams[-1] == live  # sleep ends every live request


def test_a_pass_carries_the_engines_processed_ids_and_where_its_input_starts_in_them(fake_vllm):
    import asyncio

    async def go():
        engine = FakeEngine()
        one_shot = await server.engine_pass(engine)([5, 6, 7], [2], "salt", True)
        assert (one_shot.processed, one_shot.start) == ([5, 6, 7], 0)
        session = server.open_session(engine)()
        await server.session_append(session, [1, 2, 3], [2])
        second = await server.session_append(session, [4, 5], [4])
        assert second.processed[second.start:second.start + second.encoded] == [4, 5] and second.start == 3
        await session.close()

    asyncio.run(go())


def test_the_retained_self_test_fails_when_no_append_took_the_in_place_extension(fake_vllm, monkeypatch):
    # The patch is installed, but every append falls back to stock's full re-add (9bba7a1's encoder-cache guard).
    monkeypatch.setattr(FakeEngine, "__init__", lambda self, init=FakeEngine.__init__: (
        init(self), setattr(self, "extend", False))[0])
    exits = []
    monkeypatch.setattr(server.os, "_exit", lambda code: exits.append(code) or (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(BaseException):  # noqa: B017 -- SystemExit, wrapped by the test client's task group
        with TestClient(_app()):
            pass
    assert exits == [3]


def test_the_retained_route_refuses_to_start_without_patch_0001(fake_vllm, monkeypatch):
    FakeEngine.probe = {k: v for k, v in _probe().items() if k != "arc_patch"}
    exits = []
    monkeypatch.setattr(server.os, "_exit", lambda code: exits.append(code) or (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(BaseException):  # noqa: B017 -- SystemExit, wrapped by the test client's task group
        with TestClient(_app()):
            pass
    assert exits == [3]


def test_start_up_self_test_then_a_session_and_its_append_over_http_on_the_prefix_route(fake_vllm):
    with TestClient(_app(server.Engine(sessions="prefix"))) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["stock"]["selftest"]["ok"]
        assert health["engine"].endswith("+gather-flashinfer+depth12+rope-yarn-f4+stock-u0")
        stock = health["stock"]
        assert all(c["ok"] for c in stock["detail"]["checks"]) and fake_vllm.engine.resets == 1
        assert {k: stock[k] for k in ("gdn_prefill_backend", "runner", "fused_qk", "autotune")} == {
            "gdn_prefill_backend": "flashinfer", "runner": "v2", "autotune": "timed",
            "fused_qk": {"enabled_layers": 0, "full_attention_layers": 1, "stock_choice": True}}
        assert stock["cache"]["prefix_match_unit"] is None and stock["cache"]["mamba_ssm_cache_dtype"] == "float32"
        assert stock["versions"]["vllm_wheel_sha256"] == server.VLLM_WHEEL_SHA256
        assert {"depth", "budget", "rope", "max_model_len", "model", "revision", "engine"} <= set(health)
        assert fake_vllm.args.prefix_match_unit is None and fake_vllm.args.pooler_config.tok_pooling_type == "ALL"
        tokens = list(range(1000, 1100))
        r = client.post("/encode", json={"session": "s", "base": 0, "tokens": tokens, "offsets": [99, 5]})
        assert r.status_code == 200 and r.headers["x-dim"] == "4" and r.headers["x-cached"] == "0"
        assert _rows(r).tolist() == [_hidden(1099, 99), _hidden(1005, 5)]
        r = client.post("/encode", json={"session": "s", "base": 100, "tokens": [7, 8], "offsets": [101, 100]})
        assert r.status_code == 200 and r.headers["x-cached-tokens"] == "96" and r.headers["x-encoded"] == "102"
        assert _rows(r).tolist() == [_hidden(8, 101), _hidden(7, 100)]
        assert client.post("/encode", json={"session": "s", "base": 5, "tokens": [1], "offsets": [5]}).status_code == 409
        once = client.post("/encode", json={"base": 0, "retain": False, "tokens": [3, 4], "offsets": [1]})
        assert once.status_code == 200 and _rows(once).tolist() == [_hidden(4, 1)]
        stock = client.get("/health").json()["stock"]
        assert stock["counters"] == {"over_hit": 0, "rebuild": 0, "miss": 0}
        assert stock["detail"]["counters"]["appends"] == 1 and stock["detail"]["counters"]["append_cached_tokens"] == 96
        assert client.post("/close", json={"owner": "x"}).json() == {"closed": 0}
        assert client.post("/reset_prefix_cache").status_code == 404  # a measurement endpoint, not served
        assert "images" not in client.get("/health").json()
        tokens, image = _image_body()
        r = client.post("/encode", json={"retain": False, "base": 0, "tokens": tokens, "offsets": [0],
                                         "images": [image]})
        assert (r.status_code, r.json()) == (422, {"error": "images_not_served"})


def _image_probe():
    return {**_probe(), "vision": {**_probe()["vision"], "image_limit": 32, "video_limit": 0}}


def test_an_image_engine_serves_images_in_one_shots_and_retained_sessions_at_expanded_positions(fake_vllm):
    """Start-up runs the image self-test (two images in one input: one ViT call each; a retained session of text, an
    image input and a text append that takes the in-place extension); /health and the engine id name the image
    settings. Rows are read at expanded positions in a one-shot and across a retained session's image append and
    the text after it; malformed pixels are refused, and a processed expansion other than the caller's is refused
    and drops the session."""
    FakeEngine.probe = _image_probe()
    with TestClient(_app(images="arc-image-v1")) as client:
        health = client.get("/health").json()
        selftest = health["stock"]["selftest"]["images"]
        assert health["stock"]["selftest"]["ok"] and selftest["ok"]
        assert selftest["one_shot"] == {"tokens": 196, "images": 2, "finite": True, "visual_calls": 2, "rows": [3, 4]}
        retained = dict(selftest["retained"])
        assert retained.pop("cos_vs_oneshot") == pytest.approx([1.0, 1.0])  # the fake engine is chunking-invariant
        assert retained == {"image_visual_calls": 1, "image_extends": 0, "tail_visual_calls": 0, "tail_extends": 1,
                            "finite": True, "max_abs_vs_oneshot": 0.0}
        assert "+stock-retained+images-arc-image-v1-vitfloat32-" in health["engine"]
        assert health["images"]["rule"]["rule"] == "arc-image-v1" and health["images"]["vision"] == {
            "dtype": "float32", "embed_cast": "bfloat16", "per_image_calls": True}
        assert health["images"]["processor"]["class"] == "Qwen2VLImageProcessorFast"
        assert fake_vllm.args.limit_mm_per_prompt == {"image": 32, "video": 0}
        tokens, image = _image_body()
        last, pad = len(tokens) - 1, 3 + 96
        r = client.post("/encode", json={"retain": False, "base": 0, "tokens": tokens, "offsets": [last, pad],
                                         "images": [image]})
        assert r.status_code == 200 and _rows(r).tolist() == [_hidden(14, last), _hidden(PAD, pad)]
        calls = fake_vllm.engine.visual_calls
        head = list(range(1000, 1050))
        assert client.post("/encode", json={"session": "s", "base": 0, "tokens": head, "offsets": [49]}).status_code == 200
        chunk, image = _image_body(base=50)
        r = client.post("/encode", json={"session": "s", "base": 50, "tokens": chunk, "offsets": [50 + pad],
                                         "images": [image]})
        assert r.status_code == 200 and _rows(r).tolist() == [_hidden(PAD, 50 + pad)]
        end = 50 + len(chunk)
        r = client.post("/encode", json={"session": "s", "base": end, "tokens": [7, 8], "offsets": [end + 1]})
        assert r.status_code == 200 and _rows(r).tolist() == [_hidden(8, end + 1)]
        assert fake_vllm.engine.visual_calls == calls + 1  # the image ran once; the text after it ran no ViT
        bad = client.post("/encode", json={"retain": False, "base": 0, "tokens": tokens, "offsets": [last],
                                           "images": [{**image, "offset": 3, "rgb_sha256": "0" * 64}]})
        assert bad.status_code == 422 and bad.json()["error"] == "invalid_images"
        FakeEngine.extra_pads = 1
        chunk, image = _image_body(base=end + 2)
        moved = client.post("/encode", json={"session": "s", "base": end + 2, "tokens": chunk, "offsets": [end + 2],
                                             "images": [image]})
        assert moved.status_code == 422 and moved.json()["error"] == "image_input_refused"
        after = client.post("/encode", json={"session": "s", "base": end + 2, "tokens": [9], "offsets": [end + 2]})
        assert (after.status_code, after.json()) == (409, {"error": "session_unknown"})
        once = client.post("/encode", json={"retain": False, "base": 0, "tokens": tokens, "offsets": [last],
                                            "images": [{**image, "offset": 3}]})
        assert once.status_code == 422 and once.json()["error"] == "image_input_refused"


def test_the_prefix_route_serves_image_one_shots_and_refuses_image_sessions(fake_vllm):
    FakeEngine.probe = _image_probe()
    with TestClient(_app(server.Engine(sessions="prefix"), images="arc-image-v1")) as client:
        assert client.get("/health").json()["stock"]["selftest"]["images"]["ok"]
        tokens, image = _image_body()
        once = client.post("/encode", json={"retain": False, "base": 0, "tokens": tokens, "offsets": [99],
                                            "images": [image]})
        assert once.status_code == 200 and _rows(once).tolist() == [_hidden(PAD, 99)]
        session = client.post("/encode", json={"session": "s", "base": 0, "tokens": tokens, "offsets": [99],
                                               "images": [image]})
        assert (session.status_code, session.json()) == (422, {"error": "images_need_retained"})


@pytest.mark.parametrize("images, probe, batched", [
    ("", {"gdn": [{"backend": "triton", "forward": "forward_native"}]}, False),
    ("", {"vision": {**_probe()["vision"], "param_dtypes": ["bfloat16"]}}, False),
    ("arc-image-v1", {"vision": {**_probe()["vision"], "image_limit": 32, "video_limit": 0}}, True),
    ("arc-image-v1", {"vision": {**_probe()["vision"], "image_limit": 32, "video_limit": 0}}, "extra_pad"),
])
def test_a_failed_start_up_assertion_or_self_test_stops_the_server_before_it_serves(fake_vllm, monkeypatch, images,
                                                                                     probe, batched):
    """A probe that differs from the expected engine (GDN backend, a bfloat16 ViT), or an image self-test whose two
    images ran as one ViT call or whose expansion differs from the declared one, exits 3 before anything is served."""
    FakeEngine.probe, FakeEngine.batched_vit = {**_probe(), **probe}, batched is True
    FakeEngine.extra_pads = int(batched == "extra_pad")
    exits = []
    monkeypatch.setattr(server.os, "_exit", lambda code: exits.append(code) or (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(BaseException):  # noqa: B017 -- SystemExit, wrapped by the test client's task group
        with TestClient(_app(images=images)):
            pass
    assert exits == [3]


def test_vit_input_recording_returns_each_image_requests_vit_inputs_as_a_header(fake_vllm, monkeypatch):
    """With ``ARC_STOCK_VIT_INPUTS=1`` an image request's response carries the ViT inputs recorded during it (one per
    image, in order; not the self-test's), a text request's none; with it off no response carries the header."""
    import json

    FakeEngine.probe = _image_probe()
    tokens, image = _image_body()
    body = {"retain": False, "base": 0, "tokens": tokens, "offsets": [99], "images": [image]}
    with TestClient(_app(images="arc-image-v1")) as client:
        assert server.VIT_INPUTS_HEADER not in client.post("/encode", json=body).headers
    monkeypatch.setenv("ARC_STOCK_VIT_INPUTS", "1")
    with TestClient(_app(images="arc-image-v1")) as client:
        r = client.post("/encode", json=body)
        assert r.status_code == 200
        [vit] = json.loads(r.headers[server.VIT_INPUTS_HEADER])
        assert vit["grid_thw"] == [1, 16, 24]  # 256 x 384 pixels in 16-pixel patches
        text = client.post("/encode", json={"retain": False, "base": 0, "tokens": [3, 4], "offsets": [1]})
        assert text.status_code == 200 and server.VIT_INPUTS_HEADER not in text.headers
