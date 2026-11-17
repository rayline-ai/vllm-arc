# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""Patch 0001 (``arc-v0.33.0.patch``, engine.lock's ``patch_source`` release) against vLLM 0.33.0's own scheduler and V2 model runner, on CPU.

Needs a vLLM 0.33.0 with the patch applied (``import vllm`` with ``vllm.v1.engine.async_llm.ARC_PATCH``); skipped
otherwise, so CI's environment (no vLLM) skips it. Run it with the wheel's ``vllm/`` patched and on ``PYTHONPATH``:

    python3 deploy/fetch_patches.py <patch dir>
    git -C <dir holding vllm/> apply -p1 <patch dir>/arc-v0.33.0.patch
    PYTHONPATH=<dir holding vllm/> python -m pytest tests/test_patch_0001.py

The V2 runner's buffers need a GPU (UVA, Triton write kernels); here they are CPU tensors and staged writes are
applied by a CPU loop, which is all the runner's bookkeeping needs. Covered: the streaming-input validation, the
scheduler's session lifecycle for a pooling request (open, append, pause, queued append, finish), and the runner's
in-place extension (fix 1b) leaving every buffer equal to stock's full re-add while writing only the new tokens.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
async_llm = pytest.importorskip("vllm.v1.engine.async_llm")
if getattr(async_llm, "ARC_PATCH", None) != "0001-retained-pooling-session":
    pytest.skip("vLLM without patch 0001", allow_module_level=True)

from vllm.config import DeviceConfig, VllmConfig  # noqa: E402
from vllm.exceptions import VLLMValidationError  # noqa: E402
from vllm.pooling_params import PoolingParams  # noqa: E402
from vllm.sampling_params import RequestOutputKind, SamplingParams  # noqa: E402
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.engine import FinishReason  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402
from vllm.v1.worker.gpu import buffer_utils  # noqa: E402
from vllm.v1.worker.gpu.block_table import BlockTables  # noqa: E402
from vllm.v1.worker.gpu.mm.rope import RopeState  # noqa: E402
from vllm.v1.worker.gpu.model_runner import GPUModelRunner  # noqa: E402
from vllm.v1.worker.gpu.model_states.default import DefaultModelState  # noqa: E402
from vllm.v1.worker.gpu.states import RequestState  # noqa: E402

BLOCK = 16


def gather(offsets, skip=True):
    return PoolingParams(task="token_embed", use_activation=False, skip_reading_prefix_cache=skip,
                         extra_kwargs={"arc_gather": {"v": 1, "offsets": list(offsets)}})


# --- frontend: which streamed inputs AsyncLLM accepts ---------------------------------------------------------------


def test_streaming_accepts_gather_pooling_without_prefix_reads_and_nothing_else_new():
    validate = async_llm.AsyncLLM._validate_streaming_input_sampling_params
    validate(gather([1]))
    validate(gather([]))
    for refused in (PoolingParams(task="token_embed"), PoolingParams(task="token_embed", extra_kwargs={"x": 1}),
                    gather([1], skip=False)):
        with pytest.raises(VLLMValidationError):
            validate(refused)
    validate(SamplingParams(max_tokens=4, output_kind=RequestOutputKind.DELTA))  # generation: unchanged
    with pytest.raises(VLLMValidationError):
        validate(SamplingParams(max_tokens=4, output_kind=RequestOutputKind.FINAL_ONLY))  # still refused


# --- scheduler: a pooling session's lifecycle ------------------------------------------------------------------------


def _scheduler() -> Scheduler:
    config = VllmConfig(device_config=DeviceConfig("cpu"))
    config.model_config = MagicMock()
    config.model_config.skip_tokenizer_init = True
    config.model_config.is_multimodal_model = False
    config.model_config.supports_multimodal_inputs = False
    config.model_config.is_encoder_decoder = False
    config.model_config.is_diffusion = False
    config.model_config.max_model_len = 1024
    config.model_config.enable_return_routed_experts = False
    config.cache_config = MagicMock()
    config.cache_config.num_gpu_blocks = 1000
    config.cache_config.enable_prefix_caching = False
    kv = KVCacheConfig(num_blocks=1000, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(
        ["layer"], FullAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32))])
    return Scheduler(vllm_config=config, kv_cache_config=kv, log_stats=True,
                     structured_output_manager=StructuredOutputManager(config), block_size=BLOCK,
                     hash_block_size=BLOCK)


def _input(tokens, offsets, resumable=True, req_id="session"):
    return Request(request_id=req_id, prompt_token_ids=list(tokens), sampling_params=None,
                   pooling_params=gather(offsets), resumable=resumable)


def _pooled(scheduler, out):
    """The runner's answer to one step: a pooled output for each scheduled request."""
    ids = list(out.num_scheduled_tokens)
    return scheduler.update_from_output(out, ModelRunnerOutput(
        req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)}, sampled_token_ids=[[] for _ in ids],
        logprobs=None, prompt_logprobs_dict={}, pooler_output=[torch.ones(1, 4) for _ in ids]))


def test_a_pooling_session_pauses_keeping_its_kv_and_an_append_schedules_only_the_new_tokens():
    s = _scheduler()
    first = _input(range(1, 41), [39])
    s.add_request(first)
    out = s.schedule()
    assert out.num_scheduled_tokens["session"] == 40
    [eco] = _pooled(s, out)[0].outputs
    assert eco.finish_reason == FinishReason.STOP and eco.pooling_output is not None
    session = s.requests["session"]
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ and session.num_computed_tokens == 40
    blocks = s.kv_cache_manager.get_block_ids("session")
    assert len(blocks[0]) == 3  # 40 tokens on 16-token blocks, still held while paused

    append = _input(range(41, 45), [43, 40])
    s.add_request(append)
    # Every token kept (stock 0.33 would drop the last one as a "sampled" token, and assert).
    assert session.prompt_token_ids == list(range(1, 45)) and session.num_prompt_tokens == 44
    assert session.pooling_params.extra_kwargs["arc_gather"]["offsets"] == [43, 40]
    assert session.status == RequestStatus.WAITING
    out = s.schedule()
    assert out.num_scheduled_tokens == {"session": 4}
    [new] = out.scheduled_new_reqs
    assert new.num_computed_tokens == 40 and new.prefill_token_ids == list(range(1, 45))
    assert new.pooling_params.extra_kwargs["arc_gather"]["offsets"] == [43, 40]
    assert new.block_ids[0][:3] == blocks[0]  # the same KV, extended
    _pooled(s, out)
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ and session.num_computed_tokens == 44

    # The input stream ends: a final non-resumable request finishes the session and frees its blocks.
    s.add_request(Request(request_id="session", prompt_token_ids=[0], sampling_params=None,
                          pooling_params=gather([]), resumable=False))
    assert "session" not in s.requests
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == s.kv_cache_manager.block_pool.num_gpu_blocks - 1


def test_an_append_queued_while_the_session_runs_resumes_it_right_after_its_output():
    s = _scheduler()
    s.add_request(_input(range(1, 41), [0]))
    out = s.schedule()
    s.add_request(_input(range(41, 43), [41]))  # arrives mid-step
    _pooled(s, out)
    session = s.requests["session"]
    assert session.status == RequestStatus.WAITING and session.prompt_token_ids == list(range(1, 43))
    out = s.schedule()
    assert out.num_scheduled_tokens == {"session": 2} and out.scheduled_new_reqs[0].num_computed_tokens == 40


# --- V2 runner: the in-place extension (fix 1b) ----------------------------------------------------------------------


class _CpuPool:
    """``UvaBufferPool`` without pinned memory: a CPU copy."""

    def __init__(self, size, dtype, max_concurrency=None):
        self.dtype = dtype

    def copy_to_uva(self, x):
        return torch.as_tensor(np.asarray(x)).clone()

    def copy_to_gpu(self, x, out=None):
        t = self.copy_to_uva(x)
        return t if out is None else out.copy_(t)


def _cpu_apply(self):
    """``StagedWriteTensor.apply_write`` as a CPU loop."""
    cu = [0, *self._staged_write_cu_lens]
    for k, (row, start) in enumerate(zip(self._staged_write_indices, self._staged_write_starts)):
        values = torch.tensor(self._staged_write_contents[cu[k]:cu[k + 1]], dtype=self.gpu.dtype)
        if self.gpu.dim() == 1:
            self.gpu[row] = values[0]
        else:
            self.gpu[row, start:start + len(values)] = values
    self.clear_staged_writes()


_original_stage_write = buffer_utils.StagedWriteTensor.stage_write


def _counted_stage_write(self, index, start, x):
    self.arc_written = getattr(self, "arc_written", 0) + len(x)
    return _original_stage_write(self, index, start, x)


@pytest.fixture
def cpu_buffers(monkeypatch):
    monkeypatch.setattr(buffer_utils, "UvaBufferPool", _CpuPool)
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: False)
    monkeypatch.setattr(buffer_utils.StagedWriteTensor, "apply_write", _cpu_apply)
    monkeypatch.setattr(buffer_utils.StagedWriteTensor, "stage_write", _counted_stage_write)


class _ModelState:
    """``DefaultModelState``'s request hooks over a real ``RopeState`` (M-RoPE: 3 position rows per request)."""

    add_request = DefaultModelState.add_request

    def __init__(self, positions):
        self.rope_state = RopeState(num_dims=3, max_num_reqs=4, max_num_tokens=1024, max_model_len=1024,
                                    device=torch.device("cpu"))
        self.model = SimpleNamespace(get_mrope_input_positions=positions)
        self.prompt_embeds_state = None

    def remove_request(self, req_id):
        pass

    def apply_staged_writes(self):
        self.rope_state.apply_staged_writes()


class _PoolingRunner:
    def __init__(self):
        self.params, self.states = {}, {}

    def add_request(self, req_id, req_index, params, prompt_token_ids):
        self.params[req_index], self.states[req_index] = params, object()

    def remove_request(self, req_index):
        self.params.pop(req_index, None)
        self.states.pop(req_index, None)


def text_positions(delta):
    """M-RoPE positions of a text-only tail after ``delta``'s worth of earlier multimodal stretch: ``i + delta``."""
    return lambda ids, mm: (torch.arange(len(ids)).unsqueeze(0).expand(3, -1) + delta, delta)


def _runner(positions, pooling=True, extend=True, encoder_cache=None):
    runner = Mock(spec=GPUModelRunner)
    runner.req_states = RequestState(max_num_reqs=4, max_model_len=1024, max_num_batched_tokens=1024,
                                     num_speculative_steps=0, vocab_size=32000, device=torch.device("cpu"))
    runner.model_state = _ModelState(positions)
    runner.block_tables = BlockTables(block_sizes=[BLOCK, BLOCK], max_num_reqs=4, max_num_batched_tokens=1024,
                                      max_num_blocks_per_group=[64, 64], device=torch.device("cpu"))
    runner.pooling_runner = _PoolingRunner() if pooling else None
    runner.lora_state = Mock()
    runner.pp_handler = runner.sampler = runner.prompt_logprobs_worker = None
    runner.encoder_cache = encoder_cache
    runner.adaptive_verification = None
    runner.is_last_pp_rank = False
    runner._remove_request = GPUModelRunner._remove_request.__get__(runner)
    runner.add_requests = GPUModelRunner.add_requests.__get__(runner)
    runner._arc_extend_request = (GPUModelRunner._arc_extend_request.__get__(runner) if extend
                                  else (lambda data: False))
    return runner


def _step(runner, *reqs):
    runner.add_requests(SchedulerOutput(
        scheduled_new_reqs=list(reqs), scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={}, total_num_scheduled_tokens=0, scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={}, finished_req_ids=set(), free_encoder_mm_hashes=[]))
    runner.block_tables.apply_staged_writes = lambda: [t.apply_write() for t in runner.block_tables.block_tables]
    runner.block_tables.apply_staged_writes()


def _new(req_id, tokens, computed, blocks, offsets, pooling=True, mm=()):
    return NewRequestData(req_id=req_id, prompt_token_ids=list(tokens), prefill_token_ids=list(tokens),
                          mm_features=list(mm), sampling_params=None if pooling else SamplingParams(max_tokens=1),
                          pooling_params=gather(offsets) if pooling else None, block_ids=blocks,
                          num_computed_tokens=computed, lora_request=None)


def _snapshot(runner, req_id):
    rs, rope, bt = runner.req_states, runner.model_state.rope_state, runner.block_tables
    i = rs.req_id_to_index[req_id]
    n = int(rs.prefill_len.np[i])
    return {
        "index": i, "prefill_len": n, "prompt_len": int(rs.prompt_len.np[i]), "max_seq_len": int(rs.max_seq_len[i]),
        "tokens": rs.all_token_ids.gpu[i, :n].tolist(), "total_len": int(rs.total_len.gpu[i]),
        "computed": int(rs.num_computed_tokens.gpu[i]), "computed_np": int(rs.num_computed_tokens_np[i]),
        "computed_prefill": int(rs.num_computed_prefill_tokens[i]),
        "positions": rope.prefill_positions.gpu[3 * i:3 * i + 3, :n].tolist(),
        "delta": int(rope.prefill_delta.np[i]),
        "blocks": [bt.block_tables[g].gpu[i, :int(bt.num_blocks.np[g, i])].tolist() for g in range(2)],
        "offsets": (runner.pooling_runner.params[i].extra_kwargs["arc_gather"]["offsets"]
                    if runner.pooling_runner else None),
    }


SESSION = list(range(1000, 1600))  # 600 tokens
APPEND = list(range(5000, 5064))  # 64 tokens
OLD_BLOCKS = (list(range(10, 48)), [7])  # 38 attention blocks; one GDN state block
NEW_BLOCKS = (list(range(10, 48)) + [90], [7, 8])


@pytest.mark.parametrize("delta", [0, 7])
def test_the_runner_extends_a_resumed_pooling_session_in_place_equal_to_stocks_full_re_add(cpu_buffers, delta):
    stock, patched = _runner(text_positions(delta), extend=False), _runner(text_positions(delta))
    for runner in (stock, patched):
        _step(runner, _new("other", range(30), 0, ([1, 2], [3]), [0]), _new("s", SESSION, 0, OLD_BLOCKS, [599]))
    old_state = patched.pooling_runner.states[patched.req_states.req_id_to_index["s"]]
    written = patched.req_states.all_token_ids.arc_written
    positions_written = patched.model_state.rope_state.prefill_positions.arc_written
    for runner in (stock, patched):
        _step(runner, _new("s", SESSION + APPEND, 600, NEW_BLOCKS, [663, 600]))

    want, got = _snapshot(stock, "s"), _snapshot(patched, "s")
    assert got == want
    assert got["tokens"] == SESSION + APPEND and got["prefill_len"] == 664 and got["computed"] == 600
    assert got["positions"] == [[p + delta for p in range(664)]] * 3 and got["offsets"] == [663, 600]
    assert got["blocks"] == [NEW_BLOCKS[0], NEW_BLOCKS[1]]
    assert _snapshot(patched, "other") == _snapshot(stock, "other")
    # Fix 1b: only the 64 new ids and their 3 x 64 positions were staged, against 664 and 3 x 664 for stock.
    assert patched.req_states.all_token_ids.arc_written - written == 64
    assert patched.model_state.rope_state.prefill_positions.arc_written - positions_written == 3 * 64
    assert stock.req_states.all_token_ids.arc_written - written == 664
    # A fresh pooling state, as a re-added request gets; the slot and the free list are unchanged.
    i = got["index"]
    assert patched.pooling_runner.states[i] is not old_state
    assert sorted(patched.req_states.free_indices) == sorted(stock.req_states.free_indices)


def test_a_multimodal_runner_still_extends_a_text_only_update(cpu_buffers):
    # The served ArcBaseForPooling is multimodal, so its runner always has an encoder cache; a text-only update
    # must still take the in-place extension (9bba7a1 guarded on the runner and always fell back).
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

    stock = _runner(text_positions(0), extend=False, encoder_cache=EncoderCache())
    patched = _runner(text_positions(0), encoder_cache=EncoderCache())
    for runner in (stock, patched):
        _step(runner, _new("s", SESSION, 0, OLD_BLOCKS, [599]))
    written = patched.req_states.all_token_ids.arc_written
    for runner in (stock, patched):
        _step(runner, _new("s", SESSION + APPEND, 600, NEW_BLOCKS, [663, 600]))
    assert _snapshot(patched, "s") == _snapshot(stock, "s")
    assert patched.arc_extends == 1 and patched.req_states.all_token_ids.arc_written - written == 64
    assert patched.encoder_cache.mm_features == stock.encoder_cache.mm_features == {"s": []}


def _qwen_positions():
    """Stock Qwen3-VL M-RoPE positions (the served model's), merge size 2."""
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

    config = SimpleNamespace(video_token_id=-2, vision_start_token_id=-3, vision_end_token_id=-4,
                             vision_config=SimpleNamespace(spatial_merge_size=2))

    def qwen(ids, mm):
        return Qwen3VLForConditionalGeneration._get_mrope_input_positions(ids, mm, config=config)

    return qwen


def _image(offset, h, w):
    """An image feature as a resumed session's ``mm_features`` carry it: its placeholder span (``h * w / 4``
    tokens at merge size 2) and its grid; the scheduler keeps covered features in the list (tensors stripped)."""
    return SimpleNamespace(modality="image", identifier=f"img{offset}",
                           mm_position=SimpleNamespace(offset=offset, length=h * w // 4),
                           data={"image_grid_thw": SimpleNamespace(data=torch.tensor([1, h, w]))})


def test_the_real_qwen_text_positions_extend_the_same_way(cpu_buffers):
    qwen = _qwen_positions()

    stock, patched = _runner(qwen, extend=False), _runner(qwen)
    for runner in (stock, patched):
        _step(runner, _new("s", SESSION, 0, OLD_BLOCKS, [599]))
        _step(runner, _new("s", SESSION + APPEND, 600, NEW_BLOCKS, [663]))
    assert _snapshot(patched, "s") == _snapshot(stock, "s")


def test_an_image_append_takes_the_full_re_add_and_a_later_text_append_extends(cpu_buffers):
    # A session's features accumulate across updates: after its first image every update still carries that
    # image's feature. Only an update whose own image ends past the held tokens takes stock's full re-add; a text
    # update after it extends in place, its positions continuing from the image's M-RoPE delta.
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

    qwen = _qwen_positions()
    stock = _runner(qwen, extend=False, encoder_cache=EncoderCache())
    patched = _runner(qwen, encoder_cache=EncoderCache())
    image = _image(610, 8, 12)  # 24 placeholder tokens at 610..633: positions advance by max(h, w)/2 = 6
    with_image = SESSION + list(range(7000, 7040))  # 640 tokens: the image run and text around it
    tail = with_image + APPEND  # 704
    blocks = (list(range(10, 48)) + [90], [7])
    for runner in (stock, patched):
        _step(runner, _new("s", SESSION, 0, OLD_BLOCKS, [599]))
        _step(runner, _new("s", with_image, 600, blocks, [639], mm=[image]))
    assert getattr(patched, "arc_extends", 0) == 0  # the image's own update: the full re-add
    assert _snapshot(patched, "s") == _snapshot(stock, "s")
    delta = int(patched.model_state.rope_state.prefill_delta.np[patched.req_states.req_id_to_index["s"]])
    assert delta < 0  # the image's 24 tokens span 6 x 4 positions
    written = patched.req_states.all_token_ids.arc_written
    for runner in (stock, patched):
        _step(runner, _new("s", tail, 640, blocks, [703, 640], mm=[image]))
    got, want = _snapshot(patched, "s"), _snapshot(stock, "s")
    assert got == want and got["tokens"] == tail
    assert got["positions"][0][640:] == [p + delta for p in range(640, 704)]
    assert patched.arc_extends == 1 and patched.req_states.all_token_ids.arc_written - written == 64


@pytest.mark.parametrize("case", ["not_extending", "generation", "multimodal", "shrunk_blocks"])
def test_anything_but_a_text_pooling_extension_takes_stocks_full_re_add(cpu_buffers, case):
    pooling = case != "generation"
    stock, patched = _runner(text_positions(0), pooling, extend=False), _runner(text_positions(0), pooling)
    computed, blocks, mm = 600, NEW_BLOCKS, ()
    if case == "not_extending":
        computed = 500  # resumed below the slot's length (e.g. a preemption): rewrite everything
    if case == "shrunk_blocks":
        blocks = (list(range(10, 20)), [7])
    for runner in (stock, patched):
        _step(runner, _new("s", SESSION, 0, OLD_BLOCKS, [599], pooling))
        written = runner.req_states.all_token_ids.arc_written
        if case == "multimodal":
            # An update bringing an image (its span ends past the 600 held tokens): the extension refuses.
            assert runner._arc_extend_request(_new("s", SESSION + APPEND, 600, blocks, [663], pooling,
                                                   mm=[_image(590, 8, 8)])) is False
            continue
        _step(runner, _new("s", SESSION + APPEND, computed, blocks, [663], pooling))
        assert runner.req_states.all_token_ids.arc_written - written == 664  # the full re-add
    if case != "multimodal":
        assert _snapshot(patched, "s") == _snapshot(stock, "s")
