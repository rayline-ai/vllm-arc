# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``PoolingSession`` (patch 0001's client) on CPU, against a stand-in for ``AsyncLLM.encode`` over a stream of
inputs: open, append, close, abort, and the backpressure that keeps one input in flight."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from vllm_arc.engine.pooling_session import PoolingSession, PoolingSessionError  # noqa: E402


def params(offsets=()):
    return SimpleNamespace(extra_kwargs={"arc_gather": {"v": 1, "offsets": list(offsets)}})


class StreamEngine:
    """``encode(stream, params, request_id)``: one output per streamed input, rows ``[token, position]`` at the
    input's offsets over the whole session, as GatherPooler reads them; finishes when the stream ends."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.inputs: list[list[int]] = []
        self.finished_streams: list[str] = []
        self.aborted: list[str] = []
        self.stall = asyncio.Event()
        self.stall.set()
        self.fail = self.finish = False
        self.extra_tokens = 0

    async def encode(self, stream, session_params, request_id):
        tokens: list[int] = []
        try:
            async for item in stream:
                prompt, p = item
                # A list is a text prompt; a dict stands in for one the engine expands (an image's placeholder run).
                ids = prompt if isinstance(prompt, list) else prompt["prompt_token_ids"]
                self.log.append(f"pull {len(self.inputs)}")
                self.inputs.append(list(ids))
                tokens += ids if isinstance(prompt, list) else prompt["expanded"]
                await self.stall.wait()
                if self.fail:
                    raise RuntimeError("engine failed")
                offsets = (p or session_params).extra_kwargs["arc_gather"]["offsets"]
                rows = torch.tensor([[float(tokens[o]), float(o)] for o in offsets]).reshape(len(offsets), 2)
                self.log.append(f"emit {len(self.inputs) - 1}")
                yield SimpleNamespace(finished=self.finish, prompt_token_ids=tokens + [0] * self.extra_tokens,
                                      outputs=SimpleNamespace(data=rows))
            self.finished_streams.append(request_id)
        except (asyncio.CancelledError, GeneratorExit):  # AsyncLLM.encode aborts the request on both (patch 0001)
            self.aborted.append(request_id)
            raise


def _session(engine, request_id="r1"):
    return PoolingSession(engine, pooling_params=params(), request_id=request_id, make_input=lambda ids, p: (ids, p))


def run(coro):
    return asyncio.run(coro)


def test_appends_send_only_new_tokens_and_read_offsets_over_the_whole_session_then_close_finishes():
    async def go():
        engine = StreamEngine()
        s = _session(engine)
        first = await s.append(list(range(100, 110)), len(list(range(100, 110))), params([9, 0]))
        engine.log.append("got 0")
        second = await s.append([7, 8], len([7, 8]), params([11, 10]))
        engine.log.append("got 1")
        assert engine.inputs == [list(range(100, 110)), [7, 8]] and s.num_tokens == 12
        assert first.outputs.data.tolist() == [[109.0, 9.0], [100.0, 0.0]]
        assert second.outputs.data.tolist() == [[8.0, 11.0], [7.0, 10.0]]
        # Backpressure: the engine takes input i+1 only after the caller has output i.
        assert engine.log == ["pull 0", "emit 0", "got 0", "pull 1", "emit 1", "got 1"]
        await s.close()
        await s.close()  # idempotent
        assert s.closed and engine.finished_streams == ["r1"] and engine.aborted == []
        with pytest.raises(PoolingSessionError):
            await s.append([1], len([1]), params([12]))

    run(go())


def test_offsets_outside_the_input_are_refused_before_the_engine_and_leave_the_session_usable():
    async def go():
        engine = StreamEngine()
        s = _session(engine)
        await s.append([1, 2, 3], len([1, 2, 3]), params([2]))
        for bad in ([2], [5], [3, 6]):
            with pytest.raises(ValueError):
                await s.append([4, 5], len([4, 5]), params(bad))
        with pytest.raises(ValueError):
            await s.append([], len([]), params())
        assert len(engine.inputs) == 1 and not s.closed
        out = await s.append([4, 5], len([4, 5]), params([4, 3]))
        assert out.outputs.data[:, 0].tolist() == [5.0, 4.0]
        await s.close()

    run(go())


def test_an_append_counts_the_positions_the_engine_expands_it_to_and_a_wrong_count_closes_the_session():
    async def go():
        engine = StreamEngine()
        s = _session(engine)
        await s.append([1, 2], 2, params([1]))
        image = {"prompt_token_ids": [5, 9, 6], "expanded": [5, 9, 9, 9, 6]}  # one pad becomes three
        out = await s.append(image, 5, params([6, 2]))
        assert s.num_tokens == 7 and out.outputs.data.tolist() == [[6.0, 6.0], [5.0, 2.0]]
        out = await s.append([7], 1, params([7]))  # a text append after the image
        assert s.num_tokens == 8 and out.outputs.data.tolist() == [[7.0, 7.0]]
        with pytest.raises(PoolingSessionError):  # the caller's count disagrees with the engine's expansion
            await s.append(dict(image), 3, params([8]))
        assert s.closed

    run(go())


def test_session_level_offsets_are_refused():
    with pytest.raises(ValueError):
        PoolingSession(StreamEngine(), pooling_params=params([0]), request_id="r", make_input=lambda i, p: (i, p))
    with pytest.raises(ValueError):
        PoolingSession(StreamEngine(), pooling_params=params(), request_id="", make_input=lambda i, p: (i, p))


def test_closing_an_unstarted_session_sends_nothing():
    async def go():
        engine = StreamEngine()
        s = _session(engine)
        await s.close()
        assert s.closed and engine.inputs == [] and engine.finished_streams == []

    run(go())


@pytest.mark.parametrize("fault", ["engine_error", "finished", "length"])
def test_a_broken_lifecycle_closes_the_session(fault):
    async def go():
        engine = StreamEngine()
        s = _session(engine)
        await s.append([1, 2], len([1, 2]), params([1]))
        engine.fail, engine.finish = fault == "engine_error", fault == "finished"
        engine.extra_tokens = 1 if fault == "length" else 0
        with pytest.raises(RuntimeError):  # PoolingSessionError is a RuntimeError
            await s.append([3], len([3]), params([2]))
        assert s.closed
        with pytest.raises(PoolingSessionError):
            await s.append([4], len([4]), params([3]))
        await s.close()

    run(go())


def test_a_cancelled_append_aborts_the_request_and_closes_the_session():
    async def go():
        engine = StreamEngine()
        s = _session(engine)
        await s.append([1, 2], len([1, 2]), params([1]))
        engine.stall.clear()
        task = asyncio.create_task(s.append([3], len([3]), params([2])))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert s.closed and engine.aborted == ["r1"]
        await s.close()

    run(go())
