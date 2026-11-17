# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""``examples/encode_client.py`` against a local stand-in server (CPU, no vLLM): the request it sends, the float32
rows it decodes from ``x-rows``/``x-dim``, ``/health``, and a refusal raised as ``EncodeError``."""
import importlib.util
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "examples" / "encode_client.py"
_spec = importlib.util.spec_from_file_location("encode_client", _PATH)
client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(client)

DIM = 3


class _Fake(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *args):  # quiet
        pass

    def _send(self, status, body, headers=()):
        self.send_response(status)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.path == "/health"
        self._send(200, json.dumps({"status": "ok", "engine": "fake", "depth": 12}).encode(),
                   [("content-type", "application/json")])

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).seen.append(body)
        tokens, offsets = body["tokens"], body["offsets"]
        if any(not 0 <= o < len(tokens) for o in offsets):
            self._send(422, b'{"error": "offsets_out_of_range"}', [("content-type", "application/json")])
            return
        # Row i is (offset, token at offset, i), as little-endian float32.
        data = b"".join(struct.pack("<3f", o, tokens[o], i) for i, o in enumerate(offsets))
        self._send(200, data, [("content-type", "application/octet-stream"), ("x-rows", str(len(offsets))),
                               ("x-dim", str(DIM)), ("x-cached", "0"), ("x-encoded", str(len(tokens)))])


@pytest.fixture
def url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _Fake.seen = []
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_encode_one_shot_rows(url):
    tokens = [10, 11, 12, 13, 14]
    rows, headers = client.encode(url, tokens, [4, 0, 2])
    assert _Fake.seen == [{"tokens": tokens, "offsets": [4, 0, 2], "base": 0, "retain": False}]
    assert headers["x-rows"] == "3" and headers["x-dim"] == "3" and headers["x-encoded"] == "5"
    assert [[float(v) for v in row] for row in rows] == [[4, 14, 0], [0, 10, 1], [2, 12, 2]]


def test_decode_without_numpy_matches(monkeypatch):
    body = struct.pack("<4f", 1.5, -2.0, 0.25, 3.0)
    import builtins

    real_import = builtins.__import__

    def no_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    assert client.decode_rows(body, 2, 2) == [[1.5, -2.0], [0.25, 3.0]]
    with pytest.raises(ValueError):
        client.decode_rows(body, 3, 2)


def test_health_and_refusal(url):
    assert client.health(url)["engine"] == "fake"
    with pytest.raises(client.EncodeError) as exc:
        client.encode(url, [1, 2], [2])
    assert exc.value.status == 422 and exc.value.body == {"error": "offsets_out_of_range"}
