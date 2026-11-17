# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Atlas Futures, Inc.
"""A minimal client for the ARC encoder server (``vllm-arc-server``): ``GET /health``, then one ``POST /encode``.

    vllm-arc-server --port 8765                  # on the GPU host (docs/engine.md)
    python examples/encode_client.py --url http://127.0.0.1:8765 --tokens 1000,1001,1002,1003 --offsets 3,0

Standard library only (numpy is used when installed). The request is a one-shot (``"retain": false``, ``"base": 0``):
it is encoded on its own and nothing is kept on the server, so no session or patch is involved. Token ids are the
model's (Qwen3.5's tokenizer); this client does not tokenize. The 200 response body is float32 rows, row-major,
``[x-rows, x-dim]``, one row per requested offset in request order. A refusal is a JSON body ``{"error": <code>,
...}`` with a 4xx/5xx status (codes in ``docs/engine.md``, "Endpoints"), raised here as ``EncodeError``.
"""
from __future__ import annotations

import argparse
import array
import json
import sys
import urllib.error
import urllib.request


class EncodeError(RuntimeError):
    """The server refused a request: its HTTP status and the error body's ``error`` code."""

    def __init__(self, status: int, body: dict) -> None:
        super().__init__(f"HTTP {status}: {body.get('error', body)}")
        self.status, self.body = status, body


def _request(url: str, data: bytes | None = None, timeout: float = 600.0):
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                     headers={"content-type": "application/json"} if data is not None else {})
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"error": raw.decode(errors="replace")[:200]}
        raise EncodeError(exc.code, body if isinstance(body, dict) else {"error": body}) from None


def health(url: str, timeout: float = 30.0) -> dict:
    """``GET /health``: the engine id, model, depth, RoPE, context and session state, as JSON."""
    with _request(url.rstrip("/") + "/health", timeout=timeout) as response:
        return json.loads(response.read())


def decode_rows(body: bytes, rows: int, dim: int):
    """The response body as ``rows`` x ``dim`` float32: a numpy array when numpy is installed, else lists."""
    if len(body) != rows * dim * 4:
        raise ValueError(f"{len(body)} bytes for {rows} x {dim} float32 rows")
    try:
        import numpy as np
    except ImportError:
        values = array.array("f")
        values.frombytes(body)
        if sys.byteorder != "little":  # the server writes little-endian float32
            values.byteswap()
        return [values[i * dim:(i + 1) * dim].tolist() for i in range(rows)]
    return np.frombuffer(body, dtype="<f4").reshape(rows, dim)


def encode(url: str, tokens: list[int], offsets: list[int], timeout: float = 600.0):
    """One one-shot ``POST /encode``: the rows at ``offsets`` (absolute, ``0 <= o < len(tokens)``) and the response
    headers (``x-rows``, ``x-dim``, ``x-cached``, ``x-encoded``, ...)."""
    payload = json.dumps({"tokens": list(tokens), "offsets": list(offsets), "base": 0, "retain": False}).encode()
    with _request(url.rstrip("/") + "/encode", payload, timeout=timeout) as response:
        headers = {k.lower(): v for k, v in response.headers.items()}
        body = response.read()
    return decode_rows(body, int(headers["x-rows"]), int(headers["x-dim"])), headers


def _ints(spec: str) -> list[int]:
    return [int(part) for part in spec.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="the server's base URL")
    parser.add_argument("--tokens", default=",".join(str(1000 + i) for i in range(64)),
                        help="comma-separated token ids (default: 64 synthetic ids)")
    parser.add_argument("--offsets", default=None, help="comma-separated offsets (default: the last token)")
    parser.add_argument("--timeout", type=float, default=600.0, help="request timeout in seconds")
    args = parser.parse_args()
    tokens = _ints(args.tokens)
    offsets = _ints(args.offsets) if args.offsets else [len(tokens) - 1]

    info = health(args.url, timeout=args.timeout)
    print(f"health: {info.get('status')} engine={info.get('engine')} depth={info.get('depth')} "
          f"max_model_len={info.get('max_model_len')}")
    try:
        rows, headers = encode(args.url, tokens, offsets, timeout=args.timeout)
    except EncodeError as exc:
        raise SystemExit(f"encode refused: {exc} {json.dumps(exc.body)}") from None
    print(f"encode: {len(tokens)} tokens, offsets {offsets} -> rows {headers['x-rows']} x {headers['x-dim']} "
          f"(cached {headers.get('x-cached')}, encoded {headers.get('x-encoded')})")
    for offset, row in zip(offsets, rows):
        values = [float(v) for v in row[:4]]
        print(f"  offset {offset}: {[round(v, 4) for v in values]} ...")


if __name__ == "__main__":
    main()
