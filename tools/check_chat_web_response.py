#!/usr/bin/env python3
"""Validate bounded Belka health/SSE responses; never print generated text.

This checks the Belka wire protocol, not OpenAI compatibility or model quality.
The companion shell smoke test supplies curl-verified HTTP response headers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

MAX_RESPONSE_BYTES = 1 << 20
MAX_HEADER_BYTES = 16 << 10


class ResponseError(ValueError):
    """The response is not evidence of a successful health/generation check."""


def _constant(_value: str):
    raise ResponseError("non-finite JSON value")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResponseError("duplicate JSON key")
        result[key] = value
    return result


def _json_object(text: str) -> dict:
    try:
        result = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_constant)
    except (ValueError, RecursionError) as exc:
        raise ResponseError("invalid JSON response") from exc
    if not isinstance(result, dict):
        raise ResponseError("response must be a JSON object")
    return result


def _decode(payload: bytes) -> str:
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ResponseError("response exceeds byte limit")
    try:
        return payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ResponseError("response is not valid UTF-8") from exc


def validate_headers(payload: bytes, kind: str) -> None:
    if len(payload) > MAX_HEADER_BYTES:
        raise ResponseError("HTTP headers exceed byte limit")
    # curl may include interim 100/103 or proxy headers; inspect the final block.
    blocks = re.split(r"\r?\n\r?\n", payload.decode("iso-8859-1"))
    lines = next((block.splitlines() for block in reversed(blocks) if block.strip()), [])
    if not lines or not re.fullmatch(r"HTTP/\S+ 200(?: .*)?", lines[0]):
        raise ResponseError("expected HTTP 200")
    values = [line.split(":", 1)[1].strip() for line in lines[1:]
              if line.partition(":")[0].lower() == "content-type"]
    expected = "application/json" if kind == "health" else "text/event-stream"
    if len(values) != 1 or values[0].split(";", 1)[0].strip().lower() != expected:
        raise ResponseError("unexpected response Content-Type")


def check_health(payload: bytes) -> dict:
    result = _json_object(_decode(payload))
    workers = result.get("num_gpus")
    available = result.get("available_workers")
    if (result.get("status") != "ok" or result.get("ready") is not True
            or type(workers) is not int or workers < 1
            or type(available) is not int or not 0 <= available <= workers):
        raise ResponseError("server is not ready or worker counts are invalid")
    return {"ready": True, "workers": workers}


def check_completion(payload: bytes) -> dict:
    # SSE recognizes CRLF, CR and LF, but not every Unicode line separator.
    text = _decode(payload).replace("\r\n", "\n").replace("\r", "\n")
    data: list[str] = []
    event_type = ""
    done = False
    chunks = characters = 0
    has_text = False
    for line in text.split("\n")[:-1]:  # an unterminated line is never dispatched
        if line == "":
            if data:
                if done:
                    raise ResponseError("event received after completion")
                if event_type not in ("", "message"):
                    raise ResponseError("unexpected SSE event type")
                item = _json_object("\n".join(data))
                if "error" in item:
                    raise ResponseError("server reported a generation error")
                if "done" in item:
                    if item != {"done": True} or type(item["done"]) is not bool:
                        raise ResponseError("invalid completion marker")
                    done = True
                elif "token" in item:
                    token = item["token"]
                    if (not isinstance(token, str) or not token
                            or set(item) - {"token", "gpu"}
                            or ("gpu" in item and (type(item["gpu"]) is not int or item["gpu"] < 0))):
                        raise ResponseError("invalid token event")
                    try:
                        token.encode("utf-8")
                    except UnicodeError as exc:
                        raise ResponseError("invalid Unicode token") from exc
                    chunks += 1
                    characters += len(token)
                    has_text |= bool(token.strip())
                else:
                    raise ResponseError("unrecognized generation event")
            data = []
            event_type = ""
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "data":
                data.append(value)
            elif field == "event":
                event_type = value
            # SSE id/retry/unknown fields do not contribute to event data.
    if data or text.split("\n")[-1].strip():
        raise ResponseError("truncated SSE event")
    if not done or not has_text:
        raise ResponseError("missing completion marker or nonempty generated text")
    return {"done": True, "chunks": chunks, "characters": characters}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("health", "completion"))
    parser.add_argument("--headers", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        payload = sys.stdin.buffer.read(MAX_RESPONSE_BYTES + 1)
        with args.headers.open("rb") as source:
            headers = source.read(MAX_HEADER_BYTES + 1)
        validate_headers(headers, args.kind)
        summary = (check_health if args.kind == "health" else check_completion)(payload)
    except (ResponseError, OSError) as exc:
        # Do not echo raw bodies, headers, exception payloads or credentials.
        message = str(exc) if isinstance(exc, ResponseError) else "cannot read response headers"
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
