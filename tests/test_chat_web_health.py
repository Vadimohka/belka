"""Health probe contracts: protocol units and real shell/curl + local HTTP.

The shell harness substitutes only the model launcher, not curl or the checker.
No checkpoint, GPU, external service or training entrypoint is used.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.check_chat_web_response import (
    MAX_RESPONSE_BYTES, ResponseError, check_completion, check_health, validate_headers,
)

ROOT = Path(__file__).resolve().parents[1]
GOOD = 'data: {"token":"Прывітанне!","gpu":0}\n\ndata: {"done":true}\n\n'
HEALTH = dict(status="ok", ready=True, num_gpus=1, available_workers=1)


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
def test_sse_line_endings_and_private_summary(ending):
    result = check_completion(GOOD.replace("\n", ending).encode())
    assert result == dict(done=True, chunks=1, characters=len("Прывітанне!"))
    assert "Прывітанне" not in json.dumps(result)


def test_sse_bom_comments_multiline_and_unicode_line_separator():
    payload = ('\ufeff: heartbeat\n\nevent: message\nid: 1\nretry: 100\nignored: x\n'
               'data: {"token":\ndata: "Вітаю\u2028цябе!"}\n\n'
               'data: {"done":true}\n\n: final comment\n\n')
    assert check_completion(payload.encode())["characters"] == len("Вітаю\u2028цябе!")


@pytest.mark.parametrize("payload", [
    b"", b"<html>HTTP 200 is not enough</html>", b'[]',
    b'data: {"error":"private details"}\n\ndata: {"done":true}\n\n',
    b'data: {"done":true}\n\n',
    b'data: {"token":" "}\n\ndata: {"done":true}\n\n',
    b'data: {"token":"ok"}\n\n',
    b'data: {"token":"ok"}\n\ndata: {"done":true}\n',
    b'data: {"token":"ok"}\n\ndata: {"done":true}',
    b'data: {"token":null}\n\ndata: {"done":true}\n\n',
    b'data: {"token":""}\n\ndata: {"done":true}\n\n',
    b'data: {"token":"ok","gpu":true}\n\ndata: {"done":true}\n\n',
    b'data: {"token":"ok","gpu":-1}\n\ndata: {"done":true}\n\n',
    b'data: {"token":"ok","extra":1}\n\ndata: {"done":true}\n\n',
    b'data: {"token":"\\ud800"}\n\ndata: {"done":true}\n\n',
    b'data: {"token":"ok","token":"override"}\n\ndata: {"done":true}\n\n',
    b'data: {"token":NaN}\n\ndata: {"done":true}\n\n',
    b'data: {broken}\n\n', b'data: []\n\n', b'data: [DONE]\n\n',
    b'data: {"unknown":true}\n\n', b'data: {"token":"\xff"}\n\n',
    b'data: {"token":"ok"}\n\ndata: {"done":1}\n\n',
    b'data: {"token":"ok"}\n\ndata: {"done":false}\n\n',
    b'data: {"token":"ok"}\n\ndata: {"done":true,"error":""}\n\n',
    b'event: error\ndata: {"token":"ok"}\n\ndata: {"done":true}\n\n',
    GOOD.encode() + b'data: {"error":"late error"}\n\n',
    GOOD.encode() + b'data: {"done":true}\n\n',
    GOOD.encode() + b'data: {"token":"after end"}\n\n',
    GOOD.encode() + b'data: {"error":"unterminated"}',
])
def test_sse_rejects_false_success(payload):
    with pytest.raises(ResponseError):
        check_completion(payload)


def test_response_is_bounded():
    with pytest.raises(ResponseError, match="byte limit"):
        check_completion(b" " * (MAX_RESPONSE_BYTES + 1))


@pytest.mark.parametrize("update", [
    {"ready": False}, {"ready": "true"}, {"ready": 1}, {"status": "loading"},
    {"num_gpus": 0}, {"num_gpus": True}, {"num_gpus": "1"},
    {"available_workers": -1}, {"available_workers": 2}, {"available_workers": False},
])
def test_health_rejects_not_ready_and_invalid_workers(update):
    with pytest.raises(ResponseError):
        check_health(json.dumps({**HEALTH, **update}).encode())


def test_health_allows_busy_but_loaded_workers():
    assert check_health(json.dumps({**HEALTH, "available_workers": 0}).encode()) == {
        "ready": True, "workers": 1}


@pytest.mark.parametrize("headers", [
    b"", b"HTTP/1.1 503 Busy\r\nContent-Type: text/event-stream\r\n\r\n",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n",
    b"HTTP/1.1 200 OK\r\n\r\n",
    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Type: text/event-stream\r\n\r\n",
    b" " * (16384 + 1),
])
def test_wrong_http_status_or_mime_is_not_success(headers):
    with pytest.raises(ResponseError):
        validate_headers(headers, "completion")


def test_headers_with_interim_block_and_charset():
    validate_headers(b"HTTP/1.1 103 Early Hints\r\n\r\nHTTP/1.1 200 OK\r\n"
                     b"content-type: text/event-stream; charset=utf-8\r\n\r\n", "completion")


SERVER = r'''
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
root=Path(os.environ['PACK_DIR'])
(root/'server.pid').write_text(str(os.getpid()))
(root/'launcher_args.json').write_text(json.dumps(sys.argv[1:]))
mode=os.environ.get('PROBE_TEST_MODE','ok')
if mode=='early-exit': raise SystemExit(3)
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def reply(self,payload,kind='application/json',status=200):
        self.send_response(status);self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(payload)));self.end_headers()
        self.wfile.write(payload)
    def do_GET(self):
        if mode=='slow-health': time.sleep(10)
        self.reply(json.dumps(dict(status='ok',ready=mode!='not-ready',num_gpus=1,available_workers=1)).encode())
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        key=os.environ.get('BELKA_API_KEY','')
        authorized=not key or self.headers.get('Authorization')=='Bearer '+key
        (root/'request.json').write_text(json.dumps(dict(body=body,authorized=authorized)))
        if not authorized or mode=='unauthorized': return self.reply(b'{}',status=401)
        payload='data: {"token":"PRIVATE_REPLY","gpu":0}\n\ndata: {"done":true}\n\n'
        if mode=='error':payload='data: {"error":"PRIVATE_FAILURE"}\n\ndata: {"done":true}\n\n'
        if mode=='empty':payload='data: {"done":true}\n\n'
        if mode=='truncated':payload='data: {"token":"PRIVATE_REPLY"}\n\n'
        if mode=='dtype': print('Expected query, key, and value to have the same dtype',flush=True)
        self.reply(payload.encode(),'application/json' if mode=='wrong-mime' else 'text/event-stream')
ThreadingHTTPServer(('127.0.0.1',int(os.environ['PORT'])),Handler).serve_forever()
'''


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "pack"
    for rel in ("ops/local/pack_paths.sh", "ops/local/repo_guard.sh", "configs/path_policy.env",
                "ops/local/test_chat_web_health.sh", "tools/check_chat_web_response.py"):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    (root / "fake_server.py").write_text(SERVER)
    (root / "ops/local/run_chat_web.sh").write_text(
        '#!/usr/bin/env bash\nexec python3 "$PACK_DIR/fake_server.py" "$@"\n')
    # Record curl argv, not stdin containing credentials. Still execute real curl.
    curl = shutil.which("curl")
    assert curl, "curl is required to test the real health transport"
    bin_dir = root / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "curl"
    shim.write_text('#!/usr/bin/env python3\nimport os,sys,json\nfrom pathlib import Path\n'
                    'with (Path(os.environ["PACK_DIR"])/"curl_argv.jsonl").open("a") as f:\n'
                    ' f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                    f'os.execv({curl!r},[{curl!r},*sys.argv[1:]])\n')
    shim.chmod(0o755)
    return root


def run_probe(pack, mode="ok", options=(), key="", shell_trace=False):
    # Do not inherit the caller's worktree paths, API key or proxy configuration.
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "SYSTEMROOT", "LD_LIBRARY_PATH")}
    env.update(PACK_DIR=str(pack), PROBE_TEST_MODE=mode, BELKA_API_KEY=key,
               PATH=str(pack / "bin") + os.pathsep + env["PATH"], PYTHONDONTWRITEBYTECODE="1")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    result = subprocess.run(
        ["bash", *(["-x"] if shell_trace else []), str(pack / "ops/local/test_chat_web_health.sh"),
         "--port", port, "--timeout", "5", *options],
        env=env, capture_output=True, text=True, timeout=15)
    pidfile = pack / "server.pid"
    if pidfile.exists():
        with pytest.raises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)
    return result


@pytest.mark.parametrize("key", ["", 'test-only-quote"-slash\\-key'])
def test_real_shell_curl_success_and_auth_without_secret_leak(pack, key):
    base = pack / "alternate-base"
    result = run_probe(pack, options=("--base-dir", str(base)), key=key, shell_trace=True)
    assert result.returncode == 0, result.stdout + result.stderr
    request = json.loads((pack / "request.json").read_text())
    assert request["authorized"] and request["body"]["stream"] is True
    args = json.loads((pack / "launcher_args.json").read_text())
    assert args[args.index("--base-dir") + 1] == str(base)
    summaries = list(base.glob(".health-check.*/response_summary.json"))
    assert len(summaries) == 1 and json.loads(summaries[0].read_text())["done"] is True
    assert "PRIVATE_REPLY" not in result.stdout + result.stderr + summaries[0].read_text()
    if key:
        for path in pack.rglob("*"):
            if path.is_file():
                assert key.encode() not in path.read_bytes(), path
        assert key not in result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["error", "empty", "truncated", "wrong-mime", "unauthorized", "dtype"])
def test_real_shell_rejects_false_success(pack, mode):
    result = run_probe(pack, mode)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "complete SSE generation passed" not in result.stdout
    assert "PRIVATE_FAILURE" not in result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["not-ready", "slow-health", "early-exit"])
def test_readiness_is_bounded_and_requires_live_ready_server(pack, mode):
    result = run_probe(pack, mode, options=("--timeout", "1"))
    assert result.returncode == 1, result.stdout + result.stderr
    assert not (pack / "request.json").exists()


@pytest.mark.parametrize("options", [
    ("--timeout", "0"), ("--port", "65536"), ("--port", "-1"), ("--port", "oops"),
    ("--timeout", "NaN"), ("--phase", "other"), ("--host", "example.com"), ("--base-dir",),
])
def test_bad_arguments_do_not_start_or_write(pack, options):
    result = run_probe(pack, options=options)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not (pack / "server.pid").exists()
    assert not (pack / ".workspace").exists()


def test_help_does_not_start_or_create_workspace(pack):
    assert run_probe(pack, options=("--help",)).returncode == 0
    assert not (pack / ".workspace").exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_base_override_cannot_escape_pack(pack, symlink):
    outside = pack.parent / "outside"
    target = pack / "linked-base" if symlink else outside
    if symlink:
        target.symlink_to(outside, target_is_directory=True)
    result = run_probe(pack, options=("--base-dir", str(target)))
    assert result.returncode == 2
    assert not outside.exists() and not (pack / "server.pid").exists()


def test_api_key_cannot_inject_curl_config(pack):
    result = run_probe(pack, key='key\nurl = "http://example.com"')
    assert result.returncode == 2
    assert not (pack / ".workspace").exists()


@pytest.mark.parametrize("failure", [False, True])
def test_matches_real_web_server_response(failure):
    """Real FastAPI handler/streaming producer, with only inference substituted."""
    from fastapi.testclient import TestClient
    path = ROOT / "ops/nanochat_fork/scripts/chat_web.py"
    spec = importlib.util.spec_from_file_location("belka_health_web", path)
    web = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = web
    try:
        spec.loader.exec_module(web)
        class Tokenizer:
            def render_conversation(self, *a, **kw): return [1], []
            def encode_special(self, text): return {"<|assistant_start|>": 2, "<|assistant_end|>": 3}[text]
            def get_bos_token_id(self): return 0
            def decode(self, tokens): return "Вітаю!"
        class Engine:
            model = SimpleNamespace(config=SimpleNamespace(sequence_len=128))
            def generate(self, *a, **kw):
                if failure: raise ValueError("private inference failure")
                yield [4], [1]
                yield [3], [1]
        worker = web.Worker(0, "cpu", Engine(), Tokenizer())
        with TestClient(web.create_app(web.Settings(api_key="test-only"), [worker])) as client:
            assert check_health(client.get("/health").content)["ready"]
            response = client.post("/chat/completions", headers={"Authorization": "Bearer test-only"},
                json=dict(messages=[dict(role="user", content="Вітаю")], max_tokens=8, stream=True))
            assert response.status_code == 200  # errors deliberately also use HTTP 200
            if failure:
                with pytest.raises(ResponseError, match="generation error"):
                    check_completion(response.content)
            else:
                assert check_completion(response.content)["done"]
    finally:
        sys.modules.pop(spec.name, None)
