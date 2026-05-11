#!/usr/bin/env python3
"""Run Belarusian language-lock eval against an OpenAI-compatible endpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR / "data_pipeline"))
from data_pipeline.detect_belarusian import detect_belarusian  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def complete(base_url: str, model: str, api_key: str, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "choices" in data and data["choices"]:
        msg = data["choices"][0].get("message") or {}
        return str(msg.get("content") or data["choices"][0].get("text") or "")
    return str(data)


def evaluate_response(text: str, checks: dict[str, Any]) -> dict[str, Any]:
    det = detect_belarusian(text, min_chars=10, allow_short=True, accept_threshold=float(checks.get("min_belarusian_score", 2.0)))
    must_any = checks.get("must_include_any") or []
    must_not = checks.get("must_not_include_any") or []
    lower = text.lower()
    pass_language = det.score >= float(checks.get("min_belarusian_score", 2.0))
    pass_must_any = True if not must_any else any(str(x).lower() in lower for x in must_any)
    pass_must_not = not any(str(x).lower() in lower for x in must_not)
    return {
        "score": det.score,
        "reasons": det.reasons,
        "pass_language": pass_language,
        "pass_must_include_any": pass_must_any,
        "pass_must_not_include_any": pass_must_not,
        "passed": bool(pass_language and pass_must_any and pass_must_not),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Belarusian eval against OpenAI-compatible /chat/completions")
    ap.add_argument("--eval-file", type=Path, action="append", default=[])
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("MODEL", "be-local"))
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--output", type=Path, default=Path("eval_results.jsonl"))
    args = ap.parse_args()
    files = args.eval_file or [PACK_DIR / "eval" / "belarusian_language_lock_eval.jsonl", PACK_DIR / "eval" / "meetmesh_domain_eval.be.jsonl"]
    rows = []
    for path in files:
        rows.extend(load_jsonl(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    with args.output.open("w", encoding="utf-8") as out:
        for item in rows:
            answer = complete(args.base_url, args.model, args.api_key, item["messages"], args.max_tokens, args.temperature)
            result = evaluate_response(answer, item.get("checks", {}))
            record = {"id": item.get("id"), "category": item.get("category"), "answer": answer, "result": result}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            passed += int(result["passed"])
            print(json.dumps(record, ensure_ascii=False))
    summary = {"passed": passed, "total": len(rows), "pass_rate": round(passed / max(1, len(rows)), 4), "output": str(args.output)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
