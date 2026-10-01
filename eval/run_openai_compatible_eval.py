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
from data_pipeline.sft_schema import strict_json_loads
from tools.eval_contract import completion_text, wilson


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [strict_json_loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(r,dict) for r in rows):raise ValueError('empty or invalid evaluation set')
    for row in rows:
        if 'messages' not in row and isinstance(row.get('prompt'),str):row['messages']=[{'role':'user','content':row['prompt']}]
        if not isinstance(row.get('messages'),list) or not row['messages']:raise ValueError('missing evaluation messages')
    return rows


def complete(base_url: str, model: str, api_key: str, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return completion_text(data)


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
    if not rows:ap.error('no evaluation records')
    output=args.output.resolve()
    if not output.is_relative_to(PACK_DIR):ap.error('evaluation output must stay in repository')
    if output.exists():ap.error('output already exists; use a new run-specific path')
    if any(output==p.resolve() for p in files):ap.error('output aliases an input')
    if args.max_tokens<=0 or not 0<=args.temperature<=2:ap.error('invalid decoding parameters')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    with args.output.open("x", encoding="utf-8") as out:
        for item in rows:
            answer = complete(args.base_url, args.model, args.api_key, item["messages"], args.max_tokens, args.temperature)
            result = evaluate_response(answer, item.get("checks", {}))
            record = {"id": item.get("id"), "category": item.get("category"), "answer": answer, "result": result}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            passed += int(result["passed"])
            print(json.dumps(record, ensure_ascii=False))
    summary = {"passed": passed, "total": len(rows), "pass_rate": passed / len(rows), "wilson95": wilson(passed,len(rows)), "scope": "heuristic language and lexical checks, not overall model quality", "output": str(args.output)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
