#!/usr/bin/env python3
"""Run Belarusian language-lock eval against an OpenAI-compatible endpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import math
import requests

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR / "data_pipeline"))
from data_pipeline.detect_belarusian import detect_belarusian  # noqa: E402
from eval.protocol import complete as request_completion, EvaluationTransportError


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def complete(base_url: str, model: str, api_key: str, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    return request_completion(base_url,model,api_key,messages,max_tokens,temperature)

def normalize_item(item):
    if not isinstance(item,dict):
        raise ValueError('eval record must be an object')
    ident=item.get('id',item.get('eval_id'))
    messages=item.get('messages')
    if messages is None and isinstance(item.get('prompt'),str):
        messages=[dict(role='user',content=item['prompt'])]
    if not isinstance(ident,str) or not ident or not isinstance(messages,list) or not messages:
        raise ValueError('eval record needs id/eval_id and messages/prompt')
    for msg in messages:
        if not isinstance(msg,dict) or msg.get('role') not in ('system','user','assistant') or not isinstance(msg.get('content'),str) or not msg['content'].strip():
            raise ValueError('invalid eval message')
    checks=dict(item.get('checks') or {})
    for name in ('must_include_any','min_belarusian_score'):
        if name in item: checks[name]=item[name]
    if 'must_not_include' in item: checks['must_not_include_any']=item['must_not_include']
    if 'must_not_include_any' in item: checks['must_not_include_any']=item['must_not_include_any']
    for name in ('must_include_any','must_not_include_any'):
        if name in checks and (not isinstance(checks[name],list) or any(not isinstance(x,str) or not x for x in checks[name])):
            raise ValueError('invalid keyword checks')
    weight=item.get('weight',1.0)
    if type(weight) not in (int,float) or not math.isfinite(weight) or weight<=0:
        raise ValueError('invalid eval weight')
    return dict(id=ident,messages=messages,checks=checks,weight=weight,category=item.get('category'),
                manual_criteria={key:item[key] for key in ('expected_behavior','safety_expected','fact_notes') if item.get(key)})


def evaluate_response(text: str, checks: dict[str, Any]) -> dict[str, Any]:
    det = detect_belarusian(text, min_chars=10, allow_short=True, accept_threshold=float(checks.get("min_belarusian_score", 2.0)))
    must_any = checks.get("must_include_any") or []
    must_not = checks.get("must_not_include_any") or []
    lower = text.lower()
    pass_language = det.is_belarusian
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run Belarusian eval against OpenAI-compatible /chat/completions")
    ap.add_argument("--eval-file", type=Path, action="append", default=[])
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--api-key", default=os.environ.get("BELKA_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    ap.add_argument("--model", default=os.environ.get("MODEL", "be-local"))
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--output", type=Path, default=Path("eval_results.jsonl"))
    args = ap.parse_args(argv)
    files = args.eval_file or [PACK_DIR / "eval" / "belarusian_language_lock_eval.jsonl", PACK_DIR / "eval" / "meetmesh_domain_eval.be.jsonl"]
    rows = []
    for path in files:
        rows.extend(normalize_item(item) for item in load_jsonl(path))
    if not rows or len({row['id'] for row in rows})!=len(rows):
        ap.error('eval set is empty or contains duplicate IDs')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    errors = 0
    weighted_pass = 0.0
    with args.output.open("w", encoding="utf-8") as out:
        for item in rows:
            try:
                answer = complete(args.base_url, args.model, args.api_key, item['messages'], args.max_tokens, args.temperature)
                result = evaluate_response(answer, item['checks'])
                passed += int(result['passed'])
                weighted_pass += item['weight']*int(result['passed'])
                record=dict(id=item['id'],category=item['category'],answer=answer,result=result,
                            manual_review='REQUIRED' if item['manual_criteria'] else 'NOT_REQUESTED',manual_criteria=item['manual_criteria'])
            except (requests.RequestException, EvaluationTransportError, ValueError) as exc:
                errors += 1
                record=dict(id=item['id'],answer=None,status='TRANSPORT_ERROR',error_type=type(exc).__name__)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False))
    summary = {"status":'INCOMPLETE' if errors else 'COMPLETE',"transport_errors":errors,
               "passed": passed, "total": len(rows), "pass_rate": passed/len(rows),
               "weighted_pass_rate":weighted_pass/sum(row['weight'] for row in rows),
               "scope":"automated language and keyword checks only; human criteria remain unscored", "output": str(args.output)}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(bool(errors) or passed!=len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
