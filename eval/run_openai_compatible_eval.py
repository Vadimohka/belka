#!/usr/bin/env python3
"""Run Belarusian language-lock eval against an OpenAI-compatible endpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import math
import requests

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR / "data_pipeline"))
from data_pipeline.detect_belarusian import detect_belarusian  # noqa: E402
from eval.protocol import (complete as request_completion, complete_with_metadata,
                           EvaluationTransportError, validate_sampling_options)
from tools.provenance import artifact, atomic_json


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_jsonl_with_identity(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Hash and parse the same byte snapshot, even if the file changes later."""
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f'eval input must be a nonempty file: {path}')
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f'eval input must be a nonempty file: {path}')
    identity = dict(path=str(path), size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    rows = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
    return rows, identity


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


def generation_metrics(text):
    """Descriptive Unicode word/repetition counts, not a fluency judgment."""
    words = re.findall(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", text.casefold())
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    return {
        'characters': len(text),
        'utf8_bytes': len(text.encode('utf-8')),
        'word_count': len(words),
        'immediate_word_repeats': sum(a == b for a, b in zip(words, words[1:])),
        'repeated_word_4gram_fraction': (len(grams) - len(set(grams))) / len(grams) if grams else 0.0,
    }


def sampling_confirmation(requested, reported):
    """An echoed cap may be lower because of context; absent echo proves nothing."""
    if not reported:
        return dict(status='UNCONFIRMED', missing=sorted(requested), differences={},
                    effective_max_tokens=None, max_tokens_reduced=False)
    missing = sorted(set(requested) - set(reported))
    differences = {}
    for name, value in requested.items():
        if name not in reported:
            continue
        actual = reported[name]
        if name == 'max_tokens' and type(actual) is int and 0 < actual <= value:
            continue
        equal = actual == value
        if name in ('temperature', 'min_p', 'repetition_penalty') and type(actual) in (int, float):
            equal = math.isclose(actual, value, rel_tol=1e-7, abs_tol=1e-8)
        if not equal:
            differences[name] = dict(requested=value, reported=actual)
    maximum = reported.get('max_tokens')
    reduced = type(maximum) is int and maximum < requested['max_tokens']
    return dict(status='MISMATCH' if differences else ('PARTIAL' if missing else 'CONFIRMED'),
                missing=missing, differences=differences,
                effective_max_tokens=maximum, max_tokens_reduced=reduced)


def _config_digest(config):
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run reproducible Belarusian eval against /chat/completions")
    ap.add_argument("--eval-file", type=Path, action="append", default=[])
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--api-key", default=os.environ.get("BELKA_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    ap.add_argument("--model", default=os.environ.get("MODEL", "be-local"))
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.2)
    seeds_group = ap.add_mutually_exclusive_group()
    seeds_group.add_argument("--seed", dest="seeds", type=int, action="append",
                             help="Fixed seed; repeat for a balanced prompt-by-seed evaluation.")
    seeds_group.add_argument("--seeds", dest="seeds", type=int, nargs="+",
                             help="Equivalent list of distinct fixed seeds.")
    ap.add_argument("--top-k", type=int, help="Optional endpoint extension, sent only when specified (0..200).")
    ap.add_argument("--min-p", type=float, help="Optional endpoint extension (0..1).")
    ap.add_argument("--repetition-penalty", type=float, help="Optional endpoint extension (1..2).")
    ap.add_argument("--repetition-window", type=int, help="Optional endpoint extension (1..512).")
    ap.add_argument("--variant", default="default", help="Label for comparing separate A/B runs.")
    ap.add_argument("--checkpoint", type=Path, help="Optional local checkpoint file to fingerprint.")
    ap.add_argument("--tokenizer", type=Path, help="Optional local tokenizer file to fingerprint.")
    ap.add_argument("--runtime-manifest", type=Path, help="Optional local runtime manifest to fingerprint.")
    ap.add_argument("--output", type=Path, default=Path("eval_results.jsonl"))
    args = ap.parse_args(argv)

    seeds = args.seeds if args.seeds is not None else [None]
    sampling = dict(max_tokens=args.max_tokens, temperature=args.temperature)
    extensions = {key: value for key, value in {
        'top_k': args.top_k, 'min_p': args.min_p,
        'repetition_penalty': args.repetition_penalty, 'repetition_window': args.repetition_window,
    }.items() if value is not None}
    sampling.update(extensions)
    try:
        validate_sampling_options(sampling)
        for seed in seeds:
            if seed is not None:
                validate_sampling_options({'seed': seed})
        if len(seeds) != len(set(seeds)):
            raise ValueError('seeds must be distinct')
        if not args.variant.strip():
            raise ValueError('variant must not be empty')
    except ValueError as exc:
        ap.error(str(exc))

    files = args.eval_file or [
        PACK_DIR / "eval" / "belarusian_language_lock_eval.jsonl",
        PACK_DIR / "eval" / "meetmesh_domain_eval.be.jsonl",
    ]
    summary_path = args.output.with_suffix('.summary.json')
    manifest_path = args.output.with_suffix('.manifest.json')
    supplied_paths = dict(checkpoint=args.checkpoint, tokenizer=args.tokenizer,
                          runtime_manifest=args.runtime_manifest)
    inputs = {path.resolve() for path in [*files, *[p for p in supplied_paths.values() if p is not None]]}
    outputs = {path.resolve() for path in (args.output, summary_path, manifest_path)}
    if len(outputs) != 3 or inputs & outputs:
        ap.error('output paths must be distinct from all input artifacts')

    try:
        loaded = [load_jsonl_with_identity(path) for path in files]
        eval_files = [identity for _, identity in loaded]
        supplied = {name: artifact(path) if path is not None else None for name, path in supplied_paths.items()}
        rows = [normalize_item(item) for items, _ in loaded for item in items]
    except (OSError, ValueError, TypeError) as exc:
        ap.error(f'invalid evaluation inputs: {exc}')
    if not rows or len({row['id'] for row in rows}) != len(rows):
        ap.error('eval set is empty or contains duplicate IDs')

    config = dict(model=args.model, variant=args.variant, seeds=seeds, requested_sampling=sampling)
    manifest = dict(
        schema_version=1, kind='belka_evaluation_run', status='PREPARED',
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config, config_sha256=_config_digest(config), eval_files=eval_files,
        supplied_artifacts=supplied,
        artifact_scope='Fingerprints of explicitly supplied local files; not proof of the remote model identity.',
        endpoint_identity='Recorded only when the endpoint reports model metadata; credentials and URL are not stored.',
        outputs=None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, manifest, overwrite=True)

    passed = errors = sampling_mismatches = 0
    weighted_pass = 0.0
    confirmation_counts = {name: 0 for name in ('CONFIRMED', 'PARTIAL', 'UNCONFIRMED', 'MISMATCH')}
    metric_sums = {name: 0.0 for name in generation_metrics('')}
    responses_with_immediate_repeats = 0
    per_seed = []
    weight_per_seed = sum(item['weight'] for item in rows)
    started = time.perf_counter()
    with args.output.open("w", encoding="utf-8") as out:
        for seed in seeds:
            seed_passed = seed_errors = 0
            seed_weighted_pass = 0.0
            requested = dict(sampling)
            kwargs = dict(extensions)
            if seed is not None:
                requested['seed'] = kwargs['seed'] = seed
            for item in rows:
                record = dict(
                    id=item['id'], category=item['category'], variant=args.variant,
                    seed=seed, requested_sampling=dict(requested), generation=None,
                    server_model=None, sampling_confirmation=None,
                )
                began = time.perf_counter()
                try:
                    response = complete_with_metadata(
                        args.base_url, args.model, args.api_key, item['messages'],
                        args.max_tokens, args.temperature, **kwargs,
                    )
                    result = evaluate_response(response.text, item['checks'])
                    text_metrics = generation_metrics(response.text)
                    for name, value in text_metrics.items():
                        metric_sums[name] += value
                    responses_with_immediate_repeats += text_metrics['immediate_word_repeats'] > 0
                    confirmation = sampling_confirmation(requested, response.generation)
                    confirmation_counts[confirmation['status']] += 1
                    sampling_mismatches += confirmation['status'] == 'MISMATCH'
                    is_passed = int(result['passed'])
                    passed += is_passed
                    seed_passed += is_passed
                    weighted_pass += item['weight'] * is_passed
                    seed_weighted_pass += item['weight'] * is_passed
                    record.update(
                        answer=response.text, result=result, generation=response.generation,
                        server_model=response.model, sampling_confirmation=confirmation,
                        text_metrics=text_metrics,
                        manual_review='REQUIRED' if item['manual_criteria'] else 'NOT_REQUESTED',
                        manual_criteria=item['manual_criteria'],
                    )
                except (requests.RequestException, EvaluationTransportError, ValueError) as exc:
                    errors += 1
                    seed_errors += 1
                    record.update(answer=None, status='TRANSPORT_ERROR', error_type=type(exc).__name__)
                record['elapsed_seconds'] = time.perf_counter() - began
                out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                print(json.dumps(record, ensure_ascii=False, allow_nan=False))
            per_seed.append(dict(
                seed=seed, passed=seed_passed, total=len(rows), transport_errors=seed_errors,
                pass_rate=seed_passed / len(rows),
                weighted_pass_rate=seed_weighted_pass / weight_per_seed,
            ))

    total = len(rows) * len(seeds)
    completed = total - errors
    summary = {
        "status": 'INCOMPLETE' if errors else 'COMPLETE',
        "transport_errors": errors, "sampling_mismatches": sampling_mismatches,
        "passed": passed, "total": total, "unique_prompts": len(rows), "seeds": seeds,
        "pass_rate": passed / total,
        "weighted_pass_rate": weighted_pass / (weight_per_seed * len(seeds)),
        "per_seed": per_seed, "sampling_confirmation_counts": confirmation_counts,
        "text_metrics": {
            "completed_responses": completed,
            "means": {name: value / completed if completed else None for name, value in metric_sums.items()},
            "fraction_with_immediate_repeats": responses_with_immediate_repeats / completed if completed else None,
            "scope": "Unweighted descriptive counts over completed responses; not a fluency or factuality score.",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Automated language and keyword checks only; human criteria remain unscored. "
                 "Repeated seeds share prompts and are not independent new holdout questions.",
        "sampling_scope": "CONFIRMED means the server reported matching requested settings "
                          "(a lower max_tokens cap is allowed), not cross-device determinism.",
        "output": str(args.output), "manifest": str(manifest_path),
    }
    atomic_json(summary_path, summary, overwrite=True)
    manifest.update(
        status=summary['status'], completed_at=datetime.now(timezone.utc).isoformat(),
        outputs=dict(answers=artifact(args.output), summary=artifact(summary_path)),
    )
    atomic_json(manifest_path, manifest, overwrite=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(bool(errors) or bool(sampling_mismatches) or passed != total)


if __name__ == "__main__":
    raise SystemExit(main())
