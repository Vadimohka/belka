#!/usr/bin/env python3
"""Validate complete text-only SFT conversations without executing model code.

The default retains historical assistant-only language checking. --strict-all
checks user and system text too; --schema-only makes NO language claim. A language
heuristic is not a proof of language purity. Structural errors always fail.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
from data_pipeline.contracts import conversation_messages, strict_json_loads, content_hash
from data_pipeline.detect_belarusian import detect_belarusian


def row_messages(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get('messages'), list):
        return obj['messages']
    return None


def validate_file(path: Path, *, allow_user_nonbe=True, strict_all=False,
                  min_assistant_score=2.0, verbose=False, schema_only=False, reviewed_texts=None) -> dict:
    if not math.isfinite(min_assistant_score):
        raise ValueError('language threshold must be finite')
    stats = dict(path=str(path), rows=0, messages=0, errors=0, warnings=0,
                 assistant_language_failures=0, user_language_failures=0,
                 system_language_failures=0, reviewed_language_exceptions=0)
    def error(message):
        stats['errors'] += 1
        print(f'ERROR {path}: {message}')
    try:
        with Path(path).open(encoding='utf-8', errors='strict') as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                stats['rows'] += 1
                try:
                    messages = conversation_messages(strict_json_loads(line))
                except (ValueError, RecursionError) as exc:
                    error(f'line {lineno}: {exc}')
                    continue
                for index, msg in enumerate(messages, 1):
                    stats['messages'] += 1
                    role, content = msg['role'], msg['content']
                    if schema_only or not (strict_all or role == 'assistant' or (role == 'user' and not allow_user_nonbe)):
                        continue
                    det = detect_belarusian(content, min_chars=10, allow_short=True,
                                           accept_threshold=min_assistant_score, quarantine_threshold=1.0)
                    if det.score < min_assistant_score:
                        if role+':'+content_hash(content) in (reviewed_texts or set()):
                            stats['reviewed_language_exceptions'] += 1
                            continue
                        # Retained historical ambiguity policy, explicitly reported.
                        benign = det.reasons == ['mostly_cyrillic', 'no_bel_specific_letters']
                        fatal = strict_all or (role == 'assistant' and not benign)
                        stats['errors' if fatal else 'warnings'] += 1
                        if fatal:
                            stats[role + '_language_failures'] += 1
                        print(f"{'ERROR' if fatal else 'WARN'} {path}:{lineno}: message {index} role={role} score={det.score} reasons={det.reasons}")
                    elif verbose:
                        print(f'OK {path}:{lineno}: message {index} role={role} score={det.score}')
    except (OSError, UnicodeError) as exc:
        error(f'cannot read complete UTF-8 input: {exc}')
    if stats['rows'] == 0:
        error('empty dataset')
    stats['language_scope'] = 'not_checked' if schema_only else ('all_roles' if strict_all else 'historical_assistant_policy')
    return stats


def load_language_review(path):
    data = strict_json_loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != 'belka-language-review-v1':
        raise ValueError('unsupported language review schema')
    entries = data.get('entries')
    if not isinstance(entries, dict): raise ValueError('language review entries must be a mapping')
    for key, entry in entries.items():
        if entry.get('role') not in ('system','user','assistant') or not isinstance(entry.get('text'),str):
            raise ValueError('invalid language review entry')
        if key != entry['role']+':'+content_hash(entry['text']):
            raise ValueError('language review text hash mismatch')
    return set(entries)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('paths', nargs='+', type=Path)
    ap.add_argument('--allow-user-nonbe', action='store_true', help='historical compatibility; not a Belarusian-only training policy')
    ap.add_argument('--check-user-language', action='store_true')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--strict-all', action='store_true')
    mode.add_argument('--schema-only', action='store_true')
    ap.add_argument('--min-assistant-score', type=float, default=2.0)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--language-review', type=Path, help='versioned exact role/text review; never a blanket language bypass')
    args = ap.parse_args()
    if not math.isfinite(args.min_assistant_score):
        ap.error('--min-assistant-score must be finite')
    if args.strict_all and args.allow_user_nonbe:
        ap.error('--strict-all conflicts with --allow-user-nonbe')
    review = set()
    if args.language_review:
        review = load_language_review(args.language_review)
    stats = [validate_file(p, allow_user_nonbe=args.allow_user_nonbe or not args.check_user_language,
                          strict_all=args.strict_all, schema_only=args.schema_only,
                          min_assistant_score=args.min_assistant_score, verbose=args.verbose, reviewed_texts=review) for p in args.paths]
    print(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False))
    return int(any(s['errors'] for s in stats))


if __name__ == '__main__':
    raise SystemExit(main())
