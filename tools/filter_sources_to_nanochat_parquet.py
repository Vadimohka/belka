#!/usr/bin/env python3
"""
Belarusian-only corpus filter with full record accounting, per-source thresholds,
exact+paragraph+simhash dedup, orthography split, and metadata-rich output.

Every input record MUST land in exactly one category:
  accepted, quarantine, rejected, skipped_empty, skipped_short,
  skipped_namespace, skipped_redirect, skipped_parse_error,
  skipped_duplicate_exact, skipped_duplicate_near

Assertion: raw_seen == sum(all categories)
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.contracts import HammingIndex, content_hash, group_split, corpus_generation, sha256_file, iter_jsonl as strict_iter_jsonl
from data_pipeline.normalize_text import normalize_text

from data_pipeline.source_policy import SOURCE_THRESHOLDS, SOURCE_ALIASES, BE_WORDS, RU_MARKERS, UK_MARKERS, detect_source, base_rejection
from data_pipeline.detect_belarusian import mixed_language_reasons

def score_be(text: str, source: str) -> tuple[float, list[str]]:
    """Score text for Belarusian-ness."""
    problems = mixed_language_reasons(text)
    if problems: return 0.0, problems
    low = text.lower()
    cyr = sum(map(len, re.findall(r'[а-яёіў]+', low)))
    lat = sum(map(len, re.findall(r'[a-z]+', low)))
    markers = sum(low.count(ch) for ch in 'ўі')
    words = re.findall(r"[а-яёіўʼ']+", low)
    be = sum(1 for w in words if w in BE_WORDS)
    ru = sum(1 for w in words if w in RU_MARKERS)
    uk = sum(w in UK_MARKERS for w in words) + sum(low.count(ch) for ch in 'єїґ')
    score = 0.0
    reasons = []
    if cyr >= 30:
        score += 0.25
        reasons.append('cyrillic')
    if lat > max(80, cyr * 0.25):
        score -= 0.35
        reasons.append('latin_penalty')
    if markers >= 2:
        score += 0.35
        reasons.append('be_letters')
    if be >= 2:
        score += 0.35
        reasons.append('be_words')
    if ru >= 4:
        score -= 0.25
        reasons.append('ru_words')
    if uk >= 2:
        score -= 0.25
        reasons.append('uk_markers')
    return score, reasons


class SimHash:
    """Simple word-based SimHash for near-dedup."""
    def __init__(self, bits=64):
        self.bits = bits

    def hash(self, text: str) -> int:
        words = re.findall(r'\b\w{3,}\b', text.lower())
        if not words:
            return 0
        v = [0] * self.bits
        for w in words:
            h = hashlib.md5(w.encode()).digest()
            h_int = int.from_bytes(h[:8], 'big')
            for i in range(self.bits):
                if (h_int >> i) & 1:
                    v[i] += 1
                else:
                    v[i] -= 1
        result = 0
        for i in range(self.bits):
            if v[i] > 0:
                result |= (1 << i)
        return result

    @staticmethod
    def hamming(a: int, b: int) -> int:
        return (a ^ b).bit_count()


def iter_jsonl(inp: Path):
    """Deterministic strict ingest; a bad file cannot disappear from accounting."""
    paths = sorted(set(inp.rglob('*.jsonl')) | set(inp.rglob('*.jsonl.gz')))
    if not paths:
        raise ValueError(f'no JSONL source files found in {inp}')
    for path in paths:
        source_sha = sha256_file(path)
        for lineno, obj in strict_iter_jsonl(path):
            if not isinstance(obj, dict):
                raise ValueError(f'{path}:{lineno}: expected an object')
            text = obj.get('text', obj.get('content'))
            if not isinstance(text, str):
                raise ValueError(f'{path}:{lineno}: missing/non-string text field')
            obj = dict(obj, input_file=str(path.relative_to(inp)), input_line=lineno, input_sha256=source_sha)
            yield text, str(path), obj
        if sha256_file(path) != source_sha:
            raise ValueError(f'input changed during ingestion: {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--input-dir')
    ap.add_argument('--output-dir')
    ap.add_argument('--val-ratio', type=float, default=0.02)
    ap.add_argument('--write-accounting', action='store_true', default=True)
    ap.add_argument('--strict-source-thresholds', action='store_true', default=True)
    ap.add_argument('--split-orthography', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--dedup', default='exact,paragraph,simhash')
    ap.add_argument('--max-docs', type=int, default=0)
    args = ap.parse_args()
    if not 0 < args.val_ratio < 1:
        ap.error('--val-ratio must be finite and strictly between 0 and 1')
    if args.max_docs < 0:
        ap.error('--max-docs must be nonnegative')
    modes = set(filter(None, args.dedup.split(',')))
    if not modes <= {'exact', 'paragraph', 'simhash'}:
        ap.error('unknown --dedup mode')

    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit('Missing pandas/pyarrow. Install inside .venv.') from e

    pack = Path(args.pack_dir).resolve()
    inp = Path(args.input_dir).resolve() if args.input_dir else pack / 'data_input/be_texts'
    out = Path(os.path.abspath(args.output_dir)) if args.output_dir else pack / '.workspace/nanochat_base/base_data_climbmix'
    report_dir = pack / 'reports'

    if pack not in inp.parents and inp != pack:
        raise SystemExit(f'input outside pack: {inp}')
    if out == pack or not out.resolve().is_relative_to(pack):
        raise SystemExit(f'output outside pack: {out}')

    # The live corpus is never opened for writing; publish a new generation below.

    # ----- Accounting -----
    raw_seen = 0
    parsed_ok = 0
    skipped_parse_error = 0
    skipped_no_text_field = 0
    skipped_empty = 0
    skipped_short = 0
    skipped_namespace = 0
    skipped_redirect = 0
    skipped_duplicate_exact = 0
    skipped_duplicate_near = 0

    accepted_records = []
    quarantine_records = []
    rejected_records = []
    source_stats: dict = defaultdict(lambda: Counter())

    exact_seen: set = set()
    paragraph_hashes: set = set()
    simhasher = SimHash(64)
    # Seven disjoint bands guarantee recall within Hamming radius six.
    # Candidate matches are verified using the complete 64-bit signature.
    simhash_index = HammingIndex(radius=6)

    for text, filepath, obj in iter_jsonl(inp):
        if args.max_docs and raw_seen >= args.max_docs:
            break
        raw_seen += 1
        source = str(obj.get('source') or detect_source(obj['input_file']))
        if source not in SOURCE_THRESHOLDS: source = detect_source(obj['input_file'])
        thresh = SOURCE_THRESHOLDS.get(source, {"min_score": 0.45, "min_chars": 80})
        min_score = thresh.get("min_score", 0.45)
        min_chars = thresh.get("min_chars", 80)
        orthography = thresh.get("orthography")

        source_stats[source]['raw_seen'] += 1

        # Parse error
        if text is None:
            skipped_parse_error += 1
            source_stats[source]['skipped_parse_error'] += 1
            continue

        parsed_ok += 1

        # Clean text
        text = normalize_text(text)  # preserve paragraph boundaries before dedup

        # Empty
        if not text:
            skipped_empty += 1
            source_stats[source]['skipped_empty'] += 1
            continue

        # Short
        if len(text) < min_chars:
            skipped_short += 1
            source_stats[source]['skipped_short'] += 1
            continue

        # Exclude eval-only/non-prose sources before they can influence dedup.
        if base_rejection(source, obj, obj['input_file']):
            skipped_namespace += 1
            source_stats[source]['skipped_namespace'] += 1
            continue
        if obj.get('is_redirect') or obj.get('redirect'):
            skipped_redirect += 1
            source_stats[source]['skipped_redirect'] += 1
            continue
        if obj.get('namespace', obj.get('ns', 0)) not in (0, '0', None):
            skipped_namespace += 1
            source_stats[source]['skipped_namespace'] += 1
            continue
        sc, reasons = score_be(text, source)
        h_exact = content_hash(text)
        paragraphs = [(content_hash(p),len(p)) for p in text.split('\n') if len(p.strip()) >= 30]
        sh = simhasher.hash(text) if 'simhash' in modes else None
        # Only accepted data populates duplicate indexes. A rejected document
        # must not cause a later good document to be discarded.
        if sc >= min_score and source != 'unknown':
            if 'exact' in modes and h_exact in exact_seen:
                skipped_duplicate_exact += 1
                source_stats[source]['skipped_duplicate_exact'] += 1
                continue
            repeated = sum(length for key,length in paragraphs if key in paragraph_hashes)
            para_duplicate = bool(paragraphs) and repeated >= .8 * sum(length for _,length in paragraphs)
            if (('paragraph' in modes and para_duplicate)
                    or ('simhash' in modes and simhash_index.contains_near(sh))):
                skipped_duplicate_near += 1
                source_stats[source]['skipped_duplicate_near'] += 1
                continue
            exact_seen.add(h_exact)
            paragraph_hashes.update(key for key,_ in paragraphs)
            if sh is not None: simhash_index.add(sh)

        rec = {
            'text': text,
            'source': source,
            'source_path': str(filepath),
            'input_file': obj['input_file'],
            'input_line': obj['input_line'],
            'input_sha256': obj['input_sha256'],
            'score': sc,
            'reasons': reasons,
            'chars': len(text),
            'orthography': orthography or obj.get('orthography'),
            'doc_id': f"{source}_{hashlib.sha256(text.encode()).hexdigest()[:12]}",
        }

        if sc >= min_score and source != 'unknown':
            accepted_records.append(rec)
            source_stats[source]['accepted'] += 1
        elif source == 'unknown' or sc >= min_score - 0.20:
            quarantine_records.append(rec)
            source_stats[source]['quarantine'] += 1
        else:
            rejected_records.append(rec)
            source_stats[source]['rejected'] += 1

    # Stable content-group split. It does not depend on source order or score.
    if len(accepted_records) < 2:
        raise ValueError('not enough accepted documents; existing corpus retained')
    split_records = {'train': [], 'val': []}
    for record in accepted_records:
        record['split'] = group_split(content_hash(record['text']), args.val_ratio)
        split_records[record['split']].append(record)
    if not all(split_records.values()):
        raise ValueError('stable split produced an empty partition; use more data or an explicit different ratio')
    for records in split_records.values():
        records.sort(key=lambda r: r['doc_id'])
    narkamauka = [r for r in accepted_records if r.get('orthography') == 'narkamauka']
    tarask = [r for r in accepted_records if r.get('orthography') == 'tarask']
    # ----- Accounting assertion -----
    total_accounted = (len(accepted_records) + len(quarantine_records) + len(rejected_records) +
                       skipped_parse_error + skipped_no_text_field + skipped_empty + skipped_short +
                       skipped_namespace + skipped_redirect + skipped_duplicate_exact + skipped_duplicate_near)
    if total_accounted != raw_seen:
        raise ValueError(f"ACCOUNTING FAIL: raw_seen={raw_seen} != total_accounted={total_accounted}")

    with corpus_generation(out) as stage:
        for split, records in split_records.items():
            pd.DataFrame(records).to_parquet(stage / f'{split}_00000.parquet', index=False)
        if args.split_orthography:
            reference = stage / 'orthography_reference'
            reference.mkdir()
            for orth in sorted({r.get('orthography') or 'unspecified' for r in accepted_records}):
                if not re.fullmatch(r'[a-z_-]+', orth):
                    raise ValueError(f'unsafe orthography label: {orth!r}')
                for split, records in split_records.items():
                    subset = [r for r in records if (r.get('orthography') or 'unspecified') == orth]
                    if subset:
                        pd.DataFrame(subset).to_parquet(reference / f'{orth}_{split}.parquet', index=False)
        manifest = {'schema':'belka-corpus-v2','split_policy':'belka-content-v2',
                    'val_ratio':args.val_ratio,'dedup':sorted(modes),
                    'train_docs':len(split_records['train']),'val_docs':len(split_records['val']),
                    'reference_files_are_training_shards':False,
                    'files':[{"path": p.name, "sha256":sha256_file(p)} for p in sorted(stage.glob('*.parquet'))],
                    'pipeline_sha256':sha256_file(Path(__file__)),
                    'raw_seen':raw_seen,'total_accounted':total_accounted,
                    'max_docs_requested':args.max_docs,
                    'input_scope':'bounded-prefix' if args.max_docs else 'all-discovered-jsonl'}
        (stage/'_BUILD_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    tp, vp = out/'train_00000.parquet', out/'val_00000.parquet'


    # ----- Accounting report -----
    accounting = {
        'PACK_DIR_REALPATH': str(pack),
        'raw_seen': raw_seen,
        'accepted': len(accepted_records),
        'quarantine': len(quarantine_records),
        'rejected': len(rejected_records),
        'skipped_empty': skipped_empty,
        'skipped_short': skipped_short,
        'skipped_duplicate_exact': skipped_duplicate_exact,
        'skipped_duplicate_near': skipped_duplicate_near,
        'skipped_parse_error': skipped_parse_error,
        'skipped_no_text_field': skipped_no_text_field,
        'skipped_namespace': skipped_namespace,
        'skipped_redirect': skipped_redirect,
        'TOTAL_ACCOUNTED': total_accounted,
        'ACCOUNTING_ASSERTION': 'PASS' if total_accounted == raw_seen else 'FAIL',
        'train_narkamauka': sum(r['split'] == 'train' for r in narkamauka),
        'val_narkamauka': sum(r['split'] == 'val' for r in narkamauka),
        'train_tarask': sum(r['split'] == 'train' for r in tarask),
        'val_tarask': sum(r['split'] == 'val' for r in tarask),
    }

    # Per-source stats
    per_source = {}
    for src, stats in sorted(source_stats.items()):
        per_source[src] = dict(stats)
        per_source[src]['threshold_min_score'] = SOURCE_THRESHOLDS.get(src, {}).get('min_score', 0.45)
        per_source[src]['threshold_min_chars'] = SOURCE_THRESHOLDS.get(src, {}).get('min_chars', 80)

    accounting['per_source'] = per_source

    # Reports are secondary views; the generation manifest is authoritative.
    report_dir.mkdir(parents=True, exist_ok=True)
    # Write accounting report
    (report_dir / 'source_filter_report.json').write_text(
        json.dumps(accounting, ensure_ascii=False, indent=2), encoding='utf-8')

    # Write quarantine
    with open(report_dir / 'source_quarantine.jsonl', 'w', encoding='utf-8') as f:
        for r in quarantine_records[:10000]:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Write rejected sample
    with open(report_dir / 'source_rejected_sample.jsonl', 'w', encoding='utf-8') as f:
        for r in rejected_records[:1000]:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # ----- Summary -----
    print(f"PACK_DIR_REALPATH={pack}")
    print(f"RAW_SEEN={raw_seen}")
    print(f"TOTAL_ACCOUNTED={total_accounted}")
    print(f"ACCOUNTING_ASSERTION={'PASS' if total_accounted == raw_seen else 'FAIL'}")
    print(f"ACCEPTED={len(accepted_records)}")
    print(f"QUARANTINE={len(quarantine_records)}")
    print(f"REJECTED={len(rejected_records)}")
    print(f"SKIPPED_EMPTY={skipped_empty}")
    print(f"SKIPPED_SHORT={skipped_short}")
    print(f"SKIPPED_DUPLICATE_EXACT={skipped_duplicate_exact}")
    print(f"SKIPPED_DUPLICATE_NEAR={skipped_duplicate_near}")
    print(f"SKIPPED_PARSE_ERROR={skipped_parse_error}")
    print(f"TARASK_SPLIT={'PASS' if args.split_orthography else 'NOT_REQUESTED'}")
    print(f"TRAIN_PARQUET={tp}")
    print(f"VAL_PARQUET={vp}")
    print(f"SOURCE_MANIFEST={report_dir / 'source_filter_report.json'}")


if __name__ == '__main__':
    main()
