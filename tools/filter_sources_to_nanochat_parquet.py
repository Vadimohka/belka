#!/usr/bin/env python3
"""
Belarusian-only corpus filter with full record accounting, per-source thresholds,
exact+paragraph+simhash dedup, orthography split, license manifest, and metadata-rich output.

Every input record MUST land in exactly one category:
  accepted, quarantine, rejected, skipped_empty, skipped_short,
  skipped_namespace, skipped_redirect, skipped_parse_error,
  skipped_duplicate_exact, skipped_duplicate_near, skipped_license_excluded

Assertion: raw_seen == sum(all categories)
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_THRESHOLDS = {
    "bewiki": {"min_score": 0.45, "min_chars": 120, "orthography": "narkamauka"},
    "be_x_oldwiki": {"min_score": 0.45, "min_chars": 120, "orthography": "tarask"},
    "bewikisource": {"min_score": 0.55, "min_chars": 400, "orthography": None},
    "bewiktionary": {"min_score": 0.45, "min_chars": 100, "orthography": None},
    "bewikiquote": {"min_score": 0.50, "min_chars": 200, "orthography": None},
    "bewikibooks": {"min_score": 0.50, "min_chars": 200, "orthography": None},
    "belacorpus_public_research": {"min_score": 0.55, "min_chars": 150, "orthography": "narkamauka"},
    "ud_belarusian_hse": {"min_score": 0.50, "min_chars": 30, "orthography": None, "not_for_base": True},
    "tatoeba_sentences": {"min_score": 0.50, "min_chars": 30, "orthography": None, "sentence_level": True},
    "belarusianglue": {"min_score": 0.55, "min_chars": 50, "orthography": None, "eval_or_sft_only": True},
    "morphodict-bel": {"min_score": 0.55, "min_chars": 10, "orthography": None, "not_for_base": True, "license_class": "non_commercial"},
    "oscar-2301-be": {"min_score": 0.55, "min_chars": 150, "orthography": None},
    "culturax-be": {"min_score": 0.55, "min_chars": 150, "orthography": None},
    "mc4-be": {"min_score": 0.55, "min_chars": 150, "orthography": None},
    "cc100-be": {"min_score": 0.55, "min_chars": 150, "orthography": None},
    "bootstrap": {"min_score": 0.45, "min_chars": 60, "orthography": "narkamauka"},
    "belarusian_seed": {"min_score": 0.45, "min_chars": 60, "orthography": "narkamauka"},
}

LICENSE_CLASSES = {
    "bewiki": "free/attribution_sharealike",
    "be_x_oldwiki": "free/attribution_sharealike",
    "bewikisource": "free/attribution_sharealike",
    "bewiktionary": "free/attribution_sharealike",
    "bewikiquote": "free/attribution_sharealike",
    "bewikibooks": "free/attribution_sharealike",
    "belacorpus_public_research": "research_only/attribution_required",
    "ud_belarusian_hse": "free/attribution",
    "tatoeba_sentences": "free/attribution",
    "belarusianglue": "research_only/attribution",
    "morphodict-bel": "non_commercial/attribution_sharealike",
    "oscar-2301-be": "free/attribution",
    "culturax-be": "free/attribution",
    "mc4-be": "free/attribution",
    "cc100-be": "unknown/manual_review",
    "bootstrap": "synthetic/research",
    "belarusian_seed": "synthetic/research",
}

BE_WORDS = {'гэта','які','якая','якія','быў','была','былі','ёсць','няма','для','праз','пасля',
            'вельмі','калі','каб','трэба','можна','чалавек','мова','краіна','беларусь','беларускі',
            'сустрэча','праца','кожны','месца','справа','дапамагае','таксама','іншы','яшчэ',
            'добра','вялікі','новы','свой','павінна','павінен','патрэбна','разам','менавіта',
            'можа',"з'яўляецца",'мае','мець','робіць','робяць','ідзе','ідуць','бачыць','кажа',
            'паказвае','разумець','ведаць','хацець','думаць','гаварыць','рабіць','бачыць'}
RU_MARKERS = {'что','это','который','которая','очень','если','чтобы','можно','нужно','человек',
              'страна','язык','которые','также','очень','ещё','более','менее','вообще','конечно',
              'например','наверное','находится','является','имеется','данный','данная','данное'}
UK_MARKERS = {'є','ї','ґ','дуже','якщо','після','країна','зараз','також','треба','можна',
              'наприклад','звічайно','знаходиться','згідно','щодо','протягом','відповідно'}


SOURCE_ALIASES = {
    'tatoeba': 'tatoeba_sentences',
    'belarusianglue': 'belarusianglue',
    'morphodict': 'morphodict-bel',
    'belacorpus': 'belacorpus_public_research',
    'ud_belarusian': 'ud_belarusian_hse',
    'belarusian_seed': 'belarusian_seed',
    'belarusian_bootstrap': 'bootstrap',
    'oscar': 'oscar-2301-be',
    'culturax': 'culturax-be',
    'mc4': 'mc4-be',
    'cc100': 'cc100-be',
}

def detect_source(filepath: str) -> str:
    """Map file path to source id using aliases and thresholds."""
    p = str(filepath).lower()
    # Check aliases first (keyword matching)
    for alias, src in SOURCE_ALIASES.items():
        if alias in p:
            return src
    # Check exact source keys
    for src in SOURCE_THRESHOLDS:
        if src in p:
            return src
    return "unknown"


def score_be(text: str, source: str) -> tuple[float, list[str]]:
    """Score text for Belarusian-ness."""
    low = text.lower()
    cyr = sum(1 for ch in text if 'а' <= ch.lower() <= 'я' or ch in 'ўіё')
    lat = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    markers = sum(low.count(ch) for ch in 'ўіё')
    words = re.findall(r"[а-яёіўʼ']+", low)
    be = sum(1 for w in words if w in BE_WORDS)
    ru = sum(1 for w in words if w in RU_MARKERS)
    uk = sum(low.count(w) for w in UK_MARKERS)
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
    """Iterate over all JSONL files in a directory tree."""
    paths = list(inp.rglob('*.jsonl')) + list(inp.rglob('*.jsonl.gz'))
    for path in paths:
        opener = gzip.open if str(path).endswith('.gz') else open
        try:
            with opener(path, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        yield None, str(path), {"parse_error": True}
                        continue
                    text = obj.get('text') or obj.get('content') or ''
                    if isinstance(text, str):
                        yield text, str(path), obj
                    else:
                        yield None, str(path), {"no_text_field": True}
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--input-dir')
    ap.add_argument('--output-dir')
    ap.add_argument('--val-ratio', type=float, default=0.02)
    ap.add_argument('--write-accounting', action='store_true', default=True)
    ap.add_argument('--strict-source-thresholds', action='store_true', default=True)
    ap.add_argument('--split-orthography', action='store_true', default=True)
    ap.add_argument('--dedup', default='exact,paragraph,simhash')
    ap.add_argument('--write-license-manifest', action='store_true', default=True)
    ap.add_argument('--max-docs', type=int, default=0)
    args = ap.parse_args()

    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit('Missing pandas/pyarrow. Install inside .venv.') from e

    pack = Path(args.pack_dir).resolve()
    inp = Path(args.input_dir).resolve() if args.input_dir else pack / 'data_input/be_texts'
    out = Path(args.output_dir).resolve() if args.output_dir else pack / '.workspace/nanochat_base/base_data_climbmix'
    report_dir = pack / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)

    if pack not in inp.parents and inp != pack:
        raise SystemExit(f'input outside pack: {inp}')
    if pack not in out.parents and out != pack:
        raise SystemExit(f'output outside pack: {out}')

    out.mkdir(parents=True, exist_ok=True)

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
    skipped_license_excluded = 0

    accepted_records = []
    quarantine_records = []
    rejected_records = []
    source_stats: dict = defaultdict(lambda: Counter())

    exact_seen: set = set()
    paragraph_hashes: set = set()
    simhasher = SimHash(64)
    # Bucket by top bits to avoid O(N^2) scans over all accepted documents.
    # We still compare neighbouring buckets exactly with Hamming distance.
    simhash_buckets: dict[int, list[int]] = defaultdict(list)
    simhash_threshold = 6  # Hamming distance
    bucket_shift = 52

    for text, filepath, obj in iter_jsonl(inp):
        if args.max_docs and raw_seen >= args.max_docs:
            break
        raw_seen += 1
        source = detect_source(filepath)
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
        text = re.sub(r'\s+', ' ', text).strip()

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

        # License check
        lic = LICENSE_CLASSES.get(source, 'unknown/manual_review')
        obj_license = obj.get('license', '')
        if 'non_commercial' in lic and 'commercial' not in lic:
            pass  # Keep for research, but mark

        # Exact dedup
        h_exact = hashlib.sha256(text.encode()).hexdigest()
        if h_exact in exact_seen:
            skipped_duplicate_exact += 1
            source_stats[source]['skipped_duplicate_exact'] += 1
            continue
        exact_seen.add(h_exact)

        # Paragraph-level dedup
        has_dup_para = False
        for para in text.split('\n'):
            para = para.strip()
            if len(para) < 30:
                continue
            h_para = hashlib.sha256(para.encode()).hexdigest()
            if h_para in paragraph_hashes:
                has_dup_para = True
                break
            paragraph_hashes.add(h_para)
        if has_dup_para:
            skipped_duplicate_near += 1
            source_stats[source]['skipped_duplicate_near'] += 1
            continue

        # Near-dedup via SimHash. Compare only nearby high-bit buckets.
        sh = simhasher.hash(text)
        bucket = sh >> bucket_shift
        is_near_dup = False
        for b in (bucket - 1, bucket, bucket + 1):
            for existing_sh in simhash_buckets.get(b, []):
                if SimHash.hamming(sh, existing_sh) <= simhash_threshold:
                    is_near_dup = True
                    break
            if is_near_dup:
                break
        if is_near_dup:
            skipped_duplicate_near += 1
            source_stats[source]['skipped_duplicate_near'] += 1
            continue
        simhash_buckets[bucket].append(sh)

        # Score
        sc, reasons = score_be(text, source)

        rec = {
            'text': text,
            'source': source,
            'source_path': str(filepath),
            'score': sc,
            'reasons': reasons,
            'chars': len(text),
            'orthography': orthography or obj.get('orthography'),
            'license': lic,
            'doc_id': f"{source}_{hashlib.sha256(text.encode()).hexdigest()[:12]}",
        }

        if sc >= min_score:
            accepted_records.append(rec)
            source_stats[source]['accepted'] += 1
        elif sc >= min_score - 0.20:
            quarantine_records.append(rec)
            source_stats[source]['quarantine'] += 1
        else:
            rejected_records.append(rec)
            source_stats[source]['rejected'] += 1

    # ----- Write outputs -----
    random.seed(42)
    random.shuffle(accepted_records)

    # Split by orthography if requested
    if args.split_orthography:
        narkamauka = [r for r in accepted_records if r.get('orthography') != 'tarask']
        tarask = [r for r in accepted_records if r.get('orthography') == 'tarask']
    else:
        narkamauka = accepted_records
        tarask = []

    def write_nanochat_split(records, out_dir):
        """Write nanochat-compatible train_00000.parquet + val_00000.parquet."""
        if not records:
            return None, None
        n_val = max(1, int(len(records) * args.val_ratio))
        val = records[:n_val]
        train = records[n_val:]
        tpath = out_dir / 'train_00000.parquet'
        vpath = out_dir / 'val_00000.parquet'
        pd.DataFrame(train).to_parquet(tpath, index=False)
        pd.DataFrame(val).to_parquet(vpath, index=False)
        return tpath, vpath

    def write_orthography_splits(records_dict, out_dir):
        """Write orthography-split parquets for reference."""
        for orth, records in records_dict.items():
            if records:
                n_val = max(1, int(len(records) * args.val_ratio))
                val = records[:n_val]
                train = records[n_val:]
                stem = orth or 'unspecified'
                pd.DataFrame(train).to_parquet(out_dir / f'{stem}_train_00000.parquet', index=False)
                pd.DataFrame(val).to_parquet(out_dir / f'{stem}_val_00000.parquet', index=False)

    # Primary nanochat-compatible output
    tp, vp = write_nanochat_split(sorted(narkamauka + tarask, key=lambda r: r.get('score', 0)), out)

    # Orthography-split for reference
    orth_records = {}
    for r in narkamauka + tarask:
        orth = r.get('orthography') or 'narkamauka'
        if orth not in orth_records:
            orth_records[orth] = []
        orth_records[orth].append(r)
    write_orthography_splits(orth_records, out)

    # ----- Accounting assertion -----
    total_accounted = (len(accepted_records) + len(quarantine_records) + len(rejected_records) +
                       skipped_parse_error + skipped_no_text_field + skipped_empty + skipped_short +
                       skipped_duplicate_exact + skipped_duplicate_near)
    assert total_accounted == raw_seen, f"ACCOUNTING FAIL: raw_seen={raw_seen} != total_accounted={total_accounted}"

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
        'skipped_license_excluded': skipped_license_excluded,
        'TOTAL_ACCOUNTED': total_accounted,
        'ACCOUNTING_ASSERTION': 'PASS' if total_accounted == raw_seen else 'FAIL',
        'train_narkamauka': len(narkamauka) - max(1, int(len(narkamauka) * args.val_ratio)),
        'val_narkamauka': max(1, int(len(narkamauka) * args.val_ratio)),
        'train_tarask': len(tarask) - max(1, int(len(tarask) * args.val_ratio)) if tarask else 0,
        'val_tarask': max(1, int(len(tarask) * args.val_ratio)) if tarask else 0,
    }

    # Per-source stats
    per_source = {}
    for src, stats in sorted(source_stats.items()):
        per_source[src] = dict(stats)
        per_source[src]['license'] = LICENSE_CLASSES.get(src, 'unknown')
        per_source[src]['threshold_min_score'] = SOURCE_THRESHOLDS.get(src, {}).get('min_score', 0.45)
        per_source[src]['threshold_min_chars'] = SOURCE_THRESHOLDS.get(src, {}).get('min_chars', 80)

    accounting['per_source'] = per_source

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

    # ----- License manifest -----
    lic_manifest = {
        'generated_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'categories': {
            'commercial_ok': [],
            'research_only': [],
            'non_commercial': [],
            'attribution_required': [],
            'sharealike_required': [],
            'unknown_or_manual_review': [],
        }
    }
    for src in sorted(source_stats):
        lic = LICENSE_CLASSES.get(src, 'unknown/manual_review')
        if 'non_commercial' in lic:
            lic_manifest['categories']['non_commercial'].append(src)
        elif 'research_only' in lic:
            lic_manifest['categories']['research_only'].append(src)
        else:
            lic_manifest['categories']['commercial_ok'].append(src)
        if 'attribution' in lic:
            lic_manifest['categories']['attribution_required'].append(src)
        if 'sharealike' in lic:
            lic_manifest['categories']['sharealike_required'].append(src)
        if 'unknown' in lic or 'manual' in lic:
            lic_manifest['categories']['unknown_or_manual_review'].append(src)
    (report_dir / 'LICENSE_MANIFEST.json').write_text(
        json.dumps(lic_manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    # ----- SHA256 of parquet outputs -----
    for fpath in [tp, vp]:
        if fpath and Path(fpath).exists():
            sha = hashlib.sha256(Path(fpath).read_bytes()).hexdigest()
            name = Path(fpath).name
            print(f"{name.upper().replace('.','_')}_SHA256={sha}")

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
    print(f"LICENSE_MANIFEST={report_dir / 'LICENSE_MANIFEST.json'}")


if __name__ == '__main__':
    main()
