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
import argparse, gzip, hashlib, json, os, random, re, sys, math, tempfile
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.corpus_contract import HammingIndex, content_id, group_split
from data_pipeline.sft_schema import strict_json_loads
from data_pipeline.normalize_text import normalize_text
from data_pipeline.artifact_store import publish, resolve

SOURCE_THRESHOLDS = {
    "bewiki": {"min_score": 0.45, "min_chars": 120, "orthography": "narkamauka"},
    "be_x_oldwiki": {"min_score": 0.45, "min_chars": 120, "orthography": "tarask"},
    "bewikisource": {"min_score": 0.50, "min_chars": 250, "orthography": None},
    "bewikisource_full": {"min_score": 0.50, "min_chars": 250, "orthography": None},
    "bewiktionary": {"min_score": 0.45, "min_chars": 100, "orthography": None, "not_for_base": True},
    "bewikiquote": {"min_score": 0.50, "min_chars": 150, "orthography": None},
    "bewikibooks": {"min_score": 0.45, "min_chars": 200, "orthography": None},
    "bewikibooks_full": {"min_score": 0.45, "min_chars": 200, "orthography": None},
    "belacorpus_public_research": {"min_score": 0.55, "min_chars": 150, "orthography": "narkamauka"},
    "ud_belarusian_hse": {"min_score": 0.50, "min_chars": 30, "orthography": None, "not_for_base": True},
    "tatoeba_sentences": {"min_score": 0.50, "min_chars": 30, "orthography": None, "sentence_level": True},
    "belarusianglue": {"min_score": 0.55, "min_chars": 50, "orthography": None, "eval_or_sft_only": True},
    "morphodict-bel": {"min_score": 0.55, "min_chars": 10, "orthography": None, "not_for_base": True},
    "books_clean_v2": {"min_score": 0.50, "min_chars": 300, "orthography": "narkamauka"},
    "oscar-2301-be": {"min_score": 0.55, "min_chars": 150, "orthography": None},
    "culturax-be": {"min_score": 0.55, "min_chars": 150, "orthography": None},
    "mc4-be": {"min_score": 0.55, "min_chars": 150, "orthography": None},
    "cc100-be": {"min_score": 0.55, "min_chars": 150, "orthography": None},
    "bootstrap": {"min_score": 0.45, "min_chars": 60, "orthography": "narkamauka"},
    "belarusian_seed": {"min_score": 0.45, "min_chars": 60, "orthography": "narkamauka"},
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
    'bewikisource_full': 'bewikisource_full',
    'bewikibooks_full': 'bewikibooks_full',
    'bewikiquote_full': 'bewikiquote',
    'bewiktionary_full': 'bewiktionary',
    'tatoeba': 'tatoeba_sentences',
    'belarusianglue': 'belarusianglue',
    'morphodict': 'morphodict-bel',
    'belacorpus': 'belacorpus_public_research',
    'ud_belarusian': 'ud_belarusian_hse',
    'belarusian_seed': 'belarusian_seed',
    'belarusian_bootstrap': 'bootstrap',
    'books_clean': 'books_clean_v2',
    'books_clean_v2': 'books_clean_v2',
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
    # Check exact source keys (longer/specific keys first to avoid substring false match)
    for src in sorted(SOURCE_THRESHOLDS, key=lambda s: -len(s)):
        if src in p:
            return src
    return "unknown"


def score_be(text: str, source: str) -> tuple[float, list[str]]:
    """Score text for Belarusian-ness."""
    low = text.lower()
    cyr = sum(1 for ch in text if 'а' <= ch.lower() <= 'я' or ch in 'ўіё')
    lat = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    markers = sum(low.count(ch) for ch in 'ўі')
    words = re.findall(r"[а-яёіўʼ']+", low)
    be = sum(1 for w in words if w in BE_WORDS)
    ru = sum(1 for w in words if w in RU_MARKERS)
    uk = sum(w in (UK_MARKERS - BE_WORDS) for w in words) + sum(low.count(ch) for ch in "їєґ")
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
    """Strict UTF-8 input, explicit per-record parse accounting, no silent skips."""
    paths = sorted(set(inp.rglob('*.jsonl')) | set(inp.rglob('*.jsonl.gz')))
    if not paths: raise ValueError(f'no input JSONL files: {inp}')
    for path in paths:
        if not path.resolve().is_relative_to(inp.resolve()): raise ValueError(f'input escapes root: {path}')
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt', encoding='utf-8') as f:
            for number, line in enumerate(f,1):
                if not line.strip(): continue
                try:
                    obj = strict_json_loads(line)
                    if not isinstance(obj,dict):raise ValueError('object required')
                    text = obj.get('text',obj.get('content'))
                    if not isinstance(text,str):raise ValueError('string text/content required')
                except ValueError as exc:
                    yield None, str(path), {"parse_error":str(exc), "line":number}
                    continue
                yield text, str(path), obj


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
    ap.add_argument('--max-docs', type=int, default=0)
    args = ap.parse_args()
    if not math.isfinite(args.val_ratio) or not 0 < args.val_ratio < 1: ap.error('val-ratio must be in (0,1)')
    if args.max_docs < 0: ap.error('max-docs must be nonnegative')
    modes=set(args.dedup.split(','))
    if not modes.issubset({'exact','paragraph','simhash'}): ap.error('unsupported dedup mode')

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
    skipped_source_policy = 0
    skipped_duplicate_exact = 0
    skipped_duplicate_near = 0

    accepted_records = []
    quarantine_records = []
    rejected_records = []
    source_stats: dict = defaultdict(lambda: Counter())

    exact_seen: set = set()
    paragraph_hashes: set = set()
    simhasher = SimHash(64)
    simhash_threshold = 6
    simhash_index = HammingIndex(bits=64,radius=simhash_threshold)
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
        text = normalize_text(text)
        if thresh.get('not_for_base') or thresh.get('eval_or_sft_only'):
            skipped_source_policy += 1; source_stats[source]['skipped_source_policy'] += 1
            continue
        if str(obj.get('namespace', obj.get('ns', 0))) != '0':
            skipped_namespace += 1; source_stats[source]['skipped_namespace'] += 1
            continue
        if obj.get('redirect') or obj.get('is_redirect'):
            skipped_redirect += 1; source_stats[source]['skipped_redirect'] += 1
            continue

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

        # Exact dedup
        h_exact = content_id(text)
        if 'exact' in modes and h_exact in exact_seen:
            skipped_duplicate_exact += 1
            source_stats[source]['skipped_duplicate_exact'] += 1
            continue

        # Paragraph-level dedup
        has_dup_para = False
        candidate_paragraphs = set()
        for para in text.split('\n'):
            para = para.strip()
            if len(para) < 30:
                continue
            h_para = hashlib.sha256(para.encode()).hexdigest()
            if 'paragraph' in modes and h_para in paragraph_hashes:
                has_dup_para = True
                break
            candidate_paragraphs.add(h_para)
        if has_dup_para:
            skipped_duplicate_near += 1
            source_stats[source]['skipped_duplicate_near'] += 1
            continue

        # r+1 disjoint bands guarantee every <=r Hamming neighbor is checked.
        sh = simhasher.hash(text)
        if 'simhash' in modes and simhash_index.contains_near(sh):
            skipped_duplicate_near += 1
            source_stats[source]['skipped_duplicate_near'] += 1
            continue

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
            'doc_id': f"{source}_{h_exact}",
            'group_id': str(obj.get('group_id') or h_exact),
        }

        if sc >= min_score:
            # Only accepted records can enter dedup indexes. A rejected document
            # must not poison the index and suppress a valid later candidate.
            exact_seen.add(h_exact)
            paragraph_hashes.update(candidate_paragraphs)
            simhash_index.add(sh)
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
        val = [r for r in records if group_split(r['group_id'],args.val_ratio)=='val']
        train = [r for r in records if group_split(r['group_id'],args.val_ratio)=='train']
        if not train or not val: raise ValueError('empty group split; change data/ratio explicitly')
        tpath = out_dir / 'train_00000.parquet'
        vpath = out_dir / 'val_00000.parquet'
        pd.DataFrame(train).to_parquet(tpath, index=False)
        pd.DataFrame(val).to_parquet(vpath, index=False)
        return tpath, vpath

    def write_orthography_splits(records_dict, out_dir):
        """Write orthography-split parquets for reference."""
        for orth, records in records_dict.items():
            if records:
                val = [r for r in records if group_split(r['group_id'],args.val_ratio)=='val']
                train = [r for r in records if group_split(r['group_id'],args.val_ratio)=='train']
                stem = orth or 'unspecified'
                pd.DataFrame(train).to_parquet(out_dir / f'{stem}_train_00000.parquet', index=False)
                pd.DataFrame(val).to_parquet(out_dir / f'{stem}_val_00000.parquet', index=False)

    # Primary nanochat-compatible output
    # Stage all outputs; the single corpus pointer is switched only at the end.
    staged = tempfile.TemporaryDirectory(prefix='.belka-source-',dir=out.parent)
    staging = Path(staged.name)
    tp, vp = write_nanochat_split(narkamauka + tarask, staging)

    # Orthography-split for reference
    orth_records = {}
    for r in narkamauka + tarask:
        orth = r.get('orthography') or 'narkamauka'
        if orth not in orth_records:
            orth_records[orth] = []
        orth_records[orth].append(r)
    (staging/'orthography').mkdir()
    write_orthography_splits(orth_records, staging/'orthography')

    # ----- Accounting assertion -----
    total_accounted = (len(accepted_records) + len(quarantine_records) + len(rejected_records) +
                       skipped_parse_error + skipped_no_text_field + skipped_empty + skipped_short +
                       skipped_duplicate_exact + skipped_duplicate_near + skipped_namespace + skipped_redirect + skipped_source_policy)
    if total_accounted != raw_seen: raise ValueError(f"ACCOUNTING FAIL: {raw_seen} != {total_accounted}")

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
        'skipped_source_policy': skipped_source_policy,
        'skipped_redirect': skipped_redirect,
        'TOTAL_ACCOUNTED': total_accounted,
        'ACCOUNTING_ASSERTION': 'PASS' if total_accounted == raw_seen else 'FAIL',
        'train_narkamauka': sum(group_split(r['group_id'],args.val_ratio)=='train' for r in narkamauka),
        'val_narkamauka': sum(group_split(r['group_id'],args.val_ratio)=='val' for r in narkamauka),
        'train_tarask': sum(group_split(r['group_id'],args.val_ratio)=='train' for r in tarask),
        'val_tarask': sum(group_split(r['group_id'],args.val_ratio)=='val' for r in tarask),
    }

    # Per-source stats
    per_source = {}
    for src, stats in sorted(source_stats.items()):
        per_source[src] = dict(stats)
        per_source[src]['threshold_min_score'] = SOURCE_THRESHOLDS.get(src, {}).get('min_score', 0.45)
        per_source[src]['threshold_min_chars'] = SOURCE_THRESHOLDS.get(src, {}).get('min_chars', 80)

    accounting['per_source'] = per_source

    (staging/'ACCOUNTING.json').write_text(json.dumps(accounting,ensure_ascii=False,indent=2)+'\n')
    files={str(p.relative_to(staging)):p for p in staging.rglob('*') if p.is_file()}
    publish(out,'corpus',files,{'split_policy':'belka-group-split-v2','dedup_modes':sorted(modes),'hamming_radius':simhash_threshold})
    directory,_=resolve(out,'corpus')
    tp,vp=directory/'train_00000.parquet',directory/'val_00000.parquet'
    staged.cleanup()
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
