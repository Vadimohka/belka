"""Auditable web-language diagnostics and bounded lexical near-duplicate sketches.

These are reproducible heuristics, not a substitute for native-speaker review.
Candidate sketches never alone justify removal: full shingle Jaccard is checked.
"""
import array,json,re,sys,zlib
import numpy as np
from pathlib import Path
from urllib.parse import urlsplit
from data_pipeline.detect_belarusian import BEL_WORDS,RUSSIAN_HINTS,UKRAINIAN_HINTS,mixed_language_reasons
from data_pipeline.leakage import normalized
WEB_SOURCES={'fineweb2','hplt_v2','cc100-be','leipzig','oscar-2301-be','culturax-be','mc4-be'}
BE_DISCRIMINATIVE=BEL_WORDS-RUSSIAN_HINTS-UKRAINIAN_HINTS
POLICY_PATH=Path(__file__).resolve().parents[1]/'configs/h200_quality_policy.json'
QUALITY_POLICY=json.loads(POLICY_PATH.read_text())
LETTER_RE=re.compile(r'[^\W\d_]+')
CYRILLIC_RE=re.compile(r'[а-яёіўєіїґ]+')
LATIN_WORD_RE=re.compile(r'^[a-z]+$')

def validation_coverage(splits):
    """Describe source coverage after decontamination without inventing val rows.

    Major sources require clean validation. A minor training-only category is
    reported explicitly; neither its removed validation text nor its metrics
    are silently restored. Counts come from the final serialized mixture.
    """
    policy=QUALITY_POLICY['validation_coverage'];errors=[];parts={}
    valid_count=lambda value:type(value) is int and value>=0
    for part in ('train','val'):
        summary=splits.get(part,{}) if isinstance(splits,dict) else {}
        if not isinstance(summary,dict):summary={}
        reported_rows=summary.get('rows')
        if not valid_count(reported_rows):errors.append(part+': invalid or missing row count')
        entries=summary.get('sources')
        if not isinstance(entries,dict):
            errors.append(part+': invalid or missing per-source counts');entries={}
        valid={}
        for name,stat in entries.items():
            if not isinstance(name,str) or not name or not isinstance(stat,dict) or not all(valid_count(stat.get(k)) for k in ('rows','chars')):
                errors.append(part+': invalid source/counts '+str(name));continue
            if bool(stat['rows'])!=bool(stat['chars']):
                errors.append(part+': inconsistent zero rows/characters for '+name);continue
            valid[name]=stat
        parts[part]=valid
        if valid_count(reported_rows) and sum(s['rows'] for s in valid.values())!=reported_rows:
            errors.append(part+': per-source rows do not sum to split.rows')
        if 'chars' in summary and (not valid_count(summary['chars']) or sum(s['chars'] for s in valid.values())!=summary['chars']):
            errors.append(part+': per-source characters do not sum to split.chars')
    train=parts['train'];val=parts['val']
    names=sorted(set(train)|set(val))
    total_chars=sum(stat['chars'] for source in (train,val) for stat in source.values())
    per_source={};minor=[];missing=[];val_only=[]
    for name in names:
        train_rows=train.get(name,{}).get('rows',0);val_rows=val.get(name,{}).get('rows',0)
        chars=train.get(name,{}).get('chars',0)+val.get(name,{}).get('chars',0)
        fraction=chars/max(total_chars,1)
        required=(name in policy['required_sources_when_present'] or
                  train_rows>=policy['major_source_min_train_rows'] or
                  fraction>=policy['major_source_min_char_fraction'])
        if not train_rows:
            status='VAL_ONLY';val_only.append(name)
        elif not val_rows:
            status='MISSING_REQUIRED_VALIDATION' if required else 'TRAIN_ONLY_MINOR'
            (missing if required else minor).append(name)
        else:status='COVERED'
        per_source[name]=dict(train_rows=train_rows,val_rows=val_rows,total_chars=chars,
            char_fraction=fraction,validation_required=required,status=status)
    nonempty=all(sum(s['rows'] for s in parts[part].values())>0 for part in ('train','val'))
    status='FAIL' if errors or not nonempty or missing or val_only else ('PASS_WITH_MINOR_GAPS' if minor else 'PASS')
    return dict(schema='belka-source-validation-coverage-v1',status=status,policy=dict(policy),
        source_counts_complete=not errors,source_count_errors=errors,
        final_splits_nonempty=nonempty,per_source=per_source,train_only_minor_sources=minor,
        missing_required_sources=missing,val_only_sources=val_only,
        scope='Measured final source coverage. Train-only minor categories have no source-specific held-out quality estimate; contaminated validation is never restored.')

def url_problem(url,source):
    if source not in WEB_SOURCES or not url:return None
    try:parsed=urlsplit(url);host=(parsed.hostname or '').lower().removeprefix('www.')
    except ValueError:return 'invalid_source_url'
    if any(host==domain or host.endswith('.'+domain) for domain in QUALITY_POLICY['review_domains']):
        return 'domain_requires_quality_review'
    if any(part.lower() in QUALITY_POLICY['review_path_segments'] for part in parsed.path.split('/')):
        return 'tag_or_search_index'
    return None

def language_problem(text,source):
    if problems:=mixed_language_reasons(text):return problems[0]
    if source not in WEB_SOURCES:return None
    nt=normalized(text);words=nt.split();script=QUALITY_POLICY['web_script']
    letters=sum(map(len,LETTER_RE.findall(nt)));cyrillic=sum(map(len,CYRILLIC_RE.findall(nt)))
    if letters and cyrillic/letters<script['minimum_cyrillic_letter_fraction']:return 'non_cyrillic_dominant_web_text'
    ukrainian=sum('є' in w or 'ї' in w or 'ґ' in w for w in words)
    if ukrainian>=script['ukrainian_minimum_marked_words'] and ukrainian/max(len(words),1)>=script['ukrainian_minimum_marked_fraction']:return 'ukrainian_marked_web_text'
    markers=sum(text.lower().count(c) for c in 'ўі')
    evidence=sum(w in BE_DISCRIMINATIVE for w in words)
    if markers==0 and evidence<QUALITY_POLICY['web_positive_evidence']['minimum_discriminative_words_if_no_be_letters']:return 'no_discriminative_belarusian_evidence'
    cfg=QUALITY_POLICY['russian_segment'];bad=set()
    marked=[int('и' in w or 'щ' in w or 'ъ' in w) for w in words]
    belarusian=[int('ў' in w or 'і' in w or w in BE_DISCRIMINATIVE) for w in words]
    latin=[int(bool(LATIN_WORD_RE.fullmatch(w))) for w in words];latin_bad=set()
    if cfg['discriminative_letters']!='ищъ':raise ValueError('unsupported Russian script markers')
    for start in range(0,len(words),cfg['stride_words']):
        window=words[start:start+cfg['window_words']]
        if len(window)<cfg['minimum_window_words']:continue
        ru=sum(marked[start:start+len(window)])
        be=sum(belarusian[start:start+len(window)])
        if sum(latin[start:start+len(window)])/len(window)>=script['latin_window_minimum_fraction']:latin_bad.update(range(start,start+len(window)))
        if ru>=cfg['minimum_marked_words'] and ru/len(window)>=cfg['minimum_marked_fraction'] and be/max(ru,1)<=cfg['maximum_belarusian_to_russian_ratio']:
            bad.update(range(start,start+len(window)))
    if len(bad)>=cfg['minimum_bad_words'] and len(bad)/max(len(words),1)>=cfg['minimum_bad_document_fraction']:
        return 'russian_script_dominant_web_segments'
    if len(latin_bad)>=script['latin_window_minimum_bad_words'] and len(latin_bad)/max(len(words),1)>=script['latin_window_minimum_document_fraction']:return 'latin_dominant_web_segments'
    # A names/list/menu dump is poor pretraining prose even if tagged Belarusian.
    if len(words)>=100 and len(set(words))/len(words)<.05:return 'extreme_token_repetition'
    return None

SHINGLE_POWERS=np.array([pow(0x100000001B3,i,1<<64) for i in range(4,-1,-1)],dtype=np.uint64)

def shingles(text):
    """Full 5-word rolling fingerprints (32-bit), with deterministic byte order."""
    words=normalized(text).split()
    if len(words)<5:return frozenset()
    tokens=np.fromiter((zlib.crc32(w.encode()) for w in words),dtype=np.uint64,count=len(words))
    values=np.correlate(tokens,SHINGLE_POWERS,'valid')
    values^=values>>np.uint64(33);values*=np.uint64(0xff51afd7ed558ccd);values^=values>>np.uint64(33)
    return frozenset(((values>>np.uint64(32))^values).astype(np.uint32).tolist())

def pack_shingles(values):
    a=array.array('I',sorted(values))
    if sys.byteorder!='little':a.byteswap()
    return a.tobytes()

def unpack_shingles(payload):
    a=array.array('I');a.frombytes(payload)
    if sys.byteorder!='little':a.byteswap()
    return frozenset(a)

def jaccard(a,b):
    intersection=len(a&b)
    return intersection/max(len(a)+len(b)-intersection,1)
