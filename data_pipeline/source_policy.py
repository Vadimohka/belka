"""Shared source admission policy for every base-corpus ingestion path."""
from pathlib import Path

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

SOURCE_THRESHOLDS.update({
    'fineweb2': {'min_score': .55, 'min_chars': 150, 'orthography': None},
    'hplt_v2': {'min_score': .55, 'min_chars': 150, 'orthography': None},
    'leipzig': {'min_score': .50, 'min_chars': 50, 'orthography': None},
})

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



def base_rejection(source, metadata=None, path=''):
    """Return a reason, never let a path/metadata alias bypass eval exclusions."""
    metadata = metadata or {}
    if metadata.get('eval_only') or metadata.get('never_train') or metadata.get('base_pretrain_allowed') is False:
        return 'eval_only_metadata'
    if metadata.get('status') in ('manual_review', 'quarantine', 'rejected'):
        return 'unapproved_quality_status'
    identifiers = {str(source), str(metadata.get('source', '')), detect_source(path)}
    for name in identifiers:
        rule = SOURCE_THRESHOLDS.get(name, {})
        if rule.get('not_for_base') or rule.get('eval_or_sft_only'):
            return 'excluded_base_source:' + name
    return None
