#!/usr/bin/env python3
"""Download and extract public Belarusian text sources into the pack tree.
The default is safe: no download unless --download is passed.
For Wikimedia XML dumps, this extracts a simple plain-text approximation. Run the pack language filter after extraction.
"""
from __future__ import annotations
import argparse, bz2, html, json, os, re, sys, urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

SOURCES = {
    'bewiki': {
        'url': 'https://dumps.wikimedia.org/bewiki/latest/bewiki-latest-pages-articles-multistream.xml.bz2',
        'license': 'CC-BY-SA-4.0/GFDL; verify Wikimedia terms and attribution requirements',
    },
    'bewikisource': {
        'url': 'https://dumps.wikimedia.org/bewikisource/latest/bewikisource-latest-pages-articles-multistream.xml.bz2',
        'license': 'Per-work licenses; verify compatibility before training/publishing',
    },
    'bewiktionary': {
        'url': 'https://dumps.wikimedia.org/bewiktionary/latest/bewiktionary-latest-pages-articles-multistream.xml.bz2',
        'license': 'CC-BY-SA-4.0/GFDL; verify Wikimedia terms and attribution requirements',
    },
}
TAG_RE = re.compile(r'<[^>]+>')
TEMPLATE_RE = re.compile(r'\{\{[^{}]*(?:\{\{[^{}]*\}\}[^{}]*)*\}\}', re.S)
LINK_RE = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]')
REF_RE = re.compile(r'<ref[^>]*>.*?</ref>|<ref[^/]*/>', re.S)
TABLE_RE = re.compile(r'\{\|.*?\|\}', re.S)
CATEGORY_RE = re.compile(r'\[\[(?:Катэгорыя|Category):[^\]]+\]\]', re.I)
FILE_RE = re.compile(r'\[\[(?:Файл|File|Image):[^\]]+\]\]', re.I)

def clean_wikitext(text: str) -> str:
    text = html.unescape(text or '')
    text = REF_RE.sub(' ', text)
    text = TABLE_RE.sub(' ', text)
    text = CATEGORY_RE.sub(' ', text)
    text = FILE_RE.sub(' ', text)
    for _ in range(4):
        new = TEMPLATE_RE.sub(' ', text)
        if new == text: break
        text = new
    text = LINK_RE.sub(lambda m: m.group(1), text)
    text = TAG_RE.sub(' ', text)
    text = re.sub(r"'{2,5}", '', text)
    text = re.sub(r'={2,}([^=]+)={2,}', r'\n\1\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url} -> {dest}')
    with urllib.request.urlopen(url) as r, dest.open('wb') as f:
        while True:
            b = r.read(1024 * 1024)
            if not b: break
            f.write(b)
    print(f'OK downloaded {dest} size={dest.stat().st_size}')

def iter_pages_bz2(path: Path):
    # Streaming iterparse over bz2 file. Good enough for Belarusian dumps.
    with bz2.open(path, 'rb') as f:
        context = ET.iterparse(f, events=('end',))
        for _, elem in context:
            if elem.tag.endswith('page'):
                title = elem.findtext('./{*}title') or ''
                ns = elem.findtext('./{*}ns') or '0'
                text = elem.findtext('./{*}revision/{*}text') or ''
                yield title, ns, text
                elem.clear()

def extract(source: str, dump: Path, out_dir: Path, limit_pages: int | None):
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'{source}.jsonl'
    manifest = out_dir / f'{source}.manifest.json'
    kept = 0; seen = 0
    with out.open('w', encoding='utf-8') as w:
        for title, ns, wt in iter_pages_bz2(dump):
            seen += 1
            if ns != '0':
                continue
            text = clean_wikitext(wt)
            if len(text) < 200:
                continue
            w.write(json.dumps({'text': text, 'title': title, 'source': source, 'source_url': SOURCES[source]['url'], 'license': SOURCES[source]['license'], 'lang': 'be'}, ensure_ascii=False) + '\n')
            kept += 1
            if limit_pages and kept >= limit_pages:
                break
    manifest.write_text(json.dumps({'source': source, 'dump': str(dump), 'rows': kept, 'pages_seen': seen, 'license': SOURCES[source]['license']}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'OK extracted rows={kept} pages_seen={seen} -> {out}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.getcwd())
    ap.add_argument('--source', choices=sorted(SOURCES), default='bewiki')
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--limit-pages', type=int, default=1000)
    ap.add_argument('--dump-path')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    downloads = pack / 'data_input' / 'downloads'
    out_dir = pack / 'data_input' / 'be_texts' / args.source
    dump = Path(args.dump_path) if args.dump_path else downloads / f'{args.source}-latest-pages-articles-multistream.xml.bz2'
    if args.download:
        download(SOURCES[args.source]['url'], dump)
    if args.extract:
        if not dump.exists():
            print(f'Dump not found: {dump}. Use --download first or pass --dump-path.', file=sys.stderr)
            sys.exit(2)
        extract(args.source, dump, out_dir, args.limit_pages)
    if not args.download and not args.extract:
        print('Dry run only. Use --download and/or --extract.')
        print(json.dumps(SOURCES[args.source], ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
