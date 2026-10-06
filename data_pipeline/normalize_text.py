#!/usr/bin/env python3
"""Text normalization helpers for Belarusian nanochat corpora."""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

SPACE_RE = re.compile(r"[ \t\r\f\v]+")
MANY_NL_RE = re.compile(r"\n{4,}")
DASH_REPLACEMENTS = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "—",
    "\u2212": "-",
}
QUOTE_REPLACEMENTS = {
    "\u2018": "’",
    "\u2019": "’",
    "\u02bc": "’",
    "\u201c": "\"",
    "\u201d": "\"",
    "\u00ab": "«",
    "\u00bb": "»",
}


def _control_pattern():
    # A compiled range class performs the old Unicode-category policy in C,
    # avoiding a Python callback for every character of multi-GB corpora.
    ranges=[];start=None;last=None
    for code in range(0x110000):
        bad=code not in (9,10) and unicodedata.category(chr(code)).startswith('C')
        if bad:
            if start is None:start=code
            last=code
        elif start is not None:
            ranges.append((start,last));start=None
    if start is not None:ranges.append((start,last))
    return re.compile('['+''.join('\\U%08x-\\U%08x'%(a,b) for a,b in ranges)+']')
CONTROL_RE=_control_pattern()


def normalize_text(text: str, *, keep_paragraphs: bool = True) -> str:
    """Return a stable NFC-normalized text string.

    The normalizer is intentionally conservative: it does not translate,
    spell-correct or rewrite content. It removes control garbage, normalizes
    apostrophes used in Belarusian orthography and collapses excessive spacing.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    for src, dst in DASH_REPLACEMENTS.items():
        text = text.replace(src, dst)
    for src, dst in QUOTE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.replace("\x00", " ")
    text = CONTROL_RE.sub(" ", text)
    if keep_paragraphs:
        lines = [SPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines)
        text = MANY_NL_RE.sub("\n\n\n", text)
        return text.strip()
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize UTF-8 text files for Belarusian corpus building")
    ap.add_argument("input", type=Path, nargs="?", help="Input file; stdin when omitted")
    ap.add_argument("--output", "-o", type=Path, help="Output file; stdout when omitted")
    ap.add_argument("--flat", action="store_true", help="Collapse all whitespace to single spaces")
    args = ap.parse_args()

    raw = args.input.read_text(encoding="utf-8", errors="strict") if args.input else sys.stdin.read()
    out = normalize_text(raw, keep_paragraphs=not args.flat)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out + "\n", encoding="utf-8")
    else:
        print(out)


if __name__ == "__main__":
    main()
