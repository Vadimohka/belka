#!/usr/bin/env python3
"""Verify and stream-restore the corpus+tokenizer as one immutable generation.

Existing live artifacts are never overwritten. A mismatch requires a fresh base
directory. --check-only validates the complete archive and all Parquet rows but
publishes nothing. Runtime consumers use BUNDLE_CURRENT.json.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
PACK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACK))
from data_pipeline.bundle_restore import restore


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack-dir',type=Path,default=PACK)
    ap.add_argument('--bundle-dir',type=Path)
    ap.add_argument('--base-dir',type=Path)
    ap.add_argument('--check-only',action='store_true')
    ap.add_argument('--no-link',action='store_true',help='deprecated alias for --check-only; never publish')
    ap.add_argument('--force',action='store_true',help='unsupported destructive legacy flag; use a fresh --base-dir')
    args=ap.parse_args()
    if args.force:ap.error('--force cannot overwrite corpus/tokenizer/checkpoints; choose a new --base-dir')
    pack=args.pack_dir.resolve()
    bundle=(args.bundle_dir or pack/'data_release/open_corpus_bundle').resolve()
    base=(args.base_dir or Path(os.environ.get('NANOCHAT_BASE_DIR',pack/'.workspace/nanochat_base'))).resolve()
    if any(p==pack or not p.is_relative_to(pack) for p in (bundle,base)):
        ap.error('bundle and base paths must stay below --pack-dir')
    try:report=restore(bundle,base,check_only=args.check_only or args.no_link)
    except (OSError,ValueError,KeyError,TypeError) as exc:ap.error(str(exc))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
