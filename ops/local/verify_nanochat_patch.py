#!/usr/bin/env python3
"""Verify the complete pinned runtime receipt and the active all-role SFT input.

This performs source/hash/schema checks, not a training or GPU readiness probe.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
PACK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACK))
from data_pipeline.artifact_store import resolve_sft_paths
from modernize_nanochat import verify


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nanochat-dir',type=Path,required=True)
    ap.add_argument('--pack-dir',type=Path,default=PACK)
    ap.add_argument('--base-dir',type=Path,default=None)
    ap.add_argument('--require-dtype-patch',action='store_true',help='retained for compatibility; the receipt always includes dtype checks')
    args=ap.parse_args();pack=args.pack_dir.resolve();repo=args.nanochat_dir.resolve()
    base=(args.base_dir or Path(os.environ.get('NANOCHAT_BASE_DIR',pack/'.workspace/nanochat_base'))).resolve()
    try:
        if any(p==pack or not p.is_relative_to(pack) for p in (repo,base)):raise ValueError('runtime and data must stay below PACK_DIR')
        receipt=verify(repo,pack)
        if 'BELKA_BRANDING_BANNER' not in (repo/'nanochat/common.py').read_text():raise ValueError('missing startup branding')
        if '<title>Belka</title>' not in (repo/'nanochat/ui.html').read_text():raise ValueError('missing UI branding')
        train,val=resolve_sft_paths(base)
        result=subprocess.run([sys.executable,str(pack/'tools/validate_sft_jsonl.py'),train,val,'--strict-all'],capture_output=True,text=True,timeout=120)
        if result.returncode:raise ValueError('SFT validation failed: '+result.stdout[-3000:]+result.stderr[-1000:])
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:ap.error(str(exc))
    print(json.dumps({'ok':True,'scope':'runtime hashes and active SFT schema/language; no training/GPU probe','upstream_commit':receipt['upstream_commit'],'runtime_outputs':len(receipt['outputs']),'sft_train':train,'sft_val':val,'language_policy':'strict-all heuristic'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
