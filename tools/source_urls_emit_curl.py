#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
try:
    import yaml
except Exception:
    yaml=None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default='.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--dry-run', default='0')
    ap.add_argument('--manifest', required=True)
    args=ap.parse_args()
    pack=Path(args.pack_dir).resolve(); cfg=Path(args.config)
    if yaml is None:
        raise SystemExit('PyYAML required')
    data=yaml.safe_load(cfg.read_text(encoding='utf-8'))
    dl=pack/'data_input'/'downloads'
    dry=str(args.dry_run) in ('1','true','True')
    rows=[]
    for src in data.get('sources',[]):
        out=dl/src['output']; out.parent.mkdir(parents=True,exist_ok=True)
        row={k:src.get(k) for k in ['id','name','url','type','priority','license','output']}
        row['timestamp']=datetime.now(timezone.utc).isoformat(); row['dry_run']=dry
        cmd=['curl','-L','--fail','--retry','3','--connect-timeout','20','-o',str(out),src['url']]
        row['command']=' '.join(cmd)
        print(row['command'])
        if not dry:
            r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            row['returncode']=r.returncode
            row['stdout_tail']=r.stdout[-1000:]
            row['exists']=out.exists()
            if r.returncode!=0:
                print(f"WARN: download failed for {src['id']}", file=sys.stderr)
        rows.append(row)
    man=Path(args.manifest); man.parent.mkdir(parents=True,exist_ok=True)
    with man.open('w',encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
