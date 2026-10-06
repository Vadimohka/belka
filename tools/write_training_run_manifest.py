#!/usr/bin/env python3
"""Record actual selected data/configuration before a run; never certifies quality."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.provenance import atomic_json, build_manifest

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-tag', required=True)
    ap.add_argument('--phase', choices=('base', 'sft', 'rl'), required=True)
    ap.add_argument('--config-json', type=Path, required=True)
    ap.add_argument('--dataset-dir', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--runtime-dir', type=Path)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--base-dir', type=Path, default=Path(os.environ.get('NANOCHAT_BASE_DIR', '.workspace/nanochat_base')))
    ap.add_argument('--output', type=Path)
    args = ap.parse_args(argv)
    try:
        record = build_manifest(model_tag=args.model_tag, phase=args.phase,
            config=json.loads(args.config_json.read_text()), dataset_dir=args.dataset_dir,
            tokenizer_path=args.tokenizer, runtime_dir=args.runtime_dir, checkpoint_path=args.checkpoint)
        path = args.output or args.base_dir/'run_manifests'/f'{args.model_tag}_{uuid.uuid4().hex}_RUN_MANIFEST.json'
        atomic_json(path, record)
    except (OSError, ValueError, TypeError) as exc:
        ap.error(str(exc))
    print(f'RUN_MANIFEST={path.resolve()}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
