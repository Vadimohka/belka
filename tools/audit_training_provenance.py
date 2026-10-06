#!/usr/bin/env python3
"""Verify content-bound run manifests and coverage of existing checkpoints."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.provenance import artifact, atomic_json, validate_manifest

def audit(base):
    base = Path(base).resolve()
    failures, records, checkpoints, prepared = [], [], [], []
    paths = sorted((base/'run_manifests').glob('*_RUN_MANIFEST.json'))
    if not paths:
        failures.append('no run manifests')
    bound = set()
    for path in paths:
        try:
            record = validate_manifest(json.loads(path.read_text()))
            if record['status']=='PREPARED' and not record['checkpoints']:
                prepared.append(artifact(path))
                continue
            for checkpoint in record['checkpoints']:
                expected = {'base':'base_checkpoints','sft':'chatsft_checkpoints','rl':'chatrl_checkpoints'}[record['phase']]
                ck = Path(checkpoint['path'])
                if ck.parent != (base/expected/record['model_tag']).resolve():
                    raise ValueError('checkpoint path does not match this run/model tag')
                bound.add(str(ck))
                checkpoints.append(dict(model_tag=record['model_tag'], checkpoint_path=str(ck),
                    checkpoint_sha256=checkpoint['sha256'], tokenizer_sha256=record['tokenizer']['sha256'],
                    run_manifest=artifact(path), status='PROVENANCE_VERIFIED', quality_status='NOT_EVALUATED'))
            records.append(artifact(path))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            failures.append(f'{path.name}: {exc}')
    actual = {str(p.resolve()) for dirname in ('base_checkpoints','chatsft_checkpoints','chatrl_checkpoints')
              for p in (base/dirname).glob('*/model_*.pt')}
    if not actual:
        failures.append('no checkpoints')
    failures.extend(f'checkpoint has no verified run manifest: {p}' for p in sorted(actual-bound))
    return dict(schema_version=1, audit_status='FAIL' if failures else 'PASS',
        PROVENANCE_STATUS='FAIL' if failures else 'PASS', NATIVE_TRAINING_LOG_PROVENANCE='NOT_ASSESSED',
        TRAINING_ALLOWED='NO', SFT_ALLOWED='NO', quality_status='NOT_EVALUATED',
        run_manifests=records, prepared_manifests=prepared, checkpoints=checkpoints, failures=failures)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack-dir', type=Path, default=Path(os.environ.get('PACK_DIR', '.')))
    ap.add_argument('--base-dir', type=Path)
    ap.add_argument('--output', type=Path)
    args = ap.parse_args(argv)
    base = args.base_dir or Path(os.environ.get('NANOCHAT_BASE_DIR', args.pack_dir/'.workspace/nanochat_base_d8_v3'))
    report = audit(base)
    output = args.output or args.pack_dir/'reports/audit/training_provenance_audit.json'
    atomic_json(output, report, overwrite=True)
    markdown=output.with_suffix('.md') if args.output else output.parent/'TRAINING_PROVENANCE_AUDIT.md'
    markdown.write_text('# Training provenance\n\n'+report['PROVENANCE_STATUS']+'\n\n'+
        '\n'.join('- '+x for x in report['failures'])+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(report['failures']))

if __name__ == '__main__':
    raise SystemExit(main())
