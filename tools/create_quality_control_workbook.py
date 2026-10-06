#!/usr/bin/env python3
"""Render measured evidence into QC; absent evidence remains UNKNOWN, never PASS."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.provenance import artifact, atomic_json, sha256_file, validate_manifest, verify_artifact

MODEL_RUNS_COLS = ['model_tag', 'checkpoint_path', 'checkpoint_sha256', 'tokenizer_sha256',
                   'run_manifest_path', 'run_manifest_sha256', 'provenance_status', 'quality_status']

def display_measurement(value):
    encoded=json.dumps(value,ensure_ascii=False)
    return encoded if len(encoded)<=30000 else encoded[:30000]+' [display truncated; full value in Source_Evidence]'

def prompts(path, *, strict=False):
    rows, ids = [], set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ident = row.get('eval_id', row.get('id', row.get('prompt_id')))
        text = row.get('prompt', row.get('text'))
        if not isinstance(ident, str) or ident in ids or not isinstance(text, str) or not text.strip():
            raise ValueError(f'invalid/duplicate prompt in {path}')
        if strict and row.get('never_train') is not True:
            raise ValueError('strict holdout must explicitly set never_train=true')
        ids.add(ident)
        rows.append((ident, text, row.get('category', row.get('intent')) or None, True if strict else None))
    if not rows:
        raise ValueError(f'empty prompts: {path}')
    return rows

def evidence(pack, data_report=None, provenance_report=None):
    pack = Path(pack)
    strict = pack/'eval/strict_holdout_quality_control_v2.be.jsonl'
    regression = pack/'eval/regression_quality_control_v1.be.jsonl'
    result = dict(schema_version=1, scope='checkpoint QC; not permission to train',
        inputs=dict(strict=artifact(strict), regression=artifact(regression)),
        checks=dict(corpus='UNKNOWN', leakage='UNKNOWN', provenance='UNKNOWN'),
        measurements={}, quality_pass=None, benchmark_decontamination=None, subset_derivation=None,
        reports={}, model_runs=[], dataset=[], dataset_sft=[], problems=[])
    dp = Path(data_report) if data_report else pack/'reports/data/H200_DATA_PREPARATION.json'
    if dp.exists():
        try:
            from data_pipeline.h200_evidence import read_json, validate_corpus_evidence
            data = read_json(dp)
            # Intrinsic transferred manifests are self-contained: their parent
            # directory selects the corpus and local bridges select SFT. Old
            # absolute build paths remain provenance, never transfer inputs.
            corpus = Path(data.get('corpus_dir', dp.parent)).resolve(strict=True)
            base = corpus.parent.parent
            if (base/'.corpus_current').resolve() == corpus and (base/'.sft_current').exists():
                sft = (base/'.sft_current').resolve(strict=True)
            else:
                roots = {Path(f['path']).resolve().parent for f in data['sft']['files']}
                if len(roots) != 1:
                    raise ValueError('SFT evidence must select one immutable generation')
                sft = roots.pop()
            validate_corpus_evidence(corpus, document=data, sft_dir=sft)
            files = [dict(f, path=str(corpus/f['path'])) for f in data['files']]
            if any(type(f.get('rows')) is not int or f['rows'] <= 0 for f in files):
                raise ValueError('data file has no verified rows')
            for split in ('train', 'val'):
                if sum(f['rows'] for f in files if f['split'] == split) != data['splits'][split]['rows']:
                    raise ValueError('corpus shard rows disagree with checked split counts')
            sft_files = [dict(f, path=str(sft/Path(f['path']).name)) for f in data['sft']['files']]
            derivation = data['subset_derivation']
            parent = read_json(corpus/derivation['parent_manifest']['path'])
            holdout = data['holdout']
            if (holdout.get('sha256') != result['inputs']['strict']['sha256']
                    or holdout.get('checked_train_rows') != parent['splits']['train']['rows']
                    or holdout.get('checked_sft_messages', 0) <= 0):
                raise ValueError('missing, stale or incomplete inherited holdout evidence')
            result['checks'].update(corpus='PASS', leakage='PASS')
            result['dataset'] = files
            result['dataset_sft'] = sft_files
            result['quality_pass'] = data['quality_pass']
            result['benchmark_decontamination'] = data['benchmark_decontamination']
            result['subset_derivation'] = derivation
            result['measurements'] = dict(
                evidence_mode='inherited_whole_row_subset',
                current_train_rows=data['splits']['train']['rows'],
                current_val_rows=data['splits']['val']['rows'],
                admission_removed_rows=derivation['removed_rows'],
                parent_checked_train_rows=holdout['checked_train_rows'],
                parent_checked_sft_messages=holdout['checked_sft_messages'],
                parent_holdout_evidence=holdout,
                scope='Original holdout/benchmark scan counters describe the verified full-scan parent. The current corpus inherits absence through unchanged whole-row membership; no fresh contamination scan is claimed.')
            result['reports']['data'] = artifact(dp)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            result['checks'].update(corpus='FAIL', leakage='FAIL')
            result['problems'].append(str(exc))
    pp = Path(provenance_report) if provenance_report else pack/'reports/audit/training_provenance_audit.json'
    if pp.exists():
        try:
            data = json.loads(pp.read_text())
            if data.get('schema_version') != 1 or data.get('audit_status') != 'PASS' or not data.get('run_manifests'):
                raise ValueError('missing verified checkpoint provenance')
            for item in data['run_manifests']:
                verify_artifact(item)
                manifest = validate_manifest(json.loads(Path(item['path']).read_text()), require_checkpoint=True)
                if result['checks']['corpus']=='PASS':
                    expected=result['dataset'] if manifest['phase']=='base' else result['dataset_sft']
                    identity=lambda rows:sorted((Path(f['path']).name,f['sha256']) for f in rows)
                    if identity(expected)!=identity([f for split in manifest['dataset'].values() for f in split]):
                        raise ValueError('checkpoint run used different data than the selected leakage/corpus report')
                for ck in manifest['checkpoints']:
                    result['model_runs'].append([manifest['model_tag'],ck['path'],ck['sha256'],
                        manifest['tokenizer']['sha256'],item['path'],item['sha256'],
                        'PROVENANCE_VERIFIED','NOT_EVALUATED'])
            result['checks']['provenance'] = 'PASS'
            result['reports']['provenance'] = artifact(pp)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            result['checks']['provenance'] = 'FAIL'
            result['problems'].append(str(exc))
    return result

def create(pack, output=None, data_report=None, provenance_report=None):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    pack = Path(pack)
    report = evidence(pack,data_report,provenance_report)
    strict = prompts(report['inputs']['strict']['path'], strict=True)
    regression = prompts(report['inputs']['regression']['path'])
    wb = Workbook()
    wb.active.title = 'README'
    wb.active.append(['Checkpoint QC, not training permission'])
    wb.active.append(['UNKNOWN/NOT_EVALUATED means no verified evidence; no quality or leakage result is inferred.'])
    def sheet(name, header, rows):
        ws = wb.create_sheet(name)
        ws.append(header)
        for row in rows:
            ws.append(row)
        for cell in ws[1]:
            cell.font = Font(bold=True,color='FFFFFF')
            cell.fill = PatternFill('solid',fgColor='345779')
        ws.freeze_panes = 'A2'
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(80,max(18,max(len(str(c.value or '')) for c in col)+2))
        return ws
    sheet('Strict_Holdout_LOCKED',['prompt_id','prompt_text','category','never_train'],strict).protection.sheet=True
    sheet('Regression_SeenIntent',['prompt_id','prompt_text','category','never_train'],regression)
    sheet('Model_Runs',MODEL_RUNS_COLS,report['model_runs'])
    sheet('Dataset_Registry',['path','sha256','rows','split'],[[f[k] for k in ('path','sha256','rows','split')] for f in report['dataset']])
    sheet('Regression_Dashboard',['measurement','value'],[['strict_prompts',len(strict)],['regression_prompts',len(regression)]]+
          [[k,display_measurement(v)] for k,v in report['measurements'].items()])
    sheet('Training_Gates',['gate_name','value'],[['TRAINING_ALLOWED','NO'],['SFT_ALLOWED','NO']]+
          [[k,v] for k,v in report['checks'].items()])
    for name in ('Strict_Holdout_Results','Regression_Results'):
        sheet(name,['status'],[['NOT_EVALUATED']])
    encoded=json.dumps(report,ensure_ascii=False,sort_keys=True)
    sheet('Source_Evidence',['json_chunks'],[[encoded[i:i+30000]] for i in range(0,len(encoded),30000)])
    output = Path(output) if output else pack/'reports/eval/BELKA_QUALITY_CONTROL.xlsx'
    output.parent.mkdir(parents=True,exist_ok=True)
    wb.save(output)
    return report

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack-dir',type=Path,default=Path(os.environ.get('PACK_DIR','.')))
    ap.add_argument('--output',type=Path)
    ap.add_argument('--data-report',type=Path)
    ap.add_argument('--provenance-report',type=Path)
    args=ap.parse_args(argv)
    try:
        result=create(args.pack_dir,args.output,args.data_report,args.provenance_report)
    except (OSError,ValueError,TypeError,KeyError,ImportError) as exc:
        ap.error(str(exc))
    print(json.dumps(result['checks']))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
