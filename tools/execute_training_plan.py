"""Execute an already verified plan using a stable snapshot of its input generations."""
from __future__ import annotations
import fcntl
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from data_pipeline.contracts import corpus_generation
from tools.training_artifacts import PACK, atomic_json, file_hash, inside, read_json


def commands(plan, batch, *, cpu=False, base_resume=None, sft_resume=None, base_step=None):
    p, b = plan['profile_config'], plan['budgets']
    common = ['--run=dummy', '--model-tag='+plan['model_tag'], '--device-type='+('cpu' if cpu else 'cuda')]
    early = [f'--early-stopping-patience={p["early_stopping_patience"]}']
    base = [plan['python'], '-m', 'scripts.base_train', *common, *early,
        f'--depth={p["depth"]}', f'--aspect-ratio={p["aspect_ratio"]}', f'--head-dim={p["head_dim"]}',
        f'--max-seq-len={p["max_seq_len"]}', f'--window-pattern={p["window_pattern"]}',
        f'--device-batch-size={batch}', f'--total-batch-size={p["total_batch_size"]}',
        f'--num-iterations={b["base_iterations"]}', f'--eval-every={b["eval_every"]}',
        f'--eval-tokens={b["eval_tokens"]}', f'--save-every={b["save_every"]}',
        '--core-metric-every=-1', '--sample-every=-1',
        f'--warmup-steps={min(40, max(0, b["base_iterations"]//20))}']
    if base_resume is not None:
        base += [f'--resume-from-step={base_resume}']
    sft = [plan['python'], '-m', 'scripts.chat_sft_be', *common, *early,
        f'--max-seq-len={b["sft_max_seq_len"]}', f'--device-batch-size={b["sft_device_batch_size"]}',
        f'--total-batch-size={b["sft_total_batch_size"]}', f'--num-iterations={b["sft_iterations"]}',
        f'--eval-every={b["sft_eval_every"]}', f'--eval-tokens={b["sft_eval_tokens"]}',
        f'--save-every={b["sft_eval_every"]}', '--chatcore-every=-1']
    if base_step is not None:
        sft += [f'--model-step={base_step}']
    if sft_resume is not None:
        sft += [f'--resume-from-step={sft_resume}']
    return base, sft


def snapshot(plan):
    run = inside(plan['run_base'])
    existing = run/'PLAN.json'
    if existing.exists():
        if read_json(existing) != plan:
            raise ValueError('run already belongs to a different plan; select a new model tag')
    else:
        if run.exists() and any(run.iterdir()):
            raise ValueError('unmanaged run directory retained; select a new model tag')
        run.mkdir(parents=True, exist_ok=True)
        atomic_json(existing, plan)
    for name, target in (('base_data_climbmix', plan['assets']['corpus_dir']),
                         ('tokenizer', plan['assets']['tokenizer_dir'])):
        link = run/name
        if link.is_symlink():
            if link.resolve(strict=True) != Path(target):
                raise ValueError('run input bridge was changed')
        elif link.exists():
            raise ValueError('unmanaged run input path')
        else:
            link.symlink_to(target, target_is_directory=True)
    live = run/'.sft_current'
    if not live.exists():
        with corpus_generation(live) as stage:
            for source in Path(plan['assets']['sft_dir']).iterdir():
                if source.is_symlink() or not source.is_file():
                    raise ValueError('unexpected SFT generation payload')
                shutil.copy2(source, stage/source.name)
    sft = live.resolve(strict=True)
    if sft.parent != run/'..sft_current.generations':
        raise ValueError('run SFT generation pointer was changed')
    if file_hash(sft/'SFT_BUILD_MANIFEST.json') != plan['assets']['sft_manifest_sha256']:
        raise ValueError('run SFT manifest changed')
    from tools.training_artifacts import verify_inventory
    verify_inventory(sft, plan['assets']['sft_files'])
    return run


def committed_steps(folder):
    if not folder.exists():
        return []
    # Readers below validate full commit metadata/hash membership, not names alone.
    return sorted(int(p.stem.split('_')[1]) for p in folder.glob('commit_*.json')
                  if p.stem.split('_')[-1].isdigit())


def checkpoint_choices(folder):
    from nanochat.belka_checkpoint import validate_checkpoint
    candidates = []
    for step in committed_steps(folder):
        # API validation is supplied by the runtime; validate hashes before choosing.
        validate_checkpoint(str(folder), step, load_optimizer=False, rank=0)
        meta = read_json(folder/f'meta_{step:06d}.json')
        score = meta.get('val_bpb')
        if step > 0 and meta.get('val_step') == step and type(score) in (int, float) and math.isfinite(score):
            candidates.append((score, step))
    if not candidates:
        raise ValueError(f'no committed checkpoint with a finite validation score: {folder}')
    return min(candidates)[1]


def phase_finished(folder, horizon):
    summaries = sorted(folder.glob('result_*.json')) if folder.exists() else []
    for path in reversed(summaries):
        summary = read_json(path)
        step = summary.get('completed_optimizer_steps')
        if (summary.get('schema') == 'belka-training-result-v1' and type(step) is int and 0 < step <= horizon
                and step in committed_steps(folder)
                and (step == horizon or summary.get('completion_reason') == 'early_stopping')):
            from nanochat.belka_checkpoint import validate_checkpoint
            validate_checkpoint(str(folder), step, load_optimizer=True, rank=0)
            return summary
    return None


def execute(plan, hardware, *, resume=False, cpu_smoke=False):
    from tools.training_plan import check_plan
    check_plan(plan, hardware)
    if cpu_smoke:
        if plan['profile'] != 'smoke' or plan['budgets']['base_iterations'] > 10 or plan['budgets']['sft_iterations'] > 10:
            raise ValueError('CPU acceptance is limited to the tiny smoke profile and <=10 steps per phase')
    elif hardware is None:
        raise ValueError('execute requires a matching measured H200 report; run tools/h200_probe.py first')
    if any(name in os.environ for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK')):
        raise ValueError('do not wrap the single-H200 entrypoint in torchrun/DDP')
    if hardware is not None and not cpu_smoke:
        from tools.h200_probe import hardware_identity
        if hardware_identity() != hardware['hardware']:
            raise ValueError('GPU/software identity differs from the measured probe')
        if shutil.disk_usage(plan['prepared_base']).free <= hardware['estimated_checkpoint_bytes']:
            raise ValueError('checkpoint free space decreased since the probe; free space or select a new plan')
        from tools.attention_backend import verify_probe_backend
        backend_env = verify_probe_backend(plan, hardware)
    else:
        backend_env = {}
    run = inside(plan['run_base'])
    lock_root = inside(Path(plan['prepared_base'])/'.runlocks')
    lock_root.mkdir(exist_ok=True)
    fd = os.open(lock_root/(plan['model_tag']+'.lock'), os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('another process owns this run') from exc
        folders = [run/k/plan['model_tag'] for k in ('base_checkpoints', 'chatsft_checkpoints')]
        if not resume and any(committed_steps(p) for p in folders):
            raise ValueError('checkpoints already exist; use --resume with the original plan')
        run = snapshot(plan)
        execution_identity = {'plan_sha256': plan['sha256'], 'mode': 'cpu_smoke' if cpu_smoke else 'single_h200',
                              'hardware_report_sha256': hardware['sha256'] if hardware else None}
        execution_file = run/'EXECUTION.json'
        if execution_file.exists():
            if read_json(execution_file) != execution_identity:
                raise ValueError('execution mode or hardware report changed; retain the original acceptance report for resume')
        else:
            atomic_json(execution_file, execution_identity)
            if hardware is not None:
                atomic_json(run/'HARDWARE.json', hardware)
        budget_file = run/'RUN_BUDGET.json'
        if budget_file.exists():
            wall_budget = read_json(budget_file)
            if wall_budget.get('plan_sha256') != plan['sha256']:
                raise ValueError('time budget belongs to a different plan')
            from tools.training_artifacts import positive
            positive(wall_budget['started_at_unix'], 'run start time')
            positive(wall_budget['deadline_unix'], 'run deadline')
            if not math.isclose(wall_budget['deadline_unix']-wall_budget['started_at_unix'],
                                plan['budget_hours']*3600, rel_tol=0, abs_tol=0.001):
                raise ValueError('saved wall-time budget differs from the immutable plan')
        else:
            now = time.time()
            wall_budget = {'plan_sha256': plan['sha256'], 'started_at_unix': now,
                           'deadline_unix': now + plan['budget_hours']*3600}
            atomic_json(budget_file, wall_budget)
        batch = min(plan['profile_config']['device_batch_candidates']) if cpu_smoke else hardware['selected_device_batch_size']
        env = dict(os.environ, **{k: str(v) for k, v in plan['environment'].items()})
        env.update(backend_env)
        env.update(NANOCHAT_BASE_DIR=str(run), NANOCHAT_DIR=plan['nanochat_dir'],
                   PACK_DIR=str(PACK), PYTHONNOUSERSITE='1',
                   PYTHONPATH=os.pathsep.join([plan['nanochat_dir'], str(PACK), os.environ.get('PYTHONPATH', '')]))
        if cpu_smoke:
            env.update(CUDA_VISIBLE_DEVICES='', NANOCHAT_DTYPE='', TORCHDYNAMO_DISABLE='1')
        import sys
        sys.path.insert(0, plan['nanochat_dir'])
        os.environ['NANOCHAT_BASE_DIR'] = str(run)
        os.environ['NANOCHAT_DTYPE'] = env['NANOCHAT_DTYPE']
        best_base = None
        results = {}
        for phase, folder, horizon in zip(('base', 'sft'), folders,
                                         (plan['budgets']['base_iterations'], plan['budgets']['sft_iterations'])):
            if phase == 'sft' and plan['budgets']['skip_sft']:
                break
            prior = phase_finished(folder, horizon)
            steps = committed_steps(folder)
            if resume:
                from nanochat.belka_checkpoint import recover_incomplete_checkpoints
                recovery = recover_incomplete_checkpoints(str(folder), max(steps) if steps else None)
                if recovery.get('archive'):
                    atomic_json(run/f'{phase}_RECOVERY_{time.time_ns()}.json', recovery)
            resuming = max(steps) if resume and steps and not prior else None
            base_cmd, sft_cmd = commands(plan, batch, cpu=cpu_smoke,
                                        base_resume=resuming if phase == 'base' else None,
                                        sft_resume=resuming if phase == 'sft' else None, base_step=best_base)
            cmd = base_cmd if phase == 'base' else sft_cmd
            if not prior:
                from tools.provenance import build_manifest
                config = {'argv': cmd, 'plan_sha256': plan['sha256'], 'resolved_device_batch_size': batch,
                          'phase': phase, 'horizon': horizon, 'model': plan['model'],
                          'hardware_report_sha256': execution_identity['hardware_report_sha256']}
                proof = build_manifest(model_tag=plan['model_tag'], phase=phase, config=config,
                    dataset_dir=plan['assets']['corpus_dir'] if phase == 'base' else str((run/'.sft_current').resolve()),
                    tokenizer_path=Path(plan['assets']['tokenizer_dir'])/'tokenizer.pkl', runtime_dir=plan['nanochat_dir'])
                attempt = time.time_ns()
                atomic_json(run/'run_manifests'/f'{phase}_{attempt}_PREPARED_RUN_MANIFEST.json', proof)
                log = run/f'{phase}_{attempt}.log'
                print(f'{phase}: {" ".join(cmd)}\nlog: {log}', flush=True)
                remaining = wall_budget['deadline_unix'] - time.time()
                if remaining <= 0:
                    raise ValueError('run wall-time budget exhausted; committed checkpoints are retained')
                with log.open('x') as stream:
                    try:
                        subprocess.run(cmd, cwd=plan['nanochat_dir'], env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, check=True, timeout=remaining)
                    except subprocess.TimeoutExpired as exc:
                        atomic_json(run/'BUDGET_EXHAUSTED.json', {'plan_sha256': plan['sha256'],
                            'phase': phase, 'deadline_unix': wall_budget['deadline_unix'],
                            'completed': False, 'checkpoints_retained': True}, replace=True)
                        raise ValueError('run wall-time budget exhausted; last committed checkpoints are retained') from exc
                prior = phase_finished(folder, horizon)
                if prior is None:
                    raise ValueError(f'{phase} returned without a committed completion summary')
            best = checkpoint_choices(folder)
            from tools.provenance import build_manifest
            recorded = build_manifest(model_tag=plan['model_tag'], phase=phase,
                config={'argv': cmd, 'plan_sha256': plan['sha256'], 'resolved_device_batch_size': batch,
                        'phase': phase, 'horizon': horizon, 'model': plan['model'], 'selected_step': best,
                        'hardware_report_sha256': execution_identity['hardware_report_sha256']},
                dataset_dir=plan['assets']['corpus_dir'] if phase == 'base' else str((run/'.sft_current').resolve()),
                tokenizer_path=Path(plan['assets']['tokenizer_dir'])/'tokenizer.pkl',
                runtime_dir=plan['nanochat_dir'], checkpoint_path=folder/f'model_{best:06d}.pt')
            from tools.provenance import artifact
            recorded['checkpoints'] = [artifact(folder/f'model_{step:06d}.pt') for step in committed_steps(folder)]
            atomic_json(run/'run_manifests'/f'{phase}_{time.time_ns()}_RECORDED_RUN_MANIFEST.json', recorded)
            if phase == 'base':
                best_base = best
            results[phase] = dict(prior, selected_best_step=best, checkpoint_dir=str(folder))
        result = {'schema': 'belka-training-result-v1', 'plan_sha256': plan['sha256'],
                  'status': 'CPU_SMOKE_COMPLETE' if cpu_smoke else 'TRAINING_COMPLETE',
                  'run_base': str(run), 'phases': results, 'model_quality_accepted': False}
        atomic_json(run/'RESULT.json', result, replace=True)
        return result
    finally:
        os.close(fd)
