#!/usr/bin/env python3
"""Measure real single-H200 training batches before a long run.

Each candidate runs in a fresh subprocess. This creates temporary randomly
initialized models, never changes production checkpoints or trains a final model.
"""
from __future__ import annotations
import argparse
import ast
import json
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.training_artifacts import PACK, atomic_json, inside, read_json, signed_document
from tools.training_plan import HARDWARE_SCHEMA, check_plan
from tools.attention_backend import (attention_identity, offline_environment,
    prepare_probe_backend, verify_probe_backend)


def hardware_identity():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('expose exactly one H200 with CUDA_VISIBLE_DEVICES; CUDA is unavailable or device count differs')
    props = torch.cuda.get_device_properties(0)
    if 'H200' not in props.name.upper() or props.major < 9 or props.total_memory < 120_000_000_000:
        raise ValueError(f'expected a full H200, found {props.name} with {props.total_memory} bytes')
    if not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 is unavailable')
    return {'name': props.name, 'uuid': str(getattr(props, 'uuid', 'unavailable')),
            'memory_bytes': props.total_memory, 'compute_capability': [props.major, props.minor],
            'torch': torch.__version__, 'cuda': torch.version.cuda}


def model_summary(config):
    import torch
    from nanochat.gpt import GPT, GPTConfig
    with torch.device('meta'):
        model = GPT(GPTConfig(**config))
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'flops_per_token': model.estimate_flops(), 'model_config': config}


def trainer_defaults(runtime, script):
    """Read literal CLI defaults without importing/running either trainer."""
    tree = ast.parse((Path(runtime)/'scripts'/script).read_text())
    result = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'parser'
                and node.func.attr == 'add_argument' and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            for item in node.keywords:
                if item.arg == 'default':
                    try:
                        result[node.args[0].value[2:].replace('-', '_')] = ast.literal_eval(item.value)
                    except (ValueError, TypeError):
                        pass
    return result


def probe_optimizer(model, plan, mode):
    """Match canonical trainer defaults, batch LR scaling and Muon schedules."""
    import torch
    from nanochat.gpt import GPT, GPTConfig
    base = trainer_defaults(plan['nanochat_dir'], 'base_train.py')
    if mode == 'base':
        scale = math.sqrt(plan['profile_config']['total_batch_size'] / 2**19)
        head_dim, aspect = plan['profile_config']['head_dim'], plan['profile_config']['aspect_ratio']
        dim = math.ceil(12 * aspect / head_dim) * head_dim
        reference_config = dict(plan['model'], n_layer=12, n_embd=dim, n_head=dim//head_dim, n_kv_head=dim//head_dim)
        with torch.device('meta'):
            reference = GPT(GPTConfig(**reference_config))
        def scaling(m):
            counts = m.num_scaling_params()
            return counts['transformer_matrices'] + counts['lm_head']
        decay = base['weight_decay'] * scale * scaling(reference) / scaling(model)
        optimizer = model.setup_optimizer(unembedding_lr=base['unembedding_lr']*scale,
            embedding_lr=base['embedding_lr']*scale, matrix_lr=base['matrix_lr']*scale,
            scalar_lr=base['scalar_lr']*scale, weight_decay=decay)
        horizon = plan['budgets']['base_iterations']
        warmup = min(40, max(0, horizon//20))
        warmdown = round(base['warmdown_ratio'] * horizon)
        def schedule(step):
            if step < warmup:
                lr = (step+1)/warmup
            elif step <= horizon-warmdown:
                lr = 1.0
            else:
                progress = (horizon-step)/warmdown
                lr = progress + (1-progress)*base['final_lr_frac']
            if step < 400:
                fraction = step/400
                momentum = (1-fraction)*0.85 + fraction*0.97
            elif step >= horizon-warmdown:
                progress = (step-(horizon-warmdown))/warmdown
                momentum = 0.97*(1-progress) + 0.90*progress
            else:
                momentum = 0.97
            return lr, momentum, decay * 0.5 * (1+math.cos(math.pi*step/horizon))
    else:
        sft = trainer_defaults(plan['nanochat_dir'], 'chat_sft_be.py')
        optimizer = model.setup_optimizer(unembedding_lr=base['unembedding_lr'],
            embedding_lr=base['embedding_lr'], matrix_lr=base['matrix_lr'], weight_decay=0.0)
        for group in optimizer.param_groups:
            group['lr'] *= sft['init_lr_frac']
            group['initial_lr'] = group['lr']
        horizon = plan['budgets']['sft_iterations']
        def schedule(step):
            progress = (step+1)/horizon
            if progress < sft['warmup_ratio']:
                lr = (progress+1e-8)/sft['warmup_ratio']
            elif progress <= 1-sft['warmdown_ratio']:
                lr = 1.0
            else:
                decay = (progress-(1-sft['warmdown_ratio']))/sft['warmdown_ratio']
                lr = (1-decay) + decay*sft['final_lr_frac']
            fraction = min(step/300, 1)
            return lr, (1-fraction)*0.85 + fraction*0.95, None
    def apply_schedule(step):
        # Very short smoke plans still need repeated steady-state measurements.
        lr, momentum, decay = schedule(min(step, horizon-1))
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * lr
            if group['kind'] == 'muon':
                group['momentum'] = momentum
                if decay is not None:
                    group['weight_decay'] = decay
    return optimizer, apply_schedule


def train_probe_steps(model, optimizer, loader, *, mode, total_batch, batch, sequence,
                      apply_schedule, synchronize, device, warmup_steps=2, measured_steps=3):
    """Time actual accumulated optimizer steps, including the next batch prefetch."""
    import torch
    from nanochat.belka_launch import validate_batch
    from nanochat.belka_runtime import normalize_supervised_gradients
    validate_batch(batch, sequence, 1, total_batch)
    accumulation = total_batch // (batch*sequence)
    def next_batch():
        item = next(loader)
        # Base also yields the replay cursor; SFT only yields input and target.
        return item[0], item[1]
    x, y = next_batch()
    times, losses, supervised = [], [], []
    model.zero_grad(set_to_none=True)
    for step in range(warmup_steps+measured_steps):
        synchronize()
        started = time.perf_counter()
        tokens = torch.zeros((), dtype=torch.int64, device=device)
        nats = torch.zeros((), dtype=torch.float64, device=device)
        for _ in range(accumulation):
            loss = model(x, y, loss_reduction='mean' if mode == 'base' else 'sum')
            if mode == 'sft':
                tokens += (y >= 0).sum()
                nats += loss.detach().to(nats.dtype)
                (loss / total_batch).backward()
            else:
                (loss / accumulation).backward()
            x, y = next_batch()
        apply_schedule(step)
        mean = (normalize_supervised_gradients(model, tokens, nats, gradient_divisor=total_batch)
                if mode == 'sft' else loss.detach())
        optimizer.step()
        model.zero_grad(set_to_none=True)
        synchronize()
        duration = time.perf_counter()-started
        # Numerical inspection is outside timed work: production BF16 does not
        # synchronize once per parameter to inspect every individual gradient.
        if not bool(torch.isfinite(mean)):
            raise ValueError('non-finite training loss')
        if step >= warmup_steps:
            times.append(duration)
            losses.append(float(mean))
            supervised.append(int(tokens) if mode == 'sft' else total_batch)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('non-finite model weights after optimizer update')
    return dict(gradient_accumulation_steps=accumulation, measured_optimizer_steps=measured_steps,
                median_optimizer_step_seconds=statistics.median(times), measured_losses=losses,
                supervised_tokens_per_step=supervised,
                tokens_per_second=total_batch/statistics.median(times))


def worker(plan, batch, mode):
    check_plan(plan)
    planned_batches = (plan['profile_config']['device_batch_candidates'] if mode == 'base'
                       else [plan['budgets']['sft_device_batch_size']])
    if type(batch) is not int or batch not in planned_batches:
        raise ValueError('probe batch is not part of the verified plan')
    os.environ.update({k: str(v) for k, v in plan['environment'].items()})
    os.environ['NANOCHAT_DTYPE'] = 'bfloat16'
    probe_base = inside(PACK/'.workspace/h200-probes'/plan['sha256']/'inputs')
    probe_base.mkdir(parents=True, exist_ok=True)
    for name, target in [('base_data_climbmix', plan['assets']['corpus_dir']), ('tokenizer', plan['assets']['tokenizer_dir'])]:
        p = probe_base/name
        if p.is_symlink():
            if p.resolve(strict=True) != Path(target):
                raise ValueError('probe input changed')
        elif p.exists():
            raise ValueError('unmanaged probe input')
        else:
            p.symlink_to(target, target_is_directory=True)
    os.environ['NANOCHAT_BASE_DIR'] = str(probe_base)
    sys.path.insert(0, plan['nanochat_dir'])
    import torch
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import get_tokenizer
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    from nanochat.belka_sft_stream import SFTBatchStream
    from nanochat.common import COMPUTE_DTYPE
    from nanochat.flash_attention import USE_FA3
    from data_pipeline.contracts import iter_jsonl, conversation_messages
    identity = hardware_identity()
    if any(name in os.environ for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK')):
        raise ValueError('hardware probe is single-GPU; do not wrap it in torchrun/DDP')
    if COMPUTE_DTYPE != torch.bfloat16:
        raise ValueError('probe runtime did not select BF16')
    backend, _ = attention_identity()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    with torch.device('meta'):
        model = GPT(GPTConfig(**plan['model']))
    summary = {'parameters': sum(p.numel() for p in model.parameters()), 'flops_per_token': model.estimate_flops()}
    model.to_empty(device=device)
    model.init_weights()
    optimizer, apply_schedule = probe_optimizer(model, plan, mode)
    compiled = torch.compile(model, dynamic=False)
    tokenizer = get_tokenizer()
    if mode == 'base':
        sequence = plan['model']['sequence_len']
        loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, batch, sequence, 'train', device=device)
        total_batch = plan['profile_config']['total_batch_size']
    else:
        sequence = plan['budgets']['sft_max_seq_len']
        conversations = []
        for _, row in iter_jsonl(Path(plan['assets']['sft_dir'])/'identity_conversations.jsonl'):
            conversations.append({'messages': conversation_messages(row)})
        if not conversations:
            raise ValueError('empty SFT probe')
        loader = SFTBatchStream(conversations, tokenizer, batch, sequence, device)
        total_batch = plan['budgets']['sft_total_batch_size']
    torch.cuda.reset_peak_memory_stats()
    measured = train_probe_steps(compiled, optimizer, loader, mode=mode, total_batch=total_batch,
        batch=batch, sequence=sequence, apply_schedule=apply_schedule,
        synchronize=torch.cuda.synchronize, device=device)
    peak = max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
    within = peak <= identity['memory_bytes'] * plan['memory_fraction']
    return dict(status='PASS' if within else 'MEMORY_RESERVE_EXCEEDED', mode=mode, batch=batch,
                sequence=sequence, peak_memory_bytes=peak, memory_limit_bytes=int(identity['memory_bytes']*plan['memory_fraction']),
                attention='FA3' if USE_FA3 else 'SDPA', compute_dtype=str(COMPUTE_DTYPE),
                attention_backend=backend,
                optimizer='MuonAdamW', global_batch_tokens=total_batch,
                random_initial_weights=True, hardware=identity, **summary, **measured)


def probe(plan_path, output):
    plan = read_json(inside(plan_path, exists=True))
    check_plan(plan)
    if any(name in os.environ for name in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK')):
        raise ValueError('hardware probe is single-GPU; do not wrap it in torchrun/DDP')
    identity = hardware_identity()
    folder = inside(PACK/'.workspace/h200-probes'/plan['sha256'])
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    backend_cache = folder/('backend-'+uuid.uuid4().hex)
    backend = prepare_probe_backend(plan, backend_cache)
    env = dict(os.environ, **{k: str(v) for k, v in plan['environment'].items()})
    env.update(offline_environment(backend_cache))
    env['PYTHONPATH'] = os.pathsep.join([plan['nanochat_dir'], str(PACK), os.environ.get('PYTHONPATH', '')])
    for mode, candidates in [('base', plan['profile_config']['device_batch_candidates']),
                             ('sft', [] if plan['budgets']['skip_sft'] else [plan['budgets']['sft_device_batch_size']])]:
        for batch in candidates:
            result_file = folder/f'{mode}-batch{batch}-{time.time_ns()}.json'
            cmd = [plan['python'], str(Path(__file__).resolve()), '--worker-plan', str(Path(plan_path).resolve()),
                   '--batch', str(batch), '--mode', mode, '--output', str(result_file)]
            print(f'Probe {mode}: batch {batch}', flush=True)
            with result_file.with_suffix('.log').open('x') as log:
                run = subprocess.run(cmd, cwd=PACK, env=env, stdout=log, stderr=subprocess.STDOUT)
            if not result_file.exists():
                raise RuntimeError(f'probe process failed without evidence; see {result_file.with_suffix(".log")} (exit {run.returncode})')
            result = read_json(result_file)
            results.append(result)
            if result['status'] not in ('PASS', 'OOM', 'MEMORY_RESERVE_EXCEEDED'):
                raise ValueError(f'probe failed: {result}')
            if result['status'] != 'OOM' and result.get('attention_backend') != backend:
                raise ValueError('probe worker selected a different attention backend/kernel payload')
    accepted = [r for r in results if r['mode'] == 'base' and r['status'] == 'PASS']
    if not accepted or any(r['mode']=='sft' and r['status'] != 'PASS' for r in results):
        raise ValueError('no complete base/SFT configuration meets the memory reserve; inspect probe reports')
    best = max(accepted, key=lambda r: r['tokens_per_second'])
    sft = next((r for r in results if r['mode']=='sft'), None)
    base_seconds = plan['budgets']['base_tokens']/best['tokens_per_second']
    sft_seconds = (plan['budgets']['sft_iterations'] * plan['budgets']['sft_total_batch_size'] / sft['tokens_per_second']) if sft else 0
    # Include explicit slack for evaluation/checkpoint IO and throughput variability.
    estimate = 1.35*(base_seconds+sft_seconds)/3600
    checkpoint_bytes = best['parameters']*16*18  # conservative: model+optimizer+metadata, base/SFT checkpoints
    free = __import__('shutil').disk_usage(plan['prepared_base']).free
    report = signed_document(dict(schema=HARDWARE_SCHEMA, plan_sha256=plan['sha256'],
        status='PASS' if estimate <= plan['budget_hours'] and free > checkpoint_bytes else 'BUDGET_EXCEEDED',
        hardware=identity, selected_device_batch_size=best['batch'], candidates=results,
        attention_backend=backend, backend_cache=str(backend_cache),
        estimated_training_hours=estimate, estimate_is_measurement_based_not_a_guarantee=True,
        estimated_checkpoint_bytes=checkpoint_bytes, disk_free_bytes=free,
        safety_multiplier=1.35, production_weights_written=False))
    atomic_json(inside(output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] != 'PASS':
        raise ValueError('time or disk budget insufficient; create an adjusted plan before execution')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path)
    p.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--batch', type=int, help=argparse.SUPPRESS)
    p.add_argument('--mode', choices=['base', 'sft'], default='base', help=argparse.SUPPRESS)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        if args.worker_plan:
            plan = read_json(args.worker_plan)
            try:
                result = worker(plan, args.batch, args.mode)
            except Exception as exc:
                import torch
                result = dict(status='OOM' if isinstance(exc, torch.OutOfMemoryError) else 'ERROR',
                              mode=args.mode, batch=args.batch, error=f'{type(exc).__name__}: {exc}')
            atomic_json(inside(args.output), result)
            if result['status'] != 'PASS':
                p.exit(3, result['status']+'\n')
        elif args.plan:
            probe(args.plan, args.output)
        else:
            p.error('--plan is required')
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        p.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
