#!/usr/bin/env python3
"""Install a hash-locked Belka overlay without overwriting unknown local edits.

All candidates compile before any runtime file changes. Pristine originals are
retained for repeatable regeneration. Manifest hashes detect local drift; the
manifest is written last. Installation must not run concurrently with training.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

PACK = Path(__file__).resolve().parents[2]


def digest(data): return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'upstream anchor count != 1: {old[:90]!r}')
    return text.replace(old,new,1)


def preserve_sft_warm_start_decay(text):
    """Keep fresh SFT decay values when importing pretrained optimizer state."""
    text = replace_once(text,
        '        base_lrs = [group["lr"] for group in optimizer.param_groups]',
        '        base_lrs = [group["lr"] for group in optimizer.param_groups]\n'
        '        sft_weight_decays = [group["weight_decay"] for group in optimizer.param_groups]')
    text = replace_once(text,
        '        for group, base_lr in zip(optimizer.param_groups, base_lrs):\n'
        '            group["lr"] = base_lr',
        '        for group, base_lr, sft_decay in zip(optimizer.param_groups, base_lrs, sft_weight_decays):\n'
        '            group["lr"] = base_lr\n'
        '            group["weight_decay"] = sft_decay')
    return replace_once(text,
        '(momentum buffers only, LRs reset)',
        '(state loaded, SFT LRs and weight decay retained)')


def refresh_sft_scaler_checks(text):
    """GradScaler must inspect gradients AFTER supervised normalization."""
    return replace_once(text,
        '        if is_ddp_initialized():\n'
        '            for v in scaler._found_inf_per_device(optimizer).values():\n'
        '                dist.all_reduce(v, op=dist.ReduceOp.MAX)',
        '        if scaler.is_enabled():\n'
        '            # unscale_ cached flags before normalization could overflow.\n'
        '            # Pinned Torch: inspect again with unit inverse scale.\n'
        '            scaler._check_inf_per_device(optimizer)\n'
        '            if is_ddp_initialized():\n'
        '                for v in scaler._found_inf_per_device(optimizer).values():\n'
        '                    dist.all_reduce(v, op=dist.ReduceOp.MAX)')


def add_sft_resume(text):
    """Install explicit optimizer-step checkpoint/replay over the pinned SFT loop."""
    text = replace_once(text, 'import gc\n', 'import gc\nimport math\n')
    text = replace_once(text, 'from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state',
                        'from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state, load_checkpoint')
    text = replace_once(text, 'args = parser.parse_args()', '''parser.add_argument("--resume-from-step", type=int, default=-1, help="resume an exact committed SFT step")
parser.add_argument("--save-every", type=int, default=200, help="checkpoint interval in optimizer steps (-1=final only)")
parser.add_argument("--stop-after-step", type=int, default=-1, help="checkpoint and stop this invocation; retain original training horizon")
args = parser.parse_args()
from nanochat.belka_launch import validate_model_tag, checkpoint_directory, validate_batch, validate_sft_schedule
validate_model_tag(args.model_tag)
resuming = args.resume_from_step >= 0
if args.resume_from_step < -1 or (resuming and args.model_tag is None):
    raise ValueError("SFT resume requires a model tag and a nonnegative step")
if args.stop_after_step != -1 and not max(0, args.resume_from_step) < args.stop_after_step <= args.num_iterations:
    raise ValueError("stop-after-step must be after the resume step and within the original horizon")
if args.save_every != -1 and args.save_every < 1:
    raise ValueError("save-every must be positive or -1")''')
    text = replace_once(text, 'model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)',
'''model, tokenizer, meta = load_model("sft" if resuming else "base", device, phase="train",
    model_tag=args.model_tag, step=args.resume_from_step if resuming else args.model_step)
if resuming:
    for key, value in meta.get("resolved_config", {}).items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)''')
    text = replace_once(text, 'orig_model = model\n', '''validate_sft_schedule(args)
validate_batch(args.device_batch_size, args.max_seq_len, ddp_world_size, args.total_batch_size, args.eval_tokens)
if args.max_seq_len > model.config.sequence_len:
    raise ValueError("SFT context cannot exceed the base model context")
resolved_config = vars(args).copy()
orig_model = model
''')
    text = replace_once(text, 'base_dir = get_base_dir()\nif args.load_optimizer:', '''base_dir = get_base_dir()
checkpoint_dir = checkpoint_directory(base_dir, "chatsft_checkpoints", args.model_tag or f"d{depth}")
if resuming:
    args.load_optimizer = 0  # exact SFT state is loaded after setup, below
if args.load_optimizer:''')
    start = text.index('# DataLoader is defined here')
    end = text.index('# Learning rate schedule', start)
    text = text[:start] + '''# Both training and validation use the same lossless, replayable packing.
def sft_data_generator_bos_bestfit(split, buffer_size=100):
    from nanochat.belka_sft_stream import SFTBatchStream
    if split not in ("train", "val"):
        raise ValueError("split must be train or val")
    return SFTBatchStream(train_dataset if split == "train" else val_dataset,
        tokenizer, args.device_batch_size, args.max_seq_len, device,
        ddp_rank, ddp_world_size, buffer_size)

train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
resume_meta = None
if resuming:
    from nanochat.belka_resume import prepare_sft_resume
    prepare_sft_resume(checkpoint_dir, args.resume_from_step, train_loader,
        resolved_config, COMPUTE_DTYPE, ddp_rank)
    model_state, optimizer_state, resume_meta = load_checkpoint(checkpoint_dir,
        args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    orig_model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    del model_state, optimizer_state
    if scaler is not None:
        if resume_meta.get("scaler_state") is None:
            raise ValueError("fp16 SFT checkpoint has no scaler state")
        scaler.load_state_dict(resume_meta["scaler_state"])
progress = 0

''' + text[end:]
    text = replace_once(text, 'x, y = next(train_loader) # prefetch the very first batch of data',
'''dataloader_state_dict = train_loader.state_dict()
x, y = next(train_loader) # checkpoint cursor precedes the pending prefetched batch''')
    text = replace_once(text, 'step = 0\nwhile True:', '''step = 0
val_bpb = None
if resuming:
    step = resume_meta["step"]
    state = resume_meta["loop_state"]
    min_val_bpb = float("inf") if state["min_val_bpb"] is None else state["min_val_bpb"]
    smooth_train_loss = state["smooth_train_loss"]
    total_training_time = state["total_training_time"]
    val_bpb = resume_meta["val_bpb"]
while True:''')
    text = replace_once(text, '    last_step = step >= args.num_iterations',
                        '    last_step = step >= args.num_iterations\n    stopping = args.stop_after_step != -1 and step >= args.stop_after_step')
    text = replace_once(text, '''    if last_step:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # e.g. d12
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)''',
'''    if step != args.resume_from_step and (last_step or stopping or (step > 0 and args.save_every > 0 and step % args.save_every == 0)):''')
    text = replace_once(text, '                "user_config": user_config, # inputs to the training script',
'''                "user_config": user_config, # inputs to the training script
                "trainer": "belka-sft-v1", "resolved_config": resolved_config,
                "compute_dtype": str(COMPUTE_DTYPE),
                "max_seq_len": args.max_seq_len, "device_batch_size": args.device_batch_size,
                "total_batch_size": args.total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "loop_state": {"min_val_bpb": min_val_bpb if math.isfinite(min_val_bpb) else None,
                    "smooth_train_loss": smooth_train_loss, "total_training_time": total_training_time},''')
    text = replace_once(text, '    if last_step:\n        break', '    if last_step or stopping:\n        break')
    text = replace_once(text, '        x, y = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward\n        progress = max(progress, approx_progress) # only increase progress monotonically',
'''        dataloader_state_dict = train_loader.state_dict()
        x, y = next(train_loader) # snapshot before the next prefetched batch''')
    text = text.replace('epoch: {current_epoch}', 'epoch: {train_loader.epoch}').replace('"train/epoch": current_epoch', '"train/epoch": train_loader.epoch')
    return text


def add_early_stopping(text, kind):
    text=replace_once(text, 'args = parser.parse_args()',
        'parser.add_argument("--early-stopping-patience", type=int, default=0, help="stop after this many validation checks without improvement (0 disables)")\nargs = parser.parse_args()\nif args.early_stopping_patience < 0:\n    raise ValueError("early-stopping-patience must be nonnegative")')
    meta = 'meta_data' if kind == 'base' else 'resume_meta'
    anchor='while True:\n    last_step'
    text=replace_once(text, anchor,
        f'best_step = None if not resuming else {meta}["loop_state"].get("best_step")\n'
        f'bad_eval_count = 0 if not resuming else {meta}["loop_state"].get("bad_eval_count", 0)\n'
        'while True:\n    last_step')
    text=replace_once(text, '    stopping = args.stop_after_step != -1 and step >= args.stop_after_step',
        '    stopping = (args.stop_after_step != -1 and step >= args.stop_after_step) or (args.early_stopping_patience > 0 and bad_eval_count >= args.early_stopping_patience)')
    text=replace_once(text, 'while True:\n    last_step',
        f'val_step = None if not resuming else {meta}.get("val_step")\nwhile True:\n    last_step')
    text=replace_once(text, '"val_bpb": val_bpb,', '"val_bpb": val_bpb, "val_step": val_step,')
    text=replace_once(text, '        if val_bpb < min_val_bpb:\n            min_val_bpb = val_bpb',
'''        val_step = step
        if val_bpb < min_val_bpb or (step > 0 and best_step is None):
            min_val_bpb = val_bpb
            if step > 0:
                best_step = step
            bad_eval_count = 0
        elif step > 0:
            bad_eval_count += 1
        stopping = stopping or (args.early_stopping_patience > 0 and bad_eval_count >= args.early_stopping_patience)''')
    text=replace_once(text, '"smooth_train_loss": smooth_train_loss,',
                     '"best_step": best_step, "bad_eval_count": bad_eval_count, "smooth_train_loss": smooth_train_loss,')
    if kind=='base':
        text=replace_once(text, 'if args.eval_every > 0 and (last_step or step % args.eval_every == 0):',
                         'if args.eval_every > 0 and (not resuming or step != args.resume_from_step) and (last_step or step % args.eval_every == 0):')
        text=replace_once(text, 'if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):',
                         'if args.core_metric_every > 0 and (not resuming or step != args.resume_from_step) and (last_step or (step > 0 and step % args.core_metric_every == 0)):')
        text=replace_once(text, 'if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):',
                         'if args.sample_every > 0 and master_process and (not resuming or step != args.resume_from_step) and (last_step or (step > 0 and step % args.sample_every == 0)):')
    else:
        text=replace_once(text, 'if last_step or (args.eval_every > 0 and step % args.eval_every == 0):',
                         'if (not resuming or step != args.resume_from_step) and (last_step or (args.eval_every > 0 and step % args.eval_every == 0)):')
        text=replace_once(text, 'if not _skip_generic and args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):',
                         'if not _skip_generic and args.chatcore_every > 0 and (not resuming or step != args.resume_from_step) and (last_step or (step > 0 and step % args.chatcore_every == 0)):')
    text=replace_once(text, 'step != args.resume_from_step and (last_step or stopping or',
        'step != args.resume_from_step and (last_step or stopping or (args.early_stopping_patience > 0 and best_step == step) or')
    # One summary per committed invocation end; checkpoints themselves stay append-only.
    marker='# cleanup' if kind=='sft' else '# Final cleanup'
    # Upstream base does not have a uniform cleanup header; place before common cleanup call.
    summary='''from nanochat.belka_launch import training_result
if master_process:
    training_result(checkpoint_dir, step, best_step, min_val_bpb,
        stopped_early=step < args.num_iterations,
        completion_reason=("horizon" if step >= args.num_iterations else
            "early_stopping" if args.early_stopping_patience > 0 and bad_eval_count >= args.early_stopping_patience else "invocation_stop"),
        data_state=dataloader_state_dict)
'''
    text=replace_once(text, 'compute_cleanup()',summary+'compute_cleanup()')
    return text


def load_tool(name):
    spec=importlib.util.spec_from_file_location(name,PACK/'ops/local'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def candidates(originals, scratch):
    output = dict(originals)
    fork=PACK/'ops/nanochat_fork'
    for file in sorted((fork/'nanochat').glob('belka_*.py')):
        output['nanochat/'+file.name]=file.read_text(encoding='utf-8')
    for relative in ('tasks/customjson.py','scripts/chat_web.py','nanochat/ui.html','nanochat/logo.svg'):
        if (fork/relative).is_file(): output[relative]=(fork/relative).read_text(encoding='utf-8')
    # Keep parameter layout and optimizer recipe intact; repair execution contracts.
    output['nanochat/belka_data_contracts.py']=(PACK/'data_pipeline/contracts.py').read_text(encoding='utf-8')
    output['nanochat/dataset.py']='''"""Belka explicit local split reader. No English dataset download fallback."""
import os
from pathlib import Path
from nanochat.common import get_base_dir
from nanochat.belka_runtime import split_parquet_files, corpus_data_dir
DATA_DIR = corpus_data_dir(get_base_dir())
def list_parquet_files(data_dir=None, warn_on_legacy=False):
    root = DATA_DIR if data_dir is None else data_dir
    return split_parquet_files(root, 'train') + split_parquet_files(root, 'val')
def parquets_iter_batched(split, start=0, step=1):
    import pyarrow.parquet as pq
    if type(start) is not int or type(step) is not int or start < 0 or step < 1:
        raise ValueError('invalid row-group stride')
    for path in split_parquet_files(DATA_DIR, split):
        pf=pq.ParquetFile(path)
        for rg in range(start, pf.num_row_groups, step):
            yield pf.read_row_group(rg,columns=['text']).column('text').to_pylist()
if __name__ == '__main__':
    raise SystemExit('Belka uses only its explicitly prepared Belarusian corpus; automatic upstream corpus download is disabled')
'''
    tokenizer=originals['nanochat/tokenizer.py']
    tokenizer=tokenizer.replace('    base_dir = get_base_dir()',
        '    from nanochat.belka_runtime import artifact_base_dir\n    base_dir = artifact_base_dir(get_base_dir())')
    tokenizer=replace_once(tokenizer, 'def get_token_bytes(device="cpu"):\n    import torch\n    from nanochat.common import get_base_dir\n    from nanochat.belka_runtime import artifact_base_dir\n    base_dir = artifact_base_dir(get_base_dir())\n    tokenizer_dir = os.path.join(base_dir, "tokenizer")\n    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")\n    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"\n    with open(token_bytes_path, "rb") as f:\n        token_bytes = torch.load(f, map_location=device)\n    return token_bytes',
        'def get_token_bytes(device="cpu"):\n    # Derive from the selected tokenizer; historical UTF-8 caches may be wrong.\n    from nanochat.belka_token_bytes import token_byte_lengths\n    return token_byte_lengths(get_tokenizer(), device=device)')
    output['nanochat/tokenizer.py']=tokenizer
    loader=originals['nanochat/dataloader.py']
    loader=replace_once(loader,'    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]',
        '    from nanochat.belka_runtime import split_parquet_files\n    from nanochat.dataset import DATA_DIR\n    parquet_paths = split_parquet_files(DATA_DIR, split)')
    output['nanochat/dataloader.py']=loader+'\n# Belka deliberately uses a lossless, replayable stream rather than cropping.\nfrom nanochat.belka_stream import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit\n'
    engine=originals['nanochat/engine.py']
    a=engine.index('@contextmanager\ndef timeout(');b=engine.index('class KVCache:',a)
    engine=engine[:a]+'''from nanochat.belka_runtime import safe_calculator
# BELARUSIAN_SUPERPACK_DTYPE_ENGINE: upstream uses COMPUTE_DTYPE.
def eval_with_timeout(formula, max_time=3):
    return safe_calculator(formula)
def use_calculator(expr):
    return safe_calculator(expr)

'''+engine[b:]
    engine=replace_once(engine, '        return self.cache_seqlens[0].item()',
        '        if not torch.all(self.cache_seqlens == self.cache_seqlens[0]).item():\n            raise ValueError("GPT rotary positions require a uniform batch; use separate caches")\n        return self.cache_seqlens[0].item()')
    engine=replace_once(engine, '        self.cache_seqlens += num_tokens',
        '        if type(num_tokens) is not int or num_tokens < 0 or torch.any(self.cache_seqlens + num_tokens > self.max_seq_len).item():\n            raise ValueError("KV cache capacity exceeded")\n        self.cache_seqlens += num_tokens')
    engine=replace_once(engine, '        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"',
        '        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):\n            raise ValueError("expecting a nonempty list of token IDs")\n        if type(num_samples) is not int or num_samples < 1:\n            raise ValueError("num_samples must be positive")\n        if max_tokens is None:\n            max_tokens = self.model.config.sequence_len - len(tokens)\n        if type(max_tokens) is not int or max_tokens < 0 or len(tokens) + max_tokens > self.model.config.sequence_len:\n            raise ValueError("prompt plus completion exceeds the configured context")\n        if max_tokens == 0:\n            return')
    output['nanochat/engine.py']=engine
    gpt=originals['nanochat/gpt.py']
    gpt=replace_once(gpt,'        B, T = idx.size()\n',
        '        B, T = idx.size()\n        if B < 1 or T < 1:\n            raise ValueError("GPT inputs must be nonempty")\n        if targets is not None and targets.shape != idx.shape:\n            raise ValueError("input/target shapes differ")\n')
    gpt=replace_once(gpt,'        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length',
        '        if T0 + T > self.cos.size(1) or (kv_cache is not None and T0 + T > kv_cache.max_seq_len):\n            raise ValueError("context exceeds rotary or KV cache capacity")\n        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length')
    gpt=replace_once(gpt,'            assert T > 1, "Training forward pass should have T > 1"',
        '            # A one-token forward has no predecessor; empty slices are valid.')
    gpt=replace_once(gpt,'                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)\n            elif x_pre_smear is not None:',
        '                first = x[:, :1]\n                if x_pre_smear is not None:\n                    first_gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(first[:, :, :24]))\n                    first = first + first_gate * x_pre_smear\n                x = torch.cat([first, x[:, 1:] + gate * x[:, :-1]], dim=1)\n            elif x_pre_smear is not None:')
    output['nanochat/gpt.py']=gpt
    attention=originals['nanochat/flash_attention.py']
    attention=replace_once(attention,'y = _sdpa_attention(q, k, v, window_size, enable_gqa)',
                           'y = _sdpa_attention(q, k, v, window_size, enable_gqa, causal=causal)')
    anchor='from types import SimpleNamespace\n'
    attention=replace_once(attention,anchor,'''# BELARUSIAN_SUPERPACK_DTYPE_SDPA: validated cache/window semantics.
from nanochat.belka_attention import sdpa_attention as _sdpa_attention
if not USE_FA3:
    from nanochat.belka_attention import fallback_with_kvcache as flash_attn_with_kvcache
from types import SimpleNamespace
''')
    output['nanochat/flash_attention.py']=attention
    checkpoint=originals['nanochat/checkpoint_manager.py']
    checkpoint=replace_once(checkpoint,'    log0(f"Loading optimizer state from {optimizer_path}")\n    optimizer_data = torch.load(optimizer_path, map_location=device)',
        '    log0(f"Loading optimizer state from {optimizer_path}")\n    from nanochat.belka_checkpoint import validate_checkpoint\n    validate_checkpoint(checkpoint_dir, step, rank, load_optimizer=True)\n    optimizer_data = torch.load(optimizer_path, map_location=device, weights_only=True)')
    # Both model and optimizer readers share the same contained tag boundary.
    anchor='    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)'
    if checkpoint.count(anchor) != 2:
        raise ValueError('expected two checkpoint tag joins in pinned upstream')
    checkpoint=checkpoint.replace(anchor, '    from nanochat.belka_launch import checkpoint_path_under\n    checkpoint_dir = checkpoint_path_under(checkpoints_dir, model_tag)')
    output['nanochat/checkpoint_manager.py']=checkpoint+'\nfrom nanochat.belka_checkpoint import save_checkpoint, load_checkpoint, find_last_step\n'
    base=originals['scripts/base_train.py']
    base=replace_once(base,'    model.load_state_dict(model_data, strict=True, assign=True)',
        '    from nanochat.belka_resume import validate_base_resume_config\n    validate_base_resume_config(meta_data, user_config, args.resume_from_step)\n    if meta_data["model_config"] != model_config_kwargs:\n        raise ValueError("resume model config differs from checkpoint")\n    model.load_state_dict(model_data, strict=True, assign=True)')
    base=replace_once(base,'scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None',
        'scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None\nif resuming and scaler is not None:\n    if meta_data.get("scaler_state") is None:\n        raise ValueError("fp16 resume checkpoint has no scaler state")\n    scaler.load_state_dict(meta_data["scaler_state"])')
    base=replace_once(base,'"dataloader_state_dict": dataloader_state_dict,',
        '"dataloader_state_dict": dataloader_state_dict,\n                "scaler_state": scaler.state_dict() if scaler is not None else None,')
    base=replace_once(base, '"min_val_bpb": min_val_bpb,', '"min_val_bpb": min_val_bpb if math.isfinite(min_val_bpb) else None,')
    base=replace_once(base, '    min_val_bpb = loop_state["min_val_bpb"]', '    min_val_bpb = loop_state["min_val_bpb"] if loop_state["min_val_bpb"] is not None else float("inf")')
    base=replace_once(base, 'total_tokens = total_batch_size * num_iterations',
        'if num_iterations < 1 or (resuming and args.resume_from_step > num_iterations):\n    raise ValueError("invalid training horizon or resume step exceeds horizon")\ntotal_tokens = total_batch_size * num_iterations')
    base=replace_once(base, 'args = parser.parse_args()', 'parser.add_argument("--stop-after-step", type=int, default=-1, help="checkpoint and stop invocation without changing horizon")\nargs = parser.parse_args()\nfrom nanochat.belka_launch import validate_model_tag, checkpoint_directory, validate_batch\nvalidate_model_tag(args.model_tag)\nif args.stop_after_step != -1 and not max(0, args.resume_from_step) < args.stop_after_step <= args.num_iterations:\n    raise ValueError("stop-after-step must be after resume step and within explicit horizon")')
    base=replace_once(base, 'checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)', 'checkpoint_dir = checkpoint_directory(base_dir, "base_checkpoints", output_dirname)')
    base=replace_once(base, '    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end', '    last_step = step == num_iterations\n    stopping = args.stop_after_step != -1 and step >= args.stop_after_step')
    base=replace_once(base, '    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):', '    if step != args.resume_from_step and (last_step or stopping or (step > 0 and args.save_every > 0 and step % args.save_every == 0)):')
    base=replace_once(base, '    if last_step:\n        break', '    if last_step or stopping:\n        break')
    base=replace_once(base, 'assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."', 'validate_batch(args.device_batch_size, args.max_seq_len, ddp_world_size, total_batch_size, args.eval_tokens if args.eval_every > 0 else None)')
    output['scripts/base_train.py']=add_early_stopping(base, 'base')
    output['nanochat/loss_eval.py']=originals['nanochat/loss_eval.py']+'\nfrom nanochat.belka_metrics import evaluate_bpb\n'

    # Generate the data-mixture variant from a pristine, pinned SFT source.
    src=scratch/'chat_sft.py';dst=scratch/'chat_sft_be.py'
    src.write_text(originals['scripts/chat_sft.py'],encoding='utf-8')
    report=load_tool('patch_nanochat_for_belarusian').create_variant(src,dst,overwrite=True)
    if not report.get('ok'): raise ValueError(report)
    sft=dst.read_text(encoding='utf-8')
    sft=replace_once(sft, 'identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")\nidentity_conversations_val_filepath = os.path.join(base_dir, "identity_conversations_val.jsonl")',
        'from nanochat.belka_runtime import sft_data_paths\n# v9 identity_conversations.jsonl / identity_conversations_val.jsonl in one immutable generation\nidentity_conversations_filepath, identity_conversations_val_filepath = sft_data_paths(base_dir)')
    start=sft.index('                "model_config": {')
    end=sft.index('                "user_config":',start)
    sft=sft[:start]+'                "model_config": vars(orig_model.config).copy(),\n'+sft[end:]
    sft=replace_once(sft,'args = parser.parse_args()',
        'args = parser.parse_args()\nif args.num_iterations < 1:\n    raise ValueError("Belka SFT requires an explicit positive optimizer-step budget: --num-iterations")\nfrom nanochat.belka_runtime import rendered_sft, normalize_supervised_gradients')
    sft=replace_once(sft,'ids, mask = tokenizer.render_conversation(conversation)',
        'ids, mask = rendered_sft(tokenizer, conversation, row_capacity)')
    sft=replace_once(sft,'    assert dataset_size > 0',
        '    if dataset_size < ddp_world_size:\n        raise ValueError("SFT dataset is smaller than the distributed world size")')
    # The generator counts microbatches; only the outer loop counts optimizer steps.
    sft=replace_once(sft,'while True:\n    flops_so_far',
        'while True:\n    last_step = step >= args.num_iterations\n    flops_so_far')
    sft=replace_once(sft,'    for micro_step in range(grad_accum_steps):',
        '    local_tokens = torch.zeros((), dtype=torch.int64, device=device)\n    local_nats = torch.zeros((), dtype=torch.float32 if device_type == "mps" else torch.float64, device=device)\n    for micro_step in range(grad_accum_steps):')
    sft=replace_once(sft,'        loss = model(x, y)\n        train_loss = loss.detach() # for logging\n        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here',
        '        loss_sum = model(x, y, loss_reduction="sum")\n        local_tokens += (y >= 0).sum()\n        local_nats += loss_sum.detach().to(local_nats.dtype)\n        loss = loss_sum / (args.total_batch_size // ddp_world_size)')
    sft=replace_once(sft,'    lrm = get_lr_multiplier(progress)',
        '    progress = (step + 1) / args.num_iterations\n    lrm = get_lr_multiplier(progress)')
    sft=replace_once(sft,'        scaler.unscale_(optimizer)',
        '        scaler.unscale_(optimizer)\n        train_loss = normalize_supervised_gradients(model, local_tokens, local_nats, args.total_batch_size // ddp_world_size)')
    sft=replace_once(sft,'    else:\n        optimizer.step()',
        '    else:\n        train_loss = normalize_supervised_gradients(model, local_tokens, local_nats, args.total_batch_size // ddp_world_size)\n        optimizer.step()')
    sft=sft.replace('ema_beta**(step + 1)', 'ema_beta**step')
    sft=replace_once(sft, '    CustomJSON(filepath=identity_conversations_filepath),\n    CustomJSON(filepath=identity_conversations_filepath),', '    CustomJSON(filepath=identity_conversations_filepath),')
    sft=sft.replace('CustomJSON x2','CustomJSON x1')
    sft=preserve_sft_warm_start_decay(sft)
    sft=refresh_sft_scaler_checks(sft)
    output['scripts/chat_sft_be.py']=add_early_stopping(add_sft_resume(sft), 'sft')
    common=scratch/'common.py';common.write_text(originals['nanochat/common.py'],encoding='utf-8')
    if not load_tool('patch_nanochat_branding').patch_common(common)['ok']: raise ValueError('branding failed')
    common_text=common.read_text(encoding='utf-8')
    common_text=replace_once(common_text, '    if env is not None:', '    if env:')
    # Never silently partition CPU/MPS data without a process group. GPU keeps
    # the existing NCCL reducer; CPU distributed mode uses the Gloo replica path.
    common_text=replace_once(common_text, '    else:\n        device = torch.device(device_type) # mps|cpu',
        '    else:\n        device = torch.device(device_type) # mps|cpu\n        if is_ddp_requested and ddp_world_size > 1:\n            if device_type != "cpu":\n                raise ValueError("distributed MPS is unsupported")\n            dist.init_process_group(backend="gloo")')
    # Default runtime storage follows the pack, not the user's home directory.
    start=common_text.index('def get_base_dir():')
    end=common_text.index('def download_file_with_lock',start)
    common_text=common_text[:start]+'''def get_base_dir():
    from pathlib import Path
    explicit = os.environ.get("NANOCHAT_BASE_DIR")
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs/path_policy.env").is_file():
            return str(parent / ".workspace/nanochat_base")
    raise ValueError("set NANOCHAT_BASE_DIR explicitly for an exported runtime")

'''+common_text[end:]
    common_text=replace_once(common_text, '    torch.manual_seed(42)', '    import random\n    import numpy as np\n    random.seed(42)\n    np.random.seed(42)\n    torch.manual_seed(42)')
    output['nanochat/common.py']=common_text
    optim=originals['nanochat/optim.py']
    optim=replace_once(optim, '        # Phase 1: launch all async reduce ops', '''        # Gloo's SUM is portable; use a replicated optimizer layout on CPU.
        # NCCL continues through the upstream sharded/overlapped path unchanged.
        if world_size > 1 and dist.get_backend() == "gloo":
            parameters = [p for group in self.param_groups for p in group['params']]
            present = torch.tensor([int(p.grad is not None) for p in parameters], dtype=torch.int64)
            dist.all_reduce(present, op=dist.ReduceOp.SUM)
            if any(value != world_size for value in present.tolist()):
                raise ValueError("all optimizer parameters must have gradients on every rank")
            for parameter in parameters:
                if parameter.device.type != "cpu":
                    raise ValueError("Gloo replica optimizer is CPU-only")
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(world_size)
            rank, world_size = 0, 1
        # Phase 1: launch all async reduce ops''')
    output['nanochat/optim.py']=optim
    return output


def install(repo: Path, verify_only=False):
    repo=repo.resolve(strict=True)
    lock=json.loads((PACK/'configs/nanochat_upstream.json').read_text())
    manifest_path=repo/'BELKA_RUNTIME_MANIFEST.json'
    previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    pack_files = [PACK/'configs/nanochat_upstream.json', PACK/'data_pipeline/contracts.py',
                  Path(__file__), PACK/'ops/local/patch_nanochat_for_belarusian.py',
                  PACK/'ops/local/patch_nanochat_branding.py']
    pack_files += [p for p in (PACK/'ops/nanochat_fork').rglob('*') if p.is_file() and p.suffix in ('.py','.html','.svg')]
    pack_identity = {str(p.relative_to(PACK)):digest(p.read_bytes()) for p in sorted(pack_files)}
    if verify_only:
        if previous.get('upstream_commit')!=lock['commit']: raise ValueError('runtime version not recorded or outdated')
        if previous.get('pack_inputs') != pack_identity: raise ValueError('overlay source changed; regenerate and retest the runtime')
        for name,sha in previous['files'].items():
            if digest((repo/name).read_bytes())!=sha: raise ValueError(f'runtime drift: {name}')
        return previous
    originals={}
    for name,sha in lock['files'].items():
        backup=repo/'.belka-originals'/name
        if backup.is_file(): data=backup.read_bytes()
        elif (repo/'.git').exists():
            head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
            if head!=lock['commit']: raise ValueError('checkout HEAD is not the pinned nanochat commit')
            data=subprocess.check_output(['git','-C',str(repo),'show',f"{lock['commit']}:{name}"])
        else: data=(repo/name).read_bytes()
        if digest(data)!=sha: raise ValueError(f'unsupported upstream source: {name}')
        originals[name]=data.decode('utf-8')
    with tempfile.TemporaryDirectory(prefix='.belka-build-',dir=repo) as temp:
        output=candidates(originals,Path(temp))
        for name,text in output.items():
            if name.endswith('.py'): compile(text,name,'exec',dont_inherit=True)
            path=repo/name
            allowed={digest(text.encode()),previous.get('files',{}).get(name)}
            if name in originals: allowed.add(lock['files'][name])
            # Fork files are explicitly owned by the pack, not upstream.
            if path.exists() and digest(path.read_bytes()) not in allowed:
                raise ValueError(f'unknown local edits retained: {name}; use a clean pinned checkout')
        for name,text in originals.items():
            backup=repo/'.belka-originals'/name;backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists(): backup.write_text(text,encoding='utf-8')
        hashes={}
        for i,(name,text) in enumerate(output.items()):
            path=repo/name;path.parent.mkdir(parents=True,exist_ok=True)
            stage=Path(temp)/f'candidate-{i}';stage.write_text(text,encoding='utf-8')
            if path.exists(): os.chmod(stage,path.stat().st_mode & 0o777)
            os.replace(stage,path);hashes[name]=digest(text.encode())
        manifest=dict(schema='belka-runtime-v1',upstream_commit=lock['commit'],files=hashes,pack_inputs=pack_identity,
                      pretraining_packing='lossless-stream-v1',sft_loss='global-supervised-token-mean',
                      old_checkpoints_require_explicit_legacy_policy=True)
        stage=Path(temp)/'manifest';stage.write_text(json.dumps(manifest,indent=2)+'\n')
        os.replace(stage,manifest_path)
        return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nanochat-dir',type=Path,required=True)
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    try: print(json.dumps(install(args.nanochat_dir,args.verify_only),indent=2))
    except (ValueError,OSError,subprocess.SubprocessError) as exc: ap.error(str(exc))


if __name__=='__main__': main()
