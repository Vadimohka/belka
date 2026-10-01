#!/usr/bin/env python3
"""Apply a source-hash-bound Belka runtime adaptation to pinned nanochat.

No network, model import, tokenizer loading, or training occurs. Every candidate
is compiled before the first write. The final receipt binds upstream, patcher,
overlays and generated outputs. Unknown/edited files fail closed; no force mode.
Legacy unreceipted patched workspaces must be preserved and a fresh checkout used.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

PACK=Path(__file__).resolve().parents[2]
PIN='92d63d4e8bb4df75c3b71618f31ddde2378b2bcd'
RECEIPT='_BELKA_RUNTIME.json'


def digest(data):return hashlib.sha256(data).hexdigest()


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError(f'upstream anchor not unique: {old[:80]!r}')
    return text.replace(old,new,1)


def replace_function(text,name,replacement):
    matches=[n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    if len(matches)!=1:raise ValueError(f'function {name} missing/ambiguous')
    node=matches[0];lines=text.splitlines(keepends=True)
    start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
    return ''.join(lines[:start])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])


def _load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def source_inputs(pack):
    files=[Path(__file__).resolve(),pack/'configs/nanochat_upstream.json',
           pack/'ops/local/patch_nanochat_for_belarusian.py',pack/'ops/local/patch_nanochat_branding.py',
           pack/'data_pipeline/artifact_store.py',pack/'data_pipeline/corpus_contract.py',pack/'data_pipeline/sft_schema.py']
    files += [p for p in (pack/'ops/nanochat_fork').rglob('*') if p.is_file() and p.suffix in ('.py','.html','.svg')]
    return {str(p.relative_to(pack)):digest(p.read_bytes()) for p in sorted(files)}


def verify(repo,pack):
    receipt=json.loads((repo/RECEIPT).read_text(encoding='utf-8'))
    if receipt['upstream_commit']!=PIN or receipt['source_inputs']!=source_inputs(pack):
        raise ValueError('runtime recipe changed; prepare a fresh pinned checkout (do not overwrite local edits)')
    for name,expected in receipt['outputs'].items():
        path=repo/name
        if not path.is_file() or path.is_symlink() or digest(path.read_bytes())!=expected:
            raise ValueError(f'generated runtime file changed: {name}')
    # Also bind the unchanged model/tokenizer/optimizer to the reviewed pin.
    lock=json.loads((pack/'configs/nanochat_upstream.json').read_text())
    for name,expected in lock['files'].items():
        if name not in receipt['outputs'] and digest((repo/name).read_bytes())!=expected:
            raise ValueError(f'upstream runtime drift: {name}')
    return receipt


def candidates(repo,pack):
    lock=json.loads((pack/'configs/nanochat_upstream.json').read_text())
    if lock['commit']!=PIN:raise ValueError('upstream lock and adapter disagree')
    for name,expected in lock['files'].items():
        path=repo/name
        if path.is_symlink() or not path.is_file() or digest(path.read_bytes())!=expected:
            raise ValueError(f'not the supported pristine upstream file: {name}; use a fresh pinned checkout')
    result={}
    read=lambda name:(repo/name).read_text(encoding='utf-8')
    # Empty NANOCHAT_DTYPE means automatic detection, not a dictionary KeyError.
    common=replace_once(read('nanochat/common.py'),'if env is not None:','if env:')
    common=replace_once(common,'        return _DTYPE_MAP[env],', '        if env not in _DTYPE_MAP:\n            raise ValueError(f"Unsupported NANOCHAT_DTYPE={env!r}")\n        return _DTYPE_MAP[env],')
    brand=_load(pack/'ops/local/patch_nanochat_branding.py','belka_branding_recipe')
    common=replace_function(common,'print_banner',brand.REPLACEMENT)
    result['nanochat/common.py']=common
    engine=read('nanochat/engine.py')
    engine=replace_function(engine,'timeout','')
    engine=replace_function(engine,'eval_with_timeout','def eval_with_timeout(formula, max_time=3):\n    from nanochat.belka_calculator import calculate\n    return calculate(formula)')
    engine=replace_function(engine,'use_calculator','def use_calculator(expr):\n    from nanochat.belka_calculator import calculate\n    return calculate(expr)')
    engine+='\n# BELARUSIAN_SUPERPACK_DTYPE_ENGINE: pinned upstream already uses COMPUTE_DTYPE.\n'
    result['nanochat/engine.py']=engine
    flash=read('nanochat/flash_attention.py')
    # Upstream q/k/v must already be coherent. Reject, do not silently recast.
    needle='def _sdpa_attention(q, k, v, window_size, enable_gqa):'
    flash=replace_once(flash,needle,needle+'\n    # BELARUSIAN_SUPERPACK_DTYPE_SDPA\n    if k.dtype != q.dtype or v.dtype != q.dtype:\n        raise ValueError("q/k/v compute dtype mismatch")')
    result['nanochat/flash_attention.py']=flash
    gpt=read('nanochat/gpt.py')
    gpt=replace_once(gpt,'        T0 = 0 if kv_cache is None else kv_cache.get_pos()',
        '        T0 = 0 if kv_cache is None else kv_cache.get_pos()\n        if T0 + T > self.cos.size(1) or (kv_cache is not None and T0 + T > kv_cache.max_seq_len):\n            raise ValueError("prompt plus decode exceeds model/KV context capacity")')
    result['nanochat/gpt.py']=gpt
    optimizer=read('nanochat/optim.py')
    if optimizer.count('op=dist.ReduceOp.AVG')!=3:raise ValueError('optimizer mean-reduction anchors changed')
    optimizer=optimizer.replace('op=dist.ReduceOp.AVG','op=dist.ReduceOp.SUM')
    if optimizer.count('.get_future()')!=5:raise ValueError('optimizer async Work anchors changed')
    optimizer=optimizer.replace('.get_future()','')  # Work.wait is supported by Gloo and NCCL.
    optimizer=replace_once(optimizer,'        # Phase 1: launch all async reduce ops',
        '        # Preserve optimizer-owned gradient MEAN on NCCL and Gloo.\n        # Gloo does not implement ReduceOp.AVG; predivide once, then SUM.\n        if world_size > 1:\n            for group in self.param_groups:\n                for parameter in group["params"]:\n                    if parameter.grad is None:raise ValueError("missing gradient in optimizer group")\n                    parameter.grad.div_(world_size)\n        # Phase 1: launch all async reduce ops')
    optimizer=replace_once(optimizer,'dist.all_gather_into_tensor(p, p_slice, async_op=True)',
        'dist.all_gather_into_tensor(p.detach(), p_slice.detach().clone(), async_op=True)')
    result['nanochat/optim.py']=optimizer
    token_source=read('nanochat/tokenizer.py')
    token_anchor='    tokenizer_dir = os.path.join(base_dir, "tokenizer")'
    if token_source.count(token_anchor)!=2:raise ValueError('tokenizer path anchor drift')
    token_source=token_source.replace(token_anchor,'    from nanochat.belka_artifacts import resolve_tokenizer_dir\n    tokenizer_dir = str(resolve_tokenizer_dir(base_dir))')
    token_source=token_source.replace('torch.load(f, map_location=device)','torch.load(f, map_location=device, weights_only=True)')
    result['nanochat/tokenizer.py']=token_source
    # Preserve upstream chat_sft.py; create Belarusian-only data variant.
    patch=_load(pack/'ops/local/patch_nanochat_for_belarusian.py','belka_sft_recipe')
    source=patch.ensure_imports(read('scripts/chat_sft.py'))
    converted=patch.replace_by_markers(source)
    if converted is None:raise ValueError('SFT mixture anchor missing')
    sft=patch.add_generic_evals_gate(converted[0])[0]
    sft=replace_once(sft,'os.environ.get("BELKA_DISABLE_GENERIC_EVALS", "")','os.environ.get("BELKA_DISABLE_GENERIC_EVALS", "YES")')
    sft=replace_once(sft,'import torch\n','import torch\nfrom nanochat.belka_sft import SFTBatches, normalize_accumulated_gradients, global_mean_nats, horizon_finished\n')
    sft=replace_function(sft,'sft_data_generator_bos_bestfit','''def sft_data_generator_bos_bestfit(split, buffer_size=100):
    global approx_progress, current_epoch, sft_resume_state
    if split not in ("train", "val"):
        raise ValueError("split must be train or val")
    dataset = train_dataset if split == "train" else val_dataset
    batches = SFTBatches(dataset, tokenizer, args.device_batch_size, args.max_seq_len,
                         rank=ddp_rank, world_size=ddp_world_size, device=device)
    if split == "train" and args.resume_from_step >= 0:
        batches.load_state_dict(meta["sft_loader_state"])
    for x, y, state in batches:
        if split == "train":
            sft_resume_state = state["resume_state"]
            approx_progress = state["progress"]
            current_epoch = state["epoch"]
        yield x, y
''')
    sft=replace_once(sft,'while True:\n    flops_so_far',
        'while True:\n    last_step = horizon_finished(step, args.num_iterations, approx_progress)\n    flops_so_far')
    sft=replace_once(sft,'        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)',
        '        if args.eval_tokens <= 0: raise ValueError("positive eval_tokens required")\n        eval_steps = max(1, (args.eval_tokens + world_tokens_per_fwdbwd - 1) // world_tokens_per_fwdbwd)')
    sft=replace_once(sft,'    for micro_step in range(grad_accum_steps):\n        loss = model(x, y)\n        train_loss = loss.detach() # for logging\n        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here',
        '    local_count = torch.zeros((), dtype=torch.int64, device=device)\n    local_nats = torch.zeros((), dtype=torch.float64, device=device)\n    nominal_tokens = tokens_per_fwdbwd * grad_accum_steps\n    for micro_step in range(grad_accum_steps):\n        loss = model(x, y, loss_reduction="sum")\n        local_nats += loss.detach().to(torch.float64)\n        local_count += (y >= 0).sum()\n        loss = loss / nominal_tokens')
    sft=replace_once(sft,'        progress = max(progress, approx_progress) # only increase progress monotonically',
        '        progress = min(1.0, (step + 1) / args.num_iterations) if args.num_iterations > 0 else min(1.0, approx_progress)')
    sft=replace_once(sft,'        scaler.step(optimizer)',
        '        global_count = normalize_accumulated_gradients(model.parameters(), local_count, nominal_tokens)\n        train_loss = global_mean_nats(local_nats, global_count)\n        scaler.step(optimizer)')
    sft=replace_once(sft,'    else:\n        optimizer.step()',
        '    else:\n        global_count = normalize_accumulated_gradients(model.parameters(), local_count, nominal_tokens)\n        train_loss = global_mean_nats(local_nats, global_count)\n        optimizer.step()')
    sft=sft.replace('ema_beta**(step + 1)', 'ema_beta**step')
    sft=replace_once(sft,'dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)','dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MIN)')
    sft=replace_once(sft,'    lrm = get_lr_multiplier(progress)', '    if ddp and args.num_iterations == -1:\n        progress_tensor = torch.tensor(progress, dtype=torch.float64, device=device)\n        dist.all_reduce(progress_tensor, op=dist.ReduceOp.MIN)\n        progress = progress_tensor.item()\n    lrm = get_lr_multiplier(progress)')
    result['scripts/chat_sft_be.py']=patch.candidate_banner()+sft
    checkpoint=read('nanochat/checkpoint_manager.py')
    # Definitions at the end override the unsafe legacy writer/selector, while
    # preserving build_model / public helper interfaces used by upstream.
    checkpoint += "\nfrom nanochat.belka_checkpoint import save_checkpoint, load_checkpoint, find_last_step, load_optimizer_state, find_largest_model\n"
    result['nanochat/checkpoint_manager.py']=checkpoint
    base=read('scripts/base_train.py')
    base=replace_once(base,'from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint',
        'from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint\nfrom nanochat.belka_checkpoint import restore_training_state')
    base=replace_once(base,'dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]',
        'if resuming:\n    if meta_data["model_config"] != model_config_kwargs:\n        raise ValueError("resume model configuration mismatch")\n    restore_training_state(meta_data, scaler)\ndataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]')
    base=replace_once(base,'\n            rank=ddp_rank,\n        )','\n            rank=ddp_rank,\n            scaler_state=scaler.state_dict() if scaler is not None else None,\n        )')
    base=replace_once(base,'"min_val_bpb": min_val_bpb,','"min_val_bpb": min_val_bpb if min_val_bpb != float("inf") else None,')
    base=replace_once(base,'min_val_bpb = loop_state["min_val_bpb"]','min_val_bpb = loop_state["min_val_bpb"] if loop_state["min_val_bpb"] is not None else float("inf")')
    result['scripts/base_train.py']=base
    # Exact SFT restart is separate from warm-starting momentum from pretraining.
    sft=result['scripts/chat_sft_be.py']
    sft=replace_once(sft,'args = parser.parse_args()',
        'parser.add_argument("--resume-from-step", type=int, default=-1)\nparser.add_argument("--save-every", type=int, default=-1)\nargs = parser.parse_args()')
    sft=replace_once(sft,'model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)',
        'model, tokenizer, meta = load_model("sft" if args.resume_from_step >= 0 else "base", device, phase="train", model_tag=args.model_tag, step=args.resume_from_step if args.resume_from_step >= 0 else args.model_step)')
    sft=replace_once(sft,'if args.load_optimizer:', 'if args.load_optimizer and args.resume_from_step < 0:')
    resume_setup='''resolved_config=vars(args).copy()
if args.resume_from_step >= 0:
    from nanochat.belka_checkpoint import load_checkpoint, restore_training_state
    checkpoint_dir=os.path.join(base_dir,"chatsft_checkpoints",args.model_tag or f"d{depth}")
    _, restored_optimizer, meta=load_checkpoint(checkpoint_dir,args.resume_from_step,device,True,ddp_rank,_model=False)
    saved=meta.get("resolved_sft_config",{})
    ignored={"resume_from_step","save_every","run","model_step"}
    if any(saved.get(k)!=v for k,v in resolved_config.items() if k not in ignored):
        raise ValueError("SFT resume configuration differs; this is not an exact restart")
    optimizer.load_state_dict(restored_optimizer)
    restore_training_state(meta,scaler)
    del restored_optimizer
sft_resume_state=None
'''
    sft=replace_once(sft,'# SFT data mixture and DataLoader',resume_setup+'\n# SFT data mixture and DataLoader')
    sft=replace_once(sft,'step = 0\nwhile True:', 'step = args.resume_from_step if args.resume_from_step >= 0 else 0\nif args.resume_from_step >= 0:\n    saved_loop=meta["sft_loop_state"]\n    smooth_train_loss=saved_loop["smooth_train_loss"]\n    min_val_bpb=saved_loop["min_val_bpb"] if saved_loop["min_val_bpb"] is not None else float("inf")\n    total_training_time=saved_loop["total_training_time"]\n    progress=saved_loop["progress"]\nwhile True:')
    sft=replace_once(sft,'    if last_step:\n        output_dirname', '    if (last_step or (args.save_every > 0 and step > 0 and step % args.save_every == 0)) and step != args.resume_from_step:\n        output_dirname')
    sft=replace_once(sft,'                "step": step,\n                "val_bpb": val_bpb,',
        '                "step": step,\n                "resolved_sft_config": resolved_config,\n                "sft_loader_state": sft_resume_state,\n                "sft_loop_state": {"smooth_train_loss":smooth_train_loss, "min_val_bpb":min_val_bpb if min_val_bpb != float("inf") else None, "total_training_time":total_training_time, "progress":progress},\n                "val_bpb": val_bpb,')
    sft=replace_once(sft,'min_val_bpb = float("inf")','val_bpb = None\nmin_val_bpb = float("inf")')
    result['scripts/chat_sft_be.py']=sft
    result['scripts/chat_sft_be.py']=replace_once(result['scripts/chat_sft_be.py'],
        '\n            rank=ddp_rank,\n        )','\n            rank=ddp_rank,\n            scaler_state=scaler.state_dict() if scaler is not None else None,\n        )')
    # Copy reviewed pure contracts to both task/runtime namespaces as needed.
    copies={'data_pipeline/artifact_store.py':['nanochat/belka_artifacts.py','tasks/belka_artifacts.py'],
            'data_pipeline/corpus_contract.py':['nanochat/belka_corpus.py'],
            'data_pipeline/sft_schema.py':['tasks/belka_schema.py']}
    for src,targets in copies.items():
        for target in targets:result[target]=(pack/src).read_text(encoding='utf-8')
    for src in (pack/'ops/nanochat_fork').rglob('*'):
        if src.is_file() and src.suffix in ('.py','.html','.svg'):
            result[str(src.relative_to(pack/'ops/nanochat_fork'))]=src.read_text(encoding='utf-8')
    for name,text in result.items():
        path=repo/name
        if not path.resolve().is_relative_to(repo.resolve()) or path.is_symlink():raise ValueError('runtime output path escapes checkout')
        if path.exists() and name not in lock['files']:
            raise ValueError(f'unreceipted output would be overwritten: {name}; use a fresh checkout')
        if name.endswith('.py'):compile(text,name,'exec',dont_inherit=True)
    return result


def apply(repo,pack,check_only=False):
    repo,pack=repo.resolve(),pack.resolve()
    if (repo/RECEIPT).exists():return verify(repo,pack)
    outputs=candidates(repo,pack)
    receipt={'schema_version':1,'upstream_commit':PIN,'source_inputs':source_inputs(pack),
             'outputs':{name:digest(text.encode()) for name,text in sorted(outputs.items())},
             'packing':'belka-stream-v1','sft':'lossless-supervised-chunks-v1','gradient_reduction':'optimizer-average'}
    if check_only:return receipt
    # All semantic anchors, source hashes and compilation passed before writes.
    # Receipt is written last. An interrupted installation must not be trained.
    for name,text in {**outputs,RECEIPT:json.dumps(receipt,sort_keys=True,indent=2)+'\n'}.items():
        path=repo/name;path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='.belka-runtime-',dir=path.parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as f:
                f.write(text);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:Path(tmp).unlink(missing_ok=True)
    return verify(repo,pack)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nanochat-dir',required=True,type=Path)
    ap.add_argument('--pack-dir',default=PACK,type=Path)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    try:receipt=apply(args.nanochat_dir,args.pack_dir,args.check_only)
    except (OSError,ValueError,KeyError,SyntaxError) as exc:ap.error(str(exc))
    print(json.dumps({'ok':True,'upstream_commit':receipt['upstream_commit'],'outputs':len(receipt['outputs']),'check_only':args.check_only},indent=2))
if __name__=='__main__':main()
