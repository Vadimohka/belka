#!/usr/bin/env python3
"""Apply reviewed Belka corrections to a verified upstream source contract.

Generated files are checked for drift. All candidates compile before any write.
The recorded manifest binds source, overlays and generated runtime byte hashes.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
PACK = Path(__file__).resolve().parents[2]
PIN = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_function(text, name, replacement):
    functions = [node for node in ast.parse(text).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(functions) != 1:
        raise ValueError(f"expected exactly one function {name}")
    node = functions[0]
    lines = text.splitlines(keepends=True)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    return "".join(lines[:start]) + replacement + "\n" + "".join(lines[node.end_lineno:])


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"upstream contract changed: expected one {old[:70]!r}")
    return text.replace(old, new, 1)


def apply(repo):
    repo = repo.resolve()
    lock = json.loads((PACK / "configs/nanochat_upstream.json").read_text())
    if lock.get("commit") != PIN:
        raise ValueError("upstream lock and patcher commit disagree")
    patcher_sha = digest(Path(__file__).read_bytes())
    manifest_path = repo / "BELKA_RUNTIME_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("patcher_sha256") != patcher_sha or manifest.get("source_sha256") != lock["source_sha256"] or manifest.get("prepatch_sha256") != lock["prepatch_sha256"]:
            raise ValueError("patch contract changed; use a fresh separate checkout")
        for path, sha in manifest["source_sha256"].items():
            if path not in manifest["files"] and digest((repo / path).read_bytes()) != sha:
                raise ValueError(f"modified upstream source: {path}")
        for path, sha in (manifest["files"] | manifest.get("observed_files", {})).items():
            if digest((repo / path).read_bytes()) != sha:
                raise ValueError(f"modified generated runtime: {path}; preserve it and rebuild a separate checkout")
        for path, sha in manifest["overlays"].items():
            if digest((PACK / path).read_bytes()) != sha:
                raise ValueError("overlay changed; generate a fresh separate checkout")
        return manifest
    # Pristine SFT/model/optimizer sources determine the exact supported recipe.
    for path, sha in lock["source_sha256"].items():
        if digest((repo / path).read_bytes()) != sha:
            raise ValueError(f"unsupported upstream source: {path}; expected nanochat {PIN}")
    for path, sha in lock["prepatch_sha256"].items():
        if digest((repo / path).read_bytes()) != sha:
            raise ValueError(f"unreviewed pre-patch content: {path}; preserve it and use a fresh checkout")
    candidates = {}
    overlay_hashes = {}
    for name in ("belka_runtime.py", "belka_dataset.py", "belka_stream.py", "belka_checkpoint.py", "belka_sft.py"):
        source = PACK / "ops/nanochat_fork/nanochat" / name
        candidates["nanochat/" + name] = source.read_text()
        overlay_hashes[str(source.relative_to(PACK))] = digest(source.read_bytes())
    for relative in ("tasks/customjson.py", "scripts/chat_web.py", "nanochat/ui.html", "nanochat/logo.svg"):
        source = PACK / "ops/nanochat_fork" / relative
        overlay_hashes[str(source.relative_to(PACK))] = digest(source.read_bytes())
    candidates["belka_data/__init__.py"] = "# Vendored Belka training data contracts.\n"
    for name in ("strict_io.py", "sft_schema.py", "sft_language.py", "detect_belarusian.py"):
        source = PACK / "data_pipeline" / name
        candidates["belka_data/" + name] = source.read_text().replace("from data_pipeline.", "from belka_data.")
        overlay_hashes[str(source.relative_to(PACK))] = digest(source.read_bytes())
    source = repo / "nanochat/engine.py"
    text = source.read_text()
    start = text.index("@contextmanager\ndef timeout(")
    end = text.index("# -----------------------------------------------------------------------------\nclass KVCache", start)
    text = text[:start] + "from nanochat.belka_runtime import use_calculator\n\ndef eval_with_timeout(formula, max_time=3):\n    return use_calculator(formula)\n\n" + text[end:]
    candidates["nanochat/engine.py"] = text
    text = (repo / "nanochat/flash_attention.py").read_text()
    text = replace_function(text, "_sdpa_attention", '''def _sdpa_attention(q, k, v, window_size, enable_gqa, causal=True):
    from nanochat.belka_runtime import sdpa_attention
    return sdpa_attention(q, k, v, window_size, enable_gqa, causal)
''')
    text = text.replace("_sdpa_attention(q, k, v, window_size, enable_gqa)", "_sdpa_attention(q, k, v, window_size, enable_gqa, causal)")
    text = replace_function(text, "flash_attn_with_kvcache", '''def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v,
                                         cache_seqlens=cache_seqlens, causal=causal, window_size=window_size)
    from nanochat.belka_runtime import cached_sdpa
    return cached_sdpa(q, k_cache, v_cache, k, v, cache_seqlens, causal, window_size)
''')
    text += "\n# BELARUSIAN_SUPERPACK_DTYPE_SDPA: dtype normalization is in belka_runtime.\n"
    candidates["nanochat/flash_attention.py"] = text
    text = (repo / "nanochat/dataset.py").read_text()
    for name, args in [("list_parquet_files", "data_dir=None, warn_on_legacy=False"), ("parquets_iter_batched", "split, start=0, step=1")]:
        values = "data_dir, warn_on_legacy" if name.startswith("list") else "split, start, step"
        text = replace_function(text, name, f"def {name}({args}):\n    from nanochat.belka_dataset import {name} as implementation\n    return implementation({values})\n")
    text = once(text, "base_dir = get_base_dir()", 'if not os.environ.get("NANOCHAT_BASE_DIR"):\n    raise ValueError("Belka requires an explicit NANOCHAT_BASE_DIR")\nbase_dir = get_base_dir()')
    start = text.index('if __name__ == "__main__":')
    text = text[:start] + 'if __name__ == "__main__":\n    raise SystemExit("Upstream corpus download disabled for Belarusian-only training. Restore the Belka bundle.")\n'
    candidates["nanochat/dataset.py"] = text
    text = (repo / "nanochat/dataloader.py").read_text()
    text = once(text, 'parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]',
                'from nanochat.belka_dataset import split_parquet_files\n    parquet_paths = split_parquet_files(parquet_paths, split)\n    if sum(pq.ParquetFile(p).num_row_groups for p in parquet_paths) < ddp_world_size:\n        raise ValueError("Not enough row groups for distributed loader; repartition the corpus before training")')
    text = once(text, '            else:\n                rg_idx = ddp_rank',
                '            else:\n                preceding = sum(pq.ParquetFile(p).num_row_groups for p in parquet_paths[:pq_idx])\n                rg_idx = (ddp_rank - preceding) % ddp_world_size')
    target = '    row_capacity = T + 1'
    insertion = '''    import os
    if os.environ.get("BELKA_PACKING", "bestfit") == "stream":
        from nanochat.belka_stream import stream_loader
        yield from stream_loader(tokenizer, B, T, split, tokenizer_threads, tokenizer_batch_size,
                                 device, resume_state_dict, buffer_size)
        return
    if os.environ.get("BELKA_PACKING", "bestfit") != "bestfit":
        raise ValueError("BELKA_PACKING must be bestfit or stream")
'''
    text = once(text, target, insertion + target)
    candidates["nanochat/dataloader.py"] = text
    text = (repo / "scripts/chat_sft_be.py").read_text()
    text = once(text, 'train_tasks = [',
                'from nanochat.belka_sft import validate_training_file\n_belka_sft_provenance = {"train": validate_training_file(identity_conversations_filepath), "val": validate_training_file(identity_conversations_val_filepath)}\ntrain_tasks = [')
    text = once(text, '"user_config": user_config,', '"belka_sft_provenance": _belka_sft_provenance,\n                "user_config": user_config,')
    text = once(text, 'if 0 < args.num_iterations <= it and split == "train":', 'if False:  # iteration budget is checked at optimizer-step boundaries')
    text = once(text, 'if consumed >= dataset_size:', 'if args.num_iterations <= 0 and consumed >= dataset_size:')
    text = once(text, 'while True:\n    flops_so_far', 'while True:\n    if args.num_iterations > 0:\n        last_step = step >= args.num_iterations\n        progress = min(step / args.num_iterations, 1.0)\n    flops_so_far')
    text = once(text, '            ids, mask = tokenizer.render_conversation(conversation)', '''            ids, mask = tokenizer.render_conversation(conversation, max_tokens=args.max_seq_len + 2)
            if len(ids) > row_capacity:
                raise ValueError("SFT conversation exceeds max_seq_len; explicitly curate or increase the profile")
            if not any(mask[1:]):
                raise ValueError("SFT conversation has no supervised targets")''')
    text = once(text, 'base_lrs = [group["lr"] for group in optimizer.param_groups]',
                'fresh_groups = [{k: v for k, v in group.items() if k != "params"} for group in optimizer.param_groups]')
    text = once(text, 'for group, base_lr in zip(optimizer.param_groups, base_lrs):\n            group["lr"] = base_lr',
                'for group, fresh in zip(optimizer.param_groups, fresh_groups):\n            group.update(fresh)')
    start = text.index('    for micro_step in range(grad_accum_steps):\n        loss = model(x, y)')
    end = text.index('    # step the optimizer', start)
    text = text[:start] + '''    micro_batches = []
    for micro_step in range(grad_accum_steps):
        micro_batches.append((x, y))
        x, y = next(train_loader)
        if args.num_iterations <= 0:
            progress = min(max(progress, approx_progress), 1.0)
    from nanochat.belka_runtime import supervised_scale
    scale = supervised_scale(micro_batches)
    train_loss = torch.zeros((), device=device)
    for batch_x, batch_y in micro_batches:
        loss = model(batch_x, batch_y, loss_reduction="sum") * scale
        train_loss += loss.detach()
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
    if ddp:
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        train_loss /= ddp_world_size
''' + text[end:]
    text = once(text, '    # Synchronize last_step across all ranks',
                '    if args.num_iterations <= 0 and step == 0:\n        last_step = False\n\n    # Synchronize last_step across all ranks')
    text = text.replace('ema_beta**(step + 1)', 'ema_beta**step')
    candidates["scripts/chat_sft_be.py"] = text
    text = (repo / "nanochat/checkpoint_manager.py").read_text()
    for name, arguments, values in (
        ("save_checkpoint", "checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0", "checkpoint_dir, step, model_data, optimizer_data, meta_data, rank"),
        ("load_checkpoint", "checkpoint_dir, step, device, load_optimizer=False, rank=0", "checkpoint_dir, step, device, load_optimizer, rank"),
        ("find_last_step", "checkpoint_dir", "checkpoint_dir"),
    ):
        text = replace_function(text, name, f"def {name}({arguments}):\n    from nanochat.belka_checkpoint import {name} as implementation\n    return implementation({values})\n")
    text = once(text, 'optimizer_data = torch.load(optimizer_path, map_location=device)',
                'from nanochat.belka_checkpoint import verify_checkpoint\n    verify_checkpoint(checkpoint_dir, step, load_optimizer=True, rank=rank)\n    optimizer_data = torch.load(optimizer_path, map_location=device, weights_only=True)')
    candidates["nanochat/checkpoint_manager.py"] = text
    text = (repo / "scripts/base_train.py").read_text()
    text = once(text, 'x, y, dataloader_state_dict = next(train_loader) # kick off',
                'dataloader_resume_before_batch = dataloader_resume_state_dict\nx, y, dataloader_state_dict = next(train_loader) # kick off')
    text = once(text, '        x, y, dataloader_state_dict = next(train_loader) # prefetch',
                '        dataloader_resume_before_batch = dataloader_state_dict\n        x, y, dataloader_state_dict = next(train_loader) # prefetch')
    text = once(text, '"dataloader_state_dict": dataloader_state_dict,',
                '"dataloader_state_dict": dataloader_resume_before_batch if os.environ.get("BELKA_PACKING") == "stream" else dataloader_state_dict,\n                "belka_packing": os.environ.get("BELKA_PACKING", "bestfit"),\n                "belka_compute_dtype": str(COMPUTE_DTYPE),\n                "belka_scaler_state": scaler.state_dict() if scaler is not None else None,')
    text = once(text, '"min_val_bpb": min_val_bpb,', '"min_val_bpb": min_val_bpb if math.isfinite(min_val_bpb) else None,')
    text = once(text, 'min_val_bpb = loop_state["min_val_bpb"]', 'min_val_bpb = loop_state["min_val_bpb"] if loop_state["min_val_bpb"] is not None else float("inf")')
    text = once(text, '    model.load_state_dict(model_data, strict=True, assign=True)',
                '    from nanochat.belka_checkpoint import validate_resume\n    validate_resume(meta_data, user_config, str(COMPUTE_DTYPE))\n    model.load_state_dict(model_data, strict=True, assign=True)')
    text = once(text, '    print0("GradScaler enabled for fp16 training")',
                '    print0("GradScaler enabled for fp16 training")\n    if resuming:\n        scaler.load_state_dict(meta_data["belka_scaler_state"])')
    text = once(text, '    # single training step\n',
                '    # Restore training RNG after repeated evaluation at the resume boundary.\n    if resuming and step == args.resume_from_step:\n        from nanochat.belka_checkpoint import restore_rng\n        restore_rng(meta_data.pop("_belka_rng_state"))\n    # single training step\n')
    text = once(text, '    for micro_step in range(grad_accum_steps):\n        loss = model(x, y)',
                '    train_loss = torch.zeros((), device=device)\n    for micro_step in range(grad_accum_steps):\n        loss = model(x, y)')
    text = once(text, 'train_loss = loss.detach() # for logging', 'train_loss += loss.detach() / grad_accum_steps # token-weighted mean for fixed-size base batches')
    text = once(text, '    # step the optimizer\n',
                '    if ddp:\n        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)\n        train_loss /= ddp_world_size\n    # step the optimizer\n')
    text = text.replace("dataloader_state_dict['pq_idx']", "dataloader_state_dict.get('pq_idx', '-')").replace("dataloader_state_dict['rg_idx']", "dataloader_state_dict.get('rg_idx', '-')")
    candidates["scripts/base_train.py"] = text
    for path, content in candidates.items():
        compile(content, path, "exec")
    manifest = {"version": 1, "patcher_sha256": patcher_sha, "upstream_commit": PIN, "source_sha256": lock["source_sha256"], "prepatch_sha256": lock["prepatch_sha256"],
                "observed_files": {path: sha for path, sha in lock["prepatch_sha256"].items() if path not in candidates},
                "overlays": overlay_hashes, "files": {p: digest(t.encode()) for p, t in candidates.items()},
                "packing_default": "bestfit; stream is opt-in single-rank exact resume"}
    for path, content in candidates.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as stream:
            stream.write(content)
            temporary = stream.name
        os.replace(temporary, target)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nanochat-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(apply(args.nanochat_dir), indent=2))
    except (OSError, ValueError, SyntaxError) as exc:
        parser.error(str(exc))
