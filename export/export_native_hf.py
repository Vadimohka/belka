#!/usr/bin/env python3
"""Export a committed native checkpoint to an honest custom-architecture HF bundle."""
from __future__ import annotations
import argparse
import ast
import base64
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.provenance import artifact, sha256_file


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def native_source(source):
    """Keep native inference verbatim in semantics; remove trainer dependencies."""
    names = {'GPTConfig','norm','Linear','has_ve','apply_rotary_emb','CausalSelfAttention','MLP','Block','GPT'}
    methods = {'__init__','init_weights','_precompute_rotary_embeddings','_compute_window_sizes','get_device','forward'}
    body = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            if node.name == 'GPT':
                node.body = [method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in methods]
            body.append(node)
    if {node.name for node in body} != names:
        raise ValueError('unsupported native architecture source; required definitions missing')
    preamble = '''# Derived from the selected nanochat GPT source; see NANOCHAT_LICENSE and export_manifest.json.
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_belka import sdpa_attention
COMPUTE_DTYPE = torch.float32
def print0(*args, **kwargs): pass
class ReferenceAttention:
    @staticmethod
    def flash_attn_func(q, k, v, causal=True, window_size=(-1,-1)):
        return sdpa_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), window_size, q.size(2)!=k.size(2), causal).transpose(1,2)
    @staticmethod
    def flash_attn_with_kvcache(*args, **kwargs):
        raise ValueError('HF reference export does not implement KV caching')
flash_attn = ReferenceAttention()
'''
    return preamble+'\n'+ast.unparse(ast.Module(body=body, type_ignores=[]))+'\n'


def adapter_package(directory, runtime):
    directory.mkdir(parents=True, exist_ok=True)
    for path in (ROOT/'export/hf_belka').glob('*.py'):
        shutil.copyfile(path, directory/path.name)
    (directory/'native_model.py').write_text(native_source((runtime/'nanochat/gpt.py').read_text()), encoding='utf-8')
    shutil.copyfile(runtime/'nanochat/belka_attention.py', directory/'attention_belka.py')
    shutil.copyfile(runtime/'LICENSE', directory/'NANOCHAT_LICENSE')
    name = 'belka_hf_'+uuid.uuid4().hex
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    config = importlib.import_module(name+'.configuration_belka').BelkaConfig
    model = importlib.import_module(name+'.modeling_belka').BelkaForCausalLM
    tokenizer = importlib.import_module(name+'.tokenization_belka').BelkaTokenizer
    return config, model, tokenizer


def tokenizer_json(tokenizer, directory):
    enc = tokenizer.enc
    write_json(directory/'tokenizer.tiktoken.json', dict(
        pat_str=enc._pat_str,
        mergeable_ranks={base64.b64encode(key).decode('ascii'): value for key,value in enc._mergeable_ranks.items()},
        special_tokens=enc._special_tokens))


def export_bundle(runtime, checkpoint_dir, step, tokenizer_dir, output):
    import torch
    from safetensors.torch import save_file, load_file
    runtime, checkpoint_dir, tokenizer_dir = [Path(p).resolve(strict=True) for p in (runtime,checkpoint_dir,tokenizer_dir)]
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'output already exists: {output}')
    manifest = json.loads((runtime/'BELKA_RUNTIME_MANIFEST.json').read_text())
    if manifest.get('schema') != 'belka-runtime-v1' or not manifest.get('files'):
        raise ValueError('verified Belka runtime manifest required')
    for name,digest in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256_file(runtime/name) != digest:
            raise ValueError(f'runtime manifest mismatch: {name}')
    sys.path.insert(0, str(runtime))
    # Imports must resolve to this exact selected runtime, even in a long-lived caller.
    import nanochat
    if Path(nanochat.__file__).resolve().parent != runtime/'nanochat':
        raise ValueError('a different nanochat runtime is already imported; use a fresh exporter process')
    from nanochat.belka_checkpoint import _read_commit_manifest
    from nanochat.tokenizer import RustBPETokenizer
    marker = checkpoint_dir/f'commit_{step:06d}.json'
    committed = _read_commit_manifest(marker, step)
    for name,digest in committed['files'].items():
        path=checkpoint_dir/name
        if path.is_symlink() or not path.is_file() or sha256_file(path)!=digest:
            raise ValueError(f'checkpoint integrity failure: {name}')
    for name,digest in committed['tokenizer'].items():
        if sha256_file(tokenizer_dir/name)!=digest:
            raise ValueError(f'tokenizer identity mismatch: {name}')
    weights=checkpoint_dir/f'model_{step:06d}.pt'
    metadata=checkpoint_dir/f'meta_{step:06d}.json'
    if weights.name not in committed['files'] or metadata.name not in committed['files']:
        raise ValueError('checkpoint manifest does not bind model and metadata')
    config = json.loads(metadata.read_text())['model_config']
    # This pickle is a local training artifact verified against the checkpoint.
    # The exported consumer uses JSON vocabulary and safetensors only.
    tokenizer = RustBPETokenizer.from_directory(str(tokenizer_dir))
    if config['vocab_size'] != tokenizer.get_vocab_size():
        raise ValueError('model and tokenizer vocabulary sizes differ')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.hf-export-', dir=output.parent))
    try:
        Config,Model,Tokenizer = adapter_package(staging, runtime)
        cfg = Config(**config, bos_token_id=tokenizer.get_bos_token_id(),
                     eos_token_id=tokenizer.encode_special('<|assistant_end|>'),
                     pad_token_id=tokenizer.encode_special('<|assistant_end|>'))
        cfg.auto_map={'AutoConfig':'configuration_belka.BelkaConfig','AutoModelForCausalLM':'modeling_belka.BelkaForCausalLM'}
        cfg.architectures=['BelkaForCausalLM']
        cfg.dtype='float32'
        model = Model(cfg).float().eval()
        state = torch.load(weights, map_location='cpu', weights_only=True)
        model.model.load_state_dict(state, strict=True)
        del state
        probe = torch.tensor([[tokenizer.get_bos_token_id()]+tokenizer.encode('Беларуская мова.')[:7]])
        probe = probe[:, :cfg.sequence_len]
        with torch.inference_mode():
            expected = model(probe).logits.clone()
        save_file({k:v.detach().contiguous() for k,v in model.state_dict().items()}, staging/'model.safetensors', metadata={'format':'pt'})
        cfg.save_pretrained(staging)
        model.generation_config.save_pretrained(staging)
        model.load_state_dict(load_file(staging/'model.safetensors'), strict=True)
        with torch.inference_mode():
            actual = model(probe).logits
        if not torch.isfinite(actual).all() or not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
            raise ValueError('serialized model logits failed roundtrip')
        tokenizer_json(tokenizer, staging)
        hf_tokenizer=Tokenizer(str(staging/'tokenizer.tiktoken.json'), model_max_length=cfg.sequence_len)
        hf_tokenizer.save_pretrained(staging)
        token_config=json.loads((staging/'tokenizer_config.json').read_text())
        token_config.update(auto_map={'AutoTokenizer':['tokenization_belka.BelkaTokenizer',None]})
        write_json(staging/'tokenizer_config.json', token_config)
        report=dict(schema='belka-native-hf-v1', architecture='belka_nanochat', inference_dtype='float32',
                    source_runtime=artifact(runtime/'BELKA_RUNTIME_MANIFEST.json'),
                    source_checkpoint=artifact(marker), source_weights=artifact(weights),
                    source_tokenizer=artifact(tokenizer_dir/'tokenizer.pkl'),
                    probe_tokens=probe.tolist(), max_logit_error=float((actual-expected).abs().max()),
                    quality_status='NOT_EVALUATED', limitations=['no KV cache','no padded batches','no GGUF or vLLM adapter'])
        report['files']={path.name:sha256_file(path) for path in staging.iterdir() if path.is_file()}
        write_json(staging/'export_manifest.json', report)
        shutil.copyfile(ROOT/'export/HF_MODEL_CARD.md', staging/'README.md')
        # No partial bundle or overwrite is advertised as a successful export.
        staging.rename(output)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nanochat-dir',type=Path,required=True)
    ap.add_argument('--checkpoint-dir',type=Path)
    ap.add_argument('--base-dir',type=Path)
    ap.add_argument('--phase',choices=['base','sft','rl'],default='sft')
    ap.add_argument('--model-tag')
    ap.add_argument('--step',type=int)
    ap.add_argument('--tokenizer-dir',type=Path)
    ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args(argv)
    try:
        ck=args.checkpoint_dir
        if ck is None:
            if args.base_dir is None or not args.model_tag or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',args.model_tag):
                raise ValueError('supply --checkpoint-dir or --base-dir and a safe --model-tag')
            ck=args.base_dir/{'base':'base_checkpoints','sft':'chatsft_checkpoints','rl':'chatrl_checkpoints'}[args.phase]/args.model_tag
        step=args.step
        if step is None:
            steps=[int(path.stem.split('_')[-1]) for path in ck.glob('commit_[0-9][0-9][0-9][0-9][0-9][0-9].json')]
            if not steps: raise ValueError('no committed checkpoints found')
            step=max(steps)
        tok=args.tokenizer_dir or (args.base_dir/'tokenizer' if args.base_dir else None)
        if tok is None: raise ValueError('--tokenizer-dir is required without --base-dir')
        report=export_bundle(args.nanochat_dir,ck,step,tok,args.out_dir)
    except (OSError,ValueError,KeyError,ImportError,RuntimeError) as exc:
        ap.error(str(exc))
    print(json.dumps({'status':'EXPORTED','path':str(args.out_dir),'max_logit_error':report['max_logit_error']}))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
