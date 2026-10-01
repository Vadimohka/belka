#!/usr/bin/env python3
"""Export verified native Belka weights/config/tokenizer/runtime, not a fake HF model.

Requires a complete checkpoint and the exact reviewed runtime receipt. Optimizer
state, private logs and training data are excluded. The result is an immutable
native artifact; it does not imply Transformers/vLLM/GGUF architecture support.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,tempfile
from pathlib import Path
PACK=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PACK))
from tools.find_latest_checkpoint import inspect
from data_pipeline.artifact_store import publish,resolve_tokenizer_dir,sha256_file

RUNNER='''#!/usr/bin/env python3
import argparse,hashlib,json,os,sys
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text())
for name,spec in manifest['files'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():raise ValueError('unsafe export member')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    if h.hexdigest()!=spec['sha256']:raise ValueError('export checksum mismatch: '+name)
ap=argparse.ArgumentParser();ap.add_argument('--prompt',required=True);ap.add_argument('--max-tokens',type=int,default=128);args=ap.parse_args()
os.environ['NANOCHAT_BASE_DIR']=str(root);os.environ['NANOCHAT_DTYPE']='float32'
sys.path.insert(0,str(root/'runtime'))
import torch
from nanochat.gpt import GPT,GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine
config=GPTConfig(**json.loads((root/'model_config.json').read_text()))
with torch.device('meta'):model=GPT(config)
model.to_empty(device=torch.device('cpu'));model.init_weights()
state=torch.load(root/'model.pt',map_location='cpu',weights_only=True)
model.load_state_dict(state,strict=True);model.eval()
tokenizer=get_tokenizer();ids,_=tokenizer.render_conversation({'messages':[{'role':'user','content':args.prompt}]},max_tokens=2**63-1)
ids.append(tokenizer.encode_special('<|assistant_start|>'))
if args.max_tokens<1 or len(ids)+args.max_tokens>config.sequence_len:ap.error('prompt plus decoding exceeds context')
output=[];end={tokenizer.encode_special('<|assistant_end|>'),tokenizer.get_bos_token_id()}
with torch.inference_mode():
    for column,mask in Engine(model,tokenizer).generate(ids,num_samples=1,max_tokens=args.max_tokens,temperature=0,top_k=None,seed=42):
        if column[0] in end:break
        output.append(column[0])
print(tokenizer.decode(output))
'''


def export(checkpoint_dir,step,base,runtime,destination):
    info=inspect(checkpoint_dir,step)
    meta=json.loads(Path(info['metadata_path']).read_text())
    sys.path.insert(0,str(PACK/'ops/local'))
    from modernize_nanochat import verify
    receipt=verify(runtime,PACK)
    if meta.get('runtime_sha256')!=sha256_file(runtime/'_BELKA_RUNTIME.json'):raise ValueError('checkpoint runtime identity differs from export runtime')
    tokenizer=resolve_tokenizer_dir(base)
    actual={p.name:sha256_file(p) for p in tokenizer.glob('*') if p.is_file()}
    if actual!=meta.get('tokenizer_files'):raise ValueError('checkpoint tokenizer identity differs')
    files={'model.pt':Path(info['model_path']),'model_config.json':(json.dumps(meta['model_config'],sort_keys=True)+'\n').encode(),
           'run_inference.py':RUNNER.encode(), 'README.md':b'# Native Belka artifact\n\nRun python run_inference.py --prompt TEXT from this generation after installing the pinned runtime dependencies. CPU float32; no network or automatic downloads. Tokenizer pickle is trusted only after the recorded source hashes have been verified. This is NOT Transformers, vLLM or GGUF format.\n'}
    for p in tokenizer.glob('*'):
        if p.is_file():files['tokenizer/'+p.name]=p
    for p in runtime.rglob('*'):
        if p.is_file() and not any(x.startswith('.') for x in p.relative_to(runtime).parts) and p.suffix in ('.py','.html','.svg','.toml','.lock'):
            files['runtime/'+str(p.relative_to(runtime))]=p
    files['runtime/_BELKA_RUNTIME.json']=runtime/'_BELKA_RUNTIME.json'
    return publish(destination,'native_export',files,{'checkpoint_manifest_sha256':sha256_file(info['manifest_path']),'upstream_commit':receipt['upstream_commit'],'format':'belka-native-v1','model_quality_validated':False})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint-dir',required=True,type=Path);ap.add_argument('--step',required=True,type=int);ap.add_argument('--base-dir',required=True,type=Path);ap.add_argument('--nanochat-dir',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    try:
        paths=[p.resolve() for p in (args.checkpoint_dir,args.base_dir,args.nanochat_dir,args.out)]
        if any(p==PACK or not p.is_relative_to(PACK) for p in paths):raise ValueError('all paths must stay inside repository')
        checkpoint,base,runtime,destination=paths
        if destination.exists() and any(destination.iterdir()):raise ValueError('export destination is not empty; select a new directory')
        print(json.dumps(export(checkpoint,args.step,base,runtime,destination),indent=2))
    except (OSError,ValueError,KeyError) as exc:ap.error(str(exc))
if __name__=='__main__':main()
