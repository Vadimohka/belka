#!/usr/bin/env python3
"""Read-only bundled tokenizer/data integration check, never launches training."""
import argparse,json,os,sys
from pathlib import Path
PACK=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--nanochat-dir',type=Path,required=True);ap.add_argument('--base-dir',type=Path,required=True);args=ap.parse_args()
    runtime=args.nanochat_dir.resolve();base=args.base_dir.resolve()
    if not runtime.is_relative_to(PACK) or not base.is_relative_to(PACK):ap.error('paths must stay in repository')
    os.environ['NANOCHAT_BASE_DIR']=str(base);os.environ['NANOCHAT_DTYPE']='float32'
    sys.path.insert(0,str(PACK));sys.path.insert(0,str(runtime))
    from data_pipeline.artifact_store import resolve
    directory,manifest=resolve(base,'bundle')
    from nanochat.tokenizer import get_tokenizer,get_token_bytes
    from nanochat.dataset import split_parquet_files
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
    tokenizer=get_tokenizer();token_bytes=get_token_bytes()
    phrases=['Беларуская мова.','Тарашкевіца: сьвет, жыцьцё.','Ў І Ё ў і ё','3.14 — гэта лік.','Сям’я і сям\'я.','😀 Unicode']
    for text in phrases:
        if tokenizer.decode(tokenizer.encode(text))!=text:raise ValueError('tokenizer UTF-8 roundtrip failed')
    if len(token_bytes)!=tokenizer.get_vocab_size():raise ValueError('token bytes/vocabulary mismatch')
    generator=tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer,1,32,split='train',device='cpu')
    x,y,state=next(generator)
    if tuple(x.shape)!=(1,32) or tuple(y.shape)!=(1,32):raise ValueError('loader shape mismatch')
    print(json.dumps({'ok':True,'scope':'real restored bundle/tokenizer/one CPU data batch; no model training',
                      'vocab_size':tokenizer.get_vocab_size(),'train_shards':len(split_parquet_files('train')),
                      'val_shards':len(split_parquet_files('val')),'unicode_roundtrips':len(phrases),'packing_algorithm':state.get('algorithm')},indent=2))
if __name__=='__main__':main()
