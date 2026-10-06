"""Actual two-process Gloo optimizer contract, separate from NCCL/GPU claims."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import pytest

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=Path(os.environ.get('NANOCHAT_DIR',ROOT/'.workspace/nanochat'))
if not (RUNTIME/'BELKA_RUNTIME_MANIFEST.json').is_file():
    if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS')=='1':raise RuntimeError('required runtime missing')
    pytest.skip('requires patched runtime and torch',allow_module_level=True)


def test_two_rank_gloo_matches_single_global_batch_with_zero_target_rank(tmp_path):
    program=tmp_path/'gloo_contract.py'
    program.write_text(textwrap.dedent('''
        import os, sys
        os.environ['TORCHDYNAMO_DISABLE']='1'
        os.environ['NANOCHAT_DTYPE']=''
        sys.path.insert(0,sys.argv[1])
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp
        from nanochat.optim import MuonAdamW
        from nanochat.belka_runtime import normalize_supervised_gradients
        def make():
            torch.manual_seed(3)
            model=torch.nn.Linear(4,4)
            opt=MuonAdamW([
                dict(params=[model.weight],kind='muon',lr=.005,momentum=.9,ns_steps=5,beta2=.95,weight_decay=0.),
                dict(params=[model.bias],kind='adamw',lr=.002,betas=(.9,.95),eps=1e-8,weight_decay=0.)])
            return model,opt
        def worker(rank, address, target):
            torch.set_num_threads(1)
            dist.init_process_group('gloo',init_method=address,rank=rank,world_size=2)
            model,opt=make()
            data=torch.arange(12,dtype=torch.float32).view(3,4)/12
            for _ in range(3):
                loss=(model(data)**2).sum() if rank else (model(data)*0).sum()
                count=12 if rank else 0
                loss.backward()
                normalize_supervised_gradients(model,count,loss.detach())
                opt.step();opt.zero_grad(set_to_none=True)
            torch.save(model.state_dict(),target+str(rank)+'.pt')
            dist.destroy_process_group()
        if __name__=='__main__':
            address='file://'+sys.argv[2]
            mp.spawn(worker,args=(address,sys.argv[3]),nprocs=2,join=True)
            torch.set_num_threads(1)
            model,opt=make()
            data=torch.arange(12,dtype=torch.float32).view(3,4)/12
            for _ in range(3):
                loss=(model(data)**2).mean()
                loss.backward();opt.step();opt.zero_grad(set_to_none=True)
            for rank in range(2):
                actual=torch.load(sys.argv[3]+str(rank)+'.pt',weights_only=True)
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(actual[key],value,rtol=2e-5,atol=2e-6)
            print('two-rank global supervised gradients and actual MuonAdamW updates match')
    '''))
    env=os.environ.copy();env['TORCHDYNAMO_DISABLE']='1'
    result=subprocess.run([sys.executable,str(program),str(RUNTIME),str(tmp_path/'rendezvous'),str(tmp_path/'rank')],
                          env=env,text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
