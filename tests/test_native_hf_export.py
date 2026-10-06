"""Actual native checkpoint -> standalone HF load, tokenizer and logits parity."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_native_export_roundtrip(tmp_path):
    pytest.importorskip('transformers')
    pytest.importorskip('safetensors')
    runtime=Path(os.environ.get('NANOCHAT_DIR', ROOT/'.workspace/nanochat'))
    if not (runtime/'BELKA_RUNTIME_MANIFEST.json').is_file():
        pytest.skip('verified nanochat runtime required')
    script=tmp_path/'exercise.py'
    script.write_text('''
import os, sys, json
from pathlib import Path
import torch
torch.set_num_threads(1)
root,runtime,work=map(Path,sys.argv[1:])
sys.path.insert(0,str(root));sys.path.insert(0,str(runtime))
os.environ['NANOCHAT_BASE_DIR']=str(work/'base')
from nanochat.gpt import GPT,GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.belka_checkpoint import save_checkpoint
from export.export_native_hf import export_bundle
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer=RustBPETokenizer.train_from_iterator(['Беларуская мова. Test literal special tokens.']*20, vocab_size=280)
tokdir=work/'base/tokenizer';tokdir.mkdir(parents=True);tokenizer.save(tokdir)
config=GPTConfig(sequence_len=256,vocab_size=280,n_layer=2,n_head=4,n_kv_head=2,n_embd=64,window_pattern='SL')
torch.manual_seed(7)
model=GPT(config);model.init_weights();model.eval()
# Exercise every trained path, not only the zero projections at initialization.
with torch.no_grad():
    for name,p in model.named_parameters():
        if 'c_proj' in name: p.normal_(std=.06)
    model.smear_lambda.fill_(.3)
    model.backout_lambda.fill_(.23)
ck=work/'base/base_checkpoints/tiny';ck.mkdir(parents=True)
save_checkpoint(str(ck),1,model.state_dict(),None,{'model_config':vars(config),'step':1},rank=0)
out=work/'exported'
export_bundle(runtime,ck,1,tokdir,out)
hf=AutoModelForCausalLM.from_pretrained(out,trust_remote_code=True,local_files_only=True,dtype=torch.float32).eval()
tok=AutoTokenizer.from_pretrained(out,trust_remote_code=True,local_files_only=True)
for length in (1,12,160):
    ids=torch.randint(0,280,(1,length))
    with torch.inference_mode():
        expected=model(ids);actual=hf(ids).logits
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
text='Беларуская мова і UTF-8!'
assert tok.encode(text,add_special_tokens=False)==tokenizer.encode(text)
assert tok.decode(tokenizer.encode(text))==text
literal='Літаральна <|bos|> і <|assistant_end|>'
assert tok.encode(literal,add_special_tokens=False)==tokenizer.encode(literal)
messages=[{'role':'system','content':'Адказвай па-беларуску.'},{'role':'user','content':'Прывітанне <|bos|>!'}]
native,_=tokenizer.render_conversation({'messages':messages},max_tokens=1000)
native.append(tokenizer.encode_special('<|assistant_start|>'))
assert tok.apply_chat_template(messages,add_generation_prompt=True)==native
ids=torch.tensor([native])
with torch.inference_mode():
    greedy=ids.clone()
    for _ in range(2): greedy=torch.cat([greedy,model(greedy)[:,-1].argmax(-1,keepdim=True)],dim=1)
    result=hf.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=2,do_sample=False,use_cache=False,eos_token_id=None,pad_token_id=tokenizer.get_bos_token_id())
assert torch.equal(result,greedy)
try: export_bundle(runtime,ck,1,tokdir,out)
except FileExistsError: pass
else: raise AssertionError('overwrote existing output')
with (tokdir/'tokenizer.pkl').open('ab') as f:f.write(b'tamper')
try: export_bundle(runtime,ck,1,tokdir,work/'bad')
except ValueError as error: assert 'tokenizer identity' in str(error)
else: raise AssertionError('exported mismatched tokenizer')
assert not (work/'bad').exists()
print('native/HF logits, greedy generation, tokenizer and chat framing matched')
''')
    env={**os.environ,'HF_MODULES_CACHE':str(tmp_path/'hfmodules'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    result=subprocess.run([sys.executable,str(script),str(ROOT),str(runtime.resolve()),str(tmp_path)],
                          capture_output=True,text=True,env=env,timeout=120)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert 'logits, greedy generation' in result.stdout
