"""Execute the shipped JavaScript stream parser against adversarial fragmentation."""
from pathlib import Path
import shutil
import subprocess
import pytest

def test_actual_browser_parser_all_split_boundaries_and_errors():
    node=shutil.which('node')
    if node is None:pytest.skip('Node is required for browser JavaScript contract test')
    root=Path(__file__).resolve().parents[1]
    code=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('ops/nanochat_fork/nanochat/ui.html','utf8');
const start=html.indexOf('        async function readSSE(');
const end=html.indexOf("        chatInput.addEventListener",start);
const c=vm.createContext({TextDecoder});vm.runInContext(html.slice(start,end),c);
const bytes=new TextEncoder().encode('data: {"token":"Прывітанне"}\n\ndata: {"done":true}\n\n');
async function parse(chunks){let i=0;return c.readSSE({read:async()=>i<chunks.length?{value:chunks[i++],done:false}:{done:true},cancel:async()=>{}},()=>{});}
(async()=>{
 for(let i=1;i<bytes.length;i++)assert.equal(await parse([bytes.slice(0,i),bytes.slice(i)]),'Прывітанне');
 for(const wire of ['data: {"error":"failed"}\n\ndata: {"done":true}\n\n','data: {"token":"partial"}\n\n','data: {"done":true}\n\n']){
  await assert.rejects(()=>parse([new TextEncoder().encode(wire)]));
 }
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run([node,'-e',code],cwd=root,check=True,timeout=15)
