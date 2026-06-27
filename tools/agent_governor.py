#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

try:
    import yaml
except Exception:
    yaml = None


def read_yaml(path: Path) -> dict:
    if yaml is None:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def run(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    except Exception as e:
        class R:
            returncode=99; stdout=str(e)
        return R()


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def grep_code_forbidden(pack: Path):
    bad = re.compile(r"(\$HOME/src/nanochat|~/src/nanochat|/tmp/nanochat|\$HOME/\.cache/nanochat|~/\.cache/nanochat|~/data/be_texts|/home/vadimohka/src/nanochat)")
    findings=[]
    code_ext={".sh",".py",".yaml",".yml",".json",".toml"}
    skip={".git",".workspace","dist","data_input/downloads"}
    allowed_negative_examples={
        "ops/local/repo_guard.sh",
        "tests/test_smoke_scripts_static.py",
        "tools/audit_corpus_outputs.py",
        "configs/agent_governor_policy.yaml",
        "tools/agent_governor.py",
        "MANIFEST.yaml",
    }
    for p in pack.rglob("*"):
        if not p.is_file():
            continue
        rel=p.relative_to(pack)
        rels=str(rel)
        if rels in allowed_negative_examples:
            continue
        if any(part in skip for part in rel.parts):
            continue
        if p.suffix not in code_ext:
            continue
        try:
            text=p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i,line in enumerate(text.splitlines(),1):
            if bad.search(line):
                findings.append({"path":str(rel),"line":i,"text":line[:240]})
    return findings


def parse_peak_vram_from_logs(pack: Path):
    vals=[]
    for p in list((pack/"reports").rglob("*.log"))[:500]:
        try:
            t=p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in re.finditer(r"Peak memory usage:\s*([0-9.]+)\s*MiB", t):
            vals.append({"path":str(p.relative_to(pack)),"peak_gb":float(m.group(1))/1024.0})
        for m in re.finditer(r"PEAK_VRAM_GB\s*=\s*([0-9.]+)", t):
            vals.append({"path":str(p.relative_to(pack)),"peak_gb":float(m.group(1))})
    return vals


def find_downgrade_evidence(pack: Path):
    ev=[]
    pats=["CUDA out of memory","out of memory","RuntimeError","nan","inf","dtype mismatch"]
    for p in (pack/"reports").rglob("*.log"):
        try: t=p.read_text(encoding="utf-8", errors="ignore")
        except Exception: continue
        lines=t.splitlines()
        for i,line in enumerate(lines,1):
            lo=line.lower()
            if any(x.lower() in lo for x in pats):
                ev.append({"path":str(p.relative_to(pack)),"line":i,"text":line[:300]})
    return ev


def preflight(pack: Path):
    policy=read_yaml(pack/"configs/agent_governor_policy.yaml")
    result={"timestamp":datetime.now(timezone.utc).isoformat(),"pack_dir":str(pack.resolve()),"checks":[]}
    required=["ops/local/repo_guard.sh","configs/agent_governor_policy.yaml","configs/training_ladder_8gb.yaml"]
    for r in required:
        ok=(pack/r).exists()
        result["checks"].append({"check":r,"status":"PASS" if ok else "FAIL"})
    for name in ["WORKSPACE_DIR","NANOCHAT_DIR","NANOCHAT_BASE_DIR","LOCAL_TEXT_DIR","DOWNLOAD_DIR"]:
        val=os.environ.get(name)
        if val:
            result["checks"].append({"check":f"path_contained_{name}","status":"PASS" if is_inside(Path(val),pack) else "FAIL","value":val})
    bad=grep_code_forbidden(pack)
    result["forbidden_code_refs"]=bad
    result["checks"].append({"check":"forbidden_path_refs_in_code","status":"PASS" if not bad else "FAIL","count":len(bad)})
    peaks=parse_peak_vram_from_logs(pack)
    result["observed_peak_vram"]=peaks[-10:]
    if peaks:
        last=peaks[-1]["peak_gb"]
        status="WARN" if last < policy.get("training",{}).get("target_peak_vram_gb",{}).get("warn_below",4.0) else "PASS"
        result["checks"].append({"check":"last_peak_vram_not_underused","status":status,"last_peak_gb":round(last,3)})
    ev=find_downgrade_evidence(pack)
    result["downgrade_evidence_count"]=len(ev)
    return result


def training_plan(pack: Path):
    peaks=parse_peak_vram_from_logs(pack)
    last=peaks[-1]["peak_gb"] if peaks else None
    plan={
        "timestamp":datetime.now(timezone.utc).isoformat(),
        "from_scratch_only": True,
        "pretrained_base_used": False,
        "observed_last_peak_vram_gb": last,
        "decision": "plan_only",
        "recommended_next": []
    }
    if last is None or last < 4.0:
        plan["recommended_next"].append({"step":"vram_probe","profile":"belka_d12_80m_probe","reason":"observed VRAM underuse; do not downgrade"})
        plan["recommended_next"].append({"step":"increase_seq_or_microbatch","profile":"belka_d8_40m_fast_base_v2","seq_len":"1536_or_2048","micro_batch":"probe 4..24"})
    else:
        plan["recommended_next"].append({"step":"base_v2","profile":"use best probe config","reason":"VRAM use acceptable"})
    plan["hard_rules"]=[
        "No silent downgrade without OOM/crash evidence.",
        "No more long SFT before data_foundry_v2 and base_v2.",
        "Books must be cleaned by build_books_clean_v2.py.",
        "Unknown-rights books are research_only."
    ]
    return plan


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pack-dir", default=".")
    ap.add_argument("--mode", choices=["preflight","training-plan","before-training","approve-downgrade"], default="preflight")
    ap.add_argument("--requested-profile", default="")
    ap.add_argument("--evidence-log", default="")
    ap.add_argument("--output-json")
    ap.add_argument("--output-md")
    args=ap.parse_args()
    pack=Path(args.pack_dir).resolve()
    if args.mode=="training-plan":
        result=training_plan(pack)
    elif args.mode=="approve-downgrade":
        if not args.evidence_log:
            print("DOWNGRADE_APPROVAL=DENY missing --evidence-log", file=sys.stderr); sys.exit(2)
        text=Path(args.evidence_log).read_text(encoding="utf-8",errors="ignore")
        ok=bool(re.search(r"CUDA out of memory|out of memory|NaN|Inf|dtype mismatch|RuntimeError",text,re.I))
        print("DOWNGRADE_APPROVAL=" + ("ALLOW" if ok else "DENY"))
        sys.exit(0 if ok else 3)
    else:
        result=preflight(pack)
        if args.mode=="before-training":
            ev=result.get("downgrade_evidence_count",0)
            result["requested_profile"]=args.requested_profile
            result["checks"].append({"check":"long_run_requires_user_approval","status":"WARN","message":"This tool verifies policy only; user approval must be explicit."})
    txt=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output_json:
        Path(args.output_json).write_text(txt,encoding="utf-8")
    if args.output_md:
        lines=["# Belka Governor Training Plan", "", "```json", txt, "```"]
        Path(args.output_md).write_text("\n".join(lines),encoding="utf-8")
    print(txt)
    failed=[c for c in result.get("checks",[]) if c.get("status")=="FAIL"]
    if failed:
        sys.exit(2)

if __name__=="__main__":
    main()
