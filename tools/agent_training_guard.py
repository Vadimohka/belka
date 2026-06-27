#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

AGENT_ALLOWED = {"prepare", "prepare-and-plan", "plan", "audit", "agent-prepare", "analyze", "analyze-owner-run", "post-run-analysis"}
OWNER_ACTIONS = {
    "data-download", "download", "data-build", "build-dataset", "vram-probe", "probe",
    "train-base-v2", "base-v2", "train", "safe", "aggressive", "sft", "sft-v6"
}
FORBIDDEN_PATTERNS = [
    r"torchrun\b", r"python\s+-m\s+scripts\.base_train", r"python\s+-m\s+scripts\.chat_sft",
    r"run_belka_from_scratch_safe\.sh", r"run_all_3070ti_safe\.sh", r"run_all_3070ti_aggressive\.sh",
    r"run_belka_sft_safe\.sh", r"--phase\s+(data-download|train|base|safe|probe|vram-probe|sft)"
]

def classify_phase(phase: str, mode: str) -> tuple[bool, str]:
    p = phase.strip().lower()
    if mode == "agent":
        if p in AGENT_ALLOWED:
            return True, "allowed_agent_phase"
        if p in OWNER_ACTIONS or any(k in p for k in ["train", "base", "sft", "safe", "probe", "download", "build"]):
            return False, "blocked_owner_action_for_agent"
        return False, "unknown_phase_for_agent"
    return True, "owner_mode_not_blocked_by_phase_guard"

def scan_command(command: str) -> list[str]:
    hits = []
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, command):
            hits.append(pat)
    return hits

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", default=".")
    ap.add_argument("--mode", choices=["agent", "owner"], default="agent")
    ap.add_argument("--phase", default="prepare-and-plan")
    ap.add_argument("--command", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    ok, reason = classify_phase(args.phase, args.mode)
    command_hits = scan_command(args.command) if args.command else []
    if args.mode == "agent" and command_hits:
        ok = False
        reason = "blocked_forbidden_training_command_for_agent"
    out = {
        "mode": args.mode,
        "phase": args.phase,
        "allowed": ok,
        "reason": reason,
        "command_hits": command_hits,
        "policy": "agent_must_not_download_build_probe_or_train; owner runs scripts from ops/owner_runs only"
    }
    if args.json_out:
        p = Path(args.json_out); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if ok else 7

if __name__ == "__main__":
    raise SystemExit(main())
