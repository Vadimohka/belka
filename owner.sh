#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION="1.0.0"
CANONICAL_JSON="reports/state/belka_canonical_state.json"

usage() {
    cat << EOF
Belka Owner CLI v${VERSION}
Usage: bash owner.sh <command>

Commands:
  status            Show canonical state
  qc                Run QC pipeline (39, 40, 41, 43)
  integrity         Run repo integrity audit (44)
  cleanup-dry-run   Run cleanup dry-run (45)
  cleanup-apply     Apply cleanup (46) — requires BELKA_OWNER_APPROVED_CLEANUP=YES
  corpus-plan       Show corpus expansion plan

Rules:
  - owner.sh never trains.
  - Training commands require BELKA_OWNER_APPROVED_TRAINING=YES.
  - Never runs SFT, tokenizer rebuild, or checkpoint modification.
EOF
}

check_no_training() {
    if [[ "${BELKA_OWNER_APPROVED_TRAINING:-}" == "YES" ]]; then
        echo "ERROR: BELKA_OWNER_APPROVED_TRAINING=YES — training is allowed but owner.sh does not train."
        echo "Use dedicated training scripts instead."
        exit 2
    fi
}

cmd_status() {
    echo "============================================="
    echo " BELKA CANONICAL STATE"
    echo "============================================="
    if [[ -f "$CANONICAL_JSON" ]]; then
        if command -v python3 &>/dev/null; then
            python3 -c "
import json
d = json.load(open('$CANONICAL_JSON'))
print(f'Project:        {d[\"project\"]}')
print(f'Allowed action: {d[\"current_allowed_action\"]}')
print(f'Training:       {d[\"training_gates\"][\"TRAINING_ALLOWED\"]}')
print(f'SFT:            {d[\"training_gates\"][\"SFT_ALLOWED\"]}')
print()
print('Model Status:')
for k, v in d['model_status'].items():
    if k.startswith('D'): print(f'  {k}: {v}')
print()
print('Corpus:')
for k, v in d['corpus'].items():
    print(f'  {k}: {v}')
print()
print('Holdout:')
for k, v in d['holdout'].items():
    print(f'  {k}: {v}')
print()
print('QC:')
for k, v in d['quality_control'].items():
    print(f'  {k}: {v}')
print()
print('Next allowed actions:')
for a in d['next_allowed_actions']:
    print(f'  - {a}')
print()
print('Blocked:')
for a in d['blocked_actions']:
    print(f'  - {a}')
"
        else
            cat "$CANONICAL_JSON"
        fi
    else
        echo "Canonical state not found. Run integrity audit first: bash owner.sh integrity"
        return 1
    fi
    echo ""
    echo "TRAINING_ALLOWED=NO"
    echo "SFT_ALLOWED=NO"
}

cmd_qc() {
    check_no_training
    echo "=== Running QC Pipeline ==="
    echo "Scripts: 39 (provenance), 40 (leakage), 41 (create XLSX), 43 (audit XLSX)"
    echo ""

    local all_ok=true

    for script in \
        ops/owner_runs/39_OWNER_AUDIT_TRAINING_PROVENANCE.sh \
        ops/owner_runs/40_OWNER_CHECK_HOLDOUT_LEAKAGE.sh \
        ops/owner_runs/41_OWNER_CREATE_QUALITY_CONTROL_XLSX.sh \
        ops/owner_runs/43_OWNER_AUDIT_QUALITY_CONTROL_XLSX.sh; do
        echo "--- Running: $script ---"
        if bash "$script"; then
            echo "OK: $script"
        else
            echo "FAIL: $script"
            all_ok=false
        fi
        echo ""
    done

    if $all_ok; then
        echo "QC_PIPELINE=PASS"
    else
        echo "QC_PIPELINE=FAIL (some scripts failed)"
        return 1
    fi
}

cmd_integrity() {
    check_no_training
    echo "=== Running Repo Integrity Audit (44) ==="
    bash ops/owner_runs/44_OWNER_REPO_INTEGRITY_AUDIT.sh
}

cmd_cleanup_dry_run() {
    check_no_training
    echo "=== Running Cleanup Dry-Run (45) ==="
    bash ops/owner_runs/45_OWNER_REPO_CLEANUP_DRY_RUN.sh
}

cmd_cleanup_apply() {
    check_no_training
    if [[ "${BELKA_OWNER_APPROVED_CLEANUP:-}" != "YES" ]]; then
        echo "Cleanup apply requires owner approval."
        echo "Run: BELKA_OWNER_APPROVED_CLEANUP=YES bash owner.sh cleanup-apply"
        exit 0
    fi
    echo "=== Running Cleanup Apply (46) ==="
    bash ops/owner_runs/46_OWNER_REPO_CLEANUP_APPLY.sh
}

cmd_corpus_plan() {
    check_no_training
    echo "=== Corpus Expansion Plan ==="
    echo "TRAINING_ALLOWED=NO"
    echo ""
    echo "Planning script: 36"
    for script in \
        ops/owner_runs/36_OWNER_PLAN_CORPUS_EXPANSION_200M.sh; do
        if [[ -f "$script" ]]; then
            echo "  Present: $script"
        else
            echo "  Missing: $script"
        fi
    done
    echo ""
    if [[ -f reports/strategy/CORPUS_EXPANSION_200M_SUMMARY.md ]]; then
        echo "--- Summary ---"
        cat reports/strategy/CORPUS_EXPANSION_200M_SUMMARY.md
    fi
}

# Main
check_no_training

case "${1:-}" in
    status)          cmd_status ;;
    qc)              cmd_qc ;;
    integrity)       cmd_integrity ;;
    cleanup-dry-run) cmd_cleanup_dry_run ;;
    cleanup-apply)   cmd_cleanup_apply ;;
    corpus-plan)     cmd_corpus_plan ;;
    help|--help|-h)  usage ;;
    *)
        usage
        echo ""
        echo "Unknown command: ${1:-}"
        exit 1
        ;;
esac
