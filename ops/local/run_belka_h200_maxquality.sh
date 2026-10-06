#!/usr/bin/env bash
set -euo pipefail
# Single-H200 dispatcher. No installation or training on --help/plan/check.
PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PACK_DIR"
usage() {
  cat <<'HELP'
Usage: bash ops/local/run_belka_h200_maxquality.sh COMMAND [options]
  plan     Resolve an immutable plan from prepared data; --model-tag is required.
  check    Verify a plan and optionally its measured hardware report.
  probe    Measure real base/SFT memory and speed on one H200.
  execute  Train the verified plan; requires its passing H200 report.

Examples (after ops/local/prepare_h200.sh):
  ... plan --profile h200_max --model-tag belka-h200-v1 --output .workspace/h200-ready/plan.json
  ... probe --plan .workspace/h200-ready/plan.json --output .workspace/h200-ready/hardware.json
  ... execute --plan .workspace/h200-ready/plan.json --hardware-report .workspace/h200-ready/hardware.json
  ... execute --plan .workspace/h200-ready/plan.json --hardware-report .workspace/h200-ready/hardware.json --resume

Planning options: --nanochat-dir, --base-dir, --python, --epochs or --base-iterations,
--sft-epochs or --sft-iterations, --budget-hours (default 168). See COMMAND --help.
CPU pipeline acceptance: plan --profile smoke, then execute --allow-cpu-smoke.
See docs/H200_TRAINING.md for preparation and transfer to the GPU server.
HELP
}
if [[ $# == 0 || "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
command="$1"; shift
case "$command" in plan|check|probe|execute) ;; *) usage >&2; exit 2 ;; esac
runtime="${NANOCHAT_DIR:-$PACK_DIR/.workspace/nanochat}"
py="${BELKA_PYTHON:-}"
args=("$@"); plan=""
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --nanochat-dir=*) runtime="${args[i]#*=}" ;;
    --python=*) py="${args[i]#*=}" ;;
    --plan=*) plan="${args[i]#*=}" ;;
    --nanochat-dir|--python|--plan)
      if ((i+1>=${#args[@]})) || [[ "${args[i+1]}" == --* ]]; then
        echo "ERROR: ${args[i]} requires a value" >&2; exit 2
      fi
      case "${args[i]}" in
        --nanochat-dir) runtime="${args[i+1]}" ;;
        --python) py="${args[i+1]}" ;;
        --plan) plan="${args[i+1]}" ;;
      esac ;;
  esac
done
# Inspect hardware in the same venv used by the actual trainer subprocesses.
if [[ -n "$plan" ]]; then
  py="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "$plan")"
fi
py="${py:-$runtime/.venv/bin/python}"
if [[ ! -x "$py" ]]; then
  echo "ERROR: Python environment missing: $py; run ops/local/prepare_h200.sh or set BELKA_PYTHON" >&2; exit 2
fi
if [[ "$command" == probe ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
  exec "$py" tools/h200_probe.py "$@"
fi
if [[ "$command" == execute ]]; then export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"; fi
if [[ "$command" == plan ]]; then
  exec "$py" tools/training_plan.py plan --nanochat-dir "$runtime" --python "$py" "$@"
fi
exec "$py" tools/training_plan.py "$command" "$@"
