#!/usr/bin/env bash
set -euo pipefail
cat <<'EOF'
AUTO_TUNE_3070TI_PLAN
1. Do not start with SFT. First tune base training throughput.
2. Run short 50-100 iter probes for seq_len x micro_batch candidates.
3. Pick the largest config with peak VRAM 6.0-7.2GB and no OOM.
4. If d8 profile uses <4GB even at seq_len=2048 and batch>=16, probe d12_80m.
5. Keep fp16. Do not switch to bf16 on RTX 3070 Ti.
6. Do not increase grad_accum to use VRAM; increase micro_batch/seq_len/model size.
7. Record tokens/sec, peak VRAM, val bpb after probe.
EOF
