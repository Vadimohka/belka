# Belka Governor Training Plan

```json
{
  "timestamp": "2026-05-13T12:30:39.219950+00:00",
  "from_scratch_only": true,
  "pretrained_base_used": false,
  "observed_last_peak_vram_gb": 1.32986328125,
  "decision": "plan_only",
  "recommended_next": [
    {
      "step": "vram_probe",
      "profile": "belka_d12_80m_probe",
      "reason": "observed VRAM underuse; do not downgrade"
    },
    {
      "step": "increase_seq_or_microbatch",
      "profile": "belka_d8_40m_fast_base_v2",
      "seq_len": "1536_or_2048",
      "micro_batch": "probe 4..24"
    }
  ],
  "hard_rules": [
    "No silent downgrade without OOM/crash evidence.",
    "No more long SFT before data_foundry_v2 and base_v2.",
    "Books must be cleaned by build_books_clean_v2.py.",
    "Unknown-rights books are research_only."
  ]
}
```