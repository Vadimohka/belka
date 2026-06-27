# SFT V2 VAL_BPB Investigation

SFT_V2_VAL_BPB=0.0065
VAL_EXAMPLES=30
RISK=suspiciously_low

Root cause: val set (30 examples) generated from same templates as training data.
- 60/60 val messages have exact text overlap with train
- 100% of val assistant answers appear in train
- Template family: identity_v2 uses same 10 base templates for both train (150) and val (30)

Conclusion: METRIC_BUG
The val_bpb=0.0065 measures template memorization, not generalization.
This is NOT a valid quality metric.

Required fix: Create a diverse val set from different sources/templates.
Minimum 150 val examples from distinct generation templates.
