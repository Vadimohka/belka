# D8 Pilot Accepted, Long Rejected

## Pilot (ACCEPTED)
- MODEL_TAG=belka-d8-base-v3-pilot
- CHECKPOINT=.workspace/nanochat_base_d8_v3/base_checkpoints/belka-d8-base-v3-pilot/model_030517.pt
- TOKENS_SEEN=500000000
- VAL_BPB=0.542525
- LOSS≈1.95

## Long (REJECTED)
- MODEL_TAG=belka-d8-base-v3-long
- CHECKPOINT=.workspace/nanochat_base_d8_v3/base_checkpoints/belka-d8-base-v3-long/model_061035.pt
- TOKENS_SEEN=1500000000
- REASON=multi-epoch degradation (~19 passes over ~80M unique tokens)

## Side-by-side
| Prompt | Pilot (500M) | Long (1.5B) |
|--------|-------------|------------|
| Мінск — сталіца | горад Горад ✅ | Гісторыя сучас ⚠️ |
| Скарына | Францішак Скары ✅ | нананана ❌ |
| Беларуская мова | Грам Грам ✅ | цьцьць ❌ |

## Next
- Block training until corpus expanded
- Pilot is current best d8 base
