# SFT v8 Behavior Repair Data Report

Status: **PASS_DATA_ONLY**

This pack is a behavior-repair dataset for Belka after SFT v7 was rejected for template substitution, wrong entity binding, repetition loops, and false refusals.

## Counts

- Train: 450
- Val: 120
- Eval: 180
- Format: raw CustomJSON arrays, one conversation per line.

## Training policy

Train only from:

`.workspace/nanochat_base/base_checkpoints/belka-d12-base-v2/model_006000.pt`

Do **not** continue from:

- `belka-d12-sft-v6b`
- `belka-d12-sft-v7`

## Quality gates

- Max train category share: 0.1444
- Train/val exact overlap: 0
- Train/eval exact overlap: 0
- Val/eval exact overlap: 0
- Duplicate assistant answers in train: 0
- Forbidden prefix `Галоўнае пра`: 0
- Max repeated assistant prefix in train: Я застаюся ў = 5

## Main corrections vs SFT v7

- Known fact questions must not be answered with unknown/refusal templates.
- `Якая сталіца Беларусі?` must answer `Мінск`.
- River questions must mention rivers, not unrelated entities.
- Skaryna is treated as a known factual topic.
- Harmless password recovery and OAuth explanations are helpful, not refused.
- Harmful cyber requests are refused without procedural instructions.
- MeetMesh answers are short and grounded.

## Category distribution

{
  "identity_minimal": 20,
  "language_lock_factual": 45,
  "known_belarus_facts": 65,
  "known_people_culture": 55,
  "known_places_geography": 55,
  "orthography_language": 35,
  "calibrated_unknowns": 35,
  "calibrated_safety_refusal": 35,
  "harmless_security_privacy_help": 25,
  "meetmesh_grounded": 35,
  "general_helpfulness": 45
}

## Training gate

`SFT_V8_TRAINING_ALLOWED=OWNER_ONLY_AFTER_VALIDATION`

The agent must validate this pack first and must not launch training by itself.
