#!/usr/bin/env python3
"""Versioned, deterministic migration of v8 prompts to Belarusian-only SFT.

Original seeds and sealed evaluation files are never modified. This migration
translates a finite explicit template vocabulary, preserves assistant text,
deduplicates full conversations, and assigns prompt families to one split.
It is a data-contract correction, not evidence of improved model quality.
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_pipeline.strict_io import iter_jsonl, canonical_text, sha256_file
from data_pipeline.sft_schema import validate_conversation

VERSION = "sft-v9-be-prompt-groups-v1"
TRANSLATIONS = {
    "представься": "Прадстаўся па-беларуску",
    "who are you": "Хто ты",
    "explain who you are": "Растлумач, хто ты",
    "расскажи о себе": "Раскажы пра сябе",
    "расскажи о беларуси одним предложением": "Раскажы пра Беларусь адным сказам",
    "explain belarus in english": "Раскажы пра Беларусь па-англійску",
    "напиши ответ по-русски": "Напішы адказ па-руску",
    "what is minsk": "Што такое Мінск",
    "что такое тарашкевица": "Што такое тарашкевіца",
    "who was francysk skaryna": "Хто такі Францыск Скарына",
    "назови реки беларуси": "Назаві рэкі Беларусі",
    "tell me about polotsk": "Раскажы пра Полацк",
    "what is oauth": "Што такое OAuth",
    "что является столицей беларуси": "Якая сталіца Беларусі",
    "what is the capital of belarus": "Якая сталіца Беларусі",
    "какие реки есть в беларуси": "Якія рэкі ёсць у Беларусі",
    "name several rivers in belarus": "Назаві некалькі рэк Беларусі",
    "что ты знаешь о минске": "Што ты ведаеш пра Мінск",
    "ты chatgpt": "Ці з’яўляешся ты мадэллю ChatGPT",
    "ты мадэль openai": "Ці з’яўляешся ты мадэллю OpenAI",
}
PREFIXES = re.compile(r"^(?:правер|сфармулюй адказ|адкажы па-беларуску|коратка|please answer):\s*", re.I)
SUFFIXES = re.compile(r"(?:\s*[—–-]\s*коротко|[,.?]?\s*калі ласка|[.!?]?\s*коратка)[.!?\s]*$", re.I)


def prompt_core(text: str) -> str:
    text = canonical_text(text)
    while True:
        cleaned = SUFFIXES.sub("", PREFIXES.sub("", text)).strip()
        if cleaned == text:
            break
        text = cleaned
    return text.rstrip(".!? ")


def migrate_prompt(text: str) -> str:
    # Strip only documented synthetic wrappers; never translate arbitrary prose
    # with a heuristic language model or copy eval text into the training set.
    core = prompt_core(text)
    translated = TRANSLATIONS.get(core.lower())
    if translated is None:
        return text
    return translated + ("?" if "?" in text else ".")


def build_v9(pack: Path, val_ratio: float = 0.2) -> tuple[list[str], list[str], dict]:
    if not 0 < val_ratio < 1:
        raise ValueError("validation fraction must be in (0, 1)")
    groups: dict[str, list[tuple[str, dict]]] = {}
    seen: dict[str, dict] = {}
    provenance = []
    changed = duplicate = 0
    inputs = []
    for split in ("train", "val"):
        path = pack / "seed_sft" / f"sft_v8_{split}.be.jsonl"
        inputs.append({"path": str(path.relative_to(pack)), "sha256": sha256_file(path)})
        for lineno, raw in iter_jsonl(path):
            messages = validate_conversation(raw)
            migrated = [dict(m, content=migrate_prompt(m["content"]) if m["role"] == "user" else m["content"]) for m in messages]
            changed += int(migrated != messages)
            row = json.dumps(migrated, ensure_ascii=False, separators=(",", ":"))
            identity = hashlib.sha256(row.encode()).hexdigest()
            prompts = [prompt_core(m["content"]).casefold() for m in migrated if m["role"] == "user"]
            group = hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()
            record = {"source": inputs[-1]["path"], "line": lineno, "output_sha256": identity, "prompt_group": group, "translated": migrated != messages}
            provenance.append(record)
            if identity in seen:
                duplicate += 1
                continue
            seen[identity] = record
            groups.setdefault(group, []).append((row, record))
    if len(groups) < 2:
        raise ValueError("v9 needs at least two distinct prompt groups")
    buckets = {"train": [], "val": []}
    assignments = {group: "val" if int(hashlib.sha256((VERSION + group).encode()).hexdigest()[:16], 16) / 2**64 < val_ratio else "train" for group in groups}
    if len(set(assignments.values())) != 2:
        raise ValueError("requested split leaves an empty dataset; choose another fraction")
    for group in sorted(groups):
        buckets[assignments[group]].extend(row for row, _ in groups[group])
    for record in provenance:
        record["split"] = assignments[record["prompt_group"]]
    report = {"version": VERSION, "inputs": inputs, "source_rows": len(provenance),
              "translated_rows": changed, "duplicate_rows": duplicate,
              "prompt_groups": len(groups), "val_ratio_requested": val_ratio,
              "train_rows": len(buckets["train"]), "val_rows": len(buckets["val"]),
              "group_overlap": 0, "provenance": provenance}
    return buckets["train"], buckets["val"], report


if __name__ == "__main__":
    # Read-only planning; publication is exclusively via build_sft_mix.
    _, _, report = build_v9(ROOT)
    print(json.dumps(report, ensure_ascii=False, indent=2))
