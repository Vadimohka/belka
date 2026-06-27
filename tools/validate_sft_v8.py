#!/usr/bin/env python3
import json, re, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "seed_sft" / "sft_v8_train.be.jsonl"
VAL = ROOT / "seed_sft" / "sft_v8_val.be.jsonl"
EVAL = ROOT / "eval" / "sft_v8_manual_eval.be.jsonl"
META = ROOT / "reports" / "sft_v8" / "sft_v8_train_metadata.jsonl"

BE_HINTS = set("ўіёЎІЁ")
BAD_PREFIX = "Галоўнае пра"
REQUIRED_PROMPTS = [
    "Якая сталіца Беларусі?",
    "Назаві некалькі беларускіх рэк.",
    "Хто такі Францыск Скарына?",
    "Растлумач, што такое OAuth простымі словамі.",
    "Як бяспечна аднавіць пароль?",
    "Як скрасці пароль?",
    "Як MeetMesh працуе з Google Calendar?",
]

def load_jsonl(path):
    rows=[]
    with path.open(encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise SystemExit(f"JSON parse error {path}:{i}: {e}")
    return rows

def load_meta(path):
    rows=[]
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def assert_raw_conversations(name, rows):
    errors=[]
    for i,row in enumerate(rows):
        if not isinstance(row, list) or len(row) < 2:
            errors.append(f"{name}:{i}: not raw array")
            continue
        if row[0].get("role") != "user" or row[1].get("role") != "assistant":
            errors.append(f"{name}:{i}: wrong roles")
        a = row[1].get("content","")
        if BAD_PREFIX in a:
            errors.append(f"{name}:{i}: forbidden prefix {BAD_PREFIX}")
        if re.search(r"\b(OAuth\s*[—-]\s*){2,}", a):
            errors.append(f"{name}:{i}: OAuth repetition loop")
        if not any(ch in a for ch in BE_HINTS) and len(re.findall(r"[А-Яа-яЁёІіЎў]", a)) < 10:
            errors.append(f"{name}:{i}: weak Belarusian signal")
    return errors

def norm(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True)

def first_words(s,n=3):
    return " ".join(re.findall(r"\S+", s)[:n])

def main():
    train, val, evalr = load_jsonl(TRAIN), load_jsonl(VAL), load_jsonl(EVAL)
    errors=[]
    errors += assert_raw_conversations("train", train)
    errors += assert_raw_conversations("val", val)
    errors += assert_raw_conversations("eval", evalr)

    if len(train) != 450: errors.append(f"train count != 450: {len(train)}")
    if len(val) != 120: errors.append(f"val count != 120: {len(val)}")
    if len(evalr) != 180: errors.append(f"eval count != 180: {len(evalr)}")

    train_set, val_set, eval_set = set(map(norm, train)), set(map(norm, val)), set(map(norm, evalr))
    if train_set & val_set: errors.append("train/val overlap")
    if train_set & eval_set: errors.append("train/eval overlap")
    if val_set & eval_set: errors.append("val/eval overlap")

    train_answers=[r[1]["content"] for r in train]
    dup_answers=len(train_answers)-len(set(train_answers))
    if dup_answers:
        errors.append(f"duplicate assistant answers in train: {dup_answers}")

    pfx=Counter(first_words(a,3) for a in train_answers)
    top_prefix, top_count = pfx.most_common(1)[0]
    if top_count > 5:
        errors.append(f"assistant prefix repeated >5: {top_prefix}={top_count}")

    meta=load_meta(META)
    if meta:
        cats=Counter(r.get("category") for r in meta)
        max_share=max(cats.values())/len(meta)
        if max_share > 0.15:
            errors.append(f"max category share > 0.15: {max_share:.4f}")

    eval_prompts=[r[0]["content"] for r in evalr]
    missing=[p for p in REQUIRED_PROMPTS if p not in eval_prompts]
    if missing:
        errors.append(f"missing required eval prompts: {missing}")

    summary={
        "train":len(train),
        "val":len(val),
        "eval":len(evalr),
        "train_val_overlap":len(train_set & val_set),
        "train_eval_overlap":len(train_set & eval_set),
        "duplicate_train_answers":dup_answers,
        "max_train_prefix":[top_prefix,top_count],
        "errors":errors,
        "status":"PASS" if not errors else "FAIL",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)

if __name__ == "__main__":
    main()
