#!/usr/bin/env python3
"""Practical Belarusian language detector for corpus filtering.

This is a deliberately transparent scoring filter, not a neural language ID
model. It is tuned for safety in a Belarusian-only training pipeline: doubtful
records go to quarantine rather than silently entering the training corpus.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

BEL_SPECIFIC = set("ўЎіІёЁ")
CYR_RE = re.compile(r"[А-Яа-яЁёІіЎўЄєЇїҐґ]+")
LAT_RE = re.compile(r"[A-Za-z]+")
WORD_RE = re.compile(r"[А-Яа-яЁёІіЎўЄєЇїҐґ’']+|[A-Za-z]+")
URL_RE = re.compile(r"https?://|www\.", re.I)
CODE_RE = re.compile(r"\b(def|class|import|return|function|const|let|var|SELECT|INSERT|UPDATE|DELETE)\b")

BEL_WORDS = {
    "гэта", "які", "якая", "якое", "якія", "якіх", "якім", "быў", "была", "былі",
    "ёсць", "няма", "для", "праз", "пасля", "вельмі", "калі", "каб", "трэба", "можна",
    "чалавек", "мова", "краіна", "беларусь", "беларускі", "беларуская", "беларускую",
    "адказ", "пытанне", "навучанне", "мадэль", "даных", "праца", "працоўная", "прастора",
    "сустрэча", "сустрэчы", "карыстальнік", "карыстальніка", "павінен", "павінна", "павінны",
    "звесткі", "прыватнасць", "налады", "роля", "удзельнік", "вынік", "справаздача",
}
RUSSIAN_HINTS = {
    "это", "который", "которая", "которые", "был", "была", "были", "есть", "нет", "для",
    "через", "после", "очень", "если", "чтобы", "нужно", "можно", "человек", "язык",
    "страна", "пользователь", "ответь", "пожалуйста", "сегодня", "будет", "работает",
}
UKRAINIAN_HINTS = {
    "є", "цей", "ця", "який", "яка", "після", "дуже", "якщо", "щоб", "потрібно", "людина",
    "мова", "країна", "україна", "дані", "користувач", "відповідь",
}
ENGLISH_HINTS = {
    "the", "and", "that", "this", "with", "for", "from", "please", "answer", "model", "data",
    "meeting", "workspace", "calendar", "google", "sync", "privacy", "export", "search",
}

@dataclass(slots=True)
class DetectionResult:
    score: float
    decision: str
    reasons: list[str]
    cyrillic_ratio: float
    latin_ratio: float
    bel_specific_count: int
    bel_word_hits: int
    russian_hint_hits: int
    ukrainian_hint_hits: int
    english_hint_hits: int
    length: int

    @property
    def is_belarusian(self) -> bool:
        return self.decision == "accept"


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower().replace("'", "’") for m in WORD_RE.finditer(text)]


def detect_belarusian(
    text: str,
    *,
    accept_threshold: float = 4.0,
    quarantine_threshold: float = 2.0,
    min_chars: int = 40,
    allow_short: bool = False,
) -> DetectionResult:
    text = text or ""
    tokens = _tokens(text)
    cyr = sum(len(x) for x in CYR_RE.findall(text))
    lat = sum(len(x) for x in LAT_RE.findall(text))
    total_letters = cyr + lat
    cyr_ratio = cyr / total_letters if total_letters else 0.0
    latin_ratio = lat / total_letters if total_letters else 0.0
    bel_specific_count = sum(1 for ch in text if ch in BEL_SPECIFIC)
    token_set = set(tokens)
    bel_word_hits = sum(1 for w in token_set if w in BEL_WORDS)
    russian_hint_hits = sum(1 for w in token_set if w in RUSSIAN_HINTS)
    ukrainian_hint_hits = sum(1 for w in token_set if w in UKRAINIAN_HINTS)
    english_hint_hits = sum(1 for w in token_set if w in ENGLISH_HINTS)

    score = 0.0
    reasons: list[str] = []
    if len(text.strip()) < min_chars and not allow_short:
        score -= 3.0
        reasons.append(f"short<{min_chars}")
    if total_letters == 0:
        score -= 5.0
        reasons.append("no_letters")
    if cyr_ratio >= 0.70:
        score += 1.5
        reasons.append("mostly_cyrillic")
    elif cyr_ratio >= 0.45:
        score += 0.5
        reasons.append("mixed_cyrillic")
    else:
        score -= 2.5
        reasons.append("low_cyrillic_ratio")
    if latin_ratio > 0.25:
        score -= 1.5
        reasons.append("latin_ratio_high")
    if latin_ratio > 0.50:
        score -= 3.0
        reasons.append("latin_dominant")
    if bel_specific_count:
        score += min(4.0, 0.7 * bel_specific_count)
        reasons.append(f"bel_specific_letters={bel_specific_count}")
    else:
        score -= 0.8
        reasons.append("no_bel_specific_letters")
    if bel_word_hits:
        score += min(5.0, 0.8 * bel_word_hits)
        reasons.append(f"bel_words={bel_word_hits}")
    if russian_hint_hits >= max(3, bel_word_hits + 2) and bel_specific_count == 0:
        score -= 3.0
        reasons.append(f"russian_hints={russian_hint_hits}")
    elif russian_hint_hits:
        score -= min(1.5, 0.2 * russian_hint_hits)
        reasons.append(f"some_russian_hints={russian_hint_hits}")
    if ukrainian_hint_hits >= 2:
        score -= min(3.0, 0.8 * ukrainian_hint_hits)
        reasons.append(f"ukrainian_hints={ukrainian_hint_hits}")
    if english_hint_hits >= 4 or (english_hint_hits >= 2 and latin_ratio > 0.20):
        score -= min(3.0, 0.6 * english_hint_hits)
        reasons.append(f"english_hints={english_hint_hits}")
    if URL_RE.search(text):
        score -= 0.5
        reasons.append("url_present")
    if CODE_RE.search(text) and latin_ratio > 0.15:
        score -= 2.0
        reasons.append("code_or_config_like")

    if score >= accept_threshold:
        decision = "accept"
    elif score >= quarantine_threshold:
        decision = "quarantine"
    else:
        decision = "reject"

    return DetectionResult(
        score=round(score, 3),
        decision=decision,
        reasons=reasons,
        cyrillic_ratio=round(cyr_ratio, 4),
        latin_ratio=round(latin_ratio, 4),
        bel_specific_count=bel_specific_count,
        bel_word_hits=bel_word_hits,
        russian_hint_hits=russian_hint_hits,
        ukrainian_hint_hits=ukrainian_hint_hits,
        english_hint_hits=english_hint_hits,
        length=len(text),
    )


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("rt", encoding="utf-8", errors="ignore")


def iter_jsonl_text(path: Path) -> Iterator[tuple[int, str, object]]:
    with _open_text(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield lineno, line, line
                continue
            if isinstance(obj, str):
                yield lineno, obj, obj
            elif isinstance(obj, dict):
                yield lineno, str(obj.get("text") or obj.get("content") or obj), obj
            elif isinstance(obj, list):
                joined = "\n".join(str(m.get("content", "")) for m in obj if isinstance(m, dict))
                yield lineno, joined, obj
            else:
                yield lineno, str(obj), obj


def write_detection_report(rows: Iterable[tuple[int, str, object]], out_dir: Path, **kwargs) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted = out_dir / "accepted.jsonl"
    quarantine = out_dir / "quarantine.jsonl"
    rejected = out_dir / "rejected.jsonl"
    stats = {"seen": 0, "accepted": 0, "quarantine": 0, "rejected": 0, "score_sum": 0.0}
    with accepted.open("w", encoding="utf-8") as fa, quarantine.open("w", encoding="utf-8") as fq, rejected.open("w", encoding="utf-8") as fr:
        for lineno, text, obj in rows:
            res = detect_belarusian(text, **kwargs)
            stats["seen"] += 1
            stats["score_sum"] += res.score
            record = {"lineno": lineno, "decision": res.decision, "detection": asdict(res), "data": obj}
            if res.decision == "accept":
                stats["accepted"] += 1
                fa.write(json.dumps(record, ensure_ascii=False) + "\n")
            elif res.decision == "quarantine":
                stats["quarantine"] += 1
                fq.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                stats["rejected"] += 1
                fr.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats["average_score"] = round(stats["score_sum"] / max(1, stats["seen"]), 3)
    stats["accept_rate"] = round(stats["accepted"] / max(1, stats["seen"]), 4)
    (out_dir / "report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect/filter Belarusian text with score and reasons")
    ap.add_argument("input", type=Path, nargs="?", help="Text or JSONL file; stdin text when omitted")
    ap.add_argument("--jsonl", action="store_true", help="Treat input as JSONL and produce accepted/quarantine/rejected files")
    ap.add_argument("--out-dir", type=Path, default=Path("belarusian_filter_report"))
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--accept-threshold", type=float, default=4.0)
    ap.add_argument("--quarantine-threshold", type=float, default=2.0)
    ap.add_argument("--allow-short", action="store_true")
    args = ap.parse_args()
    kwargs = {
        "min_chars": args.min_chars,
        "accept_threshold": args.accept_threshold,
        "quarantine_threshold": args.quarantine_threshold,
        "allow_short": args.allow_short,
    }
    if args.input and args.jsonl:
        stats = write_detection_report(iter_jsonl_text(args.input), args.out_dir, **kwargs)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        raise SystemExit(0 if stats["accepted"] else 1)
    text = args.input.read_text(encoding="utf-8", errors="ignore") if args.input else sys.stdin.read()
    print(json.dumps(asdict(detect_belarusian(text, **kwargs)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
