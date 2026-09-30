"""Conservative SFT language evidence, not a calibrated neural language ID.

Identifiers are ignored for scoring, never changed in training text. Shared
Cyrillic vocabulary alone is uncertain, not a Belarusian pass. The general
corpus detector remains independently versioned.
"""
from __future__ import annotations
import re
import unicodedata
from data_pipeline.detect_belarusian import detect_belarusian

VERSION = "sft-language-evidence-v1"
IDENTIFIERS = re.compile(r"\b(?:ChatGPT|OpenAI|MeetMesh|Google|Calendar|Meet|OAuth|API|JSON|Python|Belka|LLM|SFT|BPE|GPU|CPU|HTTP|HTTPS|UTF|NFC)\b")
BE_EVIDENCE = set("чым адметная хто што як дзе куды навошта чаму чаго якой цябе можаш называць створана раскажы растлумач назаві адкажы правер сфармулюй коратка скароці праверыць вядома стварыць надзейны пацверджання здагадку дакладны".split())
RU_EVIDENCE = set("кто что это расскажи представься напиши объясни коротко какие является знаешь себе пожалуйста который которая чтобы если".split())
UK_EVIDENCE = set("цей ця який яка якщо після дуже відповідь користувач".split())


def language_evidence(text: str, threshold: float = 2.0) -> dict:
    clean = IDENTIFIERS.sub(" ", unicodedata.normalize("NFC", text))
    words = set(re.findall(r"[^\W\d_]+", clean.lower()))
    foreign = sorted(words & (RU_EVIDENCE | UK_EVIDENCE))
    # These letters are evidence against this project's Cyrillic SFT profile.
    # Quoted foreign prose belongs in evaluation, not Belarusian-only training.
    if foreign or re.search(r"[иИщЩъЪєЄїЇґҐ]", clean):
        return {"decision": "reject", "reason": "foreign_language_evidence", "words": foreign}
    det = detect_belarusian(clean, min_chars=10, allow_short=True,
                            accept_threshold=threshold, quarantine_threshold=min(1.0, threshold))
    if det.decision == "accept":
        return {"decision": "accept", "reason": "detector", "score": det.score}
    if words & BE_EVIDENCE and det.latin_ratio <= 0.1:
        return {"decision": "accept", "reason": "belarusian_lexical_evidence", "score": det.score}
    return {"decision": "quarantine", "reason": "insufficient_language_evidence", "score": det.score}
