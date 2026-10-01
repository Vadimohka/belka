"""Conservative, versioned all-role gate for Belarusian training text.

This is a transparent lexical gate, not a calibrated statistical language-ID
model. Known technical identifiers are neutral; Russian/Ukrainian words and
English prose are not made Belarusian by appending one і/ў/ё. Unknown Cyrillic
text is quarantined, not accepted because it happens to use Cyrillic letters.
"""
from __future__ import annotations

import re
import unicodedata

POLICY_VERSION = "be-training-lexical-v2"
WORDS = re.compile(r"[^\W\d_]+", re.UNICODE)
BE = set("гэта што хто дзе чаму чаго чым які якая якія якіх якім калі каб трэба можна чалавек беларуская беларускі беларускую беларускамоўная беларусі мова мадэль адказ пытанне навучанне праца звесткі прыватнасць налады сустрэча вынік няма ёсць быў былі праз пасля вельмі стварыць праверыць растлумач патлумач раскажы адкажы напішы назаві прадставіся прадстаўся ведаеш называць цябе твая тваё табе сваю свае коратка правер сфармулюй сцісла надзейны надзейнага пароль трэба добра дзякуй вітаю так не чытаць пісаць значыць рэкі сталіца арфаграфія памочнік як можна ласка мове якой адказваеш можаш адказаць".split())
RU = set("что кто где почему который которая которые каких если чтобы нужно можно человек язык ответ вопрос расскажи объясни ответь напиши назови представься привет пожалуйста сегодня является какие реки знаешь коротко русском английском белорусском собой это ещё всего ёлка берёза ребёнок".split()) - BE
UK = set("що чому який яка які якщо щоб після дуже людина відповідь питання розкажи поясни скажи будь ласка українська немає ось".split()) - BE
EN = set("who what when where why how are you is the a an in on of and please answer explain tell me about name several english russian was were this that with for from hello your write".split())
IDENTIFIERS = {s.casefold() for s in "Belka ChatGPT OpenAI MeetMesh Google Calendar Meet OAuth API HTTP HTTPS JSON JSONL SQL Python LLM GPU CPU URL UTF NFC NaN Inf Linux Windows GitHub GitLab 2FA FA Xyzzor XX XIX XXI".split()}
# Reviewed ambiguous short Belarusian utterance; not a general Cyrillic bypass.
AMBIGUOUS_APPROVED = {"ты chatgpt?"}


def assess_training_text(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"decision": "reject", "reason": "empty_text", "policy": POLICY_VERSION}
    normalized = unicodedata.normalize("NFC", text).strip()
    tokens = [x.casefold() for x in WORDS.findall(normalized)]
    if not tokens:
        # Numbers/punctuation are language-neutral, but arbitrary blank/control
        # strings and unpaired surrogates are handled by the structural schema.
        return {"decision": "accept", "reason": "language_neutral", "policy": POLICY_VERSION}
    # Technical identifiers are neutral, not positive evidence of Belarusian.
    lexical = [w for w in tokens if w not in IDENTIFIERS]
    ru = sorted(set(lexical) & RU)
    uk = sorted(set(lexical) & UK)
    en = sorted(set(lexical) & EN)
    if ru or uk or en or re.search("[їЇєЄґҐ]", normalized):
        return {"decision": "reject", "reason": "foreign_language_evidence",
                "ru": ru, "uk": uk, "en": en, "policy": POLICY_VERSION}
    foreign_latin = [w for w in lexical if re.search("[a-z]", w)]
    if foreign_latin:
        return {"decision": "quarantine", "reason": "unreviewed_latin_text",
                "tokens": sorted(set(foreign_latin)), "policy": POLICY_VERSION}
    evidence = bool(set(lexical) & BE or re.search("[ўЎіІ]", normalized))
    if evidence or normalized.casefold() in AMBIGUOUS_APPROVED:
        return {"decision": "accept", "reason": "belarusian_evidence" if evidence else "reviewed_short_text",
                "policy": POLICY_VERSION}
    return {"decision": "quarantine", "reason": "ambiguous_cyrillic", "policy": POLICY_VERSION}
