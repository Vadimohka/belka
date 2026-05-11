#!/usr/bin/env python3
"""Convert a MeetMesh-like project into Belarusian domain SFT examples.

The converter reads only repository structure and selected docs. It does not copy
source code into SFT answers. It emits Belarusian paraphrases about confirmed
features and uses cautious wording when a file or marker is absent.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SECRET_PATTERNS = re.compile(r"(SECRET_KEY|API_KEY|PRIVATE|TOKEN|PASSWORD|BEGIN .*KEY|client_secret|refresh_token)", re.I)


def exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def has_text(root: Path, rel: str, pattern: str) -> bool:
    path = root / rel
    if not path.exists() or SECRET_PATTERNS.search(rel):
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return re.search(pattern, text, re.I) is not None


def conv(user: str, assistant: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def generate(root: Path) -> list[list[dict[str, str]]]:
    rows: list[list[dict[str, str]]] = []
    django = exists(root, "manage.py") and exists(root, "config/settings.py")
    rows.append(conv(
        "Што можна сказаць пра архітэктуру гэтага праекта?",
        "Паводле структуры рэпазіторыя, гэта Django-праект з асобнымі даменамі для карыстальнікаў, працоўных прастор, Google-інтэграцыі, сустрэч, аўтаматызацыі, API і вэб-інтэрфейсу." if django else "У даступнай структуры няма дастатковых падстаў дакладна апісаць Django-архітэктуру, таму трэба праверыць README і каталогі праекта."
    ))
    rows.append(conv(
        "Ці ёсць у праекце працоўныя прасторы?",
        "Так, у структуры ёсць дамен workspaces. Ён апісвае працоўную прастору, сяброўства, ролі, каманды, запрашэнні, прыватнасць, палітыку захоўвання і падпіскі." if exists(root, "apps/workspaces/models.py") else "Я не бачу пацверджанай мадэлі працоўных прастор у даступнай структуры."
    ))
    rows.append(conv(
        "Ці ёсць у праекце Google OAuth?",
        "Так, у дамене google_integration ёсць OAuth-сэрвіс і мадэлі для стану OAuth і Google-акаўнта. У SFT не трэба ўключаць сакрэты або токены." if exists(root, "apps/google_integration/services/oauth.py") else "У даступных файлах няма пацверджанага OAuth-сэрвісу, таму нельга сцвярджаць яго наяўнасць."
    ))
    rows.append(conv(
        "Як праект працуе з календаром?",
        "Калі прысутнічае calendar_sync service, праект сінхранізуе падзеі календара, шукае Google Meet-спасылкі і можа выкарыстоўваць інкрэментальныя sync token або watch channels." if exists(root, "apps/google_integration/services/calendar_sync.py") else "Па даступнай структуры нельга пацвердзіць асобны сэрвіс календарнай сінхранізацыі."
    ))
    rows.append(conv(
        "Што з артэфактамі сустрэч?",
        "Праект мае сэрвіс артэфактаў, які можа імпартаваць стэнаграмы, метаданыя запісаў і нататкі, калі яны даступныя праз падтрыманыя Google API." if exists(root, "apps/google_integration/services/artifacts.py") else "Я не бачу пацверджанага сэрвісу імпарту артэфактаў, таму трэба адказваць асцярожна."
    ))
    rows.append(conv(
        "Ці ёсць пошук па сустрэчах?",
        "Так, калі прысутнічае meetings/services/search.py, праект мае пошукавы сэрвіс па сустрэчах і стэнаграмах. Асноўны шлях варта апісваць як поўнатэкставы пошук, калі не пацверджана іншае." if exists(root, "apps/meetings/services/search.py") else "Па структуры няма пацверджанага асобнага пошуку па сустрэчах."
    ))
    rows.append(conv(
        "Ці ёсць экспарт?",
        "Так, export service сведчыць пра экспарт матэрыялаў сустрэчы. Звычайна гэта рэзюмэ, стэнаграма, задачы, рашэнні і артэфакты ў падтрымваных фарматах." if exists(root, "apps/meetings/services/exports.py") else "У даступных файлах не бачна асобнага сэрвісу экспарту."
    ))
    rows.append(conv(
        "Як апісваць AI-сэрвіс у SFT?",
        "Трэба казаць, што праект мае абстракцыю пастаўшчыка AI і fallback-логіку, калі гэта пацвярджаецца кодам. Нельга сцвярджаць, што ёсць пэўная мадэль або ключ, калі гэта не наладжана." if exists(root, "apps/meetings/services/ai.py") else "Без файла AI-сэрвісу лепш не сцвярджаць пра AI-апрацоўку."
    ))
    rows.append(conv(
        "Што нельга ўключаць у навучальныя прыклады з праекта?",
        "Нельга ўключаць сакрэты, токены, прыватныя ключы, .env з рэальнымі значэннямі і вялікія англамоўныя фрагменты кода. Трэба пераказваць сэнс па-беларуску."))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Belarusian MeetMesh domain SFT examples from project structure")
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = generate(args.project_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
