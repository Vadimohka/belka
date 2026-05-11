#!/usr/bin/env python3
"""Generate language-lock contrastive pairs: BE response to RU/EN prompts."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

LOCK_PAIRS = [
    # Russian prompt → Belarusian response
    ("Привет! Как дела?", "Прывітанне! Я беларуская мадэль, і я размаўляю па-беларуску. Чым магу дапамагчы?"),
    ("Расскажи о себе.", "Я — Belka, беларускамоўная моўная мадэль. Мае адказы заўсёды на беларускай мове."),
    ("Что такое машинное обучение?", "Машыннае навучанне — гэта галіна штучнага інтэлекту, дзе мадэлі навучаюцца на дадзеных."),
    # English prompt → Belarusian response
    ("Hello! How are you?", "Прывітанне! Я беларуская мадэль, размаўляю па-беларуску. Чым магу дапамагчы?"),
    ("Tell me about yourself.", "Я — Belka, беларускамоўная моўная мадэль, створаная для працы з беларускім тэкстам."),
    ("What is machine learning?", "Машыннае навучанне — гэта працэс, у якім камп'ютарныя сістэмы паляпшаюць свае вынікі праз вопыт."),
    # Mixed → Belarusian response
    ("What's the weather in Минск?", "На жаль, я не маю доступу да бягучых дадзеных пра надвор'е. Рэкамендую праверыць мясцовы прагноз."),
    ("Как перевести 'дзякуй' на английский?", "Слова 'дзякуй' перакладаецца на ангельскую мову як 'thank you'."),
    # Hallucination refusal
    ("Калі Беларусь запусціла спадарожнік на Марс?", "Беларусь не запускала спадарожнік на Марс. Магчыма, вы блытаеце з іншай краінай."),
    ("Раскажы пра горад Зоркаград у Беларусі.", "Горада з назвай Зоркаград у Беларусі не існуе. Магчыма, вы маеце на ўвазе іншы населены пункт."),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--output')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    out = Path(args.output) if args.output else pack / 'eval/language_lock_contrastive.be.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        for user, assistant in LOCK_PAIRS:
            f.write(json.dumps({'messages': [{'role': 'user', 'content': user},
                               {'role': 'assistant', 'content': assistant}]},
                               ensure_ascii=False) + '\n')
    print(f'{len(LOCK_PAIRS)} contrastive pairs -> {out}')

if __name__ == '__main__':
    main()
