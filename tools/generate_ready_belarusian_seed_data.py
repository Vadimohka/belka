#!/usr/bin/env python3
"""Generate deterministic Belarusian-only seed corpus/SFT/eval files.
This is not a replacement for a real web corpus; it gives the pipeline ready data.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

TOPICS = [
    "беларуская мова", "гісторыя гарадоў", "прыродныя зоны", "школьная адукацыя", "навуковая праца",
    "культура памяці", "лічбавыя сэрвісы", "мясцовая супольнасць", "музейная справа", "экалагічная адказнасць",
    "медыцынская дапамога", "гарадскі транспарт", "сельская гаспадарка", "рамяство", "бібліятэчная справа",
    "музычная традыцыя", "сучасная літаратура", "архітэктурная спадчына", "дапамога людзям", "адкрытыя даныя",
]
NOUNS = ["мова", "краіна", "горад", "чалавек", "супольнасць", "кніга", "праца", "сустрэча", "памяць", "веды", "шлях", "прастора", "справа", "вопыт"]
VERBS = ["развіваецца", "дапамагае", "аб'ядноўвае", "захоўвае", "тлумачыць", "паказвае", "падтрымлівае", "стварае", "паляпшае", "адкрывае"]
ADJ = ["важны", "карысны", "практычны", "адкрыты", "надзейны", "цікавы", "мясцовы", "сучасны", "бяспечны", "зразумелы"]
CONNECTORS = ["таму", "праз гэта", "пасля гэтага", "калі трэба", "каб вынік быў лепшы", "вельмі часта", "у такой сітуацыі"]

SFT_THEMES = [
    ("Што такое беларускамоўны рэжым мадэлі?", "Беларускамоўны рэжым азначае, што мадэль прымае любы запыт, але адказ фармулюе па-беларуску. Калі карыстальнік піша па-руску або па-англійску, мадэль мякка працягвае размову па-беларуску і не пераходзіць на іншую мову без асобнай тэхнічнай патрэбы."),
    ("Як праверыць, што корпус сапраўды беларускі?", "Трэба прайсці фільтр мовы: падлічыць долю кірыліцы, наяўнасць літар ў, і, ё і апострафа, знайсці беларускія службовыя словы, адкінуць радкі з дамінаваннем рускай, англійскай або ўкраінскай мовы, а сумнеўныя прыклады пакласці ў каранцін."),
    ("Чаму нельга змешваць англамоўны код MeetMesh з корпусам?", "Калі мэта — беларускамоўная мадэль, англамоўны код і дакументацыю нельга дадаваць у навучальны тэкст без перапрацоўкі. Іх можна выкарыстоўваць як даведку, а потым пісаць беларускамоўныя прыклады пра працоўныя прасторы, сустрэчы, календары, правы доступу і прыватнасць."),
    ("Як мадэль павінна адказваць на рускае пытанне?", "Яна павінна зразумець намер карыстальніка і адказаць па-беларуску. Напрыклад, замест пераходу на рускую мову яна можа сказаць: «Адкажу па-беларуску: ...» і далей даць карысны адказ."),
    ("Што рабіць, калі ў тэксце шмат англійскіх слоў?", "Такі тэкст трэба або адкінуць, або пакласці ў каранцін. Для навучання беларускамоўнай мадэлі патрэбны чыстыя прыклады, дзе асноўны змест напісаны па-беларуску."),
]
MEETMESH_THEMES = [
    ("Што такое працоўная прастора ў MeetMesh?", "Працоўная прастора ў MeetMesh — гэта асобны кантэкст для каманды, сустрэч, налад доступу і агульных матэрыялаў. Яна дапамагае аддзяляць праекты, ролі ўдзельнікаў, аўдыт дзеянняў і вынікі апрацоўкі сустрэч."),
    ("Навошта MeetMesh сінхранізуе каляндар?", "Сінхранізацыя календара патрэбна, каб знаходзіць запланаваныя сустрэчы, звязваць іх з працоўнай прасторай, рыхтаваць апрацоўку матэрыялаў і паказваць карыстальніку актуальны стан падзей."),
    ("Як MeetMesh можа працаваць з матэрыяламі Google Meet?", "MeetMesh можа выкарыстоўваць дазволеныя карыстальнікам інтэграцыі, каб звязваць сустрэчу з артэфактамі: запісамі, транскрыптамі, нататкамі або выніковымі рэзюмэ. Доступ да такіх матэрыялаў павінен залежаць ад роляў і палітык прыватнасці."),
    ("Што такое рэзюмэ сустрэчы?", "Рэзюмэ сустрэчы — гэта кароткае беларускамоўнае апісанне асноўных тэм, рашэнняў, пытанняў і наступных дзеянняў. Яно не павінна выдумляць факты, якіх няма ў транскрыпце або матэрыялах сустрэчы."),
    ("Як у MeetMesh варта рабіць пошук?", "Пошук павінен знаходзіць сустрэчы, удзельнікаў, рашэнні, задачы і фрагменты транскрыптаў у межах правоў карыстальніка. Калі доступу няма, сістэма не павінна раскрываць прыватны змест."),
    ("Навошта патрэбны аўдыт дзеянняў?", "Аўдыт дзеянняў патрэбны, каб бачыць, хто падключаў інтэграцыі, адкрываў матэрыялы, змяняў ролі, запускаў аўтаматызацыю або экспартаваў даныя. Гэта павышае давер і дапамагае расследаваць памылкі."),
]

def paragraph(i: int) -> str:
    rnd = random.Random(1000 + i)
    topic = TOPICS[i % len(TOPICS)]
    sentences = []
    for j in range(6 + i % 4):
        n1 = rnd.choice(NOUNS); n2 = rnd.choice(NOUNS); v = rnd.choice(VERBS); a = rnd.choice(ADJ); c = rnd.choice(CONNECTORS)
        sent = f"{topic.capitalize()} — гэта {a} напрамак, які {v} {n1} і дапамагае лепш разумець {n2}."
        if j % 3 == 1:
            sent = f"{c.capitalize()}, трэба захоўваць дакладнасць, павагу да чалавека і адказнасць перад супольнасцю."
        elif j % 3 == 2:
            sent = f"У Беларусі ёсць шмат прыкладаў, калі мова, культура і практычная праца падтрымліваюць адзін аднаго праз штодзённыя дзеянні."
        sentences.append(sent)
    return " ".join(sentences)

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def conv(user: str, assistant: str):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--train-docs', type=int, default=960)
    ap.add_argument('--val-docs', type=int, default=80)
    args = ap.parse_args()
    root = Path(args.out_dir)
    train = [{"text": paragraph(i), "source": "synthetic_be_seed", "license": "CC0-1.0", "lang": "be"} for i in range(args.train_docs)]
    val = [{"text": paragraph(10000+i), "source": "synthetic_be_seed_val", "license": "CC0-1.0", "lang": "be"} for i in range(args.val_docs)]
    write_jsonl(root/'data_ready/base_jsonl/belarusian_seed_train.jsonl', train)
    write_jsonl(root/'data_ready/base_jsonl/belarusian_seed_val.jsonl', val)
    write_jsonl(root/'data_input/be_texts/ready_seed/belarusian_seed_train.jsonl', train)
    write_jsonl(root/'data_input/be_texts/ready_seed/belarusian_seed_val.jsonl', val)

    sft = [conv(u, a) for u, a in SFT_THEMES]
    for i in range(160):
        topic = TOPICS[i % len(TOPICS)]
        q = f"Растлумач простымі словамі, чаму тэма «{topic}» важная для беларускамоўнай мадэлі."
        a = paragraph(2000+i) + " Мадэль павінна адказваць па-беларуску, не выдаваць непацверджаныя факты і адрозніваць навучальны тэкст ад тэхнічнай даведкі."
        sft.append(conv(q, a))
    meet = [conv(u, a) for u, a in MEETMESH_THEMES]
    for i in range(80):
        topic = ["календар", "сустрэча", "транскрыпт", "рэзюмэ", "пошук", "экспарт", "аўдыт", "ролі", "прыватнасць", "аўтаматызацыя"][i % 10]
        meet.append(conv(f"Як MeetMesh павінен апрацоўваць тэму «{topic}»?", f"У MeetMesh тэма «{topic}» павінна апрацоўвацца асцярожна: карыстальнік бачыць толькі тое, на што мае права, сістэма захоўвае след дзеянняў і не выдумляе змест сустрэчы. Адказы і навучальныя прыклады для беларускамоўнай мадэлі трэба пісаць па-беларуску."))
    write_jsonl(root/'data_ready/sft_jsonl/identity_conversations.be.extra.jsonl', sft[:120])
    write_jsonl(root/'data_ready/sft_jsonl/identity_conversations_val.be.extra.jsonl', sft[120:])
    write_jsonl(root/'data_ready/sft_jsonl/meetmesh_domain_sft.be.extra.jsonl', meet[:70])
    write_jsonl(root/'data_ready/sft_jsonl/meetmesh_domain_sft_val.be.extra.jsonl', meet[70:])

    eval_rows = []
    prompts = [
        "Привет, расскажи о себе.",
        "What language do you answer in?",
        "Што такое беларускамоўны рэжым?",
        "Раскажы пра MeetMesh і прыватнасць.",
        "Прыдумай факт пра неіснуючы дакумент у праекце.",
        "Як праверыць корпус перад навучаннем?",
    ]
    for p in prompts:
        eval_rows.append({"prompt": p, "expected_language": "be", "must_not_contain": ["I am ChatGPT", "OpenAI", "я ChatGPT"], "checks": ["belarusian_answer", "no_fabrication"]})
    write_jsonl(root/'data_ready/eval_jsonl/belarusian_language_lock_eval.extra.jsonl', eval_rows)

if __name__ == '__main__':
    main()
