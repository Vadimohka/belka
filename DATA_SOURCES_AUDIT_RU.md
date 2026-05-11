# Беларускамоўныя крыніцы даных: audit і план

Гэты hotfix дадае крынічны слой, а не новы superpack:

- `configs/belarusian_sources_full.yaml` — каталог крыніц, ліцэнзій і рэжымаў загрузкі;
- `tools/download_wikimedia_be.py` — Wikimedia dumps: bewiki, be_x_oldwiki, bewikisource, bewiktionary, bewikiquote, bewikibooks;
- `tools/download_belarusian_sources.py` — Belacorpus, UD Belarusian-HSE, Tatoeba;
- `tools/hf_stream_belarusian_sources.py` — BelarusianGLUE, OSCAR, CulturaX, mC4, CC100, morphodict;
- `tools/filter_sources_to_nanochat_parquet.py` — Belarusian-only filter + quarantine + dedup + train/val parquet;
- bootstrap corpus, каб pipeline не быў пустым.

## Першы прыярытэт

1. Belarusian Wikipedia `bewiki`.
2. Belarusian classical Wikipedia `be_x_oldwiki`.
3. Belarusian Wikisource `bewikisource`.
4. Belacorpus public research corpus.
5. Leipzig `bel_wikipedia_2021`.
6. OSCAR 23.01 Belarusian.
7. CulturaX Belarusian.
8. mC4 / CC100 Belarusian.

## Для SFT/eval

- BelarusianGLUE;
- UD Belarusian-HSE;
- Tatoeba / OPUS;
- Wiktionary / morphodict-bel;
- Common Voice transcripts, калі ўмовы доступу дазваляюць.

## Не рабіць

- Не змешваць raw dumps адразу ў training.
- Не навучаць на англійскіх README/code як на беларускамоўным corpus.
- Не выдаваць gated/manual sources за ўжо спампаваныя.
- Не выкідваць ліцэнзіі і attribution.
