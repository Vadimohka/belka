# Sources and licenses

## Bundled generated data

The seed SFT and eval files in this pack were generated as Belarusian paraphrases for this release. They are intended to be MIT-compatible with the pack, but you should review them before commercial use.

## nanochat

The scripts expect a local clone of `karpathy/nanochat`. The superpack does not vendor nanochat source code; it patches a user checkout at install time.

## MeetMesh

The uploaded MeetMesh project is used as a domain reference. This release does not include its English source code in training text. Domain SFT examples are Belarusian paraphrases of architecture and feature concepts observed in the project. The uploaded project declares MIT license.

## External corpora

No public corpus is bundled. `configs/dataset_sources.yaml` lists candidate sources only. Before downloading or training on any source, verify:

- dataset license;
- redistribution terms;
- attribution requirements;
- whether ML training is permitted;
- privacy and personal data constraints.

## User-provided books/texts

Place only texts you have the legal right to use under `~/data/be_texts`. The pipeline filters language quality but does not solve copyright or privacy issues.
