#!/usr/bin/env python3
"""Reject unrelated templates: native framing belongs to BelkaTokenizer."""
import argparse
import json
from pathlib import Path

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir',type=Path,required=True)
    args=ap.parse_args(argv)
    config=args.model_dir/'tokenizer_config.json'
    if not config.is_file(): ap.error('export the full native HF bundle first')
    data=json.loads(config.read_text())
    if data.get('auto_map',{}).get('AutoTokenizer')!=['tokenization_belka.BelkaTokenizer',None]:
        ap.error('unsupported tokenizer; an unrelated Jinja template cannot supply native ID parity')
    print('Native chat framing is implemented by BelkaTokenizer.apply_chat_template; no mutation needed.')
    return 0

if __name__=='__main__': raise SystemExit(main())
