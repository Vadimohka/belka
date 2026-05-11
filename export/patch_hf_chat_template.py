#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach Belarusian chat template to HF tokenizer config files when present")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--template", type=Path, default=Path(__file__).resolve().parents[1] / "templates" / "chat_template.jinja")
    args = ap.parse_args()
    template = args.template.read_text(encoding="utf-8")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    for name in ["tokenizer_config.json", "generation_config.json"]:
        path = args.model_dir / name
        data = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        if name == "tokenizer_config.json":
            data["chat_template"] = template
        if name == "generation_config.json":
            data.setdefault("temperature", 0.7)
            data.setdefault("top_p", 0.95)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.model_dir)


if __name__ == "__main__":
    main()
