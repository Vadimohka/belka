#!/usr/bin/env python3
"""Prepare a Belarusian-only nanochat base corpus.

Input: local directory with .txt, .md, .jsonl, .jsonl.gz and .parquet files.
Output: nanochat-compatible parquet shards with one column: text.

The script writes train shards first and validation shards last so that nanochat
setups that treat the final shard as validation remain compatible.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import tempfile
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit("pyarrow is required. Run local/install_nanochat_env.sh first or set PYTHON_BIN to a venv Python with pyarrow installed.") from exc

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from normalize_text import normalize_text
from detect_belarusian import detect_belarusian
from deduplicate import text_hash

if str(HERE.parent) not in sys.path: sys.path.insert(0, str(HERE.parent))
from data_pipeline.sft_schema import strict_json_loads
from data_pipeline.corpus_contract import content_id, group_split
from data_pipeline.artifact_store import publish, resolve

SMOKE_DOCS = [
    "Беларуская мова мае свае адметныя літары: ў, і, ё. Гэта важна для навучання мадэлі, якая адказвае па-беларуску.",
    "Гэта кароткі навучальны тэкст пра тое, як падрыхтаваць корпус даных для беларускай моўнай мадэлі. Тэкст павінен быць чыстым і правераным.",
    "Калі карыстальнік задае пытанне, мадэль павінна адказваць ветліва, дакладна і толькі па-беларуску. Руская і англійская мовы не павінны дамінаваць у адказе.",
    "Працоўная прастора дапамагае камандзе захоўваць сустрэчы, удзельнікаў, стэнаграмы, кароткія вынікі, задачы і рашэнні ў адным месцы.",
    "Пасля сустрэчы можна імпартаваць афіцыйныя артэфакты: стэнаграму, звесткі пра запіс, нататкі і спіс удзельнікаў. Потым сістэма стварае рэзюмэ.",
    "Для бяспекі патрэбныя ролі, аўдыт, абмежаванне доступу і тэрміны захоўвання. Гэта дапамагае кіраваць прыватнасцю і адказнасцю.",
    "Беларускамоўная мадэль павінна мякка тлумачыць, што яна працуе толькі па-беларуску, нават калі пытанне напісана на іншай мове.",
    "Пры падрыхтоўцы корпуса трэба выдаляць дублікаты, кароткія фрагменты, код, выпадковыя лагі і тэксты з вялікай доляй лацінкі.",
    "Добры навучальны набор мае маніфест крыніц, ліцэнзіі, справаздачу пра фільтрацыю, каранцін сумніўных радкоў і асобны валідацыйны падзел.",
    "Маленькая мадэль можа быць карыснай для дэманстрацыі, але для сапраўднай якасці патрэбныя вялікі корпус, доўгае навучанне і незалежная ацэнка.",
]

@dataclass(slots=True)
class BuildStats:
    seen: int = 0
    kept: int = 0
    quarantine: int = 0
    rejected: int = 0
    duplicate: int = 0
    too_short: int = 0
    parse_errors: int = 0


def _open_jsonl(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def iter_local_texts(root: Path) -> Iterator[tuple[str, str, dict]]:
    for path in sorted(root.rglob("*")):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"input symlink escapes text root: {path}")
        if path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        suffixes = "".join(path.suffixes).lower()
        rel = str(path.relative_to(root))
        if path.suffix.lower() in {".txt", ".md"}:
            yield "local", path.read_text(encoding="utf-8"), {"path": rel}
        elif suffixes.endswith(".jsonl") or suffixes.endswith(".jsonl.gz"):
            with _open_jsonl(path) as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = strict_json_loads(line)
                    except ValueError as exc:
                        raise ValueError(f"{path}:{lineno}: malformed JSONL (no raw fallback)") from exc
                    if isinstance(obj, str):
                        text = obj
                    elif isinstance(obj, dict):
                        text = obj.get("text", obj.get("content"))
                    else:
                        raise ValueError(f"{path}:{lineno}: unsupported corpus record type")
                    if not isinstance(text, str):
                        raise ValueError(f"{path}:{lineno}: corpus text must be a string")
                    group = obj.get("group_id") if isinstance(obj, dict) else None
                    yield "local_jsonl", text, {"path": rel, "line": lineno, "group_id": group}
        elif path.suffix.lower() == ".parquet":
            pf = pq.ParquetFile(path)
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=["text"] if "text" in pf.schema.names else None)
                if "text" not in table.column_names:
                    raise ValueError(f"missing text column: {path}")
                for row_idx, text in enumerate(table.column("text").to_pylist()):
                    if not isinstance(text, str): raise ValueError(f"non-string Parquet text: {path}")
                    yield "local_parquet", text, {"path": rel, "row_group": rg_idx, "row": row_idx}


def iter_smoke_texts(repeats: int = 12) -> Iterator[tuple[str, str, dict]]:
    idx = 0
    for r in range(repeats):
        for doc in SMOKE_DOCS:
            idx += 1
            suffix = f"\n\nНумар прыкладу: {idx}. Гэта дадатковы сказ, каб навучальны фрагмент меў дастатковую даўжыню і беларускія маркеры."
            yield "smoke", doc + suffix, {"smoke_index": idx, "repeat": r}


def write_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_parquet_shards(texts: list[str], output_dir: Path, *, prefix: str, shard_docs: int) -> list[dict]:
    files = []
    for i in range(0, len(texts), shard_docs):
        shard = texts[i:i + shard_docs]
        if not shard:
            continue
        path = output_dir / f"{prefix}_{len(files):05d}.parquet"
        table = pa.table({"text": pa.array(shard, type=pa.string())})
        pq.write_table(table, path, compression="zstd")
        files.append({"path": path.name, "rows": len(shard), "bytes": path.stat().st_size})
    return files


def _prepare_unpublished(
    streams: Iterable[tuple[str, str, dict]],
    *,
    output_dir: Path,
    report_dir: Path,
    min_chars: int,
    val_ratio: float,
    train_shard_docs: int,
    val_shard_docs: int,
    max_docs_total: int,
    seed: int,
    allow_short: bool,
) -> dict:
    # Validate before any output or report can be removed or overwritten.
    if not math.isfinite(val_ratio) or not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be finite and strictly between 0 and 1")
    for name, value in (("train_shard_docs", train_shard_docs), ("val_shard_docs", val_shard_docs)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if min_chars < 0:
        raise ValueError("min_chars must be nonnegative")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for old in ["accepted.jsonl", "quarantine.jsonl", "rejected.jsonl", "duplicates.jsonl"]:
        p = report_dir / old
        if p.exists():
            p.unlink()
    stats = BuildStats()
    seen_hashes: set[str] = set()
    accepted: list[tuple[str, dict, dict]] = []
    for source, raw, meta in streams:
        if max_docs_total > 0 and stats.kept >= max_docs_total:
            break
        stats.seen += 1
        text = normalize_text(raw)
        if len(text) < min_chars and not allow_short:
            stats.too_short += 1
            result = {"score": -9.0, "decision": "reject", "reasons": [f"too_short<{min_chars}"], "length": len(text)}
            write_jsonl(report_dir / "rejected.jsonl", {"source": source, "meta": meta, "detection": result, "text": text[:1000]})
            continue
        det = detect_belarusian(text, min_chars=min_chars, allow_short=allow_short)
        record = {"source": source, "meta": meta, "detection": asdict(det), "text": text[:2000]}
        if det.decision == "reject":
            stats.rejected += 1
            write_jsonl(report_dir / "rejected.jsonl", record)
            continue
        if det.decision == "quarantine":
            stats.quarantine += 1
            write_jsonl(report_dir / "quarantine.jsonl", record)
            continue
        h = content_id(text)
        if h in seen_hashes:
            stats.duplicate += 1
            write_jsonl(report_dir / "duplicates.jsonl", {"source": source, "meta": meta, "sha256_norm": h, "text": text[:1000]})
            continue
        seen_hashes.add(h)
        stats.kept += 1
        accepted.append((text, {"source": source, "meta": meta}, asdict(det)))
        write_jsonl(report_dir / "accepted.jsonl", {"text": text, "source": source, "meta": meta, "detection": asdict(det)})

    # An unsuccessful rebuild must not destroy the previous usable corpus.
    # This guard does not make the later multi-file write crash-atomic.
    if len(accepted) < 2:
        raise ValueError("At least two accepted unique documents are required for nonempty train/val splits; previous parquet shards are unchanged")

    # Group before split; source order and metadata serialization do not change
    # assignment. A changed split policy is a new corpus revision, not a resume.
    train, val = [], []
    for text, provenance, _ in accepted:
        group = provenance["meta"].get("group_id") or content_id(text)
        (val if group_split(str(group), val_ratio, f"belka-corpus-v2:{seed}") == "val" else train).append(text)
    if not train or not val:
        raise ValueError("hash split is empty; add data or explicitly choose a new ratio/seed")
    train_files = write_parquet_shards(train, output_dir, prefix="train", shard_docs=train_shard_docs)
    val_files = write_parquet_shards(val, output_dir, prefix="val", shard_docs=val_shard_docs)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "stats": asdict(stats),
        "splits": {"train_docs": len(train), "val_docs": len(val), "val_ratio_requested": val_ratio},
        "files": train_files + val_files,
        "filter": {"min_chars": min_chars, "allow_short": allow_short},
        "format": "explicit train_*.parquet / val_*.parquet; belka-group-split-v2",
    }
    (output_dir / "_BUILD_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare(streams, *, output_dir, report_dir, **kwargs):
    # Validate before staging or touching an existing corpus/report.
    ratio=kwargs["val_ratio"]
    if not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("val_ratio must be finite and strictly between 0 and 1")
    for key in ("train_shard_docs","val_shard_docs"):
        value=kwargs[key]
        if type(value) is not int or value <= 0: raise ValueError(f"{key} must be positive")
    if kwargs["min_chars"] < 0: raise ValueError("min_chars must be nonnegative")
    output_dir=Path(output_dir).resolve();report_dir=Path(report_dir).resolve()
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".belka-corpus-",dir=output_dir.parent) as temporary:
        stage=Path(temporary)
        manifest=_prepare_unpublished(streams,output_dir=stage/"data",report_dir=stage/"reports",**kwargs)
        manifest["output_dir"]=str(output_dir);manifest["report_dir"]=str(report_dir)
        (stage/"data/_BUILD_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
        files={p.name:p for p in (stage/"data").iterdir() if p.is_file()}
        files.update({"reports/"+p.name:p for p in (stage/"reports").iterdir() if p.is_file()})
        pointer=publish(output_dir,"corpus",files,{"split_policy":"belka-group-split-v2","seed":kwargs["seed"]})
        directory,_=resolve(output_dir,"corpus")
        manifest["generation"]=str(directory);manifest["pointer"]=pointer
        # Canonical accounting lives inside the generation. This convenience
        # summary may lag on an I/O failure; it is never the publication marker.
        report_dir.mkdir(parents=True,exist_ok=True)
        (report_dir/"summary.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
        return manifest


def load_streams(args) -> Iterator[tuple[str, str, dict]]:
    if args.mode == "smoke":
        yield from iter_smoke_texts(args.smoke_repeats)
    if args.local_text_dir:
        root = args.local_text_dir.expanduser()
        if not root.exists():
            raise FileNotFoundError(f"LOCAL_TEXT_DIR does not exist: {root}")
        yield from iter_local_texts(root)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Belarusian-only nanochat parquet corpus")
    ap.add_argument("--mode", choices=["smoke", "real"], default="real")
    ap.add_argument("--local-text-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--manifest-out", type=Path, default=None)
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--train-shard-docs", type=int, default=50000)
    ap.add_argument("--val-shard-docs", type=int, default=50000)
    ap.add_argument("--max-docs-total", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--smoke-repeats", type=int, default=12)
    ap.add_argument("--allow-short", action="store_true")
    args = ap.parse_args()
    report_dir = args.report_dir or args.output_dir.parent / "belarusian_filter_report"
    manifest = prepare(
        load_streams(args),
        output_dir=args.output_dir,
        report_dir=report_dir,
        min_chars=args.min_chars,
        val_ratio=args.val_ratio,
        train_shard_docs=args.train_shard_docs,
        val_shard_docs=args.val_shard_docs,
        max_docs_total=args.max_docs_total,
        seed=args.seed,
        allow_short=args.allow_short,
    )
    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["stats"] | manifest["splits"], ensure_ascii=False, indent=2))
    if manifest["stats"]["kept"] == 0:
        raise SystemExit("No Belarusian documents survived filtering; see report_dir")


if __name__ == "__main__":
    main()
