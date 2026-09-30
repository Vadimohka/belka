"""Explicit Belarusian train/validation shard routing; no download fallback."""
import os
from pathlib import Path


def list_parquet_files(data_dir=None, warn_on_legacy=False):
    if data_dir is None:
        base = os.environ.get("NANOCHAT_BASE_DIR")
        if not base:
            raise ValueError("NANOCHAT_BASE_DIR must explicitly select Belka data")
        data_dir = Path(base) / "base_data_climbmix"
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Belka corpus missing: {root}; restore the bundled corpus, not upstream English data")
    files = sorted(root.glob("*.parquet"))
    train = [str(p) for p in files if p.name.startswith("train_")]
    val = [str(p) for p in files if p.name.startswith("val_")]
    if not train or not val or len(train) + len(val) != len(files):
        raise ValueError("expected only train_*.parquet and val_*.parquet at the corpus root")
    return train + val


def split_parquet_files(paths, split):
    if split not in ("train", "val"):
        raise ValueError("split must be train or val")
    result = [path for path in paths if Path(path).name.startswith(split + "_")]
    if not result:
        raise ValueError(f"empty {split} partition")
    return result


def parquets_iter_batched(split, start=0, step=1):
    import pyarrow.parquet as pq
    if type(step) is not int or type(start) is not int or step < 1 or not 0 <= start < step:
        raise ValueError("require 0 <= start < step")
    ordinal = 0
    for path in split_parquet_files(list_parquet_files(), split):
        parquet = pq.ParquetFile(path)
        for index in range(parquet.num_row_groups):
            if ordinal % step == start:
                yield parquet.read_row_group(index, columns=["text"]).column("text").to_pylist()
            ordinal += 1
