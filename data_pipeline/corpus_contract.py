"""Explicit corpus splits, stable text identity and complete Hamming lookup."""
from __future__ import annotations
import hashlib
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
if __package__ == 'nanochat':
    from .belka_artifacts import resolve
else:
    from .artifact_store import resolve


def resolve_corpus_dir(root: str | Path) -> Path:
    root = Path(root).resolve()
    pointer = root/'CORPUS_CURRENT.json'
    if pointer.exists() or pointer.is_symlink():
        directory, _ = resolve(root,'corpus')
        return directory
    if not root.is_dir(): raise FileNotFoundError(f'corpus directory missing: {root}')
    return root


def select_shards(root: str | Path, split: str) -> list[Path]:
    if split not in ('train','val'): raise ValueError('split must be train or val')
    directory = resolve_corpus_dir(root)
    all_files = sorted(directory.glob('*.parquet'))
    bad = [p.name for p in all_files if not re.fullmatch(r'(train|val)_\d+\.parquet',p.name)]
    if bad: raise ValueError(f'ambiguous root Parquet files (move references into subdirectories): {bad}')
    groups = {s:[p for p in all_files if p.name.startswith(s+'_')] for s in ('train','val')}
    if not all(groups.values()): raise ValueError(f'explicit nonempty train_* and val_* shards required: {directory}')
    for p in all_files:
        if not p.resolve().is_relative_to(directory):
            # Legacy bundle links may point to a sibling corpus within base_dir.
            if not p.resolve().is_relative_to(directory.parent):
                raise ValueError(f'shard escapes the corpus base: {p}')
        if not p.is_file(): raise ValueError(f'not a regular Parquet file: {p}')
    return groups[split]


def canonical_text(text: str) -> str:
    if not isinstance(text,str) or not text.strip(): raise ValueError('nonempty text required')
    # Preserve punctuation, digits, case and orthographic distinctions.
    return ' '.join(unicodedata.normalize('NFC',text).split())


def content_id(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode('utf-8')).hexdigest()


def group_split(group: str, val_ratio: float, salt: str = 'belka-group-split-v1') -> str:
    if not isinstance(group,str) or not group: raise ValueError('nonempty group ID required')
    if isinstance(val_ratio,bool) or not math.isfinite(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError('validation ratio must be finite and in (0,1)')
    value = int.from_bytes(hashlib.sha256((salt+'\0'+group).encode('utf-8')).digest()[:8],'big')
    return 'val' if value < val_ratio * 2**64 else 'train'


class HammingIndex:
    """Exact radius search via r+1 disjoint bit bands and verified candidates.

    If two b-bit values differ in <=r positions, at least one of r+1 bands is
    identical (pigeonhole principle). Unlike nearby numeric high-bit buckets,
    this candidate index does not miss any such neighbor. Worst case is O(N).
    """
    def __init__(self, bits: int = 64, radius: int = 6):
        if not isinstance(bits,int) or not 1 <= bits <= 128: raise ValueError('invalid bit width')
        if not isinstance(radius,int) or not 0 <= radius < bits: raise ValueError('invalid Hamming radius')
        self.bits, self.radius = bits,radius
        self.tables = [defaultdict(set) for _ in range(radius+1)]
        self.bands = [(bits*i//(radius+1), bits*(i+1)//(radius+1)-bits*i//(radius+1)) for i in range(radius+1)]

    def _keys(self,value: int):
        if isinstance(value,bool) or not isinstance(value,int) or not 0 <= value < 1 << self.bits:
            raise ValueError('hash outside declared bit width')
        return [(value >> start) & ((1 << width)-1) for start,width in self.bands]

    def add(self,value: int) -> None:
        for table,key in zip(self.tables,self._keys(value)): table[key].add(value)

    def neighbors(self,value: int) -> set[int]:
        candidates = set()
        for table,key in zip(self.tables,self._keys(value)): candidates.update(table.get(key,()))
        return {other for other in candidates if (value ^ other).bit_count() <= self.radius}

    def contains_near(self,value: int) -> bool:
        seen = set()
        for table,key in zip(self.tables,self._keys(value)):
            for other in table.get(key,()):
                if other not in seen:
                    if (value ^ other).bit_count() <= self.radius: return True
                    seen.add(other)
        return False
