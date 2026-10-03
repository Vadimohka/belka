# Checkpoint discovery — R26-C02

Scope: `find_last_step` and the shared incomplete-save guard in
`ops/nanochat_fork/nanochat/belka_checkpoint.py`. See also
`CHECKPOINT_MANIFEST_VALIDATION.md` and GitHub issue #8. The checkpoint writer,
training loops, upstream pin and on-disk v1 format are unchanged.

## Selection contract

Discovery enumerates directory entries once. It prefers committed checkpoints
and selects their largest numeric step. Numeric commit filenames must match the
writer's canonical `commit_{step:06d}.json` spelling (at least six digits, ASCII,
without redundant leading zeros). Numeric aliases, including Unicode digits,
raise `ValueError`; unrelated nonnumeric files are ignored.

For the selected step, discovery reads the bounded, strictly validated v1 commit
manifest and checks that every listed payload exists as a regular, nonsymlink
file. A bad marker, malformed manifest or missing/nonregular payload is an error,
not permission to silently select older weights or enter the legacy path. Older
manifests are not parsed when a newer canonical committed candidate is selected.
To use an earlier checkpoint after a failure, select its step explicitly through
an existing caller that supports it and let the normal load validation run. Do
not delete or rename files merely to hide the failure.

Without any numeric commit candidates, discovery considers canonical legacy
model/metadata pairs, both regular and nonsymlink. Incomplete legacy pairs are
ineligible. A `pending_*`, `commit_*`, `meta_*_rank*` or `rng_*_rank*` entry for a
step excludes it from legacy consideration. All ranks are considered and dangling
links still count as evidence. Direct `validate_checkpoint` uses the same guard,
so `BELKA_ALLOW_LEGACY_CHECKPOINT=YES` cannot bypass an unfinished v1 save.

`optim_*_rank*.pt` alone is deliberately **not** a new-format indicator: original
nanochat already used that filename layout. Genuine legacy inference keeps its
warning; optimizer loading still requires the existing explicit opt-in. Missing
or empty checkpoint roots raise `FileNotFoundError`; non-directory roots and
other directory-enumeration errors propagate rather than masquerading as a
successful or empty scan. No checkpoint files are created, changed or removed.

## Performance and limits

Discovery reads only one commit manifest and stats its declared payloads. It
does not read/hash historical tensors, discover a tokenizer, import PyTorch or
check optimizer/backend compatibility. A structurally complete checkpoint may
still contain corrupt bytes: `validate_checkpoint` / `load_checkpoint` remain
mandatory for full SHA256, tokenizer identity and load-mode checks. A returned
step is not proof of authentic weights or exact training resumption.

This preflight does not eliminate races with concurrent filesystem changes.
Checkpoint directories and their parent paths must be trusted. A save published
after enumeration may be noticed on the next call. New saves without a commit
marker are ignored, leaving the previous committed checkpoint selectable.

## Tests and rollback

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/test_checkpoint_discovery.py
PYTHONPATH="$PWD" python -m pytest -q tests/test_checkpoint_manifest_schema.py
```

The 45 discovery cases use real temporary files. Only the final case needs
PyTorch; it checks real CPU save/load and rejection of corrupted weights before
tensor deserialization, with tokenizer identity supplied by a fixture. Lightweight
CI explicitly skips that one case without PyTorch; Runtime contracts must run it.
Both workflows must pass on the published commit before accepting this scope.
On the verified pre-change module, the same suite reports 35 failed / 10 passed.
These are test cases, not 35 independent production defects.

Rollback is a normal revert of the R26-C02 commit, including its source hashes;
there is no checkpoint migration to undo. Reverting restores the permissive
discovery behavior, so preserve the regression evidence. This scope does not
close trainer resume, owner-local d8 artifacts, GPU validation or model quality.
