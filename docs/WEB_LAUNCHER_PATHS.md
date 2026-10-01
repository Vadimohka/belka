# Web launcher path preflight — R08/R30-L01

This scope fixes CLI path validation in `ops/local/run_chat_web.sh`. It does not
change model code, checkpoints, datasets, training profiles or the server protocol.

## Defect and ordering

The previous launcher sourced `pack_paths.sh` and ran `repo_guard.sh` before
parsing arguments. Those checks validated environment/default paths, not the later
`--nanochat-dir` and `--base-dir` values. A command could therefore activate and
patch an external runtime, or pass an external asset directory to the server.
It also created default directories for `--help` and malformed invocations.

The launcher now performs these steps in order:

1. Parse the complete CLI; reject missing/empty option values and unknown options.
   `--help` exits without initializing paths or activating Python.
2. Run the existing path helper in read-only mode and the repository guard with
   the final overrides. The helper checks all 17 configured directories for
   containment, resolves symlink prefixes, rejects PACK_DIR itself and existing
   non-directory targets, and exports canonical paths.
3. Require an existing checkout and a readable activation file whose resolved
   path stays inside PACK_DIR. This matters because activation is sourced shell
   code, not passive configuration.
4. Recheck paths through the existing helper before its directory-creation pass;
   only then activate Python and run the runtime patcher/identity verifier.
5. Execute the existing server command. `--no-patch` skips only overlay refresh;
   it never skips path checks or `--verify-only`.

## Compatibility

CLI paths take precedence over inherited `NANOCHAT_DIR` / `NANOCHAT_BASE_DIR`.
Relative paths resolve against the caller's working directory, before entering
the runtime. Absolute paths and in-repository directory symlinks are accepted;
paths containing spaces remain one argument. For a path beginning with `-`, use
an absolute path or a `./` prefix. Repeated options retain last-value-wins behavior.

Defaults, server sampling, host/port feature detection and model phase selection
are unchanged. This is not new validation of every server setting: the server
still validates its non-path configuration. Patcher failures and identity-check
failures still stop execution and preserve their nonzero exit status.

A virtualenv's Python executable may normally link to a system interpreter. This
scope checks the sourced activation file, not the interpreter's symlink target.
It does not sandbox trusted repository/virtualenv code or eliminate filesystem
races; do not mutate runtime paths concurrently with a launch. Filesystem
permissions, disk capacity and concurrent changes can still cause later I/O errors.

## Regression tests

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/test_chat_web_launcher.py
bash -n ops/local/run_chat_web.sh
```

The tests execute the real Bash launcher, `pack_paths.sh`, `repo_guard.sh`, GNU
realpath and mkdir. A temporary activation script and recording Python executable
stand in for the runtime processes, so no model, server, network or checkpoint is
required. The recording executable is not evidence of a real-checkpoint launch.

Coverage includes CLI/environment escapes, PACK_DIR-as-target, external symlinks,
regular-file targets, missing checkout/activation, externally linked activation,
help/argument errors without writes, relative overrides from another directory,
spaces, in-repo symlinks, canonical environment propagation and mandatory identity
verification with `--no-patch`. Invalid-path cases compare the whole fixture tree
before and after, including files outside the fixture PACK_DIR.

The four absolute-external-path regression cases returned success on the previous
launcher. The fixed launcher rejects them before activation or runtime invocation.
Full-suite CI evidence is recorded in PR #5 against the published commit.

## Rollback

Revert the single R08/R30-L01 commit, including its source-hash update and tests.
No data/checkpoint rollback or environment rebuild is needed. Reverting restores
the old CLI path bypass, so do not use external path overrides on that version.
