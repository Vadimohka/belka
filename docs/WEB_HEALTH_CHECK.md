# Web health smoke: R30-H01

Scope: make the local health smoke an honest readiness and generation check.
This is one subtask of R30, not completion of the web-server audit.

## Why the old check was insufficient

Belka's `/chat/completions` endpoint returns SSE even when `stream:false` is
supplied. A generation exception can therefore be reported in an `error` event
inside an HTTP 200 response. `curl --fail` alone cannot detect that failure.
Likewise, HTTP 200 from `/health` does not mean `ready:true`.

The checker now requires HTTP 200 with the correct Content-Type, valid UTF-8,
valid object-shaped JSON without duplicate keys, loaded workers, and a complete
SSE sequence: at least one non-whitespace text chunk followed by exactly one
`{"done":true}` event. Error events, data after completion, missing termination,
malformed payloads and empty completions fail. SSE uses blank-line event framing
and LF/CRLF/CR line endings; it is not parsed as one JSON object per HTTP response.
Reference: https://html.spec.whatwg.org/dev/server-sent-events.html

## Owner smoke command

Use an existing checkpoint and the installed, verified runtime. This command
starts inference, makes one 32-token request, then stops the process; it does not
start training. It was not run against an owner's trained checkpoint in this scope.

```bash
bash ops/local/test_chat_web_health.sh \
  --nanochat-dir "$PWD/.workspace/nanochat" \
  --base-dir "$PWD/.workspace/nanochat_base" \
  --model-tag YOUR_EXISTING_TAG --phase sft --port 8000 --timeout 90
```

`--timeout` bounds readiness polling (1..3600 seconds). Each health request has
at most five seconds and the completion request at most 60 seconds. Cleanup sends
TERM, waits up to five seconds, then sends KILL to the launched process if needed.
Use a free local port; this is not a process-identity or concurrent-bind proof.
Loopback and wildcard bind addresses are supported. A wildcard is probed through
loopback, not as a client destination. External endpoints are deliberately rejected.

When `BELKA_API_KEY` is set, the child server inherits it and curl receives the
Authorization header through stdin configuration, not argv or a credential file.
Line breaks are rejected, quotes/backslashes are escaped, xtrace is disabled,
user curl configuration is ignored and environment proxies are bypassed. This
does not hide inherited environment variables from a privileged local observer.
Public binding still requires the server's existing API-key/TLS policy.

All CLI path overrides are checked before directories are created. Help and
invalid arguments do not start the server or create the workspace. Logs and the
count-only `response_summary.json` live in a new private `.health-check.*`
directory below the selected base. The checker does not store generated text or
print server error payloads. Always use the command's exit status; an empty or
stale output file is not evidence of a pass.

## Regression tests

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/test_chat_web_health.py
bash -n ops/local/test_chat_web_health.sh
```

The suite contains protocol tests, real Bash/curl requests to a local synthetic
HTTP server, and actual FastAPI streaming-handler tests with a substituted
inference worker. It covers successful and failed HTTP-200 streams, bad MIME,
truncation, authentication, credential non-disclosure in argv/files/output,
readiness timeouts, process cleanup, CLI validation and path escapes.

It does not test model quality, GPU behavior, the authenticated browser UI, or
full launcher migration. No server, model, tokenizer or dataset code is changed.
The existing full CI and Runtime contracts workflows collect these tests without
changing their commands or weakening their gates.

Rollback: revert the health-smoke commit as a unit (script, checker, tests,
documentation and source-hash entries). Do not revert unrelated PR #5 fixes or
change runtime/corpus/checkpoint files. Remaining R30 work includes authenticated
UI behavior, concurrent serving/cancellation and a real-checkpoint smoke.
