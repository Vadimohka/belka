"""Bounded inference benchmark backport for the hash-locked nanochat overlay.

The opt-in automatic batch sweep follows upstream 7e128a213007e29f931ca54b88744f13bf5d8429.
Keep the existing measurement and model APIs; repair input/context handling and
label partial sweeps and engine-slot throughput in the machine-readable result.
"""
from __future__ import annotations
from textwrap import indent


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f'inference benchmark anchor count != 1: {old[:90]!r}')
    return text.replace(old, new, 1)


def patch_infer_bench(text: str) -> str:
    """Transform pristine scripts/infer_bench.py; reject changed source anchors."""
    text = _replace_once(text, 'import argparse\nimport json\n',
                         'import argparse\nimport hashlib\nimport json\nimport math\n')
    text = _replace_once(text,
        'from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, get_peak_bandwidth, get_peak_flops',
        'from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, get_peak_bandwidth, get_peak_flops, COMPUTE_DTYPE')
    text = _replace_once(text,
        'def bench_generate(engine, prompt_tokens, batch_size, decode_tokens, temperature):',
        'def bench_generate(engine, prompt_tokens, batch_size, decode_tokens, temperature, seed=42):')
    text = _replace_once(text,
        '                                max_tokens=decode_tokens, temperature=temperature)',
        '                                max_tokens=decode_tokens, temperature=temperature, seed=seed)')
    measurement = '''    # The first next() runs the batch=1 prefill, the KV cache replication to
    # batch_size rows, and samples the first token: that is the TTFT.
    t_start = time.perf_counter()
    next(generator)
    torch.cuda.synchronize(device)
    ttft = time.perf_counter() - t_start

    # Every subsequent next() is one decode step for all rows.
    step_times = []
    while True:
        t0 = time.perf_counter()
        try:
            next(generator)
        except StopIteration:
            break
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)

    peak_vram = torch.cuda.max_memory_allocated(device)
    return dict(ttft=ttft, step_times=step_times, peak_vram=peak_vram)
'''
    text = _replace_once(text, measurement,
        '    try:\n' + indent(measurement, '    ') +
        '    finally:\n'
        '        # A synchronization error can leave the generator suspended with its KV cache.\n'
        '        # Release it before the caller handles OOM or propagates another exception.\n'
        '        generator.close()\n')
    text = _replace_once(text, '    paragraph = (',
        '    if type(num_tokens) is not int or num_tokens < 1:\n'
        '        raise ValueError("prompt tokens must be a positive integer")\n'
        '    paragraph = (')
    text = _replace_once(text,
        '    text = paragraph * (num_tokens // 10) # more than enough tokens\n'
        '    tokens = tokenizer.encode(text, prepend="<|bos|>")\n'
        '    assert len(tokens) >= num_tokens, "prompt text too short, increase the repetition"\n',
        '    tokens = tokenizer.encode(paragraph, prepend="<|bos|>")\n'
        '    if len(tokens) < 2:\n'
        '        raise ValueError("tokenizer did not encode the benchmark paragraph")\n'
        '    # Repeat ordinary IDs only: one BOS, exact length, no token-density assumption.\n'
        '    body = tokens[1:]\n'
        '    missing = max(0, num_tokens - len(tokens))\n'
        '    tokens.extend((body * ((missing + len(body) - 1) // len(body)))[:missing])\n')
    text = _replace_once(text, 'def main():\n', '''def parse_batch_sizes(value):
    if value == "auto":
        return [2 ** power for power in range(14)] # bounded at 8192; OOM may stop sooner
    try:
        batches = [int(part) for part in value.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ValueError("batch sizes must be positive integers or 'auto'") from exc
    if any(batch < 1 for batch in batches):
        raise ValueError("batch sizes must be positive integers or 'auto'")
    return batches


def main():
''')
    text = _replace_once(text, 'help="Comma-separated decode batch sizes")',
        'help="Comma-separated positive decode batch sizes, or auto: double 1..8192 until OOM")')
    text = _replace_once(text, '    args = parser.parse_args()\n', '''    parser.add_argument("--seed", type=int, default=42, help="Generation seed (0..2147483647)")
    args = parser.parse_args()
    if args.prompt_tokens < 1 or args.decode_tokens < 1:
        parser.error("prompt-tokens and decode-tokens must be positive")
    if not 0 <= args.seed <= 2**31 - 1:
        parser.error("seed must be between 0 and 2147483647")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("temperature must be finite and nonnegative")
    try:
        batch_sizes = parse_batch_sizes(args.batch_sizes)
    except ValueError as exc:
        parser.error(str(exc))
''')
    text = _replace_once(text, '    max_prompt = config.sequence_len - args.decode_tokens\n',
        '    if args.decode_tokens >= config.sequence_len:\n'
        '        parser.error("decode-tokens must leave room for at least one prompt token")\n'
        '    max_prompt = config.sequence_len - args.decode_tokens\n')
    text = _replace_once(text, '    payload = {\n', '''    payload = {
        "schema": "belka-inference-benchmark-v1",
        "metric_scope": "engine_decode_slots",
        "seed": args.seed,
        "compute_dtype": str(COMPUTE_DTYPE).replace("torch.", ""),
        "prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(prompt_tokens, separators=(",", ":")).encode("ascii")).hexdigest(),
        "batch_sizes_requested": args.batch_sizes,
        "auto_batch_limit": 8192 if args.batch_sizes == "auto" else None,
        "status": "COMPLETE",
        "sweep_stop_reason": "auto_batch_limit" if args.batch_sizes == "auto" else "requested_batches_completed",
        "failed_batch_size": None,
        "model_tag_requested": args.model_tag,
''')
    text = _replace_once(text,
        '    bench_generate(engine, prompt_tokens, 1, 2, args.temperature) # warmup\n'
        '    prefill_result = bench_generate(engine, prompt_tokens, 1, 2, args.temperature)\n',
        '    bench_generate(engine, prompt_tokens, 1, min(2, args.decode_tokens), args.temperature, args.seed) # warmup\n'
        '    prefill_result = bench_generate(engine, prompt_tokens, 1, min(2, args.decode_tokens), args.temperature, args.seed)\n')
    text = _replace_once(text, '    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]\n', '')
    text = _replace_once(text, '''        # warmup (cublas autotune, allocator warm, attention kernels)
        bench_generate(engine, prompt_tokens, batch_size, 8, args.temperature)
        # timed run
        result = bench_generate(engine, prompt_tokens, batch_size, args.decode_tokens, args.temperature)
''', '''        sweep_oom = False
        try:
            # Keep warmup within the same context budget as the timed run.
            bench_generate(engine, prompt_tokens, batch_size, min(8, args.decode_tokens), args.temperature, args.seed)
            result = bench_generate(engine, prompt_tokens, batch_size, args.decode_tokens, args.temperature, args.seed)
        except torch.cuda.OutOfMemoryError:
            payload.update(status="PARTIAL", sweep_stop_reason="cuda_oom", failed_batch_size=batch_size)
            print(f"{batch_size:>6}  out of memory: ending sweep; previous measurements retained")
            sweep_oom = True
        if sweep_oom:
            # The active exception traceback can retain failed forward/KV locals.
            # Leave its handler before returning their unused blocks to CUDA.
            torch.cuda.empty_cache()
            break
''')
    text = _replace_once(text, '''        if num_steps == 0:
            print(f"{batch_size:>6}  all rows terminated during warmup?! skipping")
            continue
''', '''        if num_steps == 0:
            print(f"{batch_size:>6}  no decode steps after the first token; TTFT/VRAM only")
            payload["sweep"].append({
                "batch_size": batch_size, "ttft_sec": round(result["ttft"], 6),
                "peak_vram_bytes": result["peak_vram"], "decode_steps": 0,
                "tpot_sec": None, "tok_per_sec": None, "mbu_percent": None, "mfu_percent": None,
            })
            continue
''')
    return text
