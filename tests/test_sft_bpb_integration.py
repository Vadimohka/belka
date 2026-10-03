"""SFT's native int32 inputs must reach BPB without an implicit dtype rewrite.

Unit checks use an actual Embedding/cross-entropy module. Required-runtime tests
compose the hash-verified installed SFT generator, BPE, GPT and BPB entrypoint.
Only the generator definition executes, never trainer setup or optimization.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
    import torch
else:
    torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'ops/nanochat_fork/nanochat/belka_metrics.py'
spec = importlib.util.spec_from_file_location('sft_bpb_metrics', SOURCE)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class EmbeddingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Explicit weights avoid changing global RNG during fixture setup.
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor([[.2, -.5, 1.], [.4, .3, -.2], [-1., 0., 2.]]), freeze=False)
        self.seen = []

    def get_device(self):
        return self.embedding.weight.device

    def forward(self, x, y, loss_reduction):
        self.seen.append((x.dtype, x.data_ptr(), y.dtype))
        return torch.nn.functional.cross_entropy(
            self.embedding(x).reshape(-1, 3), y.reshape(-1),
            ignore_index=-1, reduction=loss_reduction)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('noncontiguous', [False, True])
def test_bpb_preserves_supported_input_dtype_and_storage(dtype, noncontiguous):
    x = torch.tensor([[1, 2], [0, 1]], dtype=dtype)
    y = torch.tensor([[2, 1], [0, -1]], dtype=torch.int64)
    if noncontiguous:
        x, y = x.T, y.T
    model = EmbeddingLoss()
    model.embedding.weight.grad = torch.full_like(model.embedding.weight, .25)
    before = (x.clone(), y.clone(), model.embedding.weight.detach().clone(), torch.get_rng_state().clone())
    lengths = torch.tensor([0, 2, 5])
    expected = metrics.evaluate_bpb(EmbeddingLoss(), [(x.long(), y)], 1, lengths)
    actual = metrics.evaluate_bpb(model, [(x, y)], 1, lengths)
    assert actual == expected and math.isfinite(actual)
    assert model.seen == [(dtype, x.data_ptr(), torch.int64)]
    assert model.training
    assert torch.equal(x, before[0]) and torch.equal(y, before[1])
    assert torch.equal(model.embedding.weight, before[2])
    assert torch.equal(model.embedding.weight.grad, torch.full_like(model.embedding.weight, .25))
    assert torch.equal(torch.get_rng_state(), before[3])


@pytest.mark.parametrize('field,value', [('x', -1), ('x', 3), ('y', -2), ('y', 3)])
def test_native_int32_batch_still_validates_ranges_before_forward(field, value):
    x = torch.tensor([[1, 2]], dtype=torch.int32)
    y = torch.tensor([[2, 1]], dtype=torch.int64)
    (x if field == 'x' else y)[0, 0] = value
    model = EmbeddingLoss()
    with pytest.raises(ValueError, match='token ID outside'):
        metrics.evaluate_bpb(model, [(x, y)], 1, torch.tensor([0, 2, 5]))
    assert not model.seen and model.training


def generator_definition(source, filename='<sft source>'):
    """Select one trusted definition; no top-level trainer code executes."""
    nodes = [node for node in ast.parse(source, filename=filename).body
             if isinstance(node, ast.FunctionDef) and node.name == 'sft_data_generator_bos_bestfit']
    if len(nodes) != 1 or nodes[0].decorator_list:
        raise ValueError('expected one undecorated installed SFT generator')
    args = nodes[0].args
    if ([arg.arg for arg in args.args] != ['split', 'buffer_size']
            or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
            or len(args.defaults) != 1 or not isinstance(args.defaults[0], ast.Constant)
            or type(args.defaults[0].value) is not int or args.defaults[0].value != 100):
        raise ValueError('unexpected SFT generator signature')
    return compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec', dont_inherit=True)


def make_generator(code, tokenizer, rendered_sft, conversations, split, sequence_length=128):
    # The namespace is test data/runtime bindings, not a mocked generator body.
    namespace = dict(torch=torch, tokenizer=tokenizer, rendered_sft=rendered_sft,
                     train_dataset=conversations, val_dataset=conversations,
                     args=SimpleNamespace(max_seq_len=sequence_length, device_batch_size=2, num_iterations=2),
                     ddp_rank=0, ddp_world_size=1, device_type='cpu', device=torch.device('cpu'),
                     last_step=False, approx_progress=0., current_epoch=1)
    exec(code, namespace)
    return namespace['sft_data_generator_bos_bestfit'](split, buffer_size=1)


def installed_runtime():
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat')).resolve()
    manifest = runtime / 'BELKA_RUNTIME_MANIFEST.json'
    if not manifest.is_file():
        message = 'requires installed hash-verified SFT/GPT/BPE runtime'
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail(message)
        pytest.skip(message)
    identity = json.loads(manifest.read_text())
    for name in ('scripts/chat_sft_be.py', 'nanochat/tokenizer.py', 'nanochat/gpt.py',
                 'nanochat/belka_runtime.py', 'nanochat/belka_metrics.py', 'nanochat/loss_eval.py'):
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == identity['files'][name]
    assert (runtime / 'nanochat/belka_metrics.py').read_bytes() == SOURCE.read_bytes()
    sys.path.insert(0, str(runtime))
    return runtime


@pytest.mark.parametrize('split', ['train', 'val'])
def test_actual_sft_generator_bpe_gpt_and_bpb_compose(split):
    runtime = installed_runtime()
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.belka_runtime import rendered_sft
    from nanochat import belka_metrics, loss_eval
    assert Path(belka_metrics.__file__).resolve() == runtime / 'nanochat/belka_metrics.py'
    assert loss_eval.evaluate_bpb is belka_metrics.evaluate_bpb
    source = runtime / 'scripts/chat_sft_be.py'
    code = generator_definition(source.read_text(), str(source))
    text = ['Адказвай па-беларуску.', 'Як справы?', 'Добра.',
            'Сёння цёпла.', 'Дзе кніга?', 'Кніга на стале.']
    tokenizer = RustBPETokenizer.train_from_iterator(iter(text * 8), vocab_size=300)
    assert tokenizer.get_vocab_size() == 300
    conversation = {'messages': [{'role': 'system', 'content': text[0]},
                                 {'role': 'user', 'content': text[1]},
                                 {'role': 'assistant', 'content': text[2]}]}
    original = copy.deepcopy(conversation)
    iterator = make_generator(code, tokenizer, rendered_sft, [conversation], split)
    try:
        x, y = next(iterator)
    finally:
        iterator.close()
    assert x.dtype == torch.int32 and y.dtype == torch.int64
    assert x.shape == y.shape == (2, 128) and (y == -1).any()
    assert conversation == original

    # Independent expected layout: system text merges into the user, then only
    # assistant content and its end token are targets. BOS/padding stay masked.
    special = tokenizer.encode_special
    prompt = [special('<|bos|>'), special('<|user_start|>')]
    prompt += tokenizer.encode(text[0] + '\n\n' + text[1])
    prompt += [special('<|user_end|>'), special('<|assistant_start|>')]
    answer = tokenizer.encode(text[2]) + [special('<|assistant_end|>')]
    ids = prompt + answer
    mask = [0] * len(prompt) + [1] * len(answer)
    copies, padding = divmod(129, len(ids))
    assert copies >= 1 and padding > 0
    row = ids * copies + [special('<|bos|>')] * padding
    target_mask = (mask * copies + [0] * padding)[1:]
    expected_y = torch.tensor(row[1:], dtype=torch.int64)
    expected_y[torch.tensor(target_mask) == 0] = -1
    assert torch.equal(x, torch.tensor([row[:-1]] * 2, dtype=torch.int32))
    assert torch.equal(y, expected_y.expand(2, -1))

    special_ids = {special(name) for name in tokenizer.get_special_tokens()}
    lengths = torch.tensor([0 if i in special_ids else len(tokenizer.decode_single_token_bytes(i))
                            for i in range(tokenizer.get_vocab_size())], dtype=torch.int64)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29)
            model = GPT(GPTConfig(sequence_len=128, vocab_size=300, n_layer=2,
                                  n_head=2, n_kv_head=1, n_embd=32, window_pattern='SL'))
            model.init_weights()
            model.eval()
            with torch.no_grad():
                logits = model(x)
                losses = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 300), expected_y.expand(2, -1).reshape(-1),
                    ignore_index=-1, reduction='none').double()
                byte_lengths = lengths[y.clamp_min(0)].reshape(-1)
                valid = (y.reshape(-1) >= 0) & (byte_lengths > 0)
                expected = losses[valid].sum().item() / (math.log(2) * byte_lengths[valid].sum().item())
            model.train()
            original_x, original_y = x.clone(), y.clone()
            rng = torch.get_rng_state().clone()
            actual = loss_eval.evaluate_bpb(model, [(x, y)], 1, lengths)
            wide = loss_eval.evaluate_bpb(model, [(x.long(), y)], 1, lengths)
            assert actual == wide == expected and math.isfinite(actual)
            assert model.training and all(p.grad is None for p in model.parameters())
            assert torch.equal(torch.get_rng_state(), rng)
            assert torch.equal(x, original_x) and torch.equal(y, original_y)
    finally:
        torch.set_num_threads(threads)
