"""Raw-byte BPB lengths, including the actual historical published cache.

These tests never train a tokenizer/model or rewrite production artifacts.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
    import torch
else:
    torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'ops/nanochat_fork/nanochat/belka_token_bytes.py'
spec = importlib.util.spec_from_file_location('raw_byte_contract', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Tokens:
    values = [b'\x80', b'\xe2\x82', b'hello', b'<|bos|>']
    size = 4
    special = 3
    def get_vocab_size(self): return self.size
    def get_special_tokens(self): return ['<|bos|>']
    def encode_special(self, name): return self.special
    def decode_single_token_bytes(self, i): return self.values[i]
    def decode(self, ids): pytest.fail('decoded replacement text must not determine byte lengths')


def test_raw_bytes_not_unicode_replacement_lengths():
    rng = torch.get_rng_state().clone()
    actual = module.token_byte_lengths(Tokens())
    assert actual.dtype == torch.int32 and actual.device == torch.device('cpu')
    assert actual.tolist() == [1, 2, 5, 0]
    assert torch.equal(torch.get_rng_state(), rng)
    actual.fill_(99)
    assert module.token_byte_lengths(Tokens(), torch.device('cpu')).tolist() == [1, 2, 5, 0]


@pytest.mark.parametrize('size', [0, -1, True, 4.0, 2**31])
def test_invalid_vocabulary_size(size):
    t = Tokens(); t.size = size
    with pytest.raises(ValueError, match='vocabulary size'):
        module.token_byte_lengths(t)


@pytest.mark.parametrize('special', [-1, 4, True, 3.0])
def test_invalid_special_ids(special):
    t = Tokens(); t.special = special
    with pytest.raises(ValueError, match='special token ID'):
        module.token_byte_lengths(t)


@pytest.mark.parametrize('raw', ['', 'abc', b'', None])
def test_ordinary_tokens_must_supply_raw_nonempty_bytes(raw):
    t = Tokens(); t.values = [raw, b'x', b'y', b'<|bos|>']
    with pytest.raises(ValueError, match='raw byte'):
        module.token_byte_lengths(t)


def installed():
    path = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat')).resolve()
    marker = path / 'BELKA_RUNTIME_MANIFEST.json'
    if not marker.is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('required runtime is absent')
        pytest.skip('requires installed tokenizer routing')
    m = json.loads(marker.read_text())
    for name in ('nanochat/tokenizer.py', 'nanochat/belka_token_bytes.py'):
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == m['files'][name]
    assert (path / 'nanochat/belka_token_bytes.py').read_bytes() == SOURCE.read_bytes()
    sys.path.insert(0, str(path))
    return path


@pytest.mark.parametrize('cache', ['missing', 'legacy', 'unreadable'])
def test_installed_getter_does_not_trust_or_modify_derived_cache(tmp_path, monkeypatch, cache):
    runtime = installed()
    import tiktoken
    from nanochat import tokenizer as api
    from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN
    assert Path(api.__file__).resolve() == runtime / 'nanochat/tokenizer.py'
    enc = tiktoken.Encoding(name='raw-byte-test', pat_str=SPLIT_PATTERN,
                           mergeable_ranks={bytes([i]): i for i in range(256)},
                           special_tokens={name: 256+i for i, name in enumerate(SPECIAL_TOKENS)})
    t = RustBPETokenizer(enc, '<|bos|>')
    t.save(tmp_path / 'tokenizer')
    p = tmp_path / 'tokenizer/token_bytes.pt'
    if cache == 'legacy':
        torch.save(torch.tensor([1]*128 + [3]*128 + [0]*len(SPECIAL_TOKENS)), p)
    elif cache == 'unreadable':
        p.write_bytes(b'not a torch cache')
    before = {f.name: f.read_bytes() for f in (tmp_path / 'tokenizer').iterdir()}
    rng = torch.get_rng_state().clone()
    monkeypatch.setenv('NANOCHAT_BASE_DIR', str(tmp_path))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('getter must not deserialize the derived cache'))
    lengths = api.get_token_bytes()
    assert lengths.tolist() == [1]*256 + [0]*len(SPECIAL_TOKENS)
    assert {f.name: f.read_bytes() for f in (tmp_path / 'tokenizer').iterdir()} == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_actual_published_tokenizer_uses_raw_lengths_not_its_historical_cache():
    installed()
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    from nanochat.belka_runtime import artifact_base_dir
    from nanochat.common import get_base_dir
    base = Path(artifact_base_dir(get_base_dir())) / 'tokenizer'
    if not (base / 'tokenizer.pkl').is_file() or not (base / 'token_bytes.pt').is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('published tokenizer artifacts must be restored in required CI')
        pytest.skip('requires the published tokenizer, not a made-up cache')
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in base.iterdir() if f.is_file()}
    t = get_tokenizer()
    specials = {t.encode_special(s) for s in t.get_special_tokens()}
    expected = [0 if i in specials else len(t.enc.decode_single_token_bytes(i)) for i in range(t.get_vocab_size())]
    actual = get_token_bytes()
    assert actual.dtype == torch.int32 and actual.tolist() == expected
    old = torch.load(base / 'token_bytes.pt', weights_only=True)
    if before['tokenizer.pkl'] == 'd9272e817e3978aa218597f34856056a577642aeedb1576299a1301dca5e71ac':
        assert t.get_vocab_size() == 16000
        assert sum(a != b for a, b in zip(old.tolist(), expected, strict=True)) == 151
    assert {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in base.iterdir() if f.is_file()} == before
