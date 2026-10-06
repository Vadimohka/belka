"""R27-D03: real RustBPE/tiktoken + Parquet + public loader contracts.

Fit only a tiny synthetic tokenizer, never a language model or owner dataset.
Checkpoint roundtrips require the installed overlay; independent rank iterators
substitute only topology discovery, not a distributed process group.
"""
import copy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('NANOCHAT_DIR', ROOT/'.workspace/nanochat'))
if not (RUNTIME/'nanochat/tokenizer.py').is_file():
    if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
        raise RuntimeError('required pinned nanochat tokenizer source is absent')
    pytest.skip('requires pinned nanochat tokenizer source', allow_module_level=True)
sys.path.insert(0, str(RUNTIME))
torch = pytest.importorskip('torch')
pa = pytest.importorskip('pyarrow')
import pyarrow.parquet as pq
pytest.importorskip('rustbpe')
tiktoken = pytest.importorskip('tiktoken')
from nanochat import common, belka_runtime as contracts
from nanochat.tokenizer import RustBPETokenizer

spec = importlib.util.spec_from_file_location(
    'bpe_loader_test', ROOT/'ops/nanochat_fork/nanochat/belka_stream.py')
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)

TEXTS = [
    'Беларуская мова. Сёння сонца асвятляе возера.',
    'У лесе растуць высокія дрэвы. Побач бяжыць ручай.',
    'Літары ў, і, ё. Апострафы: з’ява, з\'ява, зʼява.',
    'Кніга пра гісторыю і культуру. Лічбы 12345.',
    'Літаральны запіс <|bos|> не павінен стаць мяжой дакумента.',
    'Дождж скончыўся. Над горадам зноў яснае неба.',
]


@pytest.fixture(scope='module')
def encoding():
    tok = RustBPETokenizer.train_from_iterator(iter(TEXTS * 4), vocab_size=300)
    assert len(tok.enc._mergeable_ranks) > 256  # actual learned merges, not byte-only fixture
    for text in TEXTS:
        assert tok.decode(tok.encode(text)) == text
    return tok.enc


@pytest.fixture
def tokenizer(encoding):
    return RustBPETokenizer(encoding, '<|bos|>')


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    base = tmp_path/'base'
    data = base/'base_data_climbmix'
    data.mkdir(parents=True)
    for name, texts, size in [
        ('train_00000.parquet', [], 2), ('train_00001.parquet', TEXTS[:4], 3),
        ('train_00002.parquet', TEXTS[4:], 1), ('val_00000.parquet', TEXTS[2:4], 1),
    ]:
        pq.write_table(pa.table({'text': pa.array(texts, type=pa.string())}),
                       data/name, row_group_size=size)
    monkeypatch.setenv('NANOCHAT_BASE_DIR', str(base))
    monkeypatch.setattr(common, 'get_base_dir', lambda: str(base))
    monkeypatch.setattr(common, 'get_dist_info', lambda: (False, 0, 0, 1))
    return base


def loader(tok, *, saved=None, split='train', batch=1, length=7):
    return stream.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, batch, length, split, device='cpu', resume_state_dict=saved)


def same_batch(actual, expected):
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def snapshot(base):
    return {str(p.relative_to(base)): p.read_bytes() for p in base.rglob('*') if p.is_file()}


def changed_encoding(tok, kind):
    old = tok.enc
    ranks = dict(old._mergeable_ranks)
    specials = dict(old._special_tokens)
    pattern = old._pat_str
    if kind == 'tokens':
        ranks[b'a'], ranks[b'b'] = ranks[b'b'], ranks[b'a']
    elif kind == 'specials':
        specials['<|bos|>'], specials['<|user_start|>'] = specials['<|user_start|>'], specials['<|bos|>']
    elif kind == 'pattern':
        pattern = r'\S+|\s+'
    else:
        raise AssertionError(kind)
    enc = tiktoken.Encoding(name='belka-test-changed', pat_str=pattern,
                           mergeable_ranks=ranks, special_tokens=specials)
    return RustBPETokenizer(enc, '<|bos|>')


@pytest.mark.parametrize('active', [False, True])
def test_different_bos_cannot_reuse_same_encoding_identity(corpus, tokenizer, active):
    current = loader(tokenizer)
    initial = next(current)
    saved = (next(current) if active else initial)[2]
    other = RustBPETokenizer(tokenizer.enc, '<|user_start|>')
    assert contracts.tokenizer_fingerprint(other) == contracts.tokenizer_fingerprint(tokenizer)
    assert other.get_bos_token_id() != tokenizer.get_bos_token_id()
    before = snapshot(corpus)
    with pytest.raises(ValueError, match='BOS'):
        next(loader(other, saved=copy.deepcopy(saved)))
    assert snapshot(corpus) == before


@pytest.mark.parametrize('bos', [None, True, False, -1, 0, 1.5, '291', 2**63])
def test_bad_bos_rejected_before_corpus_access(corpus, tokenizer, monkeypatch, bos):
    tokenizer.bos_token_id = bos
    def no_corpus(*args, **kwargs):
        pytest.fail('invalid BOS reached corpus discovery')
    monkeypatch.setattr(contracts, 'split_parquet_files', no_corpus)
    with pytest.raises(ValueError, match='BOS'):
        next(loader(tokenizer))


def test_missing_named_bos_is_rejected_before_corpus_access(corpus, tokenizer, monkeypatch):
    old = tokenizer.enc
    specials = dict(old._special_tokens)
    del specials['<|bos|>']
    enc = tiktoken.Encoding(name='missing-bos', pat_str=old._pat_str,
                           mergeable_ranks=old._mergeable_ranks, special_tokens=specials)
    other = RustBPETokenizer(enc, '<|user_start|>')
    monkeypatch.setattr(contracts, 'split_parquet_files', lambda *a, **k: pytest.fail('corpus read'))
    with pytest.raises(ValueError, match='BOS'):
        next(loader(other))


def test_started_loader_keeps_validated_document_boundary(corpus, tokenizer):
    current = loader(tokenizer, length=53)
    expected = loader(RustBPETokenizer(tokenizer.enc, '<|bos|>'), length=53)
    same_batch(next(current), next(expected))
    tokenizer.bos_token_id = tokenizer.enc.encode_single_token('<|user_start|>')
    # Existing iterator captures the validated boundary; a new iterator rejects.
    for _ in range(12):
        same_batch(next(current), next(expected))
    with pytest.raises(ValueError, match='BOS'):
        next(loader(tokenizer))


@pytest.mark.parametrize('rank,world,batch,length', [(0, 1, 1, 1), (0, 2, 2, 19), (1, 2, 3, 73), (2, 3, 2, 41)])
def test_real_bpe_loader_matches_direct_stream_and_json_resume(corpus, tokenizer, monkeypatch, rank, world, batch, length):
    monkeypatch.setattr(common, 'get_dist_info', lambda: (world > 1, rank, rank, world))
    current = loader(tokenizer, batch=batch, length=length)
    bos = tokenizer.get_bos_token_id()
    tokens = [t for text in TEXTS[rank::world] for t in [bos] + tokenizer.enc.encode_ordinary(text)]
    expected = itertools.cycle(tokens)
    previous = next(expected)
    for _ in range(15):
        x, y, saved = next(current)
        assert x.dtype == y.dtype == torch.int64 and x.is_contiguous() and y.is_contiguous()
        assert x.shape == y.shape == (batch, length)
        assert x[0, 0].item() == previous
        targets = list(itertools.islice(expected, batch * length))
        assert y.flatten().tolist() == targets
        assert x.flatten().tolist() == [previous] + targets[:-1]
        previous = targets[-1]
        path = corpus/'loader_checkpoint.json'
        path.write_text(json.dumps(saved), encoding='utf-8')
        resumed = loader(tokenizer, saved=json.loads(path.read_text()), batch=batch, length=length)
        same_batch(next(resumed), (x, y, saved))
        # Peek only a separate restored iterator, never consume the running one.
        following = next(resumed)
        replay = loader(tokenizer, saved=following[2], batch=batch, length=length)
        same_batch(next(replay), following)


@pytest.mark.parametrize('change', ['text', 'rename', 'add', 'split', 'rank', 'world', 'tokens', 'specials', 'pattern', 'batch', 'length'])
def test_resume_rejects_data_tokenizer_topology_and_shape_changes(corpus, tokenizer, monkeypatch, change):
    if change == 'rank':
        monkeypatch.setattr(common, 'get_dist_info', lambda: (True, 0, 0, 2))
    current = loader(tokenizer)
    next(current)
    saved = next(current)[2]
    data = corpus/'base_data_climbmix'
    options = {}
    if change == 'text':
        pq.write_table(pa.table({'text': [TEXTS[0] + ' Дадатковы сказ.']}), data/'train_00001.parquet')
    elif change == 'rename':
        (data/'train_00001.parquet').rename(data/'train_00005.parquet')
    elif change == 'add':
        pq.write_table(pa.table({'text': [TEXTS[0]]}), data/'train_00003.parquet')
    elif change == 'split':
        options['split'] = 'val'
    elif change in ('rank', 'world'):
        monkeypatch.setattr(common, 'get_dist_info', lambda: (True, int(change == 'rank'), 0, 2))
    elif change in ('tokens', 'specials', 'pattern'):
        other = changed_encoding(tokenizer, change)
        assert other.get_vocab_size() == tokenizer.get_vocab_size()
        tokenizer = other
    else:
        options[change] = 2 if change == 'batch' else 8
    before, original = snapshot(corpus), copy.deepcopy(saved)
    with pytest.raises(ValueError, match='incompatible loader resume'):
        next(loader(tokenizer, saved=saved, **options))
    assert saved == original and snapshot(corpus) == before


def test_default_bos_preserves_existing_v1_identity(corpus, tokenizer):
    digest = hashlib.sha256()
    for path in contracts.split_parquet_files(corpus/'base_data_climbmix', 'train'):
        digest.update(Path(path).name.encode())
        digest.update(Path(path).read_bytes())
    # This is the identity expression used before R27-D03; no migration needed.
    expected = f'train:0/1:{digest.hexdigest()}:{contracts.tokenizer_fingerprint(tokenizer)}'
    batch = next(loader(tokenizer))
    assert batch[2]['schema'] == 'belka-stream-v1'
    assert batch[2]['identity'] == expected
    same_batch(next(loader(tokenizer, saved=batch[2])), batch)


def checkpoint_module():
    if not (RUNTIME/'nanochat/belka_checkpoint.py').is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('required installed checkpoint overlay is absent')
        pytest.skip('requires installed checkpoint overlay, not a tokenizer-only snapshot')
    from nanochat import belka_checkpoint
    return belka_checkpoint


def test_real_checkpoint_and_saved_bpe_replay_pending_batch(corpus, tokenizer):
    checkpoint = checkpoint_module()
    tokenizer.save(corpus/'tokenizer')  # trusted, test-generated encoding only
    reloaded = RustBPETokenizer.from_directory(corpus/'tokenizer')
    current = loader(reloaded)
    next(current)
    expected = next(current)
    checkpoint.save_checkpoint(corpus/'checkpoints', 3, {'w': torch.arange(4)},
                               {'state': {}}, {'step': 3, 'dataloader_state_dict': expected[2]})
    before = snapshot(corpus)
    model, opt, meta = checkpoint.load_checkpoint(corpus/'checkpoints', 3, 'cpu', True)
    assert torch.equal(model['w'], torch.arange(4)) and opt == {'state': {}}
    same_batch(next(loader(reloaded, saved=meta['dataloader_state_dict'])), expected)
    assert snapshot(corpus) == before


def test_real_checkpoint_rejects_same_size_changed_bpe_artifact(corpus, tokenizer, monkeypatch):
    checkpoint = checkpoint_module()
    tokenizer.save(corpus/'tokenizer')
    checkpoint.save_checkpoint(corpus/'checkpoints', 1, {'w': torch.ones(1)}, None, {'step': 1})
    other = changed_encoding(tokenizer, 'tokens')
    assert other.get_vocab_size() == tokenizer.get_vocab_size()
    other.save(corpus/'tokenizer')
    before = snapshot(corpus)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('mismatch reached tensor deserialization'))
    with pytest.raises(ValueError, match='tokenizer artifact mismatch'):
        checkpoint.load_checkpoint(corpus/'checkpoints', 1, 'cpu')
    assert snapshot(corpus) == before
