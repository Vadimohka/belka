"""Executable regression contracts; synthetic tensors/data, no training jobs."""
import io
import json
import random
import tarfile
from pathlib import Path
import pytest
from data_pipeline.simhash_index import SimHashIndex
from data_pipeline.sft_language import language_evidence
from tools.prepare_sft_v9 import build_v9, prompt_core
from tools.restore_corpus_bundle import extract_regular_tar
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('radius', range(7))
def test_simhash_candidates_equal_exhaustive(radius):
    rng = random.Random(19)
    values = [rng.getrandbits(64) for _ in range(120)]
    index = SimHashIndex(radius)
    for value in values:
        index.add(value)
    for value in values[:30]:
        bits = rng.sample(range(64), radius)
        query = value
        for bit in bits:
            query ^= 1 << bit
        assert index.contains_near(query) == any((query ^ v).bit_count() <= radius for v in values)
    # Numeric-bucket neighbours miss this single high-bit flip.
    index = SimHashIndex(max(1, radius))
    index.add(0)
    assert index.contains_near(1 << 63)


@pytest.mark.parametrize('text', ['Who are you?', 'Представься.', 'Расскажи о себе.',
    'Ёж ещё несёт ёлку.', 'Цей користувач відповідає українською мовою.',
    'Что является столицей Беларуси, калі ласка?', 'Мама дома.'])
def test_foreign_or_ambiguous_sft_is_not_accepted(text):
    assert language_evidence(text)['decision'] != 'accept'


@pytest.mark.parametrize('text', ['Хто ты?', 'Што такое Палессе?', 'Як MeetMesh працуе з Google Calendar?',
    'Гэта беларуская мова, якая мае багатую гісторыю.', 'Я не бачу надзейнага пацверджання такой даты.'])
def test_belarusian_sft_and_identifiers(text):
    assert language_evidence(text)['decision'] == 'accept'


def test_v9_preserves_sources_and_has_disjoint_prompt_families():
    sources = [ROOT / f'seed_sft/sft_v8_{s}.be.jsonl' for s in ('train', 'val')]
    before = [p.read_bytes() for p in sources]
    train, val, report = build_v9(ROOT)
    assert train and val and report['source_rows'] == 570
    assert report['translated_rows'] == 108
    assert report['duplicate_rows'] == 2
    families = []
    for rows in (train, val):
        current = set()
        for row in rows:
            messages = json.loads(row)
            for message in messages:
                assert language_evidence(message['content'])['decision'] == 'accept'
            current.add(tuple(prompt_core(m['content']).casefold() for m in messages if m['role'] == 'user'))
        families.append(current)
    assert not families[0] & families[1]
    assert not set(train) & set(val)
    assert before == [p.read_bytes() for p in sources]
    assert build_v9(ROOT) == (train, val, report)


def make_tar(names, kind=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name in names:
            item = tarfile.TarInfo(name)
            if kind:
                item.type = kind
                item.linkname = '../../escape'
                archive.addfile(item)
            else:
                item.size = 3
                archive.addfile(item, io.BytesIO(b'abc'))
    stream.seek(0)
    return stream


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'tokenizer/../../escape', 'other/file', 'tokenizer/..\\escape'])
def test_tar_path_traversal_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        extract_regular_tar(make_tar([name]), tmp_path)


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE])
def test_tar_special_members_rejected(tmp_path, kind):
    with pytest.raises(ValueError):
        extract_regular_tar(make_tar(['tokenizer/file'], kind), tmp_path)


def test_tar_duplicates_and_budget(tmp_path):
    with pytest.raises(ValueError):
        extract_regular_tar(make_tar(['README.txt'] * 2), tmp_path)
    (tmp_path / 'README.txt').unlink()
    with pytest.raises(ValueError):
        extract_regular_tar(make_tar(['README.txt']), tmp_path, max_bytes=2)


@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('shape', [(4, 4), (2, 5), (1, 5)])
@pytest.mark.parametrize('window', [(-1, -1), (1, 0), (1, 2)])
def test_attention_matches_explicit_reference(causal, shape, window):
    torch = pytest.importorskip('torch')
    from ops.nanochat_fork.nanochat.belka_runtime import sdpa_attention
    tq, tk = shape
    torch.manual_seed(1)
    q, k, v = torch.randn(2, 4, tq, 8), torch.randn(2, 2, tk, 8), torch.randn(2, 2, tk, 8)
    result = sdpa_attention(q, k, v, window, True, causal)
    keys, values = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
    scores = q @ keys.transpose(-1, -2) / (8 ** 0.5)
    for i in range(tq):
        aligned = tk - tq + i
        for j in range(tk):
            allowed = (not causal or j <= aligned) and (window[0] < 0 or j >= aligned-window[0]) and (window[1] < 0 or j <= aligned+window[1])
            if not allowed:
                scores[:, :, i, j] = -float('inf')
    expected = scores.softmax(-1) @ values
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


def test_ragged_cache_and_fail_before_write():
    torch = pytest.importorskip('torch')
    from ops.nanochat_fork.nanochat.belka_runtime import cached_sdpa, sdpa_attention
    torch.manual_seed(2)
    q, k, v = [torch.randn(2, 2, 2, 8) for _ in range(3)]
    kc, vc = torch.randn(2, 8, 2, 8), torch.randn(2, 8, 2, 8)
    before = kc.clone()
    with pytest.raises(ValueError):
        cached_sdpa(q, kc, vc, k, v, torch.tensor([2, 7]))
    torch.testing.assert_close(kc, before)
    result = cached_sdpa(q, kc, vc, k, v, torch.tensor([2, 4]), True)
    for row, end in enumerate([4, 6]):
        expected = sdpa_attention(q[row:row+1].transpose(1, 2), kc[row:row+1, :end].transpose(1, 2), vc[row:row+1, :end].transpose(1, 2))
        torch.testing.assert_close(result[row:row+1], expected.transpose(1, 2))


def test_accumulation_matches_global_token_mean():
    torch = pytest.importorskip('torch')
    from ops.nanochat_fork.nanochat.belka_runtime import supervised_scale
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 4, bias=False)
    x = torch.randn(5, 3)
    y = torch.tensor([0, -1, 2, 3, -1])
    loss = torch.nn.functional.cross_entropy(model(x), y, ignore_index=-1)
    loss.backward()
    expected = model.weight.grad.clone()
    model.zero_grad()
    batches = [(x[:2], y[:2]), (x[2:], y[2:])]
    scale = supervised_scale(batches)
    for a, b in batches:
        (torch.nn.functional.cross_entropy(model(a), b, ignore_index=-1, reduction='sum') * scale).backward()
    torch.testing.assert_close(model.weight.grad, expected)
    with pytest.raises(ValueError):
        supervised_scale([(x, torch.full_like(y, -1))])


@pytest.mark.parametrize('expression,expected', [('1+2*3', 7), ('7//2', 3), ('-8/4', -2), ('"беларуская".count("а")', 2),
    ('1/0', None), ('2**1000000', None), ('__import__("os")', None), ('[0]*100', None),
    ('(1).__class__', None), ('float("nan")', None), ('True+1', None), ('9'*513, None)])
def test_bounded_calculator(expression, expected):
    pytest.importorskip('torch')
    from ops.nanochat_fork.nanochat.belka_runtime import use_calculator
    assert use_calculator(expression) == expected
