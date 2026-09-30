"""Real pinned nanochat + canonical tokenizer contracts; never launch training."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('BELKA_TEST_NANOCHAT', ROOT / '.workspace/nanochat'))
pytestmark = pytest.mark.skipif(not os.environ.get('BELKA_RUNTIME_TESTS'), reason='requires pinned runtime + restored bundle; Runtime contracts CI exercises this file')


@pytest.fixture(scope='module')
def runtime():
    sys.path.insert(0, str(RUNTIME))
    import torch
    from nanochat.tokenizer import get_tokenizer
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.engine import KVCache
    return torch, get_tokenizer(), GPT, GPTConfig, KVCache


def test_pinned_sources_and_generated_files(runtime):
    manifest = json.loads((RUNTIME / 'BELKA_RUNTIME_MANIFEST.json').read_text())
    assert manifest['upstream_commit'] == '92d63d4e8bb4df75c3b71618f31ddde2378b2bcd'
    for group in ('source_sha256', 'files'):
        for path, sha in manifest[group].items():
            if group == 'source_sha256' and path in manifest['files']:
                continue
            assert hashlib.sha256((RUNTIME / path).read_bytes()).hexdigest() == sha


@pytest.mark.parametrize('text', ['Беларуская мова: і, ў, ё; ь, ’, ʼ.', 'Тарашкевіца і наркамаўка.', 'е\u0308сць\nпрабелы\tі  лічбы 12345', 'Вітаю! 👋'])
def test_actual_tokenizer_roundtrip(runtime, text):
    tokenizer = runtime[1]
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_actual_sft_renderer_mask_and_shift(runtime):
    tokenizer = runtime[1]
    conv = {'messages': [{'role': 'user', 'content': 'Хто ты?'}, {'role': 'assistant', 'content': 'Я беларускамоўная мадэль.'}]}
    before = copy.deepcopy(conv)
    ids, mask = tokenizer.render_conversation(conv, max_tokens=256)
    assert conv == before
    assert len(ids) == len(mask)
    assert mask[0] == 0
    target = [token for token, active in zip(ids[1:], mask[1:]) if active]
    expected = tokenizer.encode('Я беларускамоўная мадэль.') + [tokenizer.encode_special('<|assistant_end|>')]
    assert target == expected


def test_actual_model_full_and_cached_decode_agree(runtime):
    torch, _, GPT, Config, Cache = runtime
    torch.manual_seed(17)
    model = GPT(Config(sequence_len=32, vocab_size=128, n_layer=2, n_head=1, n_kv_head=1, n_embd=64, window_pattern='L'))
    model.init_weights()
    with torch.no_grad():
        model.lm_head.weight.normal_(mean=0.0, std=0.05)
    model.eval()
    tokens = torch.randint(0, 128, (2, 9))
    cache = Cache(2, 1, 32, 64, 2, torch.device('cpu'), torch.float32)
    with torch.no_grad():
        full = model(tokens)
        prefill = model(tokens[:, :5], kv_cache=cache)
        chunks = [prefill]
        for i in range(5, 9):
            chunks.append(model(tokens[:, i:i+1], kv_cache=cache))
        cached = torch.cat(chunks, dim=1)
    torch.testing.assert_close(full, cached, rtol=2e-4, atol=2e-5)


def test_exact_stream_targets_and_resume(runtime, tmp_path, monkeypatch):
    torch, tokenizer, *_ = runtime
    import pyarrow as pa
    import pyarrow.parquet as pq
    from nanochat.belka_stream import stream_loader
    corpus = tmp_path / 'base_data_climbmix'
    corpus.mkdir()
    texts = ['Беларуская мова мае багатую гісторыю. ' * 8, 'Гэта асобны беларускі дакумент. ' * 8]
    pq.write_table(pa.table({'text': texts}), corpus / 'train_00000.parquet', row_group_size=1)
    pq.write_table(pa.table({'text': ['Гэта асобная праверка.']}), corpus / 'val_00000.parquet')
    monkeypatch.setenv('NANOCHAT_BASE_DIR', str(tmp_path))
    loader = stream_loader(tokenizer, 2, 16, 'train', device='cpu')
    batches = [next(loader) for _ in range(8)]
    all_ids = []
    for text in texts:
        all_ids += tokenizer.encode(text, prepend=tokenizer.get_bos_token_id())
    expected = (all_ids * 20)[1:257]
    assert torch.cat([y.reshape(-1) for _, y, _ in batches]).tolist() == expected
    state = json.loads(json.dumps(batches[2][2]))
    resumed = stream_loader(tokenizer, 2, 16, 'train', device='cpu', resume_state_dict=state)
    for x, y, old_state in batches[3:]:
        rx, ry, new_state = next(resumed)
        assert torch.equal(rx, x) and torch.equal(ry, y) and new_state == old_state
    pq.write_table(pa.table({'text': ['Зменены корпус.']}), corpus / 'train_00000.parquet')
    with pytest.raises(ValueError, match='mismatch'):
        next(stream_loader(tokenizer, 2, 16, 'train', device='cpu', resume_state_dict=state))


def test_all_validation_shards_stay_out_of_training(runtime, tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from nanochat.dataset import parquets_iter_batched
    directory = tmp_path / 'base_data_climbmix'
    directory.mkdir()
    for filename, text in [('train_00000.parquet', 'TRAIN'), ('val_00000.parquet', 'VAL1'), ('val_00001.parquet', 'VAL2')]:
        pq.write_table(pa.table({'text': [text]}), directory / filename)
    monkeypatch.setenv('NANOCHAT_BASE_DIR', str(tmp_path))
    assert list(parquets_iter_batched('train')) == [['TRAIN']]
    assert list(parquets_iter_batched('val')) == [['VAL1'], ['VAL2']]
