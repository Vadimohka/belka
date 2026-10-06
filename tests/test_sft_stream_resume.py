"""Resume cannot silently switch the SFT data, topology, tokenizer or pending batch."""
import copy
import os
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]


def setup_stream():
    torch = pytest.importorskip('torch')
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat')).resolve()
    if not (runtime / 'nanochat/belka_sft_stream.py').is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1': pytest.fail('new runtime required')
        pytest.skip('new runtime required')
    sys.path.insert(0, str(runtime))
    from nanochat.belka_sft_stream import SFTBatchStream
    from nanochat.tokenizer import RustBPETokenizer
    tok = RustBPETokenizer.train_from_iterator(iter(['Добры дзень. Як справы? Добра. Дзе кніга? На стале.'] * 8), 300)
    dataset = [{'messages': [{'role': 'user', 'content': 'Як справы?'},
                             {'role': 'assistant', 'content': a}]} for a in ['Добра.', 'На стале.']]
    return torch, SFTBatchStream, tok, dataset


def test_pending_batch_replay_and_supervised_token_accounting():
    torch, Stream, tok, dataset = setup_stream()
    a = Stream(dataset, tok, 2, 64, 'cpu', buffer_size=3)
    expected_supervised = 0
    for _ in range(3):
        _, y = next(a); expected_supervised += int((y >= 0).sum())
    state = a.state_dict(); b = Stream(dataset, tok, 2, 64, 'cpu', buffer_size=3, state=state)
    assert state['supervised_tokens'] == expected_supervised
    for _ in range(4):
        ax, ay = next(a); bx, by = next(b)
        assert torch.equal(ax, bx) and torch.equal(ay, by)
    assert a.state_dict() == b.state_dict()


@pytest.mark.parametrize('change', ['dataset', 'batch', 'world', 'buffer', 'negative', 'boolean'])
def test_resume_changes_rejected_without_mutating_loader(change):
    _, Stream, tok, dataset = setup_stream()
    a = Stream(dataset, tok, 1, 64, 'cpu'); next(a); state = a.state_dict()
    options = dict(batch_size=1, sequence_length=64, device='cpu')
    if change == 'dataset':
        dataset = copy.deepcopy(dataset); dataset[0]['messages'][-1]['content'] = 'Іншы адказ.'
    if change == 'batch': options['batch_size'] = 2
    if change == 'world': options['world_size'] = 2
    b = Stream(dataset, tok, **options); before = b.state_dict()
    if change == 'buffer': state['buffer'] = [99999]
    if change == 'negative': state['supervised_tokens'] = -1
    if change == 'boolean': state['cursor'] = True
    with pytest.raises(ValueError): b.load_state_dict(state)
    assert b.state_dict() == before
