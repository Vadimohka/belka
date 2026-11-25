"""Isolation, failure provenance and actual raw-byte fertility regressions."""
import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tools import eval_tokenizer_fertility as fertility
from tools import train_tokenizer_ablation as ablation
from tools.training_artifacts import atomic_json, file_hash, inventory
from test_training_plan import prepared_fixture


@pytest.fixture
def ablation_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(ablation, 'PACK', tmp_path)
    (tmp_path/'tools').mkdir()
    (tmp_path/'tools/prepare_h200_tokenizer.py').write_text('# subprocess test boundary\n')
    runtime = tmp_path/'runtime'
    runtime.mkdir()
    (runtime/'BELKA_RUNTIME_MANIFEST.json').write_text('{"fixture":true}')
    base = tmp_path/'active'
    base.mkdir()
    corpus, sft = base/'corpus-generation', base/'sft-generation'
    corpus.mkdir()
    sft.mkdir()
    (corpus/'H200_CORPUS_MANIFEST.json').write_text('{"fixture":"corpus"}')
    (sft/'SFT_BUILD_MANIFEST.json').write_text('{"fixture":"sft"}')
    for name in ('identity_conversations.jsonl', 'identity_conversations_val.jsonl'):
        (sft/name).write_text('[]\n')
    (base/'.corpus_current').symlink_to(corpus, target_is_directory=True)
    (base/'.sft_current').symlink_to(sft, target_is_directory=True)
    (base/'tokenizer').mkdir()
    (base/'tokenizer/tokenizer.pkl').write_bytes(b'active-tokenizer-do-not-reuse')
    return argparse.Namespace(pack_dir=tmp_path, base_dir=base, corpus_dir=None,
        nanochat_dir=runtime, python=Path(sys.executable), vocabs='300,320',
        max_chars=120, doc_cap=40, output_dir=tmp_path/'comparison')


def prepared_process(command, **kwargs):
    """Model the child process's artifact contract, not the BPE algorithm."""
    flags = dict(zip(command[2::2], command[3::2]))
    base = Path(flags['--base-dir'])
    corpus = Path(flags['--corpus-dir'])
    runtime = Path(flags['--nanochat-dir'])
    size = int(flags['--vocab-size'])
    tokenizer = base/'tokenizer'
    tokenizer.mkdir()
    (tokenizer/'tokenizer.pkl').write_bytes(f'fresh-{size}'.encode())
    (tokenizer/'token_bytes.pt').write_bytes(f'bytes-{size}'.encode())
    manifest = {
        'schema': 'belka-h200-tokenizer-v1', 'vocab_size': size,
        'corpus_dir': str(corpus), 'sft_dir': str((base/'.sft_current').resolve()),
        'runtime_manifest_sha256': file_hash(runtime/'BELKA_RUNTIME_MANIFEST.json'),
        'max_chars': int(flags['--max-chars']), 'doc_cap': int(flags['--doc-cap']),
        'training': {'source_split': 'train', 'validation_used': False, 'holdout_used': False,
                     'documents': 3, 'chars': 120, 'text_sha256': 'a'*64,
                     'sources': {'fixture': {'documents': 3, 'chars': 120}}},
        'files': inventory([tokenizer/'tokenizer.pkl', tokenizer/'token_bytes.pt'], tokenizer),
        'corpus_sha256': 'b'*64, 'corpus_files': [{'name': 'train.parquet', 'sha256': 'c'*64}],
        'sft_files': [{'name': 'identity_conversations.jsonl', 'sha256': 'd'*64}],
        'tokenizer_runtime_sha256': {'nanochat/tokenizer.py': 'e'*64},
        'counts': {'train': {'tokens': size}}, 'sft_counts': {},
    }
    atomic_json(tokenizer/'TOKENIZER_TRAINING_MANIFEST.json', manifest)
    return subprocess.CompletedProcess(command, 0, 'prepared\n', '')


def test_isolated_train_only_candidates_preserve_active_inputs(ablation_inputs, monkeypatch):
    args = ablation_inputs
    old_hash = file_hash(args.base_dir/'tokenizer/tokenizer.pkl')
    original_env = dict(ablation.os.environ)
    calls = []
    def process(command, **kwargs):
        calls.append((command, kwargs))
        return prepared_process(command, **kwargs)
    monkeypatch.setattr(ablation.subprocess, 'run', process)
    report = ablation.run_ablation(args)
    assert report['status'] == 'complete'
    assert file_hash(args.base_dir/'tokenizer/tokenizer.pkl') == old_hash
    assert dict(ablation.os.environ) == original_env
    assert len(calls) == 2
    for size, (command, kwargs) in zip((300, 320), calls):
        candidate = args.output_dir/f'vocab_{size}'
        flags = dict(zip(command[2::2], command[3::2]))
        assert Path(command[1]).name == 'prepare_h200_tokenizer.py'
        assert flags['--doc-cap'] == '40' and flags['--max-chars'] == '120'
        assert flags['--base-dir'] == kwargs['env']['NANOCHAT_BASE_DIR'] == str(candidate)
        assert (candidate/'.corpus_current').resolve() == (args.base_dir/'.corpus_current').resolve()
        local_sft = (candidate/'.sft_current').resolve()
        assert local_sft.parent == candidate/'..sft_current.generations'
        assert file_hash(local_sft/'SFT_BUILD_MANIFEST.json') == report['sources']['sft_manifest_sha256']
        assert (args.output_dir/f'tok_{size}.pkl').read_bytes() == f'fresh-{size}'.encode()
        assert report['results'][str(size)]['status'] == 'complete'
        assert 'train_stdout_tail' not in report['results'][str(size)]


@pytest.mark.parametrize('failure', ['nonzero', 'exception'])
def test_failed_subprocess_never_exposes_stale_artifacts(ablation_inputs, monkeypatch, failure):
    args = ablation_inputs
    def process(command, **kwargs):
        if failure == 'exception':
            raise OSError('interpreter could not start')
        return subprocess.CompletedProcess(command, 7, 'private corpus output', 'failure')
    monkeypatch.setattr(ablation.subprocess, 'run', process)
    with pytest.raises(ValueError, match='ablation failed'):
        ablation.run_ablation(args)
    saved = json.loads((args.output_dir/'tokenizer_ablation_raw.json').read_text())
    assert saved['status'] == saved['results']['300']['status'] == 'failed'
    assert not (args.output_dir/'tok_300.pkl').exists()
    assert not (args.output_dir/'vocab_320').exists()
    assert 'tokenizer_pkl' not in saved['results']['300']
    assert 'private corpus output' not in json.dumps(saved)
    if failure == 'nonzero':
        assert Path(saved['results']['300']['stdout_log']).read_text() == 'private corpus output'
        assert Path(saved['results']['300']['stderr_log']).read_text() == 'failure'
    assert (args.base_dir/'tokenizer/tokenizer.pkl').read_bytes() == b'active-tokenizer-do-not-reuse'


@pytest.mark.parametrize('mutation,expected', [
    ('wrong_vocab', 'vocabulary size'), ('wrong_structure', 'schema'), ('uses_validation', 'training data only'),
    ('over_cap', 'caps'), ('bad_hash', 'artifact changed'), ('different_text', 'identical train text'),
])
def test_success_exit_requires_matching_verified_artifacts(ablation_inputs, monkeypatch, mutation, expected):
    args = ablation_inputs
    def process(command, **kwargs):
        result = prepared_process(command, **kwargs)
        flags = dict(zip(command[2::2], command[3::2]))
        folder = Path(flags['--base-dir'])/'tokenizer'
        path = folder/'TOKENIZER_TRAINING_MANIFEST.json'
        manifest = json.loads(path.read_text())
        if mutation == 'wrong_structure':
            manifest = []
        elif mutation == 'wrong_vocab':
            manifest['vocab_size'] += 1
        elif mutation == 'uses_validation':
            manifest['training']['validation_used'] = True
        elif mutation == 'over_cap':
            manifest['training']['chars'] = 121
        elif mutation == 'bad_hash':
            (folder/'tokenizer.pkl').write_bytes(b'not-what-the-manifest-hashed')
        elif flags['--vocab-size'] == '320':
            manifest['training']['text_sha256'] = 'f'*64
        path.write_text(json.dumps(manifest))
        return result
    monkeypatch.setattr(ablation.subprocess, 'run', process)
    with pytest.raises(ValueError, match=expected):
        ablation.run_ablation(args)
    report = json.loads((args.output_dir/'tokenizer_ablation_raw.json').read_text())
    failed = '320' if mutation == 'different_text' else '300'
    assert report['status'] == report['results'][failed]['status'] == 'failed'
    assert not (args.output_dir/f'tok_{failed}.pkl').exists()


def test_input_manifest_drift_is_detected_before_publication(ablation_inputs, monkeypatch):
    args = ablation_inputs
    def process(command, **kwargs):
        result = prepared_process(command, **kwargs)
        (args.base_dir/'.corpus_current/H200_CORPUS_MANIFEST.json').write_text('changed')
        return result
    monkeypatch.setattr(ablation.subprocess, 'run', process)
    with pytest.raises(ValueError, match='input manifests changed'):
        ablation.run_ablation(args)
    assert not (args.output_dir/'tok_300.pkl').exists()


def test_preparation_code_drift_invalidates_the_comparison(ablation_inputs, monkeypatch):
    args = ablation_inputs
    def process(command, **kwargs):
        result = prepared_process(command, **kwargs)
        Path(command[1]).write_text('# a different preparation implementation\n')
        return result
    monkeypatch.setattr(ablation.subprocess, 'run', process)
    with pytest.raises(ValueError, match='preparation script changed'):
        ablation.run_ablation(args)
    assert not (args.output_dir/'tok_300.pkl').exists()


def test_fresh_output_and_path_confinement_precede_training(ablation_inputs, monkeypatch):
    args = ablation_inputs
    def unexpected(*args, **kwargs):
        pytest.fail('must reject unsafe/reused output before invoking preparation')
    monkeypatch.setattr(ablation.subprocess, 'run', unexpected)
    args.output_dir.mkdir()
    sentinel = args.output_dir/'existing.pkl'
    sentinel.write_bytes(b'keep')
    with pytest.raises(ValueError, match='already exists'):
        ablation.run_ablation(args)
    assert sentinel.read_bytes() == b'keep'
    for path in (args.base_dir/'new-candidate', args.nanochat_dir/'new-candidate',
                 args.pack_dir.parent/'outside-ablation'):
        args.output_dir = path
        with pytest.raises(ValueError, match='separate|below'):
            ablation.run_ablation(args)


@pytest.mark.parametrize('sizes', ['', '300,300', '299', '131073', '300,no'])
def test_invalid_vocabulary_grid_fails(sizes):
    with pytest.raises(ValueError):
        ablation.vocabulary_sizes(sizes)


def test_cli_returns_nonzero_for_failed_training(ablation_inputs, monkeypatch):
    args = ablation_inputs
    monkeypatch.setattr(ablation.subprocess, 'run',
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 3, '', ''))
    with pytest.raises(SystemExit) as error:
        ablation.main(['--pack-dir', str(args.pack_dir), '--base-dir', str(args.base_dir),
                      '--nanochat-dir', str(args.nanochat_dir), '--python', sys.executable,
                      '--output-dir', str(args.output_dir), '--vocabs', '300'])
    assert error.value.code == 2


class ByteTokenizer:
    """Actual reversible UTF-8 byte codec, including incomplete single-token UTF-8."""
    n_vocab = 257
    special_tokens_set = {'<|bos|>'}
    def encode_single_token(self, name):
        assert name == '<|bos|>'
        return 256
    def decode_single_token_bytes(self, index):
        return bytes([index])
    def encode_ordinary(self, text):
        return list(text.encode('utf-8'))
    def decode(self, ids):
        return bytes(ids).decode('utf-8')


class BrokenTokenizer(ByteTokenizer):
    def encode_ordinary(self, text):
        raise RuntimeError('encoding failed')


def test_fertility_counts_raw_bytes_and_never_invents_fallback_events():
    result = fertility.eval_tokenizer(ByteTokenizer(), ['іA'], 'be')
    assert result['be_tokens'] == 3 and result['be_chars'] == 2
    assert result['be_utf8_bytes'] == 3
    assert result['be_tokens_per_char'] == 1.5
    assert result['be_single_byte_token_rate'] == 1
    assert result['be_non_ascii_byte_token_rate'] == pytest.approx(2/3)
    assert result['be_byte_fallback_rate'] is None
    assert result['be_utf8_roundtrip'] is True
    _, _, raw = fertility.raw_vocabulary(ByteTokenizer())
    assert fertility.letter_coverage(ByteTokenizer(), raw)['і']['tokens'] == 2


def test_encode_errors_fail_instead_of_fabricating_tokens():
    with pytest.raises(ValueError, match='failed to encode'):
        fertility.eval_tokenizer(BrokenTokenizer(), ['Беларусь'], 'be')
    with pytest.raises(ValueError, match='nonempty'):
        fertility.eval_tokenizer(ByteTokenizer(), [], 'be')


def test_incorrect_raw_vocabulary_cannot_pass_unicode_check():
    _, _, raw = fertility.raw_vocabulary(ByteTokenizer())
    raw[ord('A')] = b'B'
    with pytest.raises(ValueError, match='raw-byte roundtrip'):
        fertility.eval_tokenizer(ByteTokenizer(), ['A'], 'be', raw=raw)


@pytest.fixture
def fertility_args(tmp_path, monkeypatch):
    monkeypatch.setattr(fertility, 'PACK', tmp_path)
    (tmp_path/'tokenizer.pkl').write_bytes(pickle.dumps(ByteTokenizer()))
    return argparse.Namespace(pack_dir=tmp_path, tokenizers='tokenizer.pkl',
                              eval_file=None, output=Path('fertility.json'))


def test_default_fertility_is_explicitly_smoke_only(fertility_args):
    report = fertility.evaluate(fertility_args)
    result = report['results']['tokenizer.pkl']
    assert report['evaluation_scope'] == 'smoke_only'
    assert report['sample_counts']['be'] == 8
    assert result['byte_alphabet_coverage'] == 1
    assert result['be_letter_coverage'] is True
    assert result['special_token_count'] == 1
    assert set(result['be_letter_details']) >= {'ў', 'і', 'ё', '’', 'ʼ'}


def test_external_grouped_eval_is_evaluation_only_and_provenance_bound(fertility_args):
    args = fertility_args
    path = args.pack_dir/'dev.jsonl'
    rows = [{'group': 'prose', 'text': 'Беларуская мова.'},
            {'group': 'tarask', 'text': 'Незалежнасьць.'}]
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding='utf-8')
    before = file_hash(args.pack_dir/'tokenizer.pkl')
    args.eval_file = path
    report = fertility.evaluate(args)
    assert report['evaluation_scope'] == 'provided_dataset'
    assert report['sample_source']['sha256'] == file_hash(path)
    assert report['sample_counts'] == {'prose': 1, 'tarask': 1}
    assert 'samples' not in report  # Do not duplicate potentially large input texts.
    assert file_hash(args.pack_dir/'tokenizer.pkl') == before
    assert report['results']['tokenizer.pkl']['prose_samples'] == 1


@pytest.mark.parametrize('text', ['', '{}', '[]', '{"text":"ok","group":"bad/name"}', 'invalid'])
def test_invalid_eval_inputs_are_not_silently_skipped(tmp_path, text):
    path = tmp_path/'eval.jsonl'
    path.write_text(text)
    with pytest.raises(ValueError):
        fertility.read_samples(path)


def test_plain_text_eval_and_no_word_denominator(tmp_path):
    path = tmp_path/'eval.txt'
    path.write_text('Прывітанне\n\n!!!\n', encoding='utf-8')
    assert fertility.read_samples(path) == {'custom': ['Прывітанне', '!!!']}
    result = fertility.eval_tokenizer(ByteTokenizer(), ['!!!'], 'symbols')
    assert result['symbols_tokens_per_word'] is None


def test_missing_or_broken_tokenizer_never_writes_success_report(fertility_args):
    args = fertility_args
    args.tokenizers = 'missing.pkl'
    with pytest.raises(FileNotFoundError):
        fertility.evaluate(args)
    assert not (args.pack_dir/args.output).exists()
    args.tokenizers = 'tokenizer.pkl'
    (args.pack_dir/args.tokenizers).write_bytes(pickle.dumps(BrokenTokenizer()))
    with pytest.raises(ValueError, match='failed to encode'):
        fertility.evaluate(args)
    assert not (args.pack_dir/args.output).exists()


def test_fertility_output_cannot_overwrite_its_input(fertility_args):
    args = fertility_args
    args.output = Path(args.tokenizers)
    before = (args.pack_dir/args.tokenizers).read_bytes()
    with pytest.raises(ValueError, match='aliases'):
        fertility.evaluate(args)
    assert (args.pack_dir/args.tokenizers).read_bytes() == before
    args.output = args.pack_dir/'alias.json'
    args.output.symlink_to(args.pack_dir/args.tokenizers)
    with pytest.raises(ValueError, match='symlink'):
        fertility.evaluate(args)


def test_real_tiktoken_byte_encoding_roundtrips_belarusian():
    tiktoken = pytest.importorskip('tiktoken')
    encoding = tiktoken.Encoding(name='belka-fertility-test', pat_str=r'(?s).',
        mergeable_ranks={bytes([index]): index for index in range(256)},
        special_tokens={'<|bos|>': 256})
    tokenizer = pickle.loads(pickle.dumps(encoding))
    result = fertility.eval_tokenizer(tokenizer, ['Ўў Іі Ёё 🦔 <|bos|>'], 'be')
    assert result['be_utf8_roundtrip'] is True
    assert result['be_single_byte_token_rate'] == 1


def test_real_canonical_preparation_runs_in_separate_candidate_bases(prepared_fixture):
    # Reuse the existing admitted tiny corpus fixture, not real training data.
    base, plan = prepared_fixture
    old_tokenizer = file_hash(base/'tokenizer/tokenizer.pkl')
    original_corpus = (base/'.corpus_current').resolve()
    original_sft = (base/'.sft_current').resolve()
    with tempfile.TemporaryDirectory(prefix='test-bpe-ablation-', dir=ablation.PACK/'.workspace') as folder:
        args = argparse.Namespace(pack_dir=ablation.PACK, base_dir=base, corpus_dir=None,
            nanochat_dir=Path(plan['nanochat_dir']), python=Path(sys.executable), vocabs='300,304',
            max_chars=1000, doc_cap=200, output_dir=Path(folder)/'comparison')
        report = ablation.run_ablation(args)
        assert report['status'] == 'complete'
        assert len({item['training']['text_sha256'] for item in report['results'].values()}) == 1
        for item in report['results'].values():
            assert item['training']['chars'] <= 1000
            manifest = json.loads(Path(item['manifest']).read_text())
            assert manifest['doc_cap'] == 200
            assert manifest['training']['validation_used'] is False
        assert file_hash(base/'tokenizer/tokenizer.pkl') == old_tokenizer
        assert (base/'.corpus_current').resolve() == original_corpus
        assert (base/'.sft_current').resolve() == original_sft
