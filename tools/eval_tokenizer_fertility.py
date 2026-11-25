#!/usr/bin/env python3
"""Measure byte-level tokenizer fertility and lossless Unicode roundtrips.

Pickles must be trusted local artifacts. Default phrases are smoke fixtures, not
representative validation. --eval-file accepts UTF-8 text (one sample per line) or
JSONL records with text and an optional group. It is evaluation input only.
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.contracts import strict_json_loads
from tools.training_artifacts import PACK, atomic_json, digest, file_hash, inside

BE_LETTERS = tuple('ўіёЎІЁ') + ("'", '’', 'ʼ')
BE_TEXT_SAMPLE = [
    'Беларуская мова — гэта нацыянальны скарб нашага народа.',
    'У кожным слове жыве гісторыя і культура беларусаў.',
    'Незалежнасць і свабода — найвышэйшыя каштоўнасці.',
    "Навука і адукацыя з'яўляюцца падмуркам развіцця краіны.",
    'Кожны чалавек мае права на годнае жыццё і працу.',
    'Беларуская літаратура багатая на таленавітых пісьменнікаў.',
    'Тэхналогіі змяняюць свет, але мова застаецца нязменнай.',
    'Сустрэча з сябрамі заўсёды прыносіць радасць і натхненне.',
]
TARASK_SAMPLE = [
    'Беларуская мова — гэта нацыянальны скарб нашага народу.',
    'Незалежнасьць і свабода — найвышэйшыя каштоўнасьці.',
]
RU_SAMPLE = ['Русский язык является одним из самых распространённых в мире.']
EN_SAMPLE = ['The English language has become a global lingua franca.']


def raw_vocabulary(tokenizer):
    """Read token bytes, never replacement-character lengths of decoded fragments."""
    encoding = getattr(tokenizer, 'enc', tokenizer)
    size = (tokenizer.get_vocab_size() if hasattr(tokenizer, 'get_vocab_size')
            else getattr(encoding, 'n_vocab', None))
    if type(size) is not int or size < 1:
        raise ValueError('tokenizer does not expose a valid vocabulary size')
    if hasattr(tokenizer, 'get_special_tokens') and hasattr(tokenizer, 'encode_special'):
        specials = {tokenizer.encode_special(name) for name in tokenizer.get_special_tokens()}
    elif hasattr(encoding, 'special_tokens_set') and hasattr(encoding, 'encode_single_token'):
        specials = {encoding.encode_single_token(name) for name in encoding.special_tokens_set}
    else:
        raise ValueError('tokenizer must expose special-token membership')
    if any(type(index) is not int or not 0 <= index < size for index in specials):
        raise ValueError('invalid special-token ID')
    reader = getattr(tokenizer, 'decode_single_token_bytes', None) or getattr(encoding, 'decode_single_token_bytes', None)
    if reader is None:
        raise ValueError('tokenizer must expose raw token bytes')
    raw = {}
    for index in range(size):
        if index in specials:
            continue
        try:
            value = reader(index)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f'cannot read raw bytes for ordinary token {index}') from exc
        if not isinstance(value, bytes) or not value:
            raise ValueError(f'ordinary token {index} has invalid or empty bytes')
        raw[index] = value
    return size, specials, raw


def encode_checked(tokenizer, text, raw):
    encoding = getattr(tokenizer, 'enc', tokenizer)
    # Match nanochat's ordinary-text path, including literal special-token strings.
    encode = getattr(encoding, 'encode_ordinary', None) or getattr(tokenizer, 'encode', None)
    if encode is None:
        raise ValueError('tokenizer does not expose an ordinary-text encoder')
    try:
        ids = list(encode(text))
    except Exception as exc:
        raise ValueError('tokenizer failed to encode an evaluation sample') from exc
    if not ids or any(type(index) is not int or index not in raw for index in ids):
        raise ValueError('ordinary text produced empty, unknown or special-token IDs')
    if b''.join(raw[index] for index in ids) != text.encode('utf-8'):
        raise ValueError('tokenizer raw-byte roundtrip failed')
    try:
        decoded = tokenizer.decode(ids)
    except Exception as exc:
        raise ValueError('tokenizer failed to decode an evaluation sample') from exc
    if decoded != text:
        raise ValueError('tokenizer Unicode roundtrip failed')
    return ids


def eval_tokenizer(tok, samples, label, *, raw=None):
    if not samples or any(not isinstance(text, str) or not text.strip() for text in samples):
        raise ValueError('evaluation groups must contain nonempty text samples')
    if raw is None:
        _, _, raw = raw_vocabulary(tok)
    chars = words = utf8_bytes = tokens = single_bytes = non_ascii_bytes = 0
    for text in samples:
        ids = encode_checked(tok, text, raw)
        chars += len(text)
        words += len(re.findall(r'\b\w+\b', text))
        utf8_bytes += len(text.encode('utf-8'))
        tokens += len(ids)
        single_bytes += sum(len(raw[index]) == 1 for index in ids)
        non_ascii_bytes += sum(len(raw[index]) == 1 and raw[index][0] >= 128 for index in ids)
    return {
        f'{label}_tokens_per_char': tokens/chars,
        f'{label}_tokens_per_word': tokens/words if words else None,
        f'{label}_tokens_per_utf8_byte': tokens/utf8_bytes,
        f'{label}_single_byte_token_rate': single_bytes/tokens,
        f'{label}_non_ascii_byte_token_rate': non_ascii_bytes/tokens,
        # Retain the old key without inventing a fallback-event rate for byte BPE.
        f'{label}_byte_fallback_rate': None,
        f'{label}_samples': len(samples), f'{label}_tokens': tokens,
        f'{label}_chars': chars, f'{label}_words': words, f'{label}_utf8_bytes': utf8_bytes,
        f'{label}_utf8_roundtrip': True,
    }


def letter_coverage(tokenizer, raw):
    details = {}
    for letter in BE_LETTERS:
        ids = encode_checked(tokenizer, letter, raw)
        details[letter] = {'roundtrip': True, 'tokens': len(ids),
                           'single_byte_tokens': sum(len(raw[index]) == 1 for index in ids)}
    return details


def read_samples(path):
    groups = {}
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if Path(path).suffix.lower() == '.jsonl':
                try:
                    row = strict_json_loads(line)
                except ValueError as exc:
                    raise ValueError(f'invalid evaluation JSONL at line {line_number}') from exc
                if not isinstance(row, dict):
                    raise ValueError(f'evaluation line {line_number} must be an object with text')
                text, group = row.get('text'), row.get('group', 'custom')
            else:
                text, group = line.rstrip('\r\n'), 'custom'
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'evaluation line {line_number} has no nonempty text')
            if not isinstance(group, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', group):
                raise ValueError(f'evaluation line {line_number} has an invalid group name')
            groups.setdefault(group, []).append(text)
    if not groups:
        raise ValueError('evaluation input contains no samples')
    return groups


def evaluate(args):
    pack = Path(args.pack_dir).expanduser().resolve(strict=True)
    if pack != PACK.resolve():
        raise ValueError('--pack-dir must identify the checkout containing this script')
    names = [value.strip() for value in args.tokenizers.split(',')]
    if not names or any(not name for name in names):
        raise ValueError('provide at least one tokenizer path')
    paths = [inside(name, root=pack, exists=True) for name in names]
    if len(paths) != len(set(paths)):
        raise ValueError('tokenizer paths must be distinct')
    output_arg = Path(args.output or pack/'reports/tokenizer_ablation_report.json')
    if not output_arg.is_absolute():
        output_arg = pack/output_arg
    # Do not resolve a report symlink into a tokenizer or other selected artifact.
    if output_arg.is_symlink():
        raise ValueError('refusing to replace a symlink report')
    output = inside(output_arg, root=pack)
    eval_path = inside(args.eval_file, root=pack, exists=True) if args.eval_file else None
    if output in paths or output == eval_path:
        raise ValueError('report output aliases an evaluation input')
    if eval_path:
        sample_hash = file_hash(eval_path)
        groups = read_samples(eval_path)
        source = {'path': str(eval_path), 'sha256': sample_hash}
        mode = 'provided_dataset'
    else:
        groups = {'be': BE_TEXT_SAMPLE, 'tarask': TARASK_SAMPLE, 'ru': RU_SAMPLE, 'en': EN_SAMPLE}
        source = {'fixtures_sha256': digest(groups)}
        mode = 'smoke_only'
    report = {
        'schema': 'belka-tokenizer-fertility-v2', 'status': 'complete',
        'evaluation_scope': mode, 'sample_source': source,
        'sample_counts': {group: len(texts) for group, texts in groups.items()},
        'quality_note': 'Fertility and Unicode checks do not establish language-model quality or dataset representativeness.',
        'metric_notes': {
            'single_byte_token_rate': 'Fraction of emitted ordinary tokens whose raw vocabulary entry is exactly one byte; includes ASCII.',
            'non_ascii_byte_token_rate': 'Fraction of emitted tokens which are single bytes >= 0x80.',
            'byte_fallback_rate': 'Deprecated key, null: byte-level BPE does not expose a separate fallback event.',
            'tokens_per_word': 'Unicode regex word spans; null for samples with no word spans.',
        },
        'results': {},
    }
    if mode == 'smoke_only':
        report['samples'] = groups  # Preserve the existing smoke-report shape.
    for name, path in zip(names, paths):
        before = file_hash(path)
        with path.open('rb') as stream:
            tok = pickle.load(stream)
        size, specials, raw = raw_vocabulary(tok)
        result = {'vocab_size': size, 'tokenizer_sha256': before}
        for group, samples in groups.items():
            result.update(eval_tokenizer(tok, samples, group, raw=raw))
        result['be_letter_details'] = letter_coverage(tok, raw)
        result['be_letter_coverage'] = True
        alphabet = {value[0] for value in raw.values() if len(value) == 1}
        result['byte_alphabet_coverage'] = len(alphabet)/256
        result['missing_byte_values'] = sorted(set(range(256))-alphabet)
        result['special_token_count'] = len(specials)
        if file_hash(path) != before:
            raise ValueError('tokenizer changed during evaluation')
        report['results'][name] = result
    if eval_path and file_hash(eval_path) != sample_hash:
        raise ValueError('evaluation input changed during evaluation')
    atomic_json(output, report, replace=True)
    print(f'Fertility report ({mode}): {output}')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-dir', type=Path, default=PACK)
    parser.add_argument('--tokenizers', default='reports/tokenizers/tok_16384.pkl',
                        help='comma-separated trusted local pickle paths, relative to --pack-dir')
    parser.add_argument('--eval-file', type=Path, help='UTF-8 text or JSONL with text and optional group; evaluation only')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        evaluate(args)
    except (OSError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError, ModuleNotFoundError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
