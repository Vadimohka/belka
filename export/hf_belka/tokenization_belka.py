"""Lossless tiktoken vocabulary and native chat framing for HF consumers."""
import base64
import json
from pathlib import Path
import shutil
import tiktoken
from transformers import PreTrainedTokenizer, BatchEncoding


class BelkaTokenizer(PreTrainedTokenizer):
    vocab_files_names = {'vocab_file': 'tokenizer.tiktoken.json'}
    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, vocab_file, **kwargs):
        self.vocab_file = str(vocab_file)
        data = json.loads(Path(vocab_file).read_text(encoding='utf-8'))
        self._ranks = {base64.b64decode(key, validate=True): value for key, value in data['mergeable_ranks'].items()}
        self._special = data['special_tokens']
        self.enc = tiktoken.Encoding(name='belka_export', pat_str=data['pat_str'],
                                    mergeable_ranks=self._ranks, special_tokens=self._special)
        self._vocab = {'b64:'+base64.b64encode(key).decode('ascii'): value for key, value in self._ranks.items()}
        self._vocab.update(self._special)
        self._tokens = {value:key for key,value in self._vocab.items()}
        kwargs.setdefault('bos_token', '<|bos|>')
        kwargs.setdefault('eos_token', '<|assistant_end|>')
        kwargs.setdefault('pad_token', '<|assistant_end|>')
        kwargs.setdefault('additional_special_tokens', list(self._special))
        kwargs.setdefault('clean_up_tokenization_spaces', False)
        # Native encode() always treats user-supplied marker-looking text as
        # ordinary bytes. Explicit chat framing inserts IDs, never text parsing.
        kwargs.setdefault('split_special_tokens', True)
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    def get_vocab(self):
        return dict(self._vocab, **self.added_tokens_encoder)

    def _tokenize(self, text):
        return [self._tokens[index] for index in self.enc.encode_ordinary(text)]

    def _convert_token_to_id(self, token):
        return self._vocab[token]

    def _convert_id_to_token(self, index):
        return self._tokens[index]

    def convert_tokens_to_string(self, tokens):
        # Decode byte tokens together, so multibyte characters survive boundaries.
        return b''.join(token.encode('utf-8') if token in self._special else
                        base64.b64decode(token[4:], validate=True) for token in tokens).decode('utf-8', errors='replace')

    def save_vocabulary(self, save_directory, filename_prefix=None):
        target = Path(save_directory)/((filename_prefix+'-' if filename_prefix else '')+'tokenizer.tiktoken.json')
        if Path(self.vocab_file).resolve() != target.resolve():
            shutil.copyfile(self.vocab_file, target)
        return (str(target),)

    def apply_chat_template(self, conversation, *, tokenize=True, add_generation_prompt=False,
                            return_tensors=None, return_dict=False, **kwargs):
        if kwargs:
            raise ValueError('unsupported chat options: '+', '.join(sorted(kwargs)))
        messages = [dict(message) for message in conversation]
        if not messages:
            raise ValueError('conversation must not be empty')
        if messages[0].get('role') == 'system':
            if len(messages) < 2 or messages[1].get('role') != 'user':
                raise ValueError('system must be followed by user')
            messages[1]['content'] = messages[0]['content']+'\n\n'+messages[1]['content']
            messages = messages[1:]
        ids = [self._special['<|bos|>']]
        for index, message in enumerate(messages):
            role = 'user' if index % 2 == 0 else 'assistant'
            if message.get('role') != role or not isinstance(message.get('content'), str):
                raise ValueError('native chat requires alternating user/assistant text messages')
            ids.append(self._special[f'<|{role}_start|>'])
            ids.extend(self.enc.encode_ordinary(message['content']))
            ids.append(self._special[f'<|{role}_end|>'])
        if add_generation_prompt:
            if messages[-1]['role'] != 'user':
                raise ValueError('generation prompt requires a final user message')
            ids.append(self._special['<|assistant_start|>'])
        if not tokenize:
            return self.enc.decode(ids)
        if return_tensors is not None or return_dict:
            batch = BatchEncoding({'input_ids': [ids], 'attention_mask': [[1]*len(ids)]}, tensor_type=return_tensors)
            return batch if return_dict else batch['input_ids']
        return ids
