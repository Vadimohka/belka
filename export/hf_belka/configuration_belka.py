"""Hugging Face configuration for the native nanochat architecture."""
from transformers import PretrainedConfig


class BelkaConfig(PretrainedConfig):
    model_type = 'belka_nanochat'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(self, sequence_len=2048, vocab_size=32768, n_layer=12,
                 n_head=6, n_kv_head=6, n_embd=768, window_pattern='SSSL', **kwargs):
        kwargs.pop('tie_word_embeddings', None)
        kwargs.pop('use_cache', None)
        super().__init__(tie_word_embeddings=False, **kwargs)
        for name, value in dict(sequence_len=sequence_len, vocab_size=vocab_size,
                                n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head,
                                n_embd=n_embd).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
            setattr(self, name, value)
        if n_embd < 24 or n_embd % n_head or n_head % n_kv_head or (n_embd//n_head)%2:
            raise ValueError('invalid native attention dimensions')
        if not window_pattern or set(window_pattern.upper()) - set('SL'):
            raise ValueError('window_pattern must contain only S and L')
        self.window_pattern = window_pattern
        self.max_position_embeddings = sequence_len
        self.use_cache = False

    def native_config(self):
        return {name: getattr(self, name) for name in
                ('sequence_len','vocab_size','n_layer','n_head','n_kv_head','n_embd','window_pattern')}
