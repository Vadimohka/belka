"""Reference FP32 inference adapter. Full-prefix generation; no KV-cache claim."""
import torch
from torch.nn import functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_belka import BelkaConfig
from .native_model import GPT, GPTConfig
# Keep the transitive dependency explicit for Transformers' local-code copier.
from .attention_belka import sdpa_attention


class BelkaForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = BelkaConfig
    base_model_prefix = 'model'
    _supports_cache_class = False
    _supports_flash_attn = False
    _supports_sdpa = False

    def __init__(self, config):
        super().__init__(config)
        self.model = GPT(GPTConfig(**config.native_config()))
        self.model.init_weights()

    def get_input_embeddings(self):
        return self.model.transformer.wte

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        loaded = super().from_pretrained(*args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        # HF constructs on meta; native rotary buffers are intentionally absent
        # from the checkpoint and must be rebuilt after weights reach a device.
        model.model.cos, model.model.sin = model.model._precompute_rotary_embeddings(
            model.model.rotary_seq_len, model.config.n_embd // model.config.n_head,
            device=model.model.transformer.wte.weight.device)
        return loaded

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_input_embeddings(self, value):
        self.model.transformer.wte = value

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                past_key_values=None, use_cache=False, return_dict=None, **kwargs):
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[1] > self.config.sequence_len:
            raise ValueError('input_ids must be a nonempty batch within the trained context')
        if past_key_values is not None or use_cache:
            raise ValueError('reference Belka HF export uses use_cache=False')
        if self.model.transformer.wte.weight.dtype != torch.float32:
            raise ValueError('reference Belka HF export requires torch.float32')
        if attention_mask is not None and (attention_mask.shape != input_ids.shape or not bool(attention_mask.all())):
            raise ValueError('reference Belka HF export does not support padded batches')
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape or input_ids.shape[1] < 2:
                raise ValueError('labels must match input_ids with at least two tokens')
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        if return_dict is False:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
