import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM

from .llava_meta import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlamaModel, LlavaMetaModel):
    config_class = LlavaConfig


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        print('[DEBUG]', 1, '===============================================================')
        print('[DEBUG]', 1, 'LlavaLlamaForCausalLM init')
        print('[DEBUG]', 1, 'config tie weight')
        print('[DEBUG]', 1, getattr(self.config, "tie_word_embeddings", None))
        print('[DEBUG]', 1, getattr(self.config, "tie_encoder_decoder", None))
        print('[DEBUG]', 1, '_init_weights', bool(self._init_weights))
        print('[DEBUG]', 1, 'supports gradient_checkpointing')
        print('[DEBUG]', 1, self.supports_gradient_checkpointing)
        print('[DEBUG]', 1, 'gradient_checkpointing')
        print('[DEBUG]', 1, self.model.gradient_checkpointing)
        print('[DEBUG]', 1, '===============================================================')

    def get_model(self):
        return self.model

    # def forward(self):
    #     pass

    # def generate(self):
    #     pass

    # def prepare_inputs_for_generation(self):
    #     pass


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
