from typing import Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .llava_meta import LlavaMetaModel, LlavaMetaForCausalLM, LlavaMetaConfig


# LlavaMetaConfig must be the first base class
# Otherwise, the __init__ method of it wont be called
class LlavaLlamaConfig(LlavaMetaConfig, LlamaConfig):
    model_type = "llava_llama"


# LlavaMetaModel must be the first base class
# Otherwise, the __init__ method of it wont be called
class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaLlamaConfig


# LlavaMetaForCausalLM has no __init__ method
# So it can also be the second base class
class LlavaLlamaForCausalLM(LlavaMetaForCausalLM, LlamaForCausalLM):
    config_class = LlavaLlamaConfig

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
        print('[DEBUG]', 1, 'supports gradient check:', self.supports_gradient_checkpointing)
        print('[DEBUG]', 1, 'gradient_checkpointing:', self.model.gradient_checkpointing)
        print('[DEBUG]', 1, '===============================================================')

    def get_model(self):
        return self.model

    def init_pretrained_model(self, **kwargs):
        self.model.init_vision_modules(**kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        vision_token_pos: Optional[torch.BoolTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        assert inputs_embeds is None and position_ids is None, \
            'inputs_embeds and position_ids must be None'
        assert input_ids is not None, 'input_ids is None'
        inputs_embeds = self.prepare_input_embeds_for_forward(
            input_ids=input_ids,
            images=images,
            vision_token_pos=vision_token_pos
        )  # type: ignore

        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            **kwargs
        )

    # def prepare_inputs_for_generation(self):
    #     pass

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        pass


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModelForCausalLM.register(LlavaLlamaConfig, LlavaLlamaForCausalLM)
