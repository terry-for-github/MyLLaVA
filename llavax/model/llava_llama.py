import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM

from .llava_meta import LlavaMetaModel, LlavaMetaConfig, llava_head
from ..vision import BaseAdapter, BaseVisionTower


# LlavaMetaConfig must be the first base class
# Otherwise, the __init__ method of it wont be called
class LlavaLlamaConfig(LlavaMetaConfig, LlamaConfig):
    model_type = "llava_llama"


# LlavaMetaModel must be the first base class
# Otherwise, the __init__ method of it wont be called
class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaLlamaConfig


class LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_vision_tower(self) -> BaseVisionTower:
        return self.model.vision_tower

    def get_mm_adapter(self) -> BaseAdapter:
        return self.model.mm_adapter

    @llava_head(is_forward=True)
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    @torch.no_grad()
    @llava_head(is_forward=False)
    def generate(self, *args, **kwargs):
        return super().generate(*args, **kwargs)


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModelForCausalLM.register(LlavaLlamaConfig, LlavaLlamaForCausalLM)
