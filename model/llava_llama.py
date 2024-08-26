import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM

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


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModelForCausalLM.register(LlavaLlamaConfig, LlavaLlamaForCausalLM)
