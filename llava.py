from abc import ABC, abstractmethod
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM

from vision_tower.builder import build_vision_tower
from mm_adapter.builder import build_mm_adapter


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaMetaModel:
    def __init__(self, config):
        # =====================================================================
        print('[DEBUG]', 1, 'LlavaMetaModel init')
        print('[DEBUG]', 1, 'config', config)
        assert hasattr(config, 'vision_tower') == hasattr(config, 'mm_adapter')
        # =====================================================================
        self.config = config
        # self.vision_tower = build_vision_tower(config)
        # self.mm_adapter = build_mm_adapter(config)

    def get_vision_tower(self) -> Optional[torch.nn.Module]:
        return getattr(self, 'vision_tower', None)

    def get_mm_adapter(self) -> Optional[torch.nn.Module]:
        return getattr(self, 'mm_adapter', None)

    def init_vision_modules(self, model_args):
        # =====================================================================
        assert not hasattr(self, 'vision_tower') \
            and not hasattr(self, 'mm_adapter')
        print('[DEBUG]', 1, 'init_vision_modules')
        print('[DEBUG]', 1, 'model_args', model_args)
        print('[DEBUG]', 1, 'self', self)
        # =====================================================================

        # TODO what is this means
        # if model_args.mm_patch_merge_type == 'unpad':
        #     embed_std = 1 / torch.sqrt(
        # torch.tensor(self.config.hidden_size, dtype=self.dtype))
        #     self.image_newline = torch.nn.Parameter(
        #         torch.randn(self.config.hidden_size, dtype=self.dtype) \
        # * embed_std
        #     )

        self.vision_tower = build_vision_tower(
            model_args.vision_tower,
            model_args.mm_vision_select_layer,
            model_args.mm_vision_select_feature
        )
        self.mm_adapter = build_mm_adapter(
            model_args.mm_adapter,
            self.vision_tower.hidden_size,
            self.config.hidden_size
        )
        self._update_config(model_args)

    def _update_config(self, model_args):
        self.config.vision_tower = model_args.vision_tower
        self.config.mm_adapter = model_args.mm_adapter

        self.config.mm_hidden_size = self.vision_tower.hidden_size
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_vision_select_feature = \
            model_args.mm_vision_select_feature
        self.config.mm_patch_merge_type = model_args.mm_patch_merge_type

        if len(model_args.vision_expert_list) > 0:
            self.config.vision_expert_list = model_args.vision_expert_list
            self.config.m_token_one_patch = model_args.m_token_one_patch


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self) -> LlavaMetaModel:
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_mm_adapter(self):
        return self.get_model().get_mm_adapter()

    def encode_images(self, images):
        vision_tower = self.get_vision_tower()
        mm_adapter = self.get_mm_adapter()
        return mm_adapter(vision_tower(images))

    # def prepare_inputs_labels_for_multimodal(self):
    #     pass

    def init_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            pass

        if model_args.mm_use_im_start_end:
            pass
        elif model_args.mm_use_im_patch_token:
            pass


class LlavaLlamaModel(LlamaModel, LlavaMetaModel):
    config_class = LlavaConfig


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlavaLlamaModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size,
                                       config.vocab_size,
                                       bias=False)
        # =====================================================================
        print('[DEBUG]', 1, 'LlavaLlamaForCausalLM init')
        print('[DEBUG]', 1, 'config tie weight')
        print('[DEBUG]', 1, getattr(self.config, "tie_word_embeddings", None))
        print('[DEBUG]', 1, getattr(self.config, "tie_encoder_decoder", None))
        print('[DEBUG]', 1, '_init_weights', bool(self._init_weights))
        print('[DEBUG]', 1, 'supports gradient_checkpointing')
        print('[DEBUG]', 1, self.supports_gradient_checkpointing)
        print('[DEBUG]', 1, 'gradient_checkpointing')
        print('[DEBUG]', 1, self.model.gradient_checkpointing)
        # =====================================================================

        # Initialize weights and apply final processing
        self.post_init()

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
