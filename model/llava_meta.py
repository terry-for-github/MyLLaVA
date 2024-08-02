from abc import ABC, abstractmethod
from typing import Optional

import torch

from .vision_tower.builder import build_vision_tower
from .mm_adapter.builder import build_mm_adapter


class LlavaMetaModel:
    def __init__(self, config):
        assert hasattr(config, 'vision_tower') == hasattr(config, 'mm_adapter')
        print('[DEBUG]', 1, '===============================================================')
        print('[DEBUG]', 1, 'LlavaMetaModel init')
        print('[DEBUG]', 1, 'config', config)
        print('[DEBUG]', 1, '===============================================================')
        self.config = config
        # self.vision_tower = build_vision_tower(config)
        # self.mm_adapter = build_mm_adapter(config)

    def get_vision_tower(self) -> Optional[torch.nn.Module]:
        return getattr(self, 'vision_tower', None)

    def get_mm_adapter(self) -> Optional[torch.nn.Module]:
        return getattr(self, 'mm_adapter', None)

    def init_vision_modules(self, model_args):
        assert not hasattr(self, 'vision_tower') and not hasattr(self, 'mm_adapter')
        print('[DEBUG]', 1, '===============================================================')
        print('[DEBUG]', 1, 'init_vision_modules')
        print('[DEBUG]', 1, 'model_args', model_args)
        print('[DEBUG]', 1, 'self', self)
        print('[DEBUG]', 1, '===============================================================')

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
        self.config.mm_vision_select_feature = model_args.mm_vision_select_feature
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
