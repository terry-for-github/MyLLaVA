from abc import ABC, abstractmethod
from typing import Optional

import torch

from .vision_tower.builder import build_vision_tower
from .mm_adapter.builder import build_mm_adapter


class LlavaMetaConfig:
    def __init__(
        self,
        vision_tower: str,
        mm_adapter: str,
        mm_vision_select_layer: int,
        mm_vision_select_feature: str,
        mm_patch_merge_type: str,
        max_length: int,
        pretrained_mm_adapter_path: Optional[str] = None
    ):
        self.vision_tower = vision_tower
        self.mm_adapter = mm_adapter
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_patch_merge_type = mm_patch_merge_type
        self.max_length = max_length
        self.pretrained_mm_adapter_path = pretrained_mm_adapter_path
        # TODO for mousi
        # if len(model_args.vision_expert_list) > 0:
        #     self.config.vision_expert_list = model_args.vision_expert_list
        #     self.config.m_token_one_patch = model_args.m_token_one_patch


class LlavaMetaModel:
    def __init__(self, config):
        assert hasattr(config, 'vision_tower') and hasattr(config, 'mm_adapter')
        print('[DEBUG]', 1, '===============================================================')
        print('[DEBUG]', 1, 'LlavaMetaModel init')
        print('[DEBUG]', 1, 'config', config)
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
            config.vision_tower,
            config.mm_vision_select_layer,
            config.mm_vision_select_feature,
            config.cache_dir
        )
        config.mm_hidden_size = self.vision_tower.hidden_size  # type: ignore
        self.mm_adapter = build_mm_adapter(
            config.mm_adapter,
            config.mm_hidden_size,
            config.hidden_size
        )

        # Load pretrained mm_adapter
        if config.pretrained_mm_adapter_path is not None:
            return
        state_dict = torch.load(
            config.pretrained_mm_adapter_path,
            map_location='cpu',
            weights_only=True
        )
        self.mm_adapter.load_state_dict(state_dict)
        assert isinstance(state_dict, dict)
        print('[DEBUG]', 1, '===============================================================')
        print('[DEBUG]', 1, 'Loading pretrained mm_adapter from:')
        print('[DEBUG]', 1, config.pretrained_mm_adapter_path)
        print('[DEBUG]', 1, 'mm_adapter_state_dict')
        print('[DEBUG]', 1, state_dict.keys())
        print('[DEBUG]', 1, '===============================================================')


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self) -> LlavaMetaModel:
        pass

    def get_vision_tower(self):
        return self.get_model().vision_tower

    def get_mm_adapter(self):
        return self.get_model().mm_adapter

    def encode_images(self, images):
        image_features = self.get_vision_tower()(images)  # type: ignore
        # image_features.requires_grad = True
        return self.get_mm_adapter()(image_features)  # type:ignore

    @abstractmethod
    def prepare_inputs_for_forward(self, input_ids, attention_mask, labels, images):
        pass

    def init_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            pass
        if model_args.mm_use_im_start_end:
            pass
        elif model_args.mm_use_im_patch_token:
            pass
