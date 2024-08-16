from abc import ABC, abstractmethod
from typing import Optional

import torch

from .vision_tower.builder import build_vision_tower
from .mm_adapter.builder import build_mm_adapter


class LlavaMetaConfig:
    def __init__(
        self,
        vision_tower: Optional[str] = None,
        mm_adapter: Optional[str] = None,
        mm_vision_select_layer: int = -2,
        mm_vision_select_feature: str = 'patch',
        mm_patch_merge_type: str = 'flat',
        pretrained_mm_adapter_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_tower = vision_tower
        self.mm_adapter = mm_adapter
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_patch_merge_type = mm_patch_merge_type
        self.pretrained_mm_adapter_path = pretrained_mm_adapter_path
        # TODO for mousi
        # if len(model_args.vision_expert_list) > 0:
        #     self.config.vision_expert_list = model_args.vision_expert_list
        #     self.config.m_token_one_patch = model_args.m_token_one_patch


class LlavaMetaModel:
    def __init__(self, config):
        super().__init__(config)  # type: ignore
        print('[DEBUG]', 1, '===============================================================')
        print('[DEBUG]', 1, 'LlavaMetaModel init')
        print('[DEBUG]', 1, 'config', config)
        print('[DEBUG]', 1, '===============================================================')
        if hasattr(config, 'vision_tower') and config.vision_tower is not None:
            self.init_vision_modules(config)

    def init_vision_modules(self, config=None, **kwargs):
        if not config:
            config = self.config  # type: ignore
        # TODO what is this means
        # if model_args.mm_patch_merge_type == 'unpad':
        #     embed_std = 1 / torch.sqrt(
        # torch.tensor(self.config.hidden_size, dtype=self.dtype))
        #     self.image_newline = torch.nn.Parameter(
        #         torch.randn(self.config.hidden_size, dtype=self.dtype) \
        # * embed_std
        #     )
        assert config.vision_tower is not None, 'Vision tower is not specified.'
        assert config.mm_adapter is not None, 'Multimodal adapter is not specified.'
        self.vision_tower = build_vision_tower(
            config.vision_tower,
            config.mm_vision_select_layer,
            config.mm_vision_select_feature,
            **kwargs
        )
        config.mm_hidden_size = self.vision_tower.hidden_size
        self.mm_adapter = build_mm_adapter(
            config.mm_adapter,
            config.mm_hidden_size,
            config.hidden_size
        )
        compute_dtype = kwargs.get('torch_dtype', None)
        if compute_dtype:
            self.mm_adapter.to(dtype=compute_dtype)

        if config.pretrained_mm_adapter_path is not None:
            self._load_mm_adapter(config.pretrained_mm_adapter)

    def _load_mm_adapter(self, state_dict_path: str):
        state_dict = torch.load(
            state_dict_path,
            map_location='cpu',
            weights_only=True
        )
        self.mm_adapter.load_state_dict(state_dict)
        assert isinstance(state_dict, dict)
        print('[DEBUG]', 1, '===============================================================')
        print('[DEBUG]', 1, 'Loading pretrained mm_adapter from:')
        print('[DEBUG]', 1, state_dict_path)
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

    def init_vision_modules(self, **kwargs):
        self.get_model().init_vision_modules(**kwargs)

    def init_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            pass
        if model_args.mm_use_im_start_end:
            pass
        elif model_args.mm_use_im_patch_token:
            pass
