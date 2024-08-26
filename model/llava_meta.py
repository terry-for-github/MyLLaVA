import os
from abc import ABC, abstractmethod
from typing import Optional

import torch
import deepspeed
from transformers.integrations import is_deepspeed_zero3_enabled

from .vision_tower.builder import build_vision_tower
from .mm_adapter.builder import build_mm_adapter


class LlavaMetaConfig:
    def __init__(
        self,
        vision_tower: Optional[str] = None,
        mm_adapter: Optional[str] = None,
        mm_vision_select_layer: int = -2,
        mm_patch_merge_type: str = 'flat',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_tower = vision_tower
        self.mm_adapter = mm_adapter
        self.mm_hidden_size = None
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_patch_merge_type = mm_patch_merge_type

    def update(self, model_args):
        self.vision_tower = model_args.vision_tower
        self.mm_adapter = model_args.mm_adapter
        self.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.mm_patch_merge_type = model_args.mm_patch_merge_type


class LlavaMetaModel:
    def __init__(self, config):
        super().__init__(config)  # type: ignore
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
        assert config.vision_tower is not None, 'Vision tower is not specified.'
        assert config.mm_adapter is not None, 'Multimodal adapter is not specified.'
        self.vision_tower = build_vision_tower(config)
        config.mm_hidden_size = self.vision_tower.hidden_size
        self.mm_adapter = build_mm_adapter(config)

    def init_vision_modules(self, pretrained_mm_adapter_path: Optional[str], **kwargs):
        # Load vision_tower
        self.vision_tower.load(**kwargs)
        # Load mm_adapter
        self._load_mm_adapter(pretrained_mm_adapter_path)

    def _load_mm_adapter(self, state_dict_path: Optional[str]):
        if state_dict_path is None:
            return
        assert os.path.isfile(state_dict_path), f"{state_dict_path} is not a file."
        file_name = os.path.basename(state_dict_path)
        if (
            file_name.endswith('.bin') or
            file_name.endswith('.pt') or
            file_name.endswith('.pth')
        ):
            state_dict = torch.load(
                state_dict_path,
                map_location='cpu',
                weights_only=True
            )
        elif file_name.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(state_dict_path)
        else:
            raise ValueError(f'state dict file name illegel {file_name}.')
        assert isinstance(state_dict, dict)
        no_prefix_state_dict = {}
        for name, param in state_dict.items():
            if 'mm_adapter' not in name:
                continue
            no_prefix_name = name.split('model.mm_adapter.')[1]
            no_prefix_state_dict[no_prefix_name] = param

        with deepspeed.zero.GatheredParameters(self.mm_adapter.parameters(), modifier_rank=0,
                                               enabled=is_deepspeed_zero3_enabled()):
            self.mm_adapter.load_state_dict(no_prefix_state_dict)

        print('[DEBUG]', 1, '===============================================================')
        print('[DEBUG]', 1, 'Loading pretrained mm_adapter from:')
        print('[DEBUG]', 1, state_dict_path)
        print('[DEBUG]', 1, 'mm_adapter_state_dict')
        print('[DEBUG]', 1, state_dict.keys())
        print('[DEBUG]', 1, '===============================================================')


class LlavaMetaForCausalLM(ABC):
    _keys_to_ignore_on_load_missing = ['vision_tower', 'mm_adapter']

    @abstractmethod
    def get_model(self) -> LlavaMetaModel:
        pass

    def get_vision_tower(self):
        return self.get_model().vision_tower

    def get_mm_adapter(self):
        return self.get_model().mm_adapter

    def init_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            pass
        if model_args.mm_use_im_start_end:
            pass
        elif model_args.mm_use_im_patch_token:
            pass

    def encode_images(self, images):
        image_features = self.get_vision_tower()(images)  # type: ignore
        image_features = self.get_mm_adapter()(image_features)  # type:ignore
        return image_features

    def prepare_input_embeds_for_forward(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor,
        vision_token_pos: torch.BoolTensor,
        image_masks: torch.BoolTensor
    ):
        '''
        1. If images is None, return the input embeddings directly.
        2. Embed the input_ids to input_embeds
        3. Encode the images and flatten the image features.
        4. Replace input_embeds with image features at vision_token_pos.
        '''
        embed_tokens: torch.nn.Module = self.get_input_embeddings()  # type: ignore

        input_embeds = embed_tokens(input_ids)
        # image_features: image_num x patch_num x dim
        image_features = self.encode_images(images)
        image_features = image_features.view(-1, image_features.size(-1))
        # Replace the image tokens with image features
        input_embeds[vision_token_pos] = image_features[image_masks]
        return input_embeds
