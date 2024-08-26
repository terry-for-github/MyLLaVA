import os
from typing import List, Optional, Tuple, Union

import torch
import deepspeed
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

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

    def encode_images(self, images):
        return self.mm_adapter(self.vision_tower(images))

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


class LlavaMetaForCausalLM:
    _keys_to_ignore_on_load_missing = ['vision_tower', 'mm_adapter']

    def get_vision_tower(self):
        return self.model.vision_tower  # type: ignore

    def get_mm_adapter(self):
        return self.model.mm_adapter  # type: ignore

    def init_vision_modules(self, **kwargs):
        self.model.init_vision_modules(**kwargs)  # type: ignore

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        vision_token_pos: Optional[torch.BoolTensor] = None,
        image_masks: Optional[torch.BoolTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if images is not None:
            assert vision_token_pos is not None, 'vision_token_pos is None'
            assert image_masks is not None, 'image_masks is None'
            assert inputs_embeds is None, 'inputs_embeds is not None'

            image_features = self.model.encode_images(images)  # type: ignore
            inputs_embeds = self.get_input_embeddings()(input_ids)  # type: ignore
            inputs_embeds[vision_token_pos] = image_features[image_masks]  # type: ignore

        return super().forward(  # type: ignore
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )
