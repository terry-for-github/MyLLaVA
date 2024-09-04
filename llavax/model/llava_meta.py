from typing import Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import PreTrainedTokenizerBase

from ..vision import create_mm_adapter, create_vision_tower, BaseAdapter, BaseVisionTower


class LlavaMetaConfig:
    def __init__(
        self,
        vision_tower: Optional[list[str]] = None,
        mm_adapter: Optional[str] = None,
        vision_select_layer: Optional[list[int]] = None,
        patch_merge_type: str = 'flat',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_tower = vision_tower
        self.mm_adapter = mm_adapter
        self.vision_select_layer = vision_select_layer
        self.patch_merge_type = patch_merge_type


class LlavaMetaModel:
    def __init__(self, config):
        super().__init__(config)  # type: ignore
        # TODO what is this means
        # if model_args.patch_merge_type == 'unpad':
        #     embed_std = 1 / torch.sqrt(
        # torch.tensor(self.config.hidden_size, dtype=self.dtype))
        #     self.image_newline = torch.nn.Parameter(
        #         torch.randn(self.config.hidden_size, dtype=self.dtype) \
        # * embed_std
        #     )
        assert config.vision_tower is not None, 'Vision tower is not specified.'
        assert config.mm_adapter is not None, 'Multimodal adapter is not specified.'
        self.vision_tower = create_vision_tower(
            config.vision_tower,
            config.vision_select_layer
        )
        self.mm_adapter = create_mm_adapter(
            config.mm_adapter,
            config.hidden_size,
            self.vision_tower.hidden_size,
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        image_feature = self.vision_tower(images)
        adapted_feature = self.mm_adapter(image_feature)
        return adapted_feature


class LlavaMetaForCausalLM:
    def get_vision_tower(self) -> BaseVisionTower:
        return self.model.vision_tower  # type: ignore

    def get_mm_adapter(self) -> BaseAdapter:
        return self.model.mm_adapter  # type: ignore

    # Llava cant only accept input_ids when there has images.
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
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
            image_features = self.model.encode_images(images)  # type: ignore
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)  # type: ignore

            inputs_embeds[vision_token_pos] = (
                image_features[image_masks]
                if image_masks is not None else
                image_features.view(-1, inputs_embeds.size(dim=-1))
            )

        return super().forward(  # type: ignore
            input_ids=input_ids if images is None else None,
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
