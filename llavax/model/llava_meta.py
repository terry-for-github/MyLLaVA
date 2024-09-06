from typing import Optional

import torch

from ..vision import create_mm_adapter, create_vision_tower


class LlavaMetaConfig:
    def __init__(
        self,
        *,  # FIXME Do not support positional arguments
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


def llava_head(is_forward: bool = True):
    def decorator(func):
        def warpper(
            self,
            *,  # FIXME Do not support positional arguments
            inputs_embeds: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            vision_token_pos: Optional[torch.BoolTensor] = None,
            image_masks: Optional[torch.BoolTensor] = None,
            **kwargs
        ):
            inputs_or_input_ids = (
                kwargs.pop('input_ids', None)
                if is_forward else
                kwargs.pop('inputs', None)
            )
            if images is None:
                assert vision_token_pos is None, 'vision_token_pos is not None'
                assert image_masks is None, 'image_masks is not None'
                return func(
                    self,
                    inputs_or_input_ids,
                    inputs_embeds=inputs_embeds,
                    **kwargs
                )
            assert vision_token_pos is not None, 'vision_token_pos is None'
            assert (inputs_or_input_ids is None) != (inputs_embeds is None)
            image_features = self.model.encode_images(images)
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(inputs_or_input_ids)
            inputs_embeds[vision_token_pos] = (  # type: ignore
                image_features[image_masks]
                if image_masks is not None else
                image_features.view(-1, image_features.size(dim=-1))
            )
            return func(
                self,
                inputs_embeds=inputs_embeds,
                **kwargs
            )
        return warpper
    return decorator
