from typing import Optional
import torch
import torch.nn as nn

from transformers import CLIPImageProcessor, CLIPVisionModel

feature_select_func_dict = {
    'patch': lambda x: x[:, 1:],
    'cls_patch': lambda x: x
}


class CLIPVisionTower(nn.Module):
    def __init__(self, model_name_or_path: str, select_layer: int, select_feature: str,
                 cache_dir: Optional[str] = None):
        super().__init__()
        self.name = model_name_or_path
        self.select_layer_func = lambda x: x[select_layer]
        self.select_feature_func = feature_select_func_dict[select_feature]
        self.image_processor = CLIPImageProcessor.from_pretrained(self.name, cache_dir)
        self.image_encoder = CLIPVisionModel.from_pretrained(self.name, cache_dir=cache_dir)
        # TODO compile the image_encoder to further speed up training
        # self.image_encoder = torch.compile(image_encoder)

    def feature_select(self, image_forward_outs):
        image_features = self.select_layer_func(image_forward_outs.hidden_states)
        return self.select_feature_func(image_features)

    @torch.no_grad()
    def forward(self, images):
        print('[DEBUG]', 2, '===============================================================')
        print('[DEBUG]', 2, 'CLIPVisionTower forward')
        print('[DEBUG]', 2, 'images', images.shape)
        print('[DEBUG]', 2, 'image device', images.device)
        print('[DEBUG]', 2, 'image dtype', images.dtype)
        print('[DEBUG]', 2, 'self device', self.device)
        print('[DEBUG]', 2, 'self dtype', self.dtype)
        print('[DEBUG]', 2, '===============================================================')
        image_forward_outs = self.image_encoder(images.to(dtype=self.dtype),
                                                output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs)
        return image_features.to(dtype=images.dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.image_encoder.dtype

    @property
    def device(self):
        return self.image_encoder.device

    @property
    def config(self):
        return self.image_encoder.config

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def num_patches_per_side(self) -> int:
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self) -> int:
        return self.num_patches_per_side ** 2
