import torch
import torch.nn as nn

from transformers import CLIPImageProcessor, CLIPVisionModel


class CLIPVisionTower(nn.Module):
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 mm_vision_select_layer: int,
                 mm_vision_select_feature: str):
        super().__init__()
        self.name = pretrained_model_name_or_path
        self.select_layer = mm_vision_select_layer
        self.select_feature = mm_vision_select_feature
        self.image_processor = CLIPImageProcessor.from_pretrained(self.name)
        self.image_encoder = CLIPVisionModel.from_pretrained(self.name)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: '
                             f'{self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        print('[DEBUG]', 2, 'CLIPVisionTower forward')
        print('[DEBUG]', 2, 'images', images.shape)
        print('[DEBUG]', 2, 'image device', images.device)
        print('[DEBUG]', 2, 'image dtype', images.dtype)
        print('[DEBUG]', 2, 'self device', self.device)
        print('[DEBUG]', 2, 'self dtype', self.dtype)
        image_forward_outs = self.image_encoder(images.to(dtype=self.dtype),
                                                output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs)
        return image_features.to(dtype=images.dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size,
                           device=self.device, dtype=self.dtype)

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
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2


if __name__ == '__main__':
    # Test CLIPVisionTower
    print('Test 1')
    vision_tower = CLIPVisionTower('openai/clip-vit-large-patch14-336',
                                   mm_vision_select_layer=-1,
                                   mm_vision_select_feature='patch')
    print(vision_tower)
    print(vision_tower.config)
    print(vision_tower.hidden_size)
    print(vision_tower.num_patches_per_side)
    print(vision_tower.num_patches)
    images = torch.randn(2, 3, 336, 336)
    image_features = vision_tower(images)
    print(image_features.shape)
    print(image_features.device)
    print(image_features.dtype)
    print(vision_tower.dummy_feature)
    print(vision_tower.dummy_feature.device)
    print(vision_tower.dummy_feature.dtype)
    print('Pass test 1')
    print('===========================================')
    print('Test 2')
    vision_tower = CLIPVisionTower('openai/clip-vit-large-patch14-336',
                                   mm_vision_select_layer=-2,
                                   mm_vision_select_feature='cls_patch')
    print(vision_tower)
    print(vision_tower.config)
    print(vision_tower.hidden_size)
    print(vision_tower.num_patches_per_side)
    print(vision_tower.num_patches)
    images = torch.randn(2, 3, 336, 336)
    image_features = vision_tower(images)
    print(image_features.shape)
    print(image_features.device)
    print(image_features.dtype)
    print(vision_tower.dummy_feature)
    print(vision_tower.dummy_feature.device)
    print(vision_tower.dummy_feature.dtype)
    print('Pass test 2')
