from typing import List

import torch

from .base_tower import BaseVisionTower
from .single_tower import SingleVisionTower


class MultiVisionTower(BaseVisionTower):
    def __init__(self, encoder_name: List[str], select_layer: List[int]):
        super().__init__(encoder_name, select_layer)
        self.tower_list = [
            SingleVisionTower(name, layer)
            for name, layer in zip(encoder_name, select_layer)
        ]
        patch_list = [encoder.num_patches for encoder in self.tower_list]
        self.side_list = [int(num_patches ** 0.5 + 0.1) for num_patches in patch_list]
        self.prefix_list = [0]
        for i, num_patches in enumerate(patch_list):
            self.prefix_list.append(self.prefix_list[i] + num_patches)

    @BaseVisionTower.load_decorator
    def load(self, **kwargs):
        for tower in self.tower_list:
            tower.load(**kwargs)

    @property
    def is_loaded(self):
        for tower in self.tower_list:
            if not tower.is_loaded:
                return False
        return True

    @is_loaded.setter
    def is_loaded(self, value):
        assert value == self.is_loaded

    @torch.no_grad()
    def forward(self, images):
        feature_list = []
        for i in range(len(self.prefix_list)-1):
            images_i = images[:, :, self.prefix_list[i]:self.prefix_listprefix_list[i+1]]
            b, c, _ = images_i.shape
            images_i = images_i.view(b, c, self.side_list[i], self.side_list[i])
            feature_list.append(self.tower_list[i](images_i))
        return feature_list

    @property
    def dummy_feature(self):
        return [tower.dummy_feature for tower in self.tower_list]

    @property
    def dtype(self):
        return self.tower_list[0].dtype

    @property
    def device(self):
        return self.tower_list[0].device

    @property
    def hidden_size(self) -> List[int]:
        return [tower.hidden_size for tower in self.tower_list]

    @property
    def num_patches(self) -> int:
        return sum([tower.num_patches for tower in self.tower_list])
