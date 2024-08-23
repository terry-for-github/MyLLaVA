import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPVisionConfig

from constants import CACHE_DIR, HAS_CLS_TOKEN


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_model_name: str, select_layer: int):
        super().__init__()
        self.name = vision_model_name
        self.select_layer = select_layer
        self.has_cls_token = HAS_CLS_TOKEN[vision_model_name]
        # Cant use from_pretrained here because of the nested from_pretrained
        # Nested from_pretrained will cause nan problem in ZeRO-3 training.
        # Use lazy loading instead.
        self.config = CLIPVisionConfig.from_pretrained(self.name, cache_dir=CACHE_DIR)
        self.encoder = CLIPVisionModel(self.config)  # type: ignore
        self._is_loaded = False
        # TODO compile the image_encoder to further speed up training
        # self.image_encoder = torch.compile(image_encoder)

    def load(self, **kwargs):
        if self._is_loaded:
            return
        state_dict = CLIPVisionModel.from_pretrained(
            self.name,
            cache_dir=CACHE_DIR,
            **kwargs
        ).state_dict()
        self.encoder.load_state_dict(state_dict, strict=True)
        self._is_loaded = True

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
        outs = self.encoder(images.to(dtype=self.dtype), output_hidden_states=True)
        image_features = outs.hidden_states[self.select_layer]
        if self.has_cls_token:
            image_features = image_features[:, 1:]
        return image_features.to(dtype=images.dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.encoder.dtype

    @property
    def device(self):
        return self.encoder.device

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def num_patches_per_side(self) -> int:
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self) -> int:
        return self.num_patches_per_side ** 2
