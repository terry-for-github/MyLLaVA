import torch

from llavax.constants import (
    CACHE_DIR, MODEL_CONFIG, MODEL_CLASS, HAS_CLS_TOKEN, HIDDEN_SIZE, NUM_PATCHES
)

from .base_tower import BaseVisionTower


class SingleVisionTower(BaseVisionTower):
    def __init__(self, encoder_name: str, select_layer: int):
        super().__init__(encoder_name, select_layer)
        config = MODEL_CONFIG[encoder_name].from_pretrained(encoder_name, cache_dir=CACHE_DIR)
        self.has_cls_token = HAS_CLS_TOKEN[encoder_name]
        # Cant use from_pretrained here because of the nested from_pretrained
        # Nested from_pretrained will cause nan problem in ZeRO-3 training.
        # Use lazy loading instead.
        self.encoder_class = MODEL_CLASS[encoder_name]
        self.encoder = self.encoder_class(config)
        # TODO compile the image_encoder to further speed up training
        # self.image_encoder = torch.compile(image_encoder)

    @BaseVisionTower.load_decorator
    def load(self, **kwargs):
        state_dict = self.encoder_class.from_pretrained(
            self.encoder_name,
            cache_dir=CACHE_DIR,
            **kwargs
        ).state_dict()
        self.encoder.load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def forward(self, images):
        outs = self.encoder(images.to(dtype=self.dtype), output_hidden_states=True)
        image_features = outs.hidden_states[self.select_layer]
        if self.has_cls_token:
            image_features = image_features[:, 1:]
        return image_features.to(dtype=images.dtype)

    @property
    def dummy_feature(self):
        return torch.randn(
            1,
            NUM_PATCHES[self.encoder_name],
            HIDDEN_SIZE[self.encoder_name],
            device=self.device,
            dtype=self.dtype
        )

    @property
    def dtype(self):
        return self.encoder.dtype

    @property
    def device(self):
        return self.encoder.device

    @property
    def hidden_size(self) -> int:
        return HIDDEN_SIZE[self.encoder_name]

    @property
    def num_patches(self) -> int:
        return NUM_PATCHES[self.encoder_name]
