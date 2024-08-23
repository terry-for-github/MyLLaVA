# Llama define this
from enum import Enum, auto
from transformers import CLIPVisionModel, LayoutLMv3Model, Dinov2Model, SiglipVisionModel
from transformers import CLIPVisionConfig, LayoutLMv3Config, Dinov2Config, SiglipVisionConfig
from dataclasses import dataclass

IGNORE_INDEX = -100

CACHE_DIR = None


@dataclass(frozen=True)
class MODEL_NAME:
    CLIP_L_336 = 'openai/clip-vit-large-patch14-336'
    SIGLIP_SO = 'google/siglip-so400m-patch14-384'
    SIGLIP_L = 'google/siglip-large-patch16-384'
    DINOV2_G = 'facebook/dinov2-giant'
    DINOV2_L = 'facebook/dinov2-large'
    LAYOUTLMV3_L = 'microsoft/layoutlmv3-large'


class MODEL_ATTRIBUTE(Enum):
    IMAGE_SIZE = auto()
    IMAGE_MEAN = auto()
    HIDDEN_SIZE = auto()
    PATCH_SIZE = auto()
    NUM_PATCHES = auto()
    HAS_CLS_TOKEN = auto()
    MODEL_CLASS = auto()
    MODEL_CONFIG = auto()


MODEL_CONSTANTS = {
    MODEL_NAME.CLIP_L_336: {
        MODEL_ATTRIBUTE.IMAGE_SIZE: 336,
        MODEL_ATTRIBUTE.IMAGE_MEAN: [0.48145466, 0.4578275, 0.40821073],
        MODEL_ATTRIBUTE.HIDDEN_SIZE: 1024,
        MODEL_ATTRIBUTE.PATCH_SIZE: 14,
        MODEL_ATTRIBUTE.NUM_PATCHES: 576,
        MODEL_ATTRIBUTE.HAS_CLS_TOKEN: True,
        MODEL_ATTRIBUTE.MODEL_CLASS: CLIPVisionModel,
        MODEL_ATTRIBUTE.MODEL_CONFIG: CLIPVisionConfig
    },
    MODEL_NAME.SIGLIP_SO: {
        MODEL_ATTRIBUTE.IMAGE_SIZE: 384,
        MODEL_ATTRIBUTE.IMAGE_MEAN: [0.5, 0.5, 0.5],
        MODEL_ATTRIBUTE.HIDDEN_SIZE: 1152,
        MODEL_ATTRIBUTE.PATCH_SIZE: 14,
        MODEL_ATTRIBUTE.NUM_PATCHES: 729,
        MODEL_ATTRIBUTE.HAS_CLS_TOKEN: False,
        MODEL_ATTRIBUTE.MODEL_CLASS: SiglipVisionModel,
        MODEL_ATTRIBUTE.MODEL_CONFIG: SiglipVisionConfig
    },
    MODEL_NAME.SIGLIP_L: {
        MODEL_ATTRIBUTE.IMAGE_SIZE: 384,
        MODEL_ATTRIBUTE.IMAGE_MEAN: [0.5, 0.5, 0.5],
        MODEL_ATTRIBUTE.HIDDEN_SIZE: 1024,
        MODEL_ATTRIBUTE.PATCH_SIZE: 16,
        MODEL_ATTRIBUTE.NUM_PATCHES: 576,
        MODEL_ATTRIBUTE.HAS_CLS_TOKEN: False,
        MODEL_ATTRIBUTE.MODEL_CLASS: SiglipVisionModel,
        MODEL_ATTRIBUTE.MODEL_CONFIG: SiglipVisionConfig
    },
    MODEL_NAME.DINOV2_G: {
        MODEL_ATTRIBUTE.IMAGE_SIZE: 518,
        MODEL_ATTRIBUTE.IMAGE_MEAN: [0.485, 0.456, 0.406],
        MODEL_ATTRIBUTE.HIDDEN_SIZE: 1536,
        MODEL_ATTRIBUTE.PATCH_SIZE: 14,
        MODEL_ATTRIBUTE.NUM_PATCHES: 1369,
        MODEL_ATTRIBUTE.HAS_CLS_TOKEN: True,
        MODEL_ATTRIBUTE.MODEL_CLASS: Dinov2Model,
        MODEL_ATTRIBUTE.MODEL_CONFIG: Dinov2Config
    },
    MODEL_NAME.DINOV2_L: {
        MODEL_ATTRIBUTE.IMAGE_SIZE: 518,
        MODEL_ATTRIBUTE.IMAGE_MEAN: [0.485, 0.456, 0.406],
        MODEL_ATTRIBUTE.HIDDEN_SIZE: 1024,
        MODEL_ATTRIBUTE.PATCH_SIZE: 14,
        MODEL_ATTRIBUTE.NUM_PATCHES: 1369,
        MODEL_ATTRIBUTE.HAS_CLS_TOKEN: True,
        MODEL_ATTRIBUTE.MODEL_CLASS: Dinov2Model,
        MODEL_ATTRIBUTE.MODEL_CONFIG: Dinov2Config
    },
    MODEL_NAME.LAYOUTLMV3_L: {
        MODEL_ATTRIBUTE.IMAGE_SIZE: 224,
        MODEL_ATTRIBUTE.IMAGE_MEAN: [0.5, 0.5, 0.5],
        MODEL_ATTRIBUTE.HIDDEN_SIZE: 1024,
        MODEL_ATTRIBUTE.PATCH_SIZE: 16,
        MODEL_ATTRIBUTE.NUM_PATCHES: 196,
        MODEL_ATTRIBUTE.HAS_CLS_TOKEN: True,
        MODEL_ATTRIBUTE.MODEL_CLASS: LayoutLMv3Model,
        MODEL_ATTRIBUTE.MODEL_CONFIG: LayoutLMv3Config
    }
}

IMAGE_SIZE = {k: v[MODEL_ATTRIBUTE.IMAGE_SIZE] for k, v in MODEL_CONSTANTS.items()}
IMAGE_MEAN = {k: v[MODEL_ATTRIBUTE.IMAGE_MEAN] for k, v in MODEL_CONSTANTS.items()}
HIDDEN_SIZE = {k: v[MODEL_ATTRIBUTE.HIDDEN_SIZE] for k, v in MODEL_CONSTANTS.items()}
PATCH_SIZE = {k: v[MODEL_ATTRIBUTE.PATCH_SIZE] for k, v in MODEL_CONSTANTS.items()}
NUM_PATCHES = {k: v[MODEL_ATTRIBUTE.NUM_PATCHES] for k, v in MODEL_CONSTANTS.items()}
HAS_CLS_TOKEN = {k: v[MODEL_ATTRIBUTE.HAS_CLS_TOKEN] for k, v in MODEL_CONSTANTS.items()}
MODEL_CLASS = {k: v[MODEL_ATTRIBUTE.MODEL_CLASS] for k, v in MODEL_CONSTANTS.items()}
MODEL_CONFIG = {k: v[MODEL_ATTRIBUTE.MODEL_CONFIG] for k, v in MODEL_CONSTANTS.items()}


__all__ = ['IGNORE_INDEX', 'CACHE_DIR',
           'IMAGE_SIZE', 'IMAGE_MEAN', 'HIDDEN_SIZE', 'PATCH_SIZE', 'NUM_PATCHES',
           'HAS_CLS_TOKEN', 'MODEL_CLASS', 'MODEL_CONFIG']


if __name__ == '__main__':
    import torch
    from transformers import AutoImageProcessor
    model_name = MODEL_NAME.LAYOUTLMV3_L
    model_class = MODEL_CLASS[model_name]
    model_config = MODEL_CONFIG[model_name]
    config = model_config.from_pretrained(model_name)
    print(config)
    processor = AutoImageProcessor.from_pretrained(model_name)
    print(processor)
    model = model_config.from_pretrained(model_name, torch_dtype='float16', device_map='cuda')
    dummy_image = torch.randn(1, 3, 224, 224, device='cuda', dtype=torch.float16)
    output = model(pixel_values=dummy_image, output_hidden_states=True)
    print(output.last_hidden_state.shape)
    print(output.hidden_states[-2].shape)
