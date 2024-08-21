# Llama define this
from transformers import CLIPVisionModel, LayoutLMv3Model, Dinov2Model, SiglipVisionModel
from transformers import CLIPVisionConfig, LayoutLMv3Config, Dinov2Config, SiglipVisionConfig
from dataclasses import dataclass

IGNORE_INDEX = -100

CACHE_DIR = None


@dataclass(frozen=True)
class MODEL_CONST:
    _CLIP_L_336 = 'openai/clip-vit-large-patch14-336'
    _SIGLIP_SO = 'google/siglip-so400m-patch14-384'
    _SIGLIP_L = 'google/siglip-large-patch16-384'
    _DINOV2_G = 'facebook/dinov2-giant'
    _DINOV2_L = 'facebook/dinov2-large'
    _LAYOUTLMV3_L = 'microsoft/layoutlmv3-large'

    _IMAGE = "image_size"
    _HIDDEN = "hidden_size"
    _PATCH = "patch_size"
    _TOKEN = "num_patches"
    _CLASS = "class"
    _CONFIG = "config"
    _CLS = "has_cls_token"

    _MODEL_CONFIG = {
        _CLIP_L_336: {_IMAGE: 336, _HIDDEN: 1024, _PATCH: 14, _TOKEN: 576, _CLS: True},
        _SIGLIP_SO: {_IMAGE: 384, _HIDDEN: 1152, _PATCH: 14, _TOKEN: 729, _CLS: False},
        _SIGLIP_L: {_IMAGE: 384, _HIDDEN: 1024, _PATCH: 16, _TOKEN: 576, _CLS: False},
        _DINOV2_G: {_IMAGE: 518, _HIDDEN: 1536, _PATCH: 14, _TOKEN: 1369, _CLS: True},
        _DINOV2_L: {_IMAGE: 518, _HIDDEN: 1024, _PATCH: 14, _TOKEN: 1369, _CLS: True},
        _LAYOUTLMV3_L: {_IMAGE: 224, _HIDDEN: 1024, _PATCH: 16, _TOKEN: 196, _CLS: True},
    }

    _MODEL_CLASS = {
        _CLIP_L_336: {_CLASS: CLIPVisionModel, _CONFIG: CLIPVisionConfig},
        _SIGLIP_SO: {_CLASS: SiglipVisionModel, _CONFIG: SiglipVisionConfig},
        _SIGLIP_L: {_CLASS: SiglipVisionModel, _CONFIG: SiglipVisionConfig},
        _DINOV2_G: {_CLASS: Dinov2Model, _CONFIG: Dinov2Config},
        _DINOV2_L: {_CLASS: Dinov2Model, _CONFIG: Dinov2Config},
        _LAYOUTLMV3_L: {_CLASS: LayoutLMv3Model, _CONFIG: LayoutLMv3Config},
    }


IMAGE_SIZE = {k: v[MODEL_CONST._IMAGE] for k, v in MODEL_CONST._MODEL_CONFIG.items()}
HIDDEN_SIZE = {k: v[MODEL_CONST._HIDDEN] for k, v in MODEL_CONST._MODEL_CONFIG.items()}
PATCH_SIZE = {k: v[MODEL_CONST._PATCH] for k, v in MODEL_CONST._MODEL_CONFIG.items()}
NUM_TOKEN = {k: v[MODEL_CONST._TOKEN] for k, v in MODEL_CONST._MODEL_CONFIG.items()}
HAS_CLS = {k: v[MODEL_CONST._CLS] for k, v in MODEL_CONST._MODEL_CONFIG.items()}
CLASS = {k: v[MODEL_CONST._CLASS] for k, v in MODEL_CONST._MODEL_CLASS.items()}
CONFIG = {k: v[MODEL_CONST._CONFIG] for k, v in MODEL_CONST._MODEL_CLASS.items()}


__all__ = ['IGNORE_INDEX', 'CACHE_DIR',
           'IMAGE_SIZE', 'HIDDEN_SIZE', 'PATCH_SIZE', 'NUM_TOKEN',
           'HAS_CLS', 'CLASS', 'CONFIG']
