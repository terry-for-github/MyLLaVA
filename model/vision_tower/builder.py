from .single_tower import SingleVisionTower
from .multi_tower import MultiVisionTower


def build_vision_tower(config):
    assert len(config.vision_tower) > 0
    assert len(config.vision_tower) == len(config.mm_vision_select_layer)
    if len(config.vision_tower) == 1:
        return SingleVisionTower(config.vision_tower[0], config.mm_vision_select_layer[0])
    else:
        return MultiVisionTower(config.vision_tower, config.mm_vision_select_layer)
