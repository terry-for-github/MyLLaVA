from .base_tower import BaseVisionTower
from .single_tower import SingleVisionTower
from .multi_tower import MultiVisionTower


def create_vision_tower(
    vision_tower: list[str],
    vision_select_layer: list[int]
) -> BaseVisionTower:
    assert len(vision_tower) > 0
    assert len(vision_tower) == len(vision_select_layer)
    if len(vision_tower) == 1:
        return SingleVisionTower(vision_tower[0], vision_select_layer[0])
    else:
        return MultiVisionTower(vision_tower, vision_select_layer)


__all__ = ['create_vision_tower']
