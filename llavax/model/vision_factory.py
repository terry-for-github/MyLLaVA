import re
from typing import Union, List

from torch import nn

from .vision_tower import SingleVisionTower, MultiVisionTower


def create_mm_adapter(config, vision_hidden_size: Union[int, List[int]]):
    mm_adapter_type: str = config.mm_adapter
    hidden_size = config.hidden_size
    assert isinstance(vision_hidden_size, int)
    if mm_adapter_type == 'linear':
        return nn.Linear(vision_hidden_size, hidden_size)
    if mm_adapter_type == 'identity':
        return nn.Identity()
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mm_adapter_type)
    if mlp_gelu_match:
        module_list = nn.ModuleList([nn.Linear(vision_hidden_size, hidden_size)])
        mlp_depth = int(mlp_gelu_match.group(1))
        for _ in range(1, mlp_depth):
            module_list.append(nn.GELU())
            module_list.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*module_list)
    raise NotImplementedError(f'Unsupported mm_adapter_type: {mm_adapter_type}')


def create_vision_tower(config):
    assert len(config.vision_tower) > 0
    assert len(config.vision_tower) == len(config.mm_vision_select_layer)
    if len(config.vision_tower) == 1:
        return SingleVisionTower(config.vision_tower[0], config.mm_vision_select_layer[0])
    else:
        return MultiVisionTower(config.vision_tower, config.mm_vision_select_layer)
