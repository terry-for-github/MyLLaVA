from typing import Union, Callable

from torch import nn, Tensor
import torch
import torch.nn.functional as F

from .base_adapter import BaseAdapter


class BaseMlpAdapter(BaseAdapter):
    def __init__(
        self,
        llm_hidden_size: int,
        vision_hidden_size: Union[int, list[int]],
        num_layers: int = 2,
        active_func: Callable[[Tensor], Tensor] = F.gelu,
    ):
        super().__init__(llm_hidden_size, vision_hidden_size)
        assert num_layers > 1
        self.num_layers = num_layers
        self.active_func = active_func
        self.mlp_list = nn.ModuleList([
            nn.Linear(llm_hidden_size, llm_hidden_size) for _ in range(num_layers - 1)
        ])

    def forward_mlp(self, hidden_features: Tensor) -> Tensor:
        for mlp in self.mlp_list:
            hidden_features = self.active_func(hidden_features)
            hidden_features = mlp(hidden_features)
        return hidden_features


class SingleMlpAdapter(BaseMlpAdapter):
    def __init__(
        self,
        llm_hidden_size: int,
        vision_hidden_size: int,
        num_layers: int = 2,
        active_func: Callable[[Tensor], Tensor] = F.gelu,
    ):
        super().__init__(llm_hidden_size, vision_hidden_size,
                         num_layers, active_func)
        self.projector = nn.Linear(vision_hidden_size, llm_hidden_size)

    def forward(self, vision_features: Tensor) -> Tensor:
        hidden_features = self.projector(vision_features)
        return self.forward_mlp(hidden_features)


class MultiMlpAdapter(BaseMlpAdapter):
    def __init__(
        self,
        llm_hidden_size: int,
        vision_hidden_size: list[int],
        num_layers: int = 2,
        active_func: Callable[[Tensor], Tensor] = F.gelu,
    ):
        super().__init__(llm_hidden_size, vision_hidden_size,
                         num_layers, active_func)
        self.projector_list = nn.ModuleList([
            nn.Linear(hidden_size, llm_hidden_size)
            for hidden_size in vision_hidden_size
        ])

    def forward(self, vision_features_list: list[Tensor]) -> Tensor:
        '''vision_features_list: [B x N_i x C_i, ...]'''
        hidden_features = torch.cat([
            projector(vision_features)
            for projector, vision_features in zip(self.projector_list, vision_features_list)
        ], dim=1)
        return self.forward_mlp(hidden_features)
