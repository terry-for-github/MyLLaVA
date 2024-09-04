from torch import nn, Tensor
import torch

from .base_adapter import BaseAdapter


class BaseLinearAdapter(BaseAdapter):
    pass


class SingleLinearAdapter(BaseLinearAdapter):
    def __init__(
        self,
        llm_hidden_size: int,
        vision_hidden_size: int,
    ):
        super().__init__(llm_hidden_size, vision_hidden_size)
        self.linear = nn.Linear(vision_hidden_size, llm_hidden_size)

    def forward(self, vision_features: Tensor) -> Tensor:
        return self.linear(vision_features)


class MultiLinearAdapter(BaseLinearAdapter):
    def __init__(
        self,
        llm_hidden_size: int,
        vision_hidden_size_list: list[int],
    ):
        super().__init__(llm_hidden_size, vision_hidden_size_list)
        self.linear_list = nn.ModuleList([
            nn.Linear(hidden_size, llm_hidden_size)
            for hidden_size in vision_hidden_size_list
        ])

    def forward(self, vision_features_list: list[Tensor]) -> Tensor:
        '''vision_features_list: [B x N_i x C_i, ...]'''
        return torch.cat([
            linear(vision_features)
            for linear, vision_features in zip(self.linear_list, vision_features_list)
        ], dim=1)  # cat along with the token dimension
