from abc import abstractmethod
from typing import Union

from torch import Tensor

from ..base_module import BaseModule, load_decorator


class BaseVisionTower(BaseModule):
    def __init__(self, encoder_name, select_layer, *args, **kwargs):
        super().__init__()
        self.encoder_name = encoder_name
        self.select_layer = select_layer

    @abstractmethod
    @load_decorator
    def load(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, images: Tensor) -> Union[Tensor, list[Tensor]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self) -> Union[int, list[int]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_patches(self) -> int:
        raise NotImplementedError
