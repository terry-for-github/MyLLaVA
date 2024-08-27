from abc import ABC, abstractmethod
from typing import Union

from torch import nn
import torch


class BaseVisionTower(nn.Module, ABC):
    def __init__(self, encoder_name, select_layer, *args, **kwargs):
        super().__init__()
        self.encoder_name = encoder_name
        self.select_layer = select_layer
        self._is_loaded = False

    @abstractmethod
    def load(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_decorator(func):
        def warpper(self, *args, **kwargs):
            if self.is_loaded:
                return
            func(self, *args, **kwargs)
            self.is_loaded = True
        return warpper

    @property
    def is_loaded(self):
        return self._is_loaded

    @is_loaded.setter
    def is_loaded(self, value):
        self._is_loaded = value

    @abstractmethod
    def forward(self, images):
        raise NotImplementedError

    @property
    @abstractmethod
    def dummy_feature(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> Union[str, torch.dtype]:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> Union[str, torch.device]:
        raise NotImplementedError

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_patches(self) -> int:
        raise NotImplementedError
