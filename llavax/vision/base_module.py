from abc import ABCMeta, abstractmethod

import torch
import deepspeed
from torch import nn, Tensor
from transformers.integrations import is_deepspeed_zero3_enabled


def load_decorator(load_func):
    def warpper(self, *args, **kwargs) -> None:
        if self.is_loaded:
            return
        print(f'Loading {self.__class__.__name__} parameters...')
        with deepspeed.zero.GatheredParameters(
            self.parameters(),
            modifier_rank=0,
            enabled=is_deepspeed_zero3_enabled()
        ):
            print('before loading:', self.param_mean())
            load_func(self, *args, **kwargs)
            print('after loading:', self.param_mean())

        print(f'{self.__class__.__name__} parameters loaded.')
        self.is_loaded = True
    return warpper


class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.is_loaded = False

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:
        _dtype = None
        for param in self.parameters():
            if _dtype is None:
                _dtype = param.dtype
            else:
                assert _dtype == param.dtype
        assert _dtype is not None
        return _dtype

    @property
    def device(self) -> torch.device:
        _device = None
        for param in self.parameters():
            if _device is None:
                _device = param.device
            else:
                assert _device == param.device
        assert _device is not None
        return _device

    def param_mean(self):
        is_aggregated = all(p.shape[0] != 0 for p in self.parameters())
        with deepspeed.zero.GatheredParameters(
            self.parameters(),
            enabled=is_deepspeed_zero3_enabled() and not is_aggregated
        ):
            param_sum = 0.
            param_num = 0
            for param in self.parameters():
                param_sum += torch.mean(param).item() * param.numel()
                param_num += param.numel()
        return f"{(param_sum / param_num):.8e}", param_num
