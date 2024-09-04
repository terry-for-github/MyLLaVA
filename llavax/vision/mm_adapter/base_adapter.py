import os
from typing import Union, Optional
from abc import abstractmethod

from torch import Tensor
import torch

from ..base_module import BaseModule, load_decorator


class BaseAdapter(BaseModule):
    def __init__(
        self,
        llm_hidden_size: int,
        vision_hidden_size: Union[int, list[int]],
    ):
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.vision_hidden_size = vision_hidden_size

    @load_decorator
    def load(self, state_dict_path: Optional[str] = None) -> None:
        if state_dict_path is None:
            return
        assert os.path.isfile(state_dict_path), f"{state_dict_path} is not a file."
        file_name = os.path.basename(state_dict_path)
        if (
            file_name.endswith('.bin') or
            file_name.endswith('.pt') or
            file_name.endswith('.pth')
        ):
            state_dict = torch.load(
                state_dict_path,
                map_location='cpu',
                weights_only=True
            )
        elif file_name.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(state_dict_path)
        else:
            raise ValueError(f'state dict file name illegel {file_name}.')
        assert isinstance(state_dict, dict)
        no_prefix_state_dict = {}
        for name, param in state_dict.items():
            if 'adapter.' not in name:
                continue
            no_prefix_name = name.split('adapter.')[1]
            no_prefix_state_dict[no_prefix_name] = param

        self.load_state_dict(no_prefix_state_dict)

    @abstractmethod
    def forward(self, vision_features: Union[Tensor, list[Tensor]]) -> Tensor:
        raise NotImplementedError
