import re
from typing import Union


from .base_adapter import BaseAdapter
from .linear_adapter import SingleLinearAdapter, MultiLinearAdapter
from .mlp_adapter import SingleMlpAdapter, MultiMlpAdapter


def create_mm_adapter(
    mm_adapter_type: str,
    llm_hidden_size: int,
    vision_hidden_size: Union[int, list[int]],
    **kwargs
) -> BaseAdapter:
    if mm_adapter_type == 'linear':
        if isinstance(vision_hidden_size, int):
            return SingleLinearAdapter(llm_hidden_size, vision_hidden_size)
        return MultiLinearAdapter(llm_hidden_size, vision_hidden_size)

    # default gelu active func now, can be extended in the future
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mm_adapter_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        if isinstance(vision_hidden_size, int):
            return SingleMlpAdapter(llm_hidden_size, vision_hidden_size, mlp_depth)
        return MultiMlpAdapter(llm_hidden_size, vision_hidden_size, mlp_depth)
    raise NotImplementedError(f'Unsupported mm_adapter_type: {mm_adapter_type}')


__all__ = ['create_mm_adapter']
