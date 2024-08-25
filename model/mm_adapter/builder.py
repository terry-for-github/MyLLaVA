import re
import torch.nn as nn



def build_mm_adapter(config):
    mm_adapter_type: str = config.mm_adapter
    mm_hidden_size = config.mm_hidden_size
    hidden_size = config.hidden_size
    if mm_adapter_type == 'linear':
        return nn.Linear(mm_hidden_size, hidden_size)
    if mm_adapter_type == 'identity':
        return nn.Identity()
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', mm_adapter_type)
    if mlp_gelu_match:
        module_list = nn.ModuleList([nn.Linear(mm_hidden_size, hidden_size)])
        mlp_depth = int(mlp_gelu_match.group(1))
        for _ in range(1, mlp_depth):
            module_list.append(nn.GELU())
            module_list.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*module_list)
    raise NotImplementedError(f'Unsupported mm_adapter_type: {mm_adapter_type}')
