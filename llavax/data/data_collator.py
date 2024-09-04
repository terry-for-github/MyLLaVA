import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ..constants import IGNORE_INDEX


class DataCollatorForSingleImageAtFirstDialog:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, list_input_tensor: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        '''
        Input: list_input_tensor = [{
                'input_ids': tensor.long,
                'attention_mask': tensor.long,
                'vision_token_pos': tensor.bool,
                'labels': tensor.long,
                'image': tensor.float,
                'image_mask': tensor.bool
            }]
        Output: {'input_ids': input_ids, 'labels': labels,
                 'vision_token_pos': vision_token_pos,
                 'attention_mask': attention_mask, 'images': images}
        '''
        list_input_ids = [input_tensor['input_ids'] for input_tensor in list_input_tensor]
        list_attention_mask = [input_tensor['attention_mask']
                               for input_tensor in list_input_tensor]
        list_vision_token_pos = [input_tensor['vision_token_pos']
                                 for input_tensor in list_input_tensor]
        list_labels = [input_tensor['labels'] for input_tensor in list_input_tensor]
        list_image = [input_tensor['image'] for input_tensor in list_input_tensor]
        list_image_mask = [input_tensor['image_mask'] for input_tensor in list_input_tensor]

        return {
            'input_ids': pad_sequence(list_input_ids, True, self.pad_token_id),
            'attention_mask': pad_sequence(list_attention_mask, True, 0),
            'vision_token_pos': pad_sequence(list_vision_token_pos, True, False),
            'labels': pad_sequence(list_labels, True, IGNORE_INDEX),
            'images': torch.stack(list_image),
            'image_masks': torch.stack(list_image_mask)
        }
