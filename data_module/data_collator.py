from copy import deepcopy
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase

from .template import get_template
from constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


class DataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, version: str, image_mark: str):
        self.tokenizer = deepcopy(tokenizer)
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [image_mark]  # type: ignore
        })
        self._image_token_id = self.tokenizer.additional_special_tokens_ids[0]
        self.template = get_template(version)

    def _collate_dialog(self, list_dialog: List[Dict[str, str]]):
        dialog_batch: Dict[str, torch.Tensor] = self.tokenizer.apply_chat_template(
            list_dialog,
            chat_template=self.template,
            tokenize=True,
            padding='longest',  # type: ignore
            return_tensors='pt',
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        image_token_pos = torch.where(dialog_batch['input_ids'] == self._image_token_id)
        dialog_batch['input_ids'][image_token_pos] = IMAGE_TOKEN_INDEX
        labels = torch.tensor(dialog_batch.pop('assistant_masks'), dtype=torch.long)
        labels[labels == 0] = IGNORE_INDEX
        labels[labels == 1] = dialog_batch['input_ids'][labels == 1]
        dialog_batch['labels'] = labels
        dialog_batch['attention_mask'] = dialog_batch['attention_mask'].bool()
        return dialog_batch

    def _collate_image(self, list_image: List[torch.Tensor]):
        # In case of no image
        if len(list_image) == 0:
            return {'images': None}
        return {'images': torch.stack(list_image)}

    def __call__(self, list_data_dict: List[Dict[str, str]]):
        '''
        Input: list_data_dict = [{
                'image': image,
                'dialog': [{'role': role, 'content': content}, ...]
            }]
        Output: {'input_ids': input_ids, 'labels': labels,
                 'attention_mask': attention_mask, 'images': images}
        '''
        list_image, list_dialog = [], []
        for data_dict in list_data_dict:
            # We dont wanna encode some zero images, this is a waste of calculation resource.
            # So we only append the image if it is not None,
            # this makes the length of list_image differ from list_dialog.
            # However, we can still use the image_mark to check which dialog has image later.
            # So in summary:
            # We dont append a empty image if it is None, this save a little bit of
            # calculation resource, otherwise it will always encode some empty images which is
            # meaningless for us.
            if data_dict['image'] is not None:
                list_image.append(data_dict['image'])
            list_dialog.append(data_dict['dialog'])
        batch = {}
        batch.update(self._collate_dialog(list_dialog))
        batch.update(self._collate_image(list_image))
        return batch
