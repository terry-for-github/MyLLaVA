from copy import deepcopy
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase

from .template import template_dict
from constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


class DataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, version: str, image_mark: str,
                 vision_token_num: int):
        self.tokenizer = deepcopy(tokenizer)
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [image_mark]  # type: ignore
        })
        self.image_token_id = self.tokenizer.additional_special_tokens_ids[0]
        self.template = self._get_template(version)
        self.vision_token_num = vision_token_num
        self.max_length = self.tokenizer.model_max_length

    def _get_template(self, version: str):
        return template_dict[version].get_template()

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
        image_token_pos = torch.where(dialog_batch['input_ids'] == self.image_token_id)
        dialog_batch['input_ids'][image_token_pos] = IMAGE_TOKEN_INDEX
        labels = torch.tensor(dialog_batch.pop('assistant_masks'), dtype=torch.long)
        labels[labels == 0] = IGNORE_INDEX
        labels[labels == 1] = dialog_batch['input_ids'][labels == 1]
        dialog_batch['labels'] = labels
        dialog_batch['attention_mask'] = dialog_batch['attention_mask'].bool()
        self._deal_long_dialog(dialog_batch)
        return dialog_batch

    def _deal_long_dialog(self, dialog_batch: Dict[str, torch.Tensor]):
        batch_size = dialog_batch['attention_mask'].size(dim=0)
        trunc_len = 0
        for i in range(batch_size):
            has_image = torch.any(dialog_batch['input_ids'][i] == IMAGE_TOKEN_INDEX)
            token_length = torch.sum(dialog_batch['attention_mask'][i]).item()
            token_length += (self.vision_token_num - 1) if has_image else 0
            if token_length <= self.max_length:
                continue
            trunc_len = max(trunc_len, token_length - self.max_length)
            print(f'[WARN] {token_length} > max_length: {self.max_length} in dialog.'
                  f'Set labels to {IGNORE_INDEX}.\n {dialog_batch["input_ids"][i]}')
            dialog_batch['labels'][i] = IGNORE_INDEX
        # We truncate the dialog here because it is more efficient to do it here
        # than in the model forward function. (Maybe. I'm not sure.)
        # (multi-core cpu here vs. gpu in `forward` function.)
        # (At least save some gpu resources)
        if trunc_len > 0:
            print(f'[WARN] Truncate {trunc_len} tokens in dialog.')
            dialog_batch['input_ids'] = dialog_batch['input_ids'][:, :-trunc_len]
            dialog_batch['labels'] = dialog_batch['labels'][:, :-trunc_len]
            dialog_batch['attention_mask'] = dialog_batch['attention_mask'][:, :-trunc_len]

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
