import re
from copy import deepcopy
from typing import Dict

import torch
from transformers import PreTrainedTokenizerBase

from ..constants import IGNORE_INDEX, IMAGE_MARK
from .template import TemplateFactory


class DataCollatorForSingleImageAtFirstDialog:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 version: str):
        self.tokenizer = deepcopy(tokenizer)
        self.version = version
        self._init_from_tokenizer()

    def _init_from_tokenizer(self):
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [IMAGE_MARK]  # type: ignore
        })

        assert self.tokenizer.padding_side in ['right']
        assert self.tokenizer.pad_token_id is not None
        self.pad_token_id = self.tokenizer.pad_token_id
        self.image_mark_id = self.tokenizer.additional_special_tokens_ids[0]

    def _deal_with_plain_template(self, list_dialog):
        if self.version == 'plain':
            return
        for dialog in list_dialog:
            result = re.findall(rf'{IMAGE_MARK}', dialog[0]['content'])
            dialog[0]['content'] = ''.join(result)

    def __call__(self, list_data_dict: list) -> Dict[str, torch.Tensor]:
        '''
        Input: list_data_dict = [{
                'image': image,
                'dialog': [{'role': role, 'content': content}, ...]
                'image_mask': tensor.bool
            }]
        Output: {'input_ids': input_ids, 'labels': labels,
                 'vision_token_pos': vision_token_pos,
                 'attention_mask': attention_mask, 'images': images}
        '''
        template = TemplateFactory.create_template(self.version)
        list_image = [data_dict['image'] for data_dict in list_data_dict]
        list_dialog = [template.add_default_system_message(data_dict['dialog'])
                       for data_dict in list_data_dict]
        list_image_mask = [data_dict['image_mask'] for data_dict in list_data_dict]
        self._deal_with_plain_template(list_dialog)
        input_dict: Dict[str, torch.Tensor] = self.tokenizer.apply_chat_template(
            list_dialog,
            chat_template=template.get_template(),
            tokenize=True,
            truncation=True,
            padding='longest',  # type: ignore
            return_tensors='pt',
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_dict['assistant_masks'] = torch.tensor(input_dict['assistant_masks'])

        self._drop_max_length_sample(input_dict, len(list_dialog))

        labels = input_dict.pop('assistant_masks')
        labels[labels == 0] = IGNORE_INDEX
        labels[labels == 1] = input_dict['input_ids'][labels == 1]
        input_dict['labels'] = labels
        vision_token_pos = input_dict['input_ids'] == self.image_mark_id
        input_dict['input_ids'][vision_token_pos] = self.pad_token_id
        input_dict['vision_token_pos'] = vision_token_pos

        input_dict['images'] = torch.stack(list_image)
        input_dict['image_masks'] = torch.stack(list_image_mask)

        # verify group_by_length
        # print([torch.sum(mask == 1).item() for mask in input_dict['attention_mask']])
        return input_dict

    def _drop_max_length_sample(self, input_dict, batch_size):
        '''
        Drop the sample which length reach the model_max_length.
        They are very likely samples that were truncated after exceeding model_max_length.
        '''
        drop_num = 0
        for i in range(batch_size):
            length = torch.sum(input_dict['attention_mask'][i] == 1).item()
            if length < self.tokenizer.model_max_length:
                continue
            assert length == self.tokenizer.model_max_length
            drop_num += 1
            input_dict['attention_mask'][i][:] = 0
            input_dict['assistant_masks'][i][:] = 0
        if drop_num > 0:
            print(f'Dropped {drop_num} samples exceed max_length')
