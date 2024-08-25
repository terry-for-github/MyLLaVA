import os
import re
from copy import deepcopy
from typing import Dict, Optional

import torch
from transformers import PreTrainedTokenizerBase

from constants import IGNORE_INDEX, IMAGE_MARK

from .template import template_dict


class DataCollatorForSingleImageAtFirstDialog:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 version: str):
        # huggingface/tokenizers: The current process just got forked, after parallelism has
        # already been used. Disabling parallelism to avoid deadlocks...
        # To disable this warning, you can either:
        #     - Avoid using `tokenizers` before the fork if possible
        #     - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = deepcopy(tokenizer)
        self.is_plain = version == 'plain'
        self._init_from_tokenizer(version)

    def _init_from_tokenizer(self, version):
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': [IMAGE_MARK]  # type: ignore
        })
        template = template_dict[version].get_template()
        if template:
            self.tokenizer.chat_template = template

        assert self.tokenizer.padding_side in ['right', 'left']
        self.pad_func = ((lambda x, y: torch.cat([x, y]))
                         if self.tokenizer.padding_side == 'right'
                         else (lambda x, y: torch.cat([y, x])))
        assert self.tokenizer.pad_token_id is not None
        self.pad_token_id = self.tokenizer.pad_token_id
        self.image_mark_id = self.tokenizer.additional_special_tokens_ids[0]

    def _deal_with_plain_template(self, list_dialog):
        if not self.is_plain:
            return
        for dialog in list_dialog:
            result = re.findall(rf'{IMAGE_MARK}', dialog[0]['content'])
            dialog[0]['content'] = ''.join(result)

    def __call__(self, list_data_dict: list) -> Dict[str, Optional[torch.Tensor]]:
        '''
        Input: list_data_dict = [{
                'image': image,
                'dialog': [{'role': role, 'content': content}, ...]
            }]
        Output: {'input_ids': input_ids, 'labels': labels,
                 'vision_token_pos': vision_token_pos,
                 'attention_mask': attention_mask, 'images': images}
        '''
        list_image = [data_dict['image'] for data_dict in list_data_dict]
        list_dialog = [data_dict['dialog'] for data_dict in list_data_dict]
        self._deal_with_plain_template(list_dialog)
        input_dict: Dict[str, torch.Tensor] = self.tokenizer.apply_chat_template(
            list_dialog,
            tokenize=True,
            truncation=True,
            padding='longest',  # type: ignore
            return_tensors='pt',
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_dict['assistant_masks'] = torch.tensor(input_dict['assistant_masks'])

        self._drop_max_length_sample(input_dict, list_image)

        labels = input_dict.pop('assistant_masks')
        labels[labels == 0] = IGNORE_INDEX
        labels[labels == 1] = input_dict['input_ids'][labels == 1]
        input_dict['labels'] = labels
        vision_token_pos = input_dict['input_ids'] == self.image_mark_id
        input_dict['input_ids'][vision_token_pos] = self.pad_token_id
        input_dict['vision_token_pos'] = vision_token_pos

        batch_data_dict: Dict[str, Optional[torch.Tensor]] = input_dict  # type: ignore
        not_none_list_image = [image for image in list_image if image is not None]
        if len(not_none_list_image) == 0:
            batch_data_dict['images'] = None
            batch_data_dict['vision_token_pos'] = None
        else:
            batch_data_dict['images'] = torch.stack([
                image for image in list_image if image is not None
            ])
        # verify group_by_length
        # print([torch.sum(mask == 1).item() for mask in input_dict['attention_mask']])
        return batch_data_dict

    def _drop_max_length_sample(self, input_dict, list_image):
        '''
        Drop the sample which length reach the model_max_length.
        They are very likely samples that were truncated after exceeding model_max_length.
        '''
        batch_size = len(list_image)
        drop_num = 0
        for i in range(batch_size):
            length = torch.sum(input_dict['attention_mask'][i] == 1).item()
            if length < self.tokenizer.model_max_length:
                continue
            drop_num += 1
            input_ids = input_dict['input_ids'][i]
            attention_mask = input_dict['attention_mask'][i]
            assistent_mask = input_dict['assistant_masks'][i]

            input_ids[input_ids == self.image_mark_id] = self.pad_token_id
            attention_mask[attention_mask == 1] = 0
            assistent_mask[assistent_mask == 1] = 0
            list_image[i] = None
        if drop_num > 0:
            print(f'Dropped {drop_num} samples exceed max_length')

    def _deal_with_overflow(self, input_dict, list_image):
        '''deprecated'''
        mapping_list = input_dict.pop('overflow_to_sample_mapping').tolist()
        if len(mapping_list) == len(list_image):
            return
        idx = 0
        last_idx = -1
        num_overflow = 0
        total_overflow = len(mapping_list) - len(list_image)
        while idx < len(mapping_list):
            if mapping_list[idx] != last_idx:
                last_idx = mapping_list[idx]
                idx += 1
            num_overflow += 1
            del list_image[idx]
            for v in input_dict.values():
                # delete the idx-th sample
                v = torch.cat([v[:idx], v[idx+1:]])

            input_ids = input_dict['input_ids'][idx - 1]
            attention_mask = input_dict['attention_mask'][idx - 1]
            assistent_mask = input_dict['assistant_masks'][idx - 1]

            input_ids[input_ids == self.image_mark_id] = self.pad_token_id
            attention_mask[attention_mask == 1] = 0
            assistent_mask[assistent_mask == 1] = 0
            list_image[idx - 1] = None
        assert num_overflow == total_overflow
        print(f'{total_overflow} samples exceed max_length. Filtered')
