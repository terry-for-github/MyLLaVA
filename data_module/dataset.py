import json
import os
import re
from typing import List

from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from constants import IMAGE_MARK

tqdm_off = os.environ.get('LOCAL_RANK', '-1') not in ['0', '-1']


class LazySingleImageAtFirstDialogDataset(Dataset):
    def __init__(self,
                 data_args,
                 image_loader,
                 vision_token_num,
                 model_max_length):
        super().__init__()
        self.data_args = data_args
        self.image_loader = image_loader
        self.image_mark = data_args.image_mark
        self.vision_token_num = vision_token_num
        self.model_max_length = model_max_length
        self._prepare()

    def __len__(self) -> int:
        return len(self.list_data_dict)

    @property
    def lengths(self) -> List[int]:
        if not hasattr(self, '_lengths'):
            self._cal_lengths()
        return self._lengths

    def _prepare(self):
        print('Start preparing dataset.')
        (
            self._load_json_file()
                ._filter_columns()
                ._check_dataset()
                ._duplicate_image_mark()
        )
        print('Dataset preparing completed. Num data:', len(self))

    def _load_json_file(self):
        json_path = self.data_args.json_path
        print('Loading json file:', json_path)
        self.list_data_dict = json.load(open(json_path, "r"))
        print('Total num_data:', len(self.list_data_dict))
        return self

    def _filter_columns(self):
        dialog_key = self.data_args.dialog_key
        image_key = self.data_args.image_key
        role_key = self.data_args.role_key
        content_key = self.data_args.content_key
        print(f'Keep only "{dialog_key}" and "{image_key}" columns.')
        # If no 'image' column, set it to None
        self.list_data_dict = [{
            'dialog': [{'role': message[role_key], 'content': message[content_key]}
                       for message in data_dict[dialog_key]],
            'image': data_dict[image_key] if image_key in data_dict else None
        } for data_dict in tqdm(self.list_data_dict, disable=tqdm_off)]
        return self

    def _check_dataset(self):
        '''
        Check the dataset.
        You should run this at least once to make sure the dataset is correct.
        This confirms that:
        1.  The image mark should only exist in the first message.
        #   (dialog stands for data_dict['dialog'])
        2.  The dialog should have even length (2 if is_plain).
        3.  dialog[i]['role'] is human if i is even
        4.  dialog[i]['role'] is gpt if i is odd
        5.  dialog[i]['content'] is always a string
        '''
        if not self.data_args.check_dataset:
            return self
        is_plain_dataset = self.data_args.is_plain_dataset
        _human = self.data_args.human_key
        _gpt = self.data_args.gpt_key

        print('Start checking the dataset.')
        for data_dict in tqdm(self.list_data_dict, disable=tqdm_off):
            has_image = (data_dict['image'] is not None)
            if is_plain_dataset:
                assert has_image
            len_dialog = len(data_dict['dialog'])
            # check for point 2.
            assert (len_dialog == 2) if is_plain_dataset else (len_dialog % 2 == 0)
            for i, message in enumerate(data_dict['dialog']):
                # check for point 3 and 4
                assert message['role'] == (_human if (i % 2 == 0) else _gpt)
                # check for point 5
                assert isinstance(message['content'], str)
                # check for point 1
                has_mark = self.image_mark in message['content']
                assert has_mark == (i == 0 and has_image)
                if has_mark:
                    assert message['content'].count(self.image_mark) == 1
        print('Check completed.')
        return self

    def _duplicate_image_mark(self):
        # image_mark is a special token. It will always be tokenized to one single input_id.
        # So we can replace image_mark with vision_token_num*image_mark before it turns to
        # input_ids. Then we can easily replace input_embeds with image_feature in forward()
        print('Duplicate image_mark in all dialogs.')
        for data_dict in tqdm(self.list_data_dict, disable=tqdm_off):
            first_message = data_dict['dialog'][0]
            first_message['content'] = first_message['content'].replace(
                self.image_mark, self.vision_token_num*IMAGE_MARK
            )
        print('Duplication Completed.')
        return self

    def _cal_lengths(self):
        print('Use group_by_lengths == True Calculate the lengths of all dialogs.')
        self._lengths = []
        pattern = r"[\n\t\r!\"#$%&'()*+,\-./:;=?@[\]^_`{}~]"
        for data_dict in tqdm(self.list_data_dict, disable=tqdm_off):
            dialog = data_dict['dialog']
            length = 0
            for message in dialog:
                # use number of words to estimate the length
                length += len(message['content'].split())
                length += message['content'].count(IMAGE_MARK)
                length += len(re.findall(pattern, message['content']))
            self._lengths.append(length)
        print('Calculation Completed.')
        return self

    def __getitem__(self, idx: int):
        data_dict = self.list_data_dict[idx]
        image_path = data_dict['image']
        data_dict['image'] = self.image_loader(image_path)
        data_dict['image_mask'] = (
            torch.zeros(self.vision_token_num, dtype=torch.bool)
            if image_path is None else
            torch.ones(self.vision_token_num, dtype=torch.bool)
        )
        return data_dict
