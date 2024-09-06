import json
import os
from typing import Optional, TypedDict

from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from ..constants import IMAGE_MARK
from .image_loader import BaseImageLoader
from .template import TemplateApplier

_tqdm_off = os.environ.get('LOCAL_RANK', '-1') not in ['0', '-1']
messages_dict = TypedDict(
    'messages_dict',
    {
        'dialog': list[dict[str, str]],
        'image': Optional[str]
    }
)


class LazySingleImageAtFirstDialogDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_folder: str,
        image_loader: BaseImageLoader,
        image_mark: str,
        template_applier: TemplateApplier,
        is_plain_dataset: bool = False,
        check_dataset: bool = True,
        dialog_key: str = 'dialog',
        image_key: str = 'image',
        role_key: str = 'role',
        content_key: str = 'content',
        user_key: str = 'user',
        assistant_key: str = 'assistant',
    ):
        super().__init__()
        self.json_path = json_path
        self.image_folder = image_folder
        self.image_loader = image_loader
        self.image_mark = image_mark
        self.template_applier = template_applier
        self.num_vision_token = template_applier.num_vision_token
        self.is_plain_dataset = is_plain_dataset
        self.check_dataset = check_dataset
        self.dialog_key = dialog_key
        self.image_key = image_key
        self.role_key = role_key
        self.content_key = content_key
        self.user_key = user_key
        self.assistant_key = assistant_key
        self._prepare()

    def __len__(self) -> int:
        return len(self.list_data_dict)

    @property
    def lengths(self) -> list[int]:
        if not hasattr(self, '_lengths'):
            self._cal_lengths()
        return self._lengths

    def _prepare(self):
        print('Start preparing dataset.')
        (
            self._load_json_file()
                ._check_dataset()
                ._format_dataset()
        )
        print('Dataset preparing completed. Num data:', len(self))

    def _load_json_file(self) -> 'LazySingleImageAtFirstDialogDataset':
        print('Loading json file:', self.json_path)
        self.list_data_dict: list[messages_dict] = \
            json.load(open(self.json_path, "r"))
        print('Total num_data:', len(self.list_data_dict))
        return self

    def _format_dataset(self) -> 'LazySingleImageAtFirstDialogDataset':
        _dialog = self.dialog_key
        _image = self.image_key
        _role = self.role_key
        _content = self.content_key
        _user = self.user_key
        print(f'Keep only "{_dialog}" and "{_image}" columns.')
        # If no 'image' column, set it to None
        self.list_data_dict = [{
            'dialog': [dict(role='user' if message[_role] == _user else 'assistant',
                            content=message[_content].replace(self.image_mark, IMAGE_MARK))
                       for message in data_dict[_dialog]],
            'image': data_dict[_image] if _image in data_dict else None
        } for data_dict in tqdm(self.list_data_dict, disable=_tqdm_off)]
        return self

    def _check_dataset(self) -> 'LazySingleImageAtFirstDialogDataset':
        '''
        Check the dataset.
        You should run this at least once to make sure the dataset is correct.
        This confirms that:
        1.  The image mark should only exist in the first message.
        #   (dialog stands for data_dict['dialog'])
        2.  The dialog should have even length (2 if is_plain).
        3.  dialog[i]['role'] is user if i is even
        4.  dialog[i]['role'] is assistant if i is odd
        5.  dialog[i]['content'] is always a string
        '''
        if not self.check_dataset:
            return self
        is_plain_dataset = self.is_plain_dataset
        _user = self.user_key
        _assistant = self.assistant_key
        _dialog = self.dialog_key
        _image = self.image_key
        _role = self.role_key
        _content = self.content_key

        print('Start checking the dataset.')
        for data_dict in tqdm(self.list_data_dict, disable=_tqdm_off):
            has_image = (_image in data_dict)
            if is_plain_dataset:
                assert has_image
            len_dialog = len(data_dict[_dialog])
            # check for point 2.
            assert (len_dialog == 2) if is_plain_dataset else (len_dialog % 2 == 0)
            messages: list[dict[str, str]] = data_dict[_dialog]  # type: ignore
            for i, message in enumerate(messages):
                # check for point 3 and 4
                assert message[_role] == (_user if (i % 2 == 0) else _assistant)
                # check for point 5
                assert isinstance(message[_content], str)
                # check for point 1
                has_mark = self.image_mark in message[_content]
                assert has_mark == (i == 0 and has_image)
                if has_mark:
                    assert message[_content].count(self.image_mark) == 1
        print('Check completed.')
        return self

    def _cal_lengths(self):
        print('Use group_by_lengths == True Calculate the lengths of all dialogs.')
        self._lengths = []
        sep = " \n\t\r!\"#$%&'()*+,-./:;=?@[]^_`{}~"
        for data_dict in tqdm(self.list_data_dict, disable=_tqdm_off):
            dialog = data_dict['dialog']
            length = 0
            for message in dialog:
                # use number of words to estimate the length
                length += len(list(filter(lambda x: (x != ""),
                                          message['content'].split(sep))))
                length += message['content'].count(IMAGE_MARK)
            self._lengths.append(length)
        print('Calculation Completed.')
        return self

    def __getitem__(self, idx: int):
        dialog = self.list_data_dict[idx]['dialog']
        image_file = self.list_data_dict[idx]['image']
        image_path = (
            os.path.join(self.image_folder, image_file)
            if image_file is not None else None
        )
        input_tensor = self.template_applier.dialog_to_input(dialog)
        input_tensor['image_mask'] = torch.full(
            (self.num_vision_token,),
            image_path is not None,
            dtype=torch.bool
        )
        input_tensor['image'] = self.image_loader.load_image(image_path)
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
        # print('--------------------------------------------------------')
        # length = len(input_tensor['input_ids'])
        # for i in range(length):
        #     print(input_tensor['input_ids'][i].item(),
        #           input_tensor['attention_mask'][i].item(),
        #           input_tensor['labels'][i].item(),
        #           input_tensor['vision_token_pos'][i].item(),
        #           tokenizer.decode(input_tensor['input_ids'][i]), sep='\t')
        # print('--------------------------------------------------------')
        # raise ValueError('Stop here')
        return input_tensor
