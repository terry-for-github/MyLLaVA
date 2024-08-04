from copy import deepcopy
import json
from typing import Tuple

from torch.utils.data import Dataset

from .image_loader import ImageLoader


class LazyMMDialogDataset(Dataset):
    '''Lazy Multimodal Dialog Dataset'''
    def __init__(self, data_path: str,
                 image_loader: ImageLoader,
                 roles: Tuple[str, str] = ('human', 'gpt'),
                 conv_keys: Tuple[str, str] = ('from', 'value'),
                 data_keys: Tuple[str, str] = ('conversations', 'image')):
        super().__init__()
        self.data_path = data_path
        self.image_loader = image_loader
        self._human, self._gpt = roles
        _role, _content = conv_keys
        _dialog, _image = data_keys
        self.image_mark = image_loader.image_mark

        print('Loading json data from:', data_path)
        raw_list_data_dict = json.load(open(data_path, "r"))
        self.num_data = len(raw_list_data_dict)
        print('Load json file complete:', self.num_data, 'data')

        print(f'Keep only "{_dialog}" and "{_image}" columns')
        # If no 'image' column, set it to None
        self.list_data_dict = [{
            'dialog': {'role': data_dict[_dialog][_role],
                       'content': data_dict[_dialog][_content]},
            'image': data_dict[_image] if _image in data_dict else None
        } for data_dict in raw_list_data_dict]
        print('Filtering columns complete.')
        # Change the flag to True if you are running a new dataset
        # else set it to False to speed up the initialization
        self.check_dataset(False)

    def check_dataset(self, check_dataset: bool):
        '''
        Check the dataset.
        You should run this at least once to make sure the dataset is correct.
        After that you can set the check_dataset to False to speed up the initialization.
        We confirm that:
        1.  The image mark should only exist in the first message.
        #   (dialog stands for data_dict['dialog'])
        2.  The dialog should have even length.
        3.  dialog[i]['role'] is 'human' if i is even
        4.  dialog[i]['role'] is 'gpt' if i is odd
        5.  dialog[i]['content'] is always a string

        '''
        if not check_dataset:
            return
        for data_dict in self.list_data_dict:
            has_image = (data_dict['image'] is not None)
            # check for point 2.
            assert len(data_dict['dialog']) % 2 == 0
            for i, message in enumerate(data_dict['dialog']):
                # check for point 3 and 4
                if i % 2 == 0:
                    assert message['role'] == self._human
                else:
                    assert message['role'] == self._gpt
                # check for point 5
                assert isinstance(message['content'], str)
                # check for point 1
                if not has_image:
                    continue
                if i == 0:
                    assert self.image_mark in message['content']
                else:
                    assert self.image_mark not in message['content']

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx: int):
        '''load image and get dialog'''
        assert isinstance(idx, int)
        raw_data_dict = self.list_data_dict[idx]
        # Load image from the image_file name
        image = self.image_loader(raw_data_dict['image'])
        # Copy the dialog to avoid changing the original data
        dialog = deepcopy(raw_data_dict['dialog'])
        return {'image': image, 'dialog': dialog}
