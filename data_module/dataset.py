from copy import deepcopy
import json

from torch.utils.data import Dataset

from .image_loader import ImageLoader


class LazyMMDialogDataset(Dataset):
    '''Lazy Multimodal Dialog Dataset'''
    def __init__(self,
                 json_path: str,
                 image_loader: ImageLoader,
                 dialog_key: str,
                 image_key: str,
                 role_key: str,
                 content_key: str,
                 human_key: str,
                 gpt_key: str):
        super().__init__()
        self.json_path = json_path
        self.image_loader = image_loader

    def __len__(self) -> int:
        return len(self.list_data_dict)

    @property
    def lengths(self) -> List[int]:
        return self._lengths

        # If no 'image' column, set it to None
        self.list_data_dict = [{
            'dialog': [{'role': message[role_key], 'content': message[content_key]}
                       for message in data_dict[dialog_key]],
            'image': data_dict[image_key] if image_key in data_dict else None

    def _cal_lengths(self):
        print('Calculate the lengths of all dialogs.')
        self._lengths = []
        pattern = r"[\n\t\r!\"#$%&'()*+,\-./:;=?@[\]^_`{}~]"
        for data_dict in tqdm(self.list_data_dict, disable=tqdm_off):
            dialog = data_dict['dialog']
            length = 0
            for message in dialog:
                num_image = message['content'].count(self.image_mark)
                # use number of words to estimate the length
                length += len(message['content'].split())
                length += len(re.findall(pattern, message['content']))
                length += num_image * (self.vision_token_num - 1)
            self._lengths.append(length)
        print('Calculation Completed.')
        return self
        You should run this at least once to make sure the dataset is correct.
        This confirms that:
        1.  The image mark should only exist in the first message.
        #   (dialog stands for data_dict['dialog'])
        2.  The dialog should have even length (2 if is_plain).
        3.  dialog[i]['role'] is self._human if i is even
        4.  dialog[i]['role'] is self._gpt if i is odd
        5.  dialog[i]['content'] is always a string
        '''

        for data_dict in self.list_data_dict:
            has_image = (data_dict['image'] is not None)
            len_dialog = len(data_dict['dialog'])
            # check for point 2.
            assert (len_dialog == 2) if is_plain else (len_dialog % 2 == 0)
            for i, message in enumerate(data_dict['dialog']):
                # check for point 3 and 4
                assert message['role'] == (self.human_key if (i % 2 == 0) else self.gpt_key)
                # check for point 5
                assert isinstance(message['content'], str)
                # check for point 1
                has_mark = self.image_mark in message['content']
                assert has_mark == (i == 0 and has_image)
                if has_mark:
                    assert message['content'].count(self.image_mark) == 1

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
