import torch
import json


class LazySupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path
        print('Loading json data from:', data_path)
        self.list_data_dict = json.load(open(data_path, "r"))
        self.num_data = len(self.list_data_dict)
        print('Load json file complete: ', self.num_data, 'data')

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx: int):
        return self.list_data_dict[idx]
