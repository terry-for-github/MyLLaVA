from copy import deepcopy
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase

# from constants import IGNORE_INDEX
from .template import get_template

# Llama define this
IGNORE_INDEX = -100


class DataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, version: str, image_mark: str):
        self.tokenizer = deepcopy(tokenizer)
        self.tokenizer.add_special_tokens({'additional_special_tokens': [image_mark]})
        self.template = get_template(version)
        print(self.tokenizer)

    def __call__(self, list_data_dict: List[Dict[str, str]]):
        '''
        Input: list_data_dict = [{
                'image': image,
                'dialog': [{'role': role, 'content': content}, ...]
            }]
        Output: {'input_ids': input_ids, 'labels': labels, 'images': images}
        '''
        list_image, list_dialog = [], []
        for data_dict in list_data_dict:
            list_image.append(data_dict['image'])
            list_dialog.append(data_dict['dialog'])
        result_dict: Dict[str, torch.Tensor] = self.tokenizer.apply_chat_template(
            list_dialog,
            chat_template=self.template,
            tokenize=True,
            padding='longest',
            return_tensors='pt',
            return_dict=True,
            return_assistant_tokens_mask=True,
        )  # type: ignore
        batch = {}
        batch['input_ids'] = result_dict['input_ids']
        batch['attention_mask'] = result_dict['attention_mask']
        assistant_masks = torch.tensor(result_dict['assistant_masks'], dtype=torch.long)
        labels = torch.full_like(assistant_masks, IGNORE_INDEX)
        labels[assistant_masks == 1] = batch['input_ids'][assistant_masks == 1]
        batch['labels'] = labels
        batch['images'] = torch.stack(list_image)
        return batch


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    data_collector = DataCollator(tokenizer, 'vicuna', '<image>')
    print(data_collector.template)
    list_data_dict = [
        {'image': 'image1.jpg',
         'dialog': [{'role': 'human', 'content': '<image>\nHello!'},
                    {'role': 'gpt', 'content': "Hi!"},
                    {'role': 'human', 'content': 'How are you?'},
                    {'role': 'gpt', 'content': "I'm fine, thank you."}]},
        {'image': 'image2.jpg',
         'dialog': [{'role': 'human', 'content': 'Hello, how are you?'},
                    {'role': 'gpt', 'content': "I'm fine, thank you."},
                    {'role': 'human', 'content': 'Hello, how are you?'},
                    {'role': 'gpt', 'content': "I'm fine, thank you."}]}
    ]
    batch = data_collector(list_data_dict)
    print(batch['input_ids'])
    print(batch['attention_mask'])
    print(batch['labels'])
