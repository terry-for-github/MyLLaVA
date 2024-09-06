from typing import Optional
from copy import deepcopy
import logging

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase, BatchEncoding

from ...constants import IGNORE_INDEX

from .template_strategy import TemplateStrategy
from .plain_strategy import PlainStrategy
from .vicuna_strategy import VicunaStrategy
from .llama3_strategy import Llama3Strategy


class TemplateApplier:
    def __init__(
        self,
        strategy: str,
        tokenizer: PreTrainedTokenizerBase,
        num_vision_token: int,
        is_training: bool = True
    ):
        self.strategy = strategy
        self.tokenizer = deepcopy(tokenizer)
        self.num_vision_token = num_vision_token
        self.is_training = is_training
        self.template_strategy = self._get_template_strategy()
        self.chat_template = self.template_strategy.get_chat_template()
        assert self.tokenizer.padding_side == 'right'
        assert self.tokenizer.pad_token_id is not None
        self.pad_token_id = self.tokenizer.pad_token_id

    def _get_template_strategy(self) -> TemplateStrategy:
        if self.strategy == 'plain':
            assert self.is_training
            return PlainStrategy(self.num_vision_token, self.tokenizer.pad_token)
        elif self.strategy == 'vicuna':
            return VicunaStrategy(self.num_vision_token, self.tokenizer.pad_token)
        elif self.strategy == 'llama3':
            return Llama3Strategy(self.num_vision_token, self.tokenizer.pad_token)
        else:
            raise ValueError(f'Unknown strategy: {self.strategy}')

    def _add_system_message(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str],
    ) -> list[dict[str, str]]:
        if messages[0]['role'] != 'system':
            if system_prompt is None:
                system_prompt = self.template_strategy.system_prompt
            if system_prompt is not None and system_prompt != '':
                messages = [dict(role='system', content=system_prompt)] + messages
        else:
            if system_prompt is not None:
                raise ValueError('system_prompt is not None but the first message is system')
            else:
                logging.warn('add_system_message is True but the system message is '
                             ' already in the first message. Skip adding.')
        return messages

    def dialog_to_input(
        self,
        messages: list[dict[str, str]],
        add_system_message: bool = True,
        system_prompt: Optional[str] = None
    ) -> dict[str, Tensor]:
        if len(messages) == 0:
            raise ValueError('Empty messages. Something wrong with the data.')
        if add_system_message:
            messages = self._add_system_message(messages, system_prompt)
        formated_messages = self.template_strategy.format_dialog(messages)
        templated_result: BatchEncoding = self.tokenizer.apply_chat_template(
            conversation=formated_messages,
            chat_template=self.chat_template,
            tokenize=True,
            truncation=True,
            return_tensors='pt',
            return_dict=True,
            return_attention_mask=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=not self.is_training
        )  # type: ignore
        input_tensor: dict[str, Tensor] = {
            'input_ids': templated_result['input_ids'].squeeze(0),  # type: ignore
            'attention_mask': templated_result['attention_mask'].squeeze(0),  # type: ignore
            'assistant_masks': torch.tensor(templated_result['assistant_masks'])
        }  # type: ignore
        input_tensor = self._drop_max_length_sample(input_tensor)
        if self.is_training:
            labels = input_tensor.pop('assistant_masks')
            labels[labels == 0] = IGNORE_INDEX
            labels[labels == 1] = input_tensor['input_ids'][labels == 1]
            input_tensor['labels'] = labels
        else:
            input_tensor.pop('assistant_masks')

        input_tensor['vision_token_pos'] = input_tensor['input_ids'] == self.pad_token_id

        return input_tensor  # input_ids, attention_mask, vision_token_pos, labels(optional)

    def _drop_max_length_sample(self, input_tensor: dict[str, Tensor]) -> dict[str, Tensor]:
        '''
        Drop the sample which length reach the model_max_length.
        They are very likely samples that were truncated after exceeding model_max_length.
        '''
        if len(input_tensor['input_ids']) < self.tokenizer.model_max_length:
            return input_tensor
        # Rare case
        assert len(input_tensor['input_ids']) == self.tokenizer.model_max_length
        input_tensor['assistant_masks'][:] = 0
        print('Dropped 1 sample exceed max_length', rank0_only=False)  # type: ignore
        return input_tensor
