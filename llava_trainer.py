from typing import List, Optional
import random

from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer_utils import has_length


class LengthGroupedRandomSampler(Sampler):
    def __init__(
        self,
        lengths: List[int],
        per_device_batch_size: int,
        world_size: int,
        gradient_acc_steps: int
    ):
        self.per_device_batch_size = per_device_batch_size
        self.step_batch_size = per_device_batch_size * world_size
        self.world_batch_size = self.step_batch_size * gradient_acc_steps
        self.mega_batch_size = self.world_batch_size * 4
        self.num_batches = len(lengths) // self.per_device_batch_size

        self.list_indices = list(range(len(lengths)))
        random.shuffle(self.list_indices)
        for i in range(0, len(lengths), self.mega_batch_size):
            if i + self.mega_batch_size > len(lengths):
                break
            self.list_indices[i:i+self.mega_batch_size] = sorted(
                self.list_indices[i:i+self.mega_batch_size],
                key=lambda i: lengths[i],
                reverse=True
            )
        # Shuffle all the world_batches.
        # Otherwise, the training will be stuck in the first few batches.
        for i in range(0, len(lengths), self.world_batch_size*2):
            if i + self.world_batch_size*2 > len(lengths):
                break
            self.list_indices[i:i+self.world_batch_size*2] = random.sample(
                self.list_indices[i:i+self.world_batch_size*2],
                k=self.world_batch_size*2
            )
        self.batch_indices = [self.list_indices[i:i+self.step_batch_size]
                              for i in range(0, len(lengths), self.step_batch_size)
                              if i+self.step_batch_size <= len(lengths)]
        random.shuffle(self.batch_indices)

    def __len__(self):
        return len(self.batch_indices) * self.step_batch_size

    def __iter__(self):
        for batch_index in self.batch_indices:
            yield from batch_index


class LLaVATrainer(Trainer):

    def train(self, *args, **kwargs):
        change_back = False
        if getattr(self.model.config, 'use_cache', False):
            self.model.config.use_cache = False
            change_back = True
        super().train(*args, **kwargs)
        if change_back:
            self.model.config.use_cache = True

    # Overload the Trainer method
    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if not self.args.group_by_length or not hasattr(self.train_dataset, 'lengths'):
            return super()._get_train_sampler()
        return LengthGroupedRandomSampler(
            lengths=self.train_dataset.lengths,  # type: ignore
            per_device_batch_size=self.args.per_device_train_batch_size,
            world_size=self.args.world_size,
            gradient_acc_steps=self.args.gradient_accumulation_steps
        )
