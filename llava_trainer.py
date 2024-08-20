import os
from collections import OrderedDict
from typing import List

import torch
from transformers import Trainer
from transformers.trainer import PREFIX_CHECKPOINT_DIR


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

    def _save_rng_state(self, output_dir):
        if not self.control.should_training_stop:
            super()._save_rng_state(output_dir)

    # Hack the source code to delete all the checkpoints when training ends.
    # This function will be called inside the _rotate_checkpoints function.
    # It skips the check `return if save_total_limit == 0`
    # Finally it will delete all the `checkpoint-xxx` directories.
    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        if self.control.should_training_stop:
            self.args.save_total_limit = 0
        return super()._sorted_checkpoints(output_dir, checkpoint_prefix, use_mtime)

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        tmp_save_total_limit = self.args.save_total_limit
        super()._rotate_checkpoints(use_mtime, output_dir)
        self.args.save_total_limit = tmp_save_total_limit

    def _save(self, output_dir, state_dict):
        if not self.control.should_training_stop:
            super()._save(output_dir, state_dict)
            return
        # restore the `use_cache` config
        self.model.config.use_cache = True
        # Save only the required state_dict
        assert isinstance(state_dict, OrderedDict) and state_dict, type(state_dict)
        if output_dir is None:
            output_dir = self.args.output_dir
        # Stage 1: only tune mm_adapter
        if (
            not self.tune_backbone and
            not self.tune_vision_tower and
            self.tune_mm_adapter
        ):
            mm_adapter_state_dict = {}
            # remove 'model.mm_adapter' prefix
            for key in list(state_dict.keys()):
                if 'mm_adapter' not in key:
                    continue
                no_prefix_key = key.split('model.mm_adapter.')[1]
                mm_adapter_state_dict[no_prefix_key] = state_dict.pop(key)
            # Save the config file only for recording.
            self.model.config.save_pretrained(output_dir)
            torch.save(mm_adapter_state_dict, os.path.join(output_dir, 'mm_adapter.bin'))
        # Stage 2: tune mm_adapter and backbone
        elif (
            self.tune_backbone and
            not self.tune_vision_tower and
            self.tune_mm_adapter
        ):
            for key in list(state_dict.keys()):
                if 'vision_tower' in key:
                    state_dict.pop(key)
            super()._save(output_dir, state_dict)
        else:
            raise NotImplementedError('Only support stage 1 and 2.')
