import os
from collections import OrderedDict
from typing import List

import torch
from transformers import Trainer
from transformers.trainer import PREFIX_CHECKPOINT_DIR


class LLaVATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.tune_backbone = kwargs.pop('tune_backbone')
        self.tune_vision_tower = kwargs.pop('tune_vision_tower')
        self.tune_mm_adapter = kwargs.pop('tune_mm_adapter')
        super().__init__(*args, **kwargs)
        # Only use cache in inference
        self.model.config.use_cache = False

    # Dont save optimizer, scheduler, rng_state when training ends.
    def _save_optimizer_and_scheduler(self, output_dir):
        if not self.control.should_training_stop:
            super()._save_optimizer_and_scheduler(output_dir)

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
        assert isinstance(state_dict, OrderedDict) and not state_dict
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
