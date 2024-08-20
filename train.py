import os
import glob
import builtins

import transformers

from llava_trainer import LLaVATrainer, SkipFinalSaveCallback
from arguments import ModelArguments, DataArguments, TrainingArguments
from model import get_causal_lm, LlavaLlamaForCausalLM
from data_module import get_dataset_and_data_collator
from constants import CACHE_DIR


def set_ignore_when_save(model: LlavaLlamaForCausalLM, model_args: ModelArguments):
    ignore_keys = []
    for param_name in model.state_dict().keys():
        # vision_tower params
        if 'vision_tower' in param_name:
            if not model_args.tune_vision_tower:
                ignore_keys.append(param_name)
        elif 'mm_adapter' in param_name:
            if not model_args.tune_mm_adapter:
                ignore_keys.append(param_name)
        elif not model_args.tune_backbone:
            ignore_keys.append(param_name)
    # Use the class attribute to ignore these keys
    LlavaLlamaForCausalLM._keys_to_ignore_on_save = ignore_keys  # type: ignore



def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)  # type: ignore
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print('[DEBUG]', 1, '===================================================================')
    print('[DEBUG]', 1, model_args)
    print('[DEBUG]', 1, data_args)
    print('[DEBUG]', 1, training_args)
    print('[DEBUG]', 1, '===================================================================')

    # You need to have sentencepiece installed to convert a slow tokenizer to a fast one.
    # --> pip install sentencepiece protobuf
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right"
    )

    causal_lm = get_causal_lm(model_args, training_args)
    set_ignore_when_save(causal_lm, model_args)

    train_dataset, data_collator = get_dataset_and_data_collator(
        tokenizer=tokenizer,
        data_args=data_args,
        vision_model_name=model_args.vision_tower,
        vision_token_num=causal_lm.get_vision_tower().num_patches,
        version=model_args.version
    )

    trainer = LLaVATrainer(
        model=causal_lm,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[SkipFinalSaveCallback()]
    )

    has_checkpoints = bool(glob.glob(os.path.join(training_args.output_dir, 'checkpoint-*')))
    trainer.train(resume_from_checkpoint=has_checkpoints)

    trainer.save_model(training_args.output_dir)
    # FIXME Update causal_llm.config
    # causal_lm.config.image_aspect_ratio = data_args.image_aspect_ratio


if __name__ == '__main__':
    from accelerate import PartialState
    is_local_main_process = PartialState().is_local_main_process
    builtins_print = builtins.print

    if 'LLAVA_DEBUG' not in os.environ:
        debug_level = 0
    else:
        debug_level = int(os.environ['LLAVA_DEBUG'])
    assert debug_level >= 0

    def custom_print(*args, **kwargs):
        now_debug_level = 0
        if isinstance(args[0], str) and args[0].startswith('[DEBUG]'):
            assert isinstance(args[1], int) and 1 <= args[1] <= 9
            now_debug_level = args[1]
        # hack accelerator.print
        if debug_level >= now_debug_level and is_local_main_process:
            builtins_print(*args, **kwargs)

    builtins.print = custom_print

    main()
