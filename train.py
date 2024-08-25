import os
import glob
import builtins

import transformers

from llava_trainer import LLaVATrainer
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


def set_logger():
    import logging
    from datetime import datetime
    from transformers import logging as transformers_logging
    from deepspeed import logger as deepspeed_logger

    transformers_logging.set_verbosity_info()
    deepspeed_logger.setLevel(logging.INFO)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/log_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    transformers_logger = logging.getLogger('transformers')
    for handler in transformers_logger.handlers[:]:
        transformers_logger.removeHandler(handler)
    for handlder in deepspeed_logger.handlers[:]:
        deepspeed_logger.removeHandler(handlder)
    transformers_logger.propagate = False
    deepspeed_logger.propagate = False

    info_handler = logging.FileHandler(os.path.join(log_dir, 'info.log'))
    info_handler.setLevel(logging.INFO)

    warning_handler = logging.FileHandler(os.path.join(log_dir, 'warn.log'))
    warning_handler.setLevel(logging.WARNING)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    info_handler.setFormatter(formatter)
    warning_handler.setFormatter(formatter)

    transformers_logger.addHandler(info_handler)
    transformers_logger.addHandler(warning_handler)
    deepspeed_logger.addHandler(info_handler)
    deepspeed_logger.addHandler(warning_handler)


def get_tokenizer(model_args: ModelArguments):
    # You need to have sentencepiece installed to convert a slow tokenizer to a fast one.
    # --> pip install sentencepiece protobuf
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=CACHE_DIR,
        model_max_length=model_args.model_max_length,
        padding_side="right"
    )
    if 'llama-3' in model_args.model_name_or_path.lower():
        # <|finetune_right_pad_id|> or <|reserved_special_token_2|>
        tokenizer.pad_token_id = 128004  # type: ignore
    return tokenizer


def get_num_vision_token(model) -> int:
    vision_tower = model.get_vision_tower()
    mm_adapter = model.get_mm_adapter()
    dummy_feature = vision_tower.dummy_feature
    from transformers.integrations import is_deepspeed_zero3_enabled
    if is_deepspeed_zero3_enabled():
        import deepspeed
        gather_params = deepspeed.zero.GatheredParameters
        with gather_params(mm_adapter.parameters(), modifier_rank=0):
            dummy_output = mm_adapter(dummy_feature)
    else:
        dummy_output = mm_adapter(dummy_feature)
    return dummy_output.shape[1]


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

    tokenizer = get_tokenizer(model_args)

    causal_lm = get_causal_lm(model_args, training_args)
    set_ignore_when_save(causal_lm, model_args)

    num_vision_token = get_num_vision_token(causal_lm)

    train_dataset, data_collator = get_dataset_and_data_collator(
        tokenizer=tokenizer,
        data_args=data_args,
        vision_tower=model_args.vision_tower,
        num_vision_token=num_vision_token,
        version=model_args.version
    )

    trainer = LLaVATrainer(
        model=causal_lm,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    has_checkpoints = bool(glob.glob(os.path.join(training_args.output_dir, 'checkpoint-*')))
    trainer.train(resume_from_checkpoint=has_checkpoints)

    trainer.save_state()
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
        if len(args) == 0:
            builtins_print(**kwargs)
            return
        now_debug_level = 0
        if isinstance(args[0], str) and args[0].startswith('[DEBUG]'):
            assert isinstance(args[1], int) and 1 <= args[1] <= 9
            now_debug_level = args[1]
        # hack accelerator.print
        if debug_level >= now_debug_level and is_local_main_process:
            builtins_print(*args, **kwargs)

    if is_local_main_process:
        set_logger()

    builtins.print = custom_print

    main()
