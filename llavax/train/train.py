import os
import glob

import torch
import transformers
import deepspeed
from transformers import BitsAndBytesConfig
from transformers.integrations import is_deepspeed_zero3_enabled

from .llava_trainer import LLaVATrainer
from .train_factory import TrainFactory
from ..arguments import ModelArguments, DataArguments, TrainingArguments
from ..model import LlavaLlamaForCausalLM


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


def get_num_vision_token(model) -> int:
    vision_tower = model.get_vision_tower()
    mm_adapter = model.get_mm_adapter()
    dummy_feature = vision_tower.dummy_feature
    with deepspeed.zero.GatheredParameters(mm_adapter.parameters(),
                                           enabled=is_deepspeed_zero3_enabled()):
        dummy_output = mm_adapter(dummy_feature)
    return dummy_output.shape[1]


def get_bnb_args(training_args: TrainingArguments, compute_dtype: torch.dtype) -> dict:
    '''Get BitsAndBytes args for from_pretrained method.'''

    if training_args.bits not in [4, 8]:
        return {}

    return dict(
        device_map={"": training_args.device},
        load_in_4bit=(training_args.bits == 4),
        load_in_8bit=(training_args.bits == 8),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=(training_args.bits == 4),
            load_in_8bit=(training_args.bits == 8),
            llm_int8_skip_modules=["mm_adapter"],
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
        )
    )


def get_compute_dtype(training_args) -> torch.dtype:
    if training_args.fp16 and training_args.bf16:
        raise ValueError("Only one of fp16 and bf16 can be set to True")
    elif training_args.fp16:
        return torch.float16
    elif training_args.bf16:
        return torch.bfloat16
    else:
        return torch.float32


def build_and_train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)  # type: ignore
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    factory = TrainFactory(model_args, data_args, training_args)
    print('[DEBUG]', 1, '===================================================================')
    print('[DEBUG]', 1, model_args)
    print('[DEBUG]', 1, data_args)
    print('[DEBUG]', 1, training_args)
    print('[DEBUG]', 1, '===================================================================')

    tokenizer = factory.create_tokenizer()

    compute_dtype = get_compute_dtype(training_args=training_args)

    bnb_args = get_bnb_args(training_args, compute_dtype)

    causal_lm = factory.create_causal_lm(compute_dtype, bnb_args, tokenizer.pad_token_id)
    set_ignore_when_save(causal_lm, model_args)

    num_vision_token = get_num_vision_token(causal_lm)

    train_dataset = factory.create_dataset(num_vision_token)

    data_collator = factory.create_data_collator(tokenizer)

    trainer = LLaVATrainer(
        model=causal_lm,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        skip_save_after_last_step=training_args.skip_save_after_last_step
    )

    has_checkpoints = bool(glob.glob(os.path.join(training_args.output_dir, 'checkpoint-*')))
    trainer.train(resume_from_checkpoint=has_checkpoints)

    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    # FIXME Update causal_llm.config
    # causal_lm.config.image_aspect_ratio = data_args.image_aspect_ratio
