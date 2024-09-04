import os
import glob

import torch
import deepspeed
from transformers import BitsAndBytesConfig
from transformers.integrations import is_deepspeed_zero3_enabled

from .llava_trainer import LLaVATrainer
from .builder import (
    build_tokenizer,
    build_template_applier,
    build_image_loader,
    build_model_config,
    build_causal_lm,
    build_dataset,
    build_datacollator,
)
from .arguments import ModelArguments, DataArguments, TrainingArguments
from .data import BaseImageLoader
from .model import LlavaLlamaForCausalLM


def get_num_vision_token(
    causal_lm: LlavaLlamaForCausalLM,
    image_loader: BaseImageLoader
) -> int:
    '''Get the number of vision tokens.'''
    image_tensor = image_loader(None).unsqueeze(0)
    vision_tower = causal_lm.get_vision_tower()
    mm_adapter = causal_lm.get_mm_adapter()

    vision_tower_training = vision_tower.training
    mm_adapter_training = mm_adapter.training
    vision_tower.eval()
    mm_adapter.eval()

    image_tensor = image_tensor.to(dtype=vision_tower.dtype, device=vision_tower.device)
    with deepspeed.zero.GatheredParameters(
        list(vision_tower.parameters()) + list(mm_adapter.parameters()),
        enabled=is_deepspeed_zero3_enabled()
    ):
        with torch.no_grad():
            # B x N x C
            dummy_output = causal_lm.model.encode_images(image_tensor)
    vision_tower.train(vision_tower_training)
    mm_adapter.train(mm_adapter_training)
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


def build_and_train(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments
):
    tokenizer = build_tokenizer(model_args.model_name_or_path, model_args.model_max_length)

    compute_dtype = get_compute_dtype(training_args=training_args)

    bnb_kwargs = get_bnb_args(training_args, compute_dtype)

    model_config = build_model_config(
        model_name_or_path=model_args.model_name_or_path,
        vision_tower=model_args.vision_tower,
        vision_select_layer=model_args.vision_select_layer,
        mm_adapter=model_args.mm_adapter,
        pad_token_id=tokenizer.pad_token_id,
        patch_merge_type=model_args.patch_merge_type,
    )

    print('Building the causal language model...')
    causal_lm = build_causal_lm(
        model_name_or_path=model_args.model_name_or_path,
        model_config=model_config,
        compute_dtype=compute_dtype,
        attn_implementation=training_args.attn_impl,
        pretrained_mm_adapter_path=model_args.pretrained_mm_adapter_path,
        bnb_kwargs=bnb_kwargs,
        tune_backbone=model_args.tune_backbone,
        tune_vision_tower=model_args.tune_vision_tower,
        tune_mm_adapter=model_args.tune_mm_adapter,
    )

    print('Building the image loader...')
    image_loader = build_image_loader(
        vision_model=model_args.vision_tower,
        image_process_mode=data_args.image_process_mode
    )

    print('Getting the number of vision tokens...')
    num_vision_token = get_num_vision_token(causal_lm, image_loader)

    print('Building the template applier...')
    template_applier = build_template_applier(
        strategy=model_args.strategy,
        tokenizer=tokenizer,
        num_vision_token=num_vision_token,
        is_training=True
    )

    print('Building the training dataset and data collator...')
    train_dataset = build_dataset(
        json_path=data_args.json_path,
        image_folder=data_args.image_folder,
        image_loader=image_loader,
        image_mark=data_args.image_mark,
        template_applier=template_applier,
        is_plain_dataset=data_args.is_plain_dataset,
        check_dataset=data_args.check_dataset,
        dialog_key=data_args.dialog_key,
        image_key=data_args.image_key,
        role_key=data_args.role_key,
        content_key=data_args.content_key,
        user_key=data_args.user_key,
        assistant_key=data_args.assistant_key,
    )

    data_collator = build_datacollator(
        pad_token_id=tokenizer.pad_token_id,
    )

    trainer = LLaVATrainer(
        model=causal_lm,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        skip_save_after_last_step=training_args.skip_save_after_last_step
    )

    print('Start to train...')
    has_checkpoints = bool(glob.glob(os.path.join(training_args.output_dir, 'checkpoint-*')))
    trainer.train(resume_from_checkpoint=has_checkpoints)

    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    # FIXME Update causal_llm.config
    # causal_lm.config.image_aspect_ratio = data_args.image_aspect_ratio
