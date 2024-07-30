#!/bin/bash
# microsoft/layoutlmv3-large facebook/dinov2-giant openai/clip-vit-large-patch14-336 
# --vision_experts_list # --m_token_one_patch 1

MODEL_NAME=llava-llama3
PRETRAIN_MODEL_NAME=${MODEL_NAME}-pretrain
DATA_PATH=./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_100.json
IMAGE_FOLDER=./playground/data/LLaVA-Pretrain/images

command="accelerate launch --config_file=single.yaml train.py"
command+=" --deepspeed=./zero2.json"
command+=" --output_dir=./checkpoints/$PRETRAIN_MODEL_NAME-toy"
command+=" --report_to=none"

command+=" --model_name_or_path=lmsys/vicuna-7b-v1.5"
command+=" --version=plain"

command+=" --tune_backbone=false"
command+=" --tune_encoder=false"
command+=" --tune_mm_adapter=true"

command+=" --vision_tower=openai/clip-vit-large-patch14-336"
command+=" --mm_adapter=linear"

command+=" --mm_use_im_start_end=false"
command+=" --mm_use_im_patch_token=false"
command+=" --mm_vision_select_layer=-2"
# command+=" --vision_experts_list openai/clip-vit-large-patch14-336"
# command+=" --m_token_one_patch 1"

command+=" --data_path=$DATA_PATH"
command+=" --image_folder=$IMAGE_FOLDER"
command+=" --lazy_preprocess=true"
command+=" --model_max_length=2048"

command+=" --num_train_epochs=1"
command+=" --per_device_train_batch_size=16"
command+=" --per_device_eval_batch_size=4"
command+=" --gradient_accumulation_steps=2"
command+=" --dataloader_num_workers=4"
command+=" --gradient_checkpointing=true"
command+=" --eval_strategy=no"

command+=" --bf16=true"
command+=" --tf32=true"
command+=" --learning_rate=1e-3"
command+=" --weight_decay=0."
command+=" --warmup_ratio=0.03"
command+=" --lr_scheduler_type=cosine"
command+=" --logging_steps=1"

command+=" --save_strategy=steps"
command+=" --save_steps=24000"
command+=" --save_total_limit=1"


eval $command