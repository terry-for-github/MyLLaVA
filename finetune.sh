#!/bin/bash
# microsoft/layoutlmv3-large facebook/dinov2-giant openai/clip-vit-large-patch14-336 
# --vision_experts_list # --m_token_one_patch 1
# export TRANSFORMERS_OFFLINE=1

MODEL_NAME=llava-7b-v1.5
PRETRAIN_MODEL_NAME=${MODEL_NAME}-pretrain

JSON_PATH=llava_v1_5_mix665k.json
IMAGE_FOLDER=/userhome/Dataset/LLaVA-Finetune
MM_ADAPTER_PATH=checkpoints/llava-7b-v1.5-pretrain-toy/model.safetensors

# launch params
command="accelerate launch"
command+=" --config_file=single.yaml"
command+=" train.py"
command+=" --deepspeed=./zero3.json"
command+=" --output_dir=./checkpoints/$MODEL_NAME-toy"
command+=" --report_to=none"

# model params
command+=" --model_name_or_path=lmsys/vicuna-7b-v1.5"
command+=" --version=vicuna"

# tuning params
command+=" --tune_backbone=true"
command+=" --tune_vision_tower=false"
command+=" --tune_mm_adapter=true"

# vision module params
command+=" --vision_tower=openai/clip-vit-large-patch14-336"
command+=" --mm_adapter=linear"
command+=" --pretrained_mm_adapter_path=$MM_ADAPTER_PATH"

# vision params
command+=" --mm_use_im_start_end=false"
command+=" --mm_use_im_patch_token=false"
command+=" --mm_vision_select_layer=-2"
# command+=" --vision_experts_list openai/clip-vit-large-patch14-336"
# command+=" --m_token_one_patch 1"

# data params
command+=" --json_path=$JSON_PATH"
command+=" --image_folder=$IMAGE_FOLDER"
command+=" --model_max_length=2048"
command+=" --check_dataset=false"
command+=" --is_plain_dataset=false"

# training params
command+=" --num_train_epochs=1"
command+=" --per_device_train_batch_size=8"
command+=" --gradient_accumulation_steps=2"
command+=" --dataloader_num_workers=4"
command+=" --dataloader_prefetch_factor=2"
command+=" --group_by_length=true"
command+=" --gradient_checkpointing=true"
command+=' --gradient_checkpointing_kwargs={\"use_reentrant\":false}'
command+=" --ddp_backend=nccl"
command+=" --eval_strategy=no"

command+=" --bf16=true"
command+=" --tf32=true"
command+=" --learning_rate=2e-5"
command+=" --weight_decay=0."
command+=" --warmup_ratio=0.03"
command+=" --lr_scheduler_type=cosine"

# logging params
command+=" --logging_steps=1"

# save params
command+=" --save_strategy=steps"
command+=" --save_steps=100"
command+=" --save_total_limit=1"

# seed params
command+=" --seed=42" # default value
command+=" --data_seed=814" # default equals seed

eval $command