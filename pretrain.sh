#!/bin/bash
# microsoft/layoutlmv3-large facebook/dinov2-giant openai/clip-vit-large-patch14-336 
# --vision_experts_list # --m_token_one_patch 1

MODEL_NAME=llava-7b-v1.5
PRETRAIN_MODEL_NAME=${MODEL_NAME}-pretrain

JSON_PATH=./tests/test_dataset.json
IMAGE_FOLDER=/home/lanxy/Dataset/LLaVA-Pretrain/images

# launch params
command="accelerate launch"
command+=" --config_file=single.yaml"
command+=" train.py"
command+=" --deepspeed=./zero2.json"
command+=" --output_dir=./checkpoints/$PRETRAIN_MODEL_NAME-toy"
command+=" --report_to=none"

# model params
command+=" --model_name_or_path=lmsys/vicuna-7b-v1.5"
command+=" --version=plain"

# tuning params
command+=" --tune_backbone=false"
command+=" --tune_vision_tower=false"
command+=" --tune_mm_adapter=true"

# vision module params
command+=" --vision_tower=openai/clip-vit-large-patch14-336"
command+=" --mm_adapter=linear"

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

# training params
command+=" --num_train_epochs=1"
command+=" --per_device_train_batch_size=2"
command+=" --gradient_accumulation_steps=2"
command+=" --dataloader_num_workers=4"
command+=" --dataloader_prefetch_factor=2"
command+=" --gradient_checkpointing=true"
command+=' --gradient_checkpointing_kwargs={\"use_reentrant\":false}'
command+=" --ddp_backend=nccl"
command+=" --eval_strategy=no"

command+=" --bf16=true"
command+=" --tf32=true"
command+=" --learning_rate=1e-3"
command+=" --weight_decay=0."
command+=" --warmup_ratio=0.03"
command+=" --lr_scheduler_type=cosine"

# logging params
command+=" --logging_steps=1"

# save params
command+=" --save_strategy=steps" # For resuming training
command+=" --save_steps=2000"
command+=" --save_total_limit=1"

# seed params
command+=" --seed=42" # default value
command+=" --data_seed=814" # default equals seed


eval $command