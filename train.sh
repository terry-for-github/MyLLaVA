#!/bin/bash
# microsoft/layoutlmv3-large facebook/dinov2-giant openai/clip-vit-large-patch14-336 
# --vision_experts_list # --m_token_one_patch 1
# export TRANSFORMERS_OFFLINE=1

# 
if [[ $1 == "pretrain" ]]; then
    is_pretrain=1
elif [[ $1 == "finetune" ]]; then
    is_pretrain=0
else
    echo "First params must be 'pretrain' or 'finetune'."
    exit 1
fi
shift

export OMP_NUM_THREADS=1

MODEL_NAME=llava-test

# LLM_MODEL=lmsys/vicuna-7b-v1.5
# STRATEGY=vicuna
LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
# LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
STRATEGY=llama3

ACC_NUM=4

if [ $is_pretrain -eq 1 ]; then
    MODEL_NAME=${MODEL_NAME}-pretrain
    JSON_PATH=json/blip_laion_cc_sbu_558k.json
    IMAGE_FOLDER=/userhome/Dataset/LLaVA-Pretrain/images
    ZERO_JSON=configs/zero2.json
    STRATEGY=plain
    TUNE_LLM=false
    IS_PLAIN=true
    BATCH_SIZE=256
else
    MM_ADAPTER_PATH=checkpoints/llava-llama3.1-8b-mlp2x-double-pretrain/model.safetensors
    JSON_PATH=json/llava_v1_5_mix665k.json
    IMAGE_FOLDER=/userhome/Dataset/LLaVA-Finetune
    ZERO_JSON=configs/zero3.json
    TUNE_LLM=true
    IS_PLAIN=false
    BATCH_SIZE=128
fi

# launch params
command="accelerate launch"

if [ $# -eq 0 ]; then
    command+=" --config_file=configs/single.yaml"
    PER_BATCH_SIZE=$(($BATCH_SIZE / 8 / $ACC_NUM))
elif [ $# -eq 2 ]; then
    command+=" --config_file=configs/double.yaml"
    command+=" --machine_rank=$1"
    command+=" --main_process_ip=$2"
    PER_BATCH_SIZE=$(($BATCH_SIZE / 16 / $ACC_NUM))
else
    echo "You must provide another 0 or 2 arguments."
    exit 1
fi

command+=" main.py"
command+=" --deepspeed=$ZERO_JSON"
command+=" --output_dir=./checkpoints/$MODEL_NAME"
command+=" --report_to=wandb"
command+=" --run_name=$MODEL_NAME"

# model params
command+=" --model_name_or_path=$LLM_MODEL"
command+=" --strategy=$STRATEGY"

# tuning params
command+=" --tune_backbone=$TUNE_LLM"
command+=" --tune_vision_tower=false"
command+=" --tune_mm_adapter=true"

# vision module params
command+=" --vision_tower=openai/clip-vit-large-patch14-336"
command+=" --vision_select_layer=-2"
command+=" --mm_adapter=mlp2x_gelu"
if [ $is_pretrain -eq 0 ]; then
    command+=" --pretrained_mm_adapter_path=$MM_ADAPTER_PATH"
fi

# vision params
command+=" --mm_use_im_start_end=false"
command+=" --mm_use_im_patch_token=false"
# command+=" --m_patch_one_token 1"

# data params
command+=" --json_path=$JSON_PATH"
command+=" --image_folder=$IMAGE_FOLDER"
command+=" --dialog_key=conversations"
command+=" --image_key=image"
command+=" --role_key=from"
command+=" --content_key=value"
command+=" --user_key=human"
command+=" --assistant_key=gpt"
command+=" --model_max_length=2048"
if [ $is_pretrain -eq 1 ]; then
    command+=" --image_process_mode=no"
else
    command+=" --image_process_mode=pad"
fi
command+=" --check_dataset=true"
command+=" --is_plain_dataset=$IS_PLAIN"

# training params
command+=" --num_train_epochs=1"
command+=" --max_steps=-1" # default -1
command+=" --per_device_train_batch_size=$PER_BATCH_SIZE"
command+=" --gradient_accumulation_steps=$ACC_NUM"
command+=" --dataloader_num_workers=4"
command+=" --dataloader_prefetch_factor=2"
command+=" --group_by_length=false"
command+=" --gradient_checkpointing=true"
command+=' --gradient_checkpointing_kwargs={\"use_reentrant\":false}'
command+=" --ddp_backend=nccl"
command+=" --skip_save_after_last_step=true"

command+=" --bf16=true"
command+=" --tf32=true"
if [ $is_pretrain -eq 1 ]; then
    command+=" --learning_rate=1e-3"
else
    command+=" --learning_rate=2e-5"
fi
command+=" --weight_decay=0."
command+=" --warmup_ratio=0.03"
command+=" --lr_scheduler_type=cosine"

# logging params
command+=" --logging_steps=1"

# save params
command+=" --save_strategy=steps"
command+=" --save_steps=1000"
command+=" --save_total_limit=1"

# seed params
command+=" --seed=219" # default value
command+=" --data_seed=44" # default equals seed

eval $command