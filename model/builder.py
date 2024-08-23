import torch
from transformers import BitsAndBytesConfig

from arguments import ModelArguments, TrainingArguments
from constants import CACHE_DIR

from .llava_llama import LlavaLlamaForCausalLM, LlavaLlamaConfig


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


def get_causal_lm(model_args: ModelArguments,
                  training_args: TrainingArguments
                  ) -> LlavaLlamaForCausalLM:
    '''Get causal language model.'''
    # compute_dtype in [torch.float16, torch.bfloat16, torch.float32]
    compute_dtype = get_compute_dtype(training_args)
    # Not support bits_and_bytes right now
    bnb_args = {} if True else get_bnb_args(training_args, compute_dtype)

    # Temporarily change model_type to llama to avoid the warning
    LlavaLlamaConfig.model_type = 'llama'
    llava_config: LlavaLlamaConfig = LlavaLlamaConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=CACHE_DIR,
    )  # type: ignore
    LlavaLlamaConfig.model_type = 'llava_llama'

    # We are loading a model without vision tower. (LLM only)
    # So we need to initialize vision modules by ourselves.
    # This wont happened in evaluation if we save the config properly.
    if llava_config.vision_tower is None:
        llava_config.vision_tower = model_args.vision_tower
        llava_config.mm_adapter = model_args.mm_adapter
        llava_config.mm_vision_select_layer = model_args.mm_vision_select_layer
        llava_config.mm_patch_merge_type = model_args.mm_patch_merge_type

    # Load causal_llm
    causal_lm: LlavaLlamaForCausalLM = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=llava_config,
        cache_dir=CACHE_DIR,
        attn_implementation=training_args.attn_impl,
        torch_dtype=compute_dtype,
        **bnb_args
    )  # type: ignore

    causal_lm.init_pretrained_model(
        pretrained_mm_adapter_path=model_args.pretrained_mm_adapter_path,
        attn_implementation=training_args.attn_impl,
        # torch_dtype=compute_dtype,
        **bnb_args
    )

    backbone: torch.nn.Module = causal_lm.get_model()
    vision_tower: torch.nn.Module = causal_lm.get_vision_tower()
    mm_adapter: torch.nn.Module = causal_lm.get_mm_adapter()
    print('[DEBUG]', 1, '===================================================================')
    print('[DEBUG]', 1, causal_lm)
    print('[DEBUG]', 1, causal_lm.config)
    print('[DEBUG]', 1, 'default requires_grad and dtype:')
    backbone_param = next(backbone.parameters())
    tower_param = next(vision_tower.parameters())
    mm_adapter_param = next(mm_adapter.parameters())
    print('[DEBUG]', 1, 'backbone', backbone_param.requires_grad, backbone_param.dtype)
    print('[DEBUG]', 1, 'vision_tower', tower_param.requires_grad, tower_param.dtype)
    print('[DEBUG]', 1, 'mm_adapter', mm_adapter_param.requires_grad, mm_adapter_param.dtype)
    print('[DEBUG]', 1, '===================================================================')
    backbone.requires_grad_(model_args.tune_backbone)
    # FIXME Remember the lm_head, might tune backbone but not lm_head in the future?
    causal_lm.lm_head.requires_grad_(model_args.tune_backbone)
    vision_tower.requires_grad_(model_args.tune_vision_tower)
    mm_adapter.requires_grad_(model_args.tune_mm_adapter)

    return causal_lm  # type: ignore

    # FIXME no quantization
    # if training_args.bits in [4, 8]:
    #     from peft.utils.other import prepare_model_for_kbit_training
    #     print('[DEBUG]', 'is_loaded_in_8bit:', causal_lm.is_loaded_in_8bit)
    #     print('[DEBUG]', 'is_loaded_in_4bit:', causal_lm.is_loaded_in_4bit)
    #     causal_lm = prepare_model_for_kbit_training(causal_lm,
    # use_gradient_checkpointing=training_args.gradient_checkpointing)

    # FIXME no lora
    # ignore this temporaly
    # if training_args.lora_enable:
    #     from peft.tuners.lora import LoraConfig
    #     from peft.mapping import get_peft_model
    #     def find_all_linear_names(model):
    #         linear_module_cls = torch.nn.Linear
    #         lora_module_names = set()
    #         mm_keyword_list = ['mm_adapter', 'vision_tower']
    #         for name, module in model.named_modules():
    #             # skip mm modules
    #             if any(mm_keyword in name for mm_keyword in mm_keyword_list):
    #                 continue
    #             if isinstance(module, linear_module_cls):
    #                 names = name.split('.')
    #                 lora_module_names.add(names[-1])

    #         if 'lm_head' in lora_module_names: # needed for 16-bit
    #             lora_module_names.remove('lm_head')
    #         print('[DEBUG]', 1, lora_module_names)
    #         return list(lora_module_names)

    #     lora_config = LoraConfig(
    #         r=training_args.lora_r,
    #         target_modules=find_all_linear_names(causal_lm),
    #         lora_alpha=training_args.lora_alpha,
    #         lora_dropout=training_args.lora_dropout,
    #         bias=training_args.lora_bias
    #     )
    #     # FIXME haven't use training_args.lora_weight_path
    #     causal_lm = get_peft_model(causal_lm, lora_config)
    #     # debug
    #     print('[DEBUG]', 1, 'LoraConfig:', lora_config)
    #     print('[DEBUG]', 1, 'different part requires_grad:')
    #     print('[DEBUG]', 1,next(causal_lm.base_model.get_model()
    # .parameters()).requires_grad)
    #     print('[DEBUG]', 1, next(causal_lm.base_model
    # .get_vision_tower().parameters()).requires_grad)
    #     print('[DEBUG]', 1, next(causal_lm.base_model
    # .get_mm_adapter().parameters()).requires_grad)

    # FIXME no gradient_checkpointing
    # FIXME gradient_checkpointing for input embeddings
    # if training_args.gradient_checkpointing:
    #     causal_lm.get_model().gradient_checkpointing = True
    #     causal_lm.gradient_checkpointing_enable()
    #     causal_lm.enable_input_require_grads()
