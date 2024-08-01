import os
import builtins

import torch
import transformers

from arguments import ModelArguments, DataArguments, TrainingArguments
from llava import LlavaLlamaForCausalLM
import conversation


def get_compute_dtype(training_args) -> torch.dtype:
    if training_args.fp16 and training_args.bf16:
        raise ValueError("Only one of fp16 and bf16 can be set to True")
    elif training_args.fp16:
        return torch.float16
    elif training_args.bf16:
        return torch.bfloat16
    else:
        return torch.float32


def get_bnb_args(training_args) -> dict:
    '''
    Get BitsAndBytes args for from_pretrained method.
    '''

    if training_args.bits not in [4, 8]:
        return {}

    from transformers import BitsAndBytesConfig
    return dict(
        device_map={"": training_args.device},
        load_in_4bit=training_args.bits == 4,
        load_in_8bit=training_args.bits == 8,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            llm_int8_skip_modules=["mm_adapter"],
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=get_compute_dtype(training_args),
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
        )
    )


def get_causal_language_model(model_args: ModelArguments,
                              training_args: TrainingArguments,
                              compute_dtype: torch.dtype,
                              bnb_args):
    # Load causal_llm
    causal_lm = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_impl,
        torch_dtype=compute_dtype,
        **bnb_args
    )

    # Load vision module
    vision_tower = causal_lm.get_vision_tower()
    mm_adapter = causal_lm.get_mm_adapter()
    if vision_tower is None:
        assert mm_adapter is None
        causal_lm.get_model().init_vision_modules(model_args)
        vision_tower = causal_lm.get_vision_tower()
        mm_adapter = causal_lm.get_mm_adapter()
    else:
        assert mm_adapter is not None
        # Load pretrained mm_adapter
        if model_args.pretrained_mm_adapter_path is None:
            raise ValueError('pretrained_mm_adapter_path is required for '
                             'fine-tuning multimodal adapter')

        mm_adapter_state_dict = torch.load(
            model_args.pretrained_mm_adapter_path,
            map_location='cpu', weights_only=True
        )

        assert isinstance(mm_adapter_state_dict, dict)
        print('[DEBUG]', 1, 'Loading pretrained mm_adapter from:')
        print('[DEBUG]', 1, model_args.pretrained_mm_adapter_path)
        print('[DEBUG]', 1, 'mm_adapter_state_dict')
        print('[DEBUG]', 1, mm_adapter_state_dict.keys())
        # remove 'model.mm_adapter' prefix
        no_prefix_state_dict = {}
        for k, v in mm_adapter_state_dict.items():
            no_prefix_state_dict[k.split('model.mm_adapter.')[1]] = v
        mm_adapter.load_state_dict(no_prefix_state_dict)

    # Set requires_grad
    assert mm_adapter is not None and vision_tower is not None
    # =========================================================================
    print('[DEBUG]', 1, 'default requires_grad:')
    print('[DEBUG]', 1, 'backbone',
          next(causal_lm.get_model().parameters()).requires_grad)
    print('[DEBUG]', 1, 'vision_tower',
          next(vision_tower.parameters()).requires_grad)
    print('[DEBUG]', 1, 'mm_adapter',
          next(mm_adapter.parameters()).requires_grad)
    # =========================================================================
    causal_lm.get_model().requires_grad_(model_args.tune_backbone)
    vision_tower.requires_grad_(model_args.tune_vision_tower)
    mm_adapter.requires_grad_(model_args.tune_mm_adapter)

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

    return causal_lm


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = \
        parser.parse_args_into_dataclasses()
    # =========================================================================
    print('[DEBUG]', 1, model_args)
    print('[DEBUG]', 1, data_args)
    print('[DEBUG]', 1, training_args)
    # =========================================================================

    # compute_dtype in [torch.float16, torch.bfloat16, torch.float32]
    compute_dtype = get_compute_dtype(training_args)
    # Not support bits_and_bytes right now
    bnb_args = {} if True else get_bnb_args(training_args)

    # You need to have sentencepiece installed to convert a slow tokenizer to
    # a fast one. --> pip install sentencepiece protobuf
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right"
    )
    if model_args.version == "plain":
        tokenizer.pad_token = tokenizer.unk_token
        conversation.set_default(model_args.version)

    causal_lm = get_causal_language_model(
        model_args, training_args, compute_dtype, bnb_args
    )
    causal_lm.init_vision_tokenizer(model_args, tokenizer)

    # =========================================================================
    print('[DEBUG]', 1, causal_lm)
    print('[DEBUG]', 1, tokenizer)
    # =========================================================================
    # FIXME Update causal_llm.config
    # FIXME use cache or not?
    # causal_lm.config.use_cache = False
    # causal_lm.config.image_aspect_ratio = data_args.image_aspect_ratio
    # causal_lm.config.tokenizer_padding_side = tokenizer.padding_side
    # causal_lm.config.tokenizer_model_max_length = tokenizer.model_max_length
    # causal_lm.config.tune_backbone = model_args.tune_backbone
    # causal_lm.config.tune_vision_tower = model_args.tune_vision_tower
    # causal_lm.config.tune_mm_adapter = model_args.tune_mm_adapter


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

    train()
