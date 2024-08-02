
import torch

from arguments import ModelArguments, TrainingArguments
from .llava_llama import LlavaLlamaForCausalLM


def get_bnb_args(training_args: TrainingArguments,
                 compute_dtype: torch.dtype) -> dict:
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
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
        )
    )


def get_causal_language_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    compute_dtype: torch.dtype
) -> LlavaLlamaForCausalLM:

    # Not support bits_and_bytes right now
    bnb_args = {} if True else get_bnb_args(training_args, compute_dtype)
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
        assert model_args.pretrained_mm_adapter_path is not None
        mm_adapter_state_dict = torch.load(
            model_args.pretrained_mm_adapter_path,
            map_location='cpu', weights_only=True
        )
        assert isinstance(mm_adapter_state_dict, dict)
        print('[DEBUG]', 1, '================================================')
        print('[DEBUG]', 1, 'Loading pretrained mm_adapter from:')
        print('[DEBUG]', 1, model_args.pretrained_mm_adapter_path)
        print('[DEBUG]', 1, 'mm_adapter_state_dict')
        print('[DEBUG]', 1, mm_adapter_state_dict.keys())
        print('[DEBUG]', 1, '================================================')
        # remove 'model.mm_adapter' prefix
        no_prefix_state_dict = {}
        for k, v in mm_adapter_state_dict.items():
            no_prefix_state_dict[k.split('model.mm_adapter.')[1]] = v
        mm_adapter.load_state_dict(no_prefix_state_dict)

    # Set requires_grad
    assert mm_adapter is not None and vision_tower is not None
    print('[DEBUG]', 1, '====================================================')
    print('[DEBUG]', 1, 'default requires_grad:')
    print('[DEBUG]', 1, 'backbone',
          next(causal_lm.get_model().parameters()).requires_grad)
    print('[DEBUG]', 1, 'vision_tower',
          next(vision_tower.parameters()).requires_grad)
    print('[DEBUG]', 1, 'mm_adapter',
          next(mm_adapter.parameters()).requires_grad)
    print('[DEBUG]', 1, '====================================================')
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
