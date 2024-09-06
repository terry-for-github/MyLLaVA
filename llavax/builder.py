import os
from typing import Callable, Optional, Any

import transformers
import deepspeed
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import PreTrainedTokenizerBase
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import (
    SingleTowersImageLoader,
    MultiTowersImageLoader,
    BaseImageLoader,
    TemplateApplier,
    DataCollatorForSingleImageAtFirstDialog,
    LazySingleImageAtFirstDialogDataset
)
from .model import LlavaLlamaForCausalLM, LlavaLlamaConfig
from .constants import CACHE_DIR


def build_tokenizer(
    model_name_or_path: str,
    model_max_length: Optional[int] = None
) -> PreTrainedTokenizerBase:
    # You need to have sentencepiece installed to convert a slow tokenizer to a fast one.
    # --> pip install sentencepiece protobuf
    # huggingface/tokenizers: The current process just got forked, after parallelism has
    # already been used. Disabling parallelism to avoid deadlocks...
    # To disable this warning, you can either:
    #     - Avoid using `tokenizers` before the fork if possible
    #     - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=CACHE_DIR,
        model_max_length=model_max_length,
        padding_side="right"
    )
    if (
        'llama-3' in model_name_or_path.lower() or
        'llama3' in model_name_or_path.lower()
    ):
        tokenizer.pad_token = "<|reserved_special_token_1|>"
    assert tokenizer.pad_token_id is not None
    return tokenizer


def build_image_loader(
    vision_model: list[str],
    image_process_mode: str
) -> BaseImageLoader:
    if len(vision_model) > 1:
        return MultiTowersImageLoader(
            vision_model_list=vision_model,
            image_process_mode=image_process_mode
        )
    else:
        return SingleTowersImageLoader(
            vision_model_name=vision_model[0],
            image_process_mode=image_process_mode
        )


def _get_num_vision_token(
    causal_lm: LlavaLlamaForCausalLM,
    image_loader: BaseImageLoader
) -> int:
    '''Get the number of vision tokens.'''
    image_tensor = image_loader.load_image(None).unsqueeze(0)
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


def build_template_applier(
    strategy: str,
    model: LlavaLlamaForCausalLM,
    image_loader: BaseImageLoader,
    tokenizer: PreTrainedTokenizerBase,
    is_training: bool = True
) -> TemplateApplier:
    print('Getting the number of vision tokens...')
    num_vision_token = _get_num_vision_token(model, image_loader)
    print('Building the template applier...')
    return TemplateApplier(
        strategy=strategy,
        tokenizer=tokenizer,
        num_vision_token=num_vision_token,
        is_training=is_training
    )


def build_dataset(
    json_path: str,
    image_folder: str,
    image_loader: BaseImageLoader,
    image_mark: str,
    template_applier: TemplateApplier,
    is_plain_dataset: bool = False,
    check_dataset: bool = True,
    dialog_key: str = 'dialog',
    image_key: str = 'image',
    role_key: str = 'role',
    content_key: str = 'content',
    user_key: str = 'user',
    assistant_key: str = 'assistant',
) -> Dataset:
    return LazySingleImageAtFirstDialogDataset(
        json_path=json_path,
        image_folder=image_folder,
        image_loader=image_loader,
        image_mark=image_mark,
        template_applier=template_applier,
        is_plain_dataset=is_plain_dataset,
        check_dataset=check_dataset,
        dialog_key=dialog_key,
        image_key=image_key,
        role_key=role_key,
        content_key=content_key,
        user_key=user_key,
        assistant_key=assistant_key
    )


def build_datacollator(
    pad_token_id: Optional[int]
) -> Callable[[list[dict[str, Tensor]]], dict[str, Tensor]]:
    assert pad_token_id is not None
    return DataCollatorForSingleImageAtFirstDialog(
        pad_token_id=pad_token_id
    )


def build_model_config(
    model_name_or_path: str,
    pad_token_id: Optional[int] = None,
    vision_tower: Optional[list[str]] = None,
    vision_select_layer: Optional[list[int]] = None,
    mm_adapter: Optional[str] = None,
    patch_merge_type: Optional[str] = None,
) -> LlavaLlamaConfig:
    # model_name_or_path is a LLM model
    # we need to configure the model with vision tower and mm adapter

    # 1. Temporarily change model_type to llama to avoid the warning
    LlavaLlamaConfig.model_type = 'llama'
    # 2. Load llama config
    llava_config: LlavaLlamaConfig = LlavaLlamaConfig.from_pretrained(
        model_name_or_path,
        cache_dir=CACHE_DIR,
    )  # type: ignore
    # 2.5. Debug option
    if os.environ.get('LLAVA_DEBUG', None) is not None:
        llava_config.num_hidden_layers = 4
    # 3. Set pad_token_id if not set
    if llava_config.pad_token_id is None:
        assert pad_token_id is not None
        llava_config.pad_token_id = pad_token_id
    # 4. Set the model_type back
    LlavaLlamaConfig.model_type = 'llava_llama'

    # 5. Set vision_tower and mm_adapter
    assert vision_tower is not None
    assert mm_adapter is not None
    assert vision_select_layer is not None
    assert patch_merge_type is not None

    llava_config.vision_tower = vision_tower
    llava_config.mm_adapter = mm_adapter
    llava_config.vision_select_layer = vision_select_layer
    llava_config.patch_merge_type = patch_merge_type
    return llava_config


def build_causal_lm(
    model_name_or_path: str,
    model_config: LlavaLlamaConfig,
    compute_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = 'flash_attention_2',
    pretrained_mm_adapter_path: Optional[str] = None,
    bnb_kwargs: Optional[dict[str, Any]] = None,
    tune_backbone: Optional[bool] = None,
    tune_vision_tower: Optional[bool] = None,
    tune_mm_adapter: Optional[bool] = None
) -> LlavaLlamaForCausalLM:
    '''Get causal language model.'''
    # 1. Deal with bnb_kwargs None
    if bnb_kwargs is None:
        _bnb_kwargs: dict[str, Any] = dict()
    else:
        _bnb_kwargs = bnb_kwargs

    # 2. Set keys to ignore on load missing
    LlavaLlamaForCausalLM._keys_to_ignore_on_load_missing = (  # type: ignore
        ['vision_tower', 'mm_adapter']
        if pretrained_mm_adapter_path is None else
        ['vision_tower']
    )

    # 3. Load causal_llm
    causal_lm = LlavaLlamaForCausalLM.from_pretrained(
        model_name_or_path,
        config=model_config,
        cache_dir=CACHE_DIR,
        attn_implementation=attn_implementation,
        torch_dtype=compute_dtype,
        **_bnb_kwargs
    )
    # 4. Set keys to ignore on save
    LlavaLlamaForCausalLM._keys_to_ignore_on_save = (  # type: ignore
        [name for name, _ in causal_lm.named_parameters() if 'mm_adapter' not in name]
        if pretrained_mm_adapter_path is None else None
    )

    assert isinstance(causal_lm, LlavaLlamaForCausalLM)
    backbone = causal_lm.model
    vision_tower = causal_lm.get_vision_tower()
    mm_adapter = causal_lm.get_mm_adapter()

    # 5. Load vision module
    vision_tower.load()
    mm_adapter.load(state_dict_path=pretrained_mm_adapter_path)

    print('[DEBUG]', 1, '===============================================================')
    print('[DEBUG]', 1, 'default requires_grad and dtype:')
    backbone_param = next(backbone.parameters())
    tower_param = next(vision_tower.parameters())
    adapter_param = next(mm_adapter.parameters())
    print('[DEBUG]', 1, 'backbone', backbone_param.requires_grad, backbone_param.dtype)
    print('[DEBUG]', 1, 'vision_tower', tower_param.requires_grad, tower_param.dtype)
    print('[DEBUG]', 1, 'mm_adapter', adapter_param.requires_grad, adapter_param.dtype)
    print('[DEBUG]', 1, '===============================================================')

    # 5. Set requires_grad
    assert tune_backbone is not None
    assert tune_vision_tower is not None
    assert tune_mm_adapter is not None
    # FIXME Remember the lm_head, might tune backbone but not lm_head in the future?
    backbone.requires_grad_(tune_backbone)
    causal_lm.lm_head.requires_grad_(tune_backbone)
    vision_tower.requires_grad_(tune_vision_tower)
    mm_adapter.requires_grad_(tune_mm_adapter)

    # 6. Set train
    backbone.train(tune_backbone)
    causal_lm.lm_head.train(tune_backbone)
    vision_tower.train(tune_vision_tower)
    mm_adapter.train(tune_mm_adapter)

    return causal_lm

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
