import os

import transformers
import torch

from ..arguments import ModelArguments, DataArguments, TrainingArguments
from ..data_module import (
    SingleTowersImageLoader,
    MultiTowersImageLoader,
    LazySingleImageAtFirstDialogDataset,
    DataCollatorForSingleImageAtFirstDialog
)
from ..model import LlavaLlamaConfig, LlavaLlamaForCausalLM
from ..constants import CACHE_DIR


class TrainFactory:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def create_tokenizer(self):
        # You need to have sentencepiece installed to convert a slow tokenizer to a fast one.
        # --> pip install sentencepiece protobuf
        # huggingface/tokenizers: The current process just got forked, after parallelism has
        # already been used. Disabling parallelism to avoid deadlocks...
        # To disable this warning, you can either:
        #     - Avoid using `tokenizers` before the fork if possible
        #     - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=CACHE_DIR,
            model_max_length=self.model_args.model_max_length,
            padding_side="right"
        )
        if (
            'llama-3' in self.model_args.model_name_or_path.lower() or
            'llama3' in self.model_args.model_name_or_path.lower()
        ):
            tokenizer.pad_token = "<|reserved_special_token_1|>"
        assert tokenizer.pad_token_id is not None
        return tokenizer

    def _create_image_loader(self):
        if len(self.model_args.vision_tower) > 1:
            image_loader = MultiTowersImageLoader(
                image_folder=self.data_args.image_folder,
                vision_model_list=self.model_args.vision_tower,
                image_process_mode=self.data_args.image_process_mode
            )
        else:
            image_loader = SingleTowersImageLoader(
                image_folder=self.data_args.image_folder,
                vision_model_name=self.model_args.vision_tower[0],
                image_process_mode=self.data_args.image_process_mode
            )
        return image_loader

    def create_dataset(self, num_vision_token):
        image_loader = self._create_image_loader()
        return LazySingleImageAtFirstDialogDataset(
            data_args=self.data_args,
            image_loader=image_loader,
            vision_token_num=num_vision_token,
            model_max_length=self.model_args.model_max_length
        )

    def create_data_collator(self, tokenizer):
        return DataCollatorForSingleImageAtFirstDialog(
            tokenizer=tokenizer,
            version=self.model_args.version
        )

    def _update_config_with_model_args(self, config: LlavaLlamaConfig):
        config.vision_tower = self.model_args.vision_tower
        config.mm_adapter = self.model_args.mm_adapter
        config.mm_vision_select_layer = self.model_args.mm_vision_select_layer
        config.mm_patch_merge_type = self.model_args.mm_patch_merge_type

    def _create_llava_config(self, pad_token_id: int) -> LlavaLlamaConfig:
        '''Create llava config.'''
        # Temporarily change model_type to llama to avoid the warning
        LlavaLlamaConfig.model_type = 'llama'
        llava_config: LlavaLlamaConfig = LlavaLlamaConfig.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=CACHE_DIR,
        )  # type: ignore
        if os.environ.get('LLAVA_DEBUG', None) is not None:
            llava_config.num_hidden_layers = 4
        llava_config.pad_token_id = pad_token_id
        LlavaLlamaConfig.model_type = 'llava_llama'

        # We are loading a model without vision tower. (LLM only)
        # So we need to initialize vision modules by ourselves.
        # This wont happened in evaluation if we save the config properly.
        if llava_config.vision_tower is None:
            self._update_config_with_model_args(llava_config)
            LlavaLlamaForCausalLM._keys_to_ignore_on_load_missing += ['mm_adapter']
        return llava_config

    def _update_require_grad(self, causal_lm: LlavaLlamaForCausalLM):
        backbone: torch.nn.Module = causal_lm.model
        vision_tower: torch.nn.Module = causal_lm.get_vision_tower()
        mm_adapter: torch.nn.Module = causal_lm.get_mm_adapter()
        print('[DEBUG]', 1, '===============================================================')
        print('[DEBUG]', 1, 'default requires_grad and dtype:')
        backbone_param = next(backbone.parameters())
        tower_param = next(vision_tower.parameters())
        adapter_param = next(mm_adapter.parameters())
        print('[DEBUG]', 1, 'backbone', backbone_param.requires_grad, backbone_param.dtype)
        print('[DEBUG]', 1, 'vision_tower', tower_param.requires_grad, tower_param.dtype)
        print('[DEBUG]', 1, 'mm_adapter', adapter_param.requires_grad, adapter_param.dtype)
        print('[DEBUG]', 1, '===============================================================')
        backbone.requires_grad_(self.model_args.tune_backbone)
        # FIXME Remember the lm_head, might tune backbone but not lm_head in the future?
        causal_lm.lm_head.requires_grad_(self.model_args.tune_backbone)
        vision_tower.requires_grad_(self.model_args.tune_vision_tower)
        mm_adapter.requires_grad_(self.model_args.tune_mm_adapter)

    def create_causal_lm(
        self,
        compute_dtype: torch.dtype,
        bnb_args: dict,
        pad_token_id: int
    ) -> LlavaLlamaForCausalLM:
        '''Get causal language model.'''
        llava_config = self._create_llava_config(pad_token_id)

        # Load causal_llm
        causal_lm = LlavaLlamaForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            config=llava_config,
            cache_dir=CACHE_DIR,
            attn_implementation=self.training_args.attn_impl,
            torch_dtype=compute_dtype,
            **bnb_args
        )  # type: ignore
        assert isinstance(causal_lm, LlavaLlamaForCausalLM)

        causal_lm.init_vision_modules(
            pretrained_mm_adapter_path=self.model_args.pretrained_mm_adapter_path,
            attn_implementation=self.training_args.attn_impl,
            torch_dtype=compute_dtype,
            **bnb_args
        )

        self._update_require_grad(causal_lm)

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
