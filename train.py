import os
import builtins

import torch
import transformers

from arguments import ModelArguments, DataArguments, TrainingArguments
from model import get_causal_language_model
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


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = \
        parser.parse_args_into_dataclasses()
    print('[DEBUG]', 1, '====================================================')
    print('[DEBUG]', 1, model_args)
    print('[DEBUG]', 1, data_args)
    print('[DEBUG]', 1, training_args)
    print('[DEBUG]', 1, '====================================================')

    # compute_dtype in [torch.float16, torch.bfloat16, torch.float32]
    compute_dtype = get_compute_dtype(training_args)

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
        model_args, training_args, compute_dtype
    )
    causal_lm.init_vision_tokenizer(model_args, tokenizer)

    print('[DEBUG]', 1, '====================================================')
    print('[DEBUG]', 1, causal_lm)
    print('[DEBUG]', 1, tokenizer)
    print('[DEBUG]', 1, causal_lm.config)
    print('[DEBUG]', 1, '====================================================')
    # FIXME Update causal_llm.config
    # FIXME use cache or not?
    # causal_lm.config.use_cache = False
    # causal_lm.config.image_aspect_ratio = data_args.image_aspect_ratio
    # causal_lm.config.tokenizer_padding_side = tokenizer.padding_side
    # causal_lm.config.tokenizer_model_max_length = tokenizer.model_max_length
    # causal_lm.config.tune_backbone = model_args.tune_backbone
    # causal_lm.config.tune_vision_tower = model_args.tune_vision_tower
    # causal_lm.config.tune_mm_adapter = model_args.tune_mm_adapter
    #     """
    # A DataCollator is a function that takes a list of samples from a Dataset
    # and collate them into a batch, as a dictionary
    # of PyTorch/TensorFlow tensors or NumPy arrays.
    # """
    # DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]],
    # Dict[str, Any]])


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
