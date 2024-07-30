import builtins
import torch
import transformers
from accelerate import Accelerator
from .arguments import ModelArguments, DataArguments, TrainingArguments
accelerator = Accelerator()
builtins.print = accelerator.print


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    # print(model_args)
    # print(data_args)
    # print(training_args)


if __name__ == '__main__':
    train()