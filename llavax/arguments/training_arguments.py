from dataclasses import asdict, dataclass, field, fields
from typing import Literal, Optional

import transformers


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    attn_impl: Literal['eager', 'sdpa', 'flash_attention_2'] = \
        field(default="flash_attention_2", metadata={
            "help": "Type of attention implementation to use: eager, sdpa, flash_attention_2"
        })
    skip_save_after_last_step: bool = field(default=False, metadata={
        "help": "Whether to skip saving a checkpoint after the last training step."
    })

    # quantization
    double_quant: bool = field(default=True, metadata={
        "help": "Compress the quantization statistics through double quantization."
    })
    quant_type: Literal['nf4', 'fp4'] = field(default="nf4", metadata={
        "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
    })
    bits: int = field(default=16, metadata={"help": "How many bits to use."})

    # lora
    lora_enable: bool = field(default=False, metadata={"help": "Whether to enable LoRA"})
    lora_r: int = field(default=64, metadata={"help": "LoRA r"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_weight_path: Optional[str] = field(default=None, metadata={
        "help": "Path to LoRA weight file"
    })
    # FIXME why none?
    lora_bias: Literal['none', 'all', 'lora_only'] = field(default="none", metadata={
        "help": "LoRA bias"
    })

    # FIXME why none?
    # Not enable now
    mm_adapter_lr: Optional[float] = field(default=None, metadata={
        "help": "Learning rate for multimodal adapter"
    })

    def __post_init__(self):
        super().__post_init__()
        assert self.lora_bias in ["none", "all", "lora_only"], \
            f"lora_bias should be one of `none`, `all`, or `lora_only` got {self.lora_bias}"
        assert self.quant_type in ["fp4", "nf4"], \
            f"quant_type should be one of `fp4` or `nf4`, got {self.quant_type}"
        assert self.attn_impl in ["eager", "sdpa", "flash_attention_2"], \
            f"attn_impl should be one of `eager`, `sdpa`, or `flash_attention_2`, got " \
            f"{self.attn_impl}"
        # just set False, we remove unused columns by ourselves
        self.remove_unused_columns = False

    def __str__(self):
        self_as_dict = asdict(self)
        # hide the token values
        token_key = [k for k in self_as_dict.keys() if k.endswith("_token")]
        self_as_dict.update({k: f"<{k.upper()}>" for k in token_key})

        # seperate the super class attributes
        super_key = [f.name for f in fields(transformers.TrainingArguments)]
        super_as_dict = {k: self_as_dict.pop(k) for k in super_key}
        # add the sorted super class attributes to the self_as_dict
        self_as_dict.update({k: super_as_dict[k] for k in sorted(super_as_dict)})

        # Remove deprecated arguments. That code should be removed once
        # those deprecated arguments are removed from TrainingArguments. (TODO: v5)
        del self_as_dict["per_gpu_train_batch_size"]
        del self_as_dict["per_gpu_eval_batch_size"]

        attrs_as_str = [f"{k}={v},\n" for k, v in self_as_dict.items()]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"
