from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={
        "help": "Path to pretrained model or model identifier from "
                "huggingface.co/models"
    })
    version: str = field(metadata={
        "help": "Version of the training mode"
    })

    tune_backbone: bool = field(metadata={
        "help": "Whether to tune the llm backbone of the model"
    })
    tune_vision_tower: bool = field(metadata={
        "help": "Whether to tune the vision encoder of the model"
    })
    tune_mm_adapter: bool = field(metadata={
        "help": "Whether to tune the multimodal adapter of the model"
    })

    vision_tower: str = field(metadata={"help": "Name of the vision tower"})
    mm_adapter: str = field(metadata={
        "help": "Name of the multimodal adapter"
    })
    pretrained_mm_adapter_path: Optional[str] = field(default=None, metadata={
        "help": "Path to pretrained multimodal adapter"
    })

    mm_use_im_start_end: bool = field(default=False, metadata={
        "help": "Whether to use image start and end tokens in multimodal "
                "adapter"
    })
    mm_use_im_patch_token: bool = field(default=True, metadata={
        "help": "Whether to use image patch tokens in multimodal adapter"
    })
    mm_patch_merge_type: str = field(default='flat', metadata={
        "help": "Type of merging image patch tokens"
    })
    mm_vision_select_layer: int = field(default=-1, metadata={
        "help": "Layer of the vision encoder to use for multimodal adapter"
    })
    mm_vision_select_feature: Literal['patch', 'cls_patch'] = \
        field(default='patch', metadata={
            "help": "Feature of the vision encoder to use for multimodal "
                    "adapter"
        })

    # moe_vision_tower arguments:
    vision_expert_list: List[str] = field(default_factory=list,  metadata={
        "help": "List of experts for vision tower"
    })
    m_patch_one_token: List[int] = field(default_factory=list, metadata={
        "help": "List of number of patches combined into one token for each "
                "expert"
    })

    def __post_init__(self):
        assert self.mm_patch_merge_type in ["flat", "unpad"], \
            f"mm_patch_merge_type should be one of `flat` or `unpad`, got " \
            f"{self.mm_patch_merge_type}"
        assert self.mm_vision_select_feature in ["patch", "cls_patch"], \
            f"mm_vision_select_feature should be one of `patch` or " \
            f"`cls_patch`, got {self.mm_vision_select_feature}"
        assert not self.tune_vision_tower, "Tuning vision tower is not " \
                                           "supported yet"

    def __str__(self):
        attrs_as_str = [f"{k}={v},\n" for k, v in asdict(self).items()]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__
