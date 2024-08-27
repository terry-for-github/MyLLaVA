from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class DataArguments:
    json_path: str = field(metadata={"help": "Path to json data file"})
    image_folder: str = field(metadata={"help": "Path to image folder"})

    check_dataset: bool = field(default=False, metadata={
        "help": "Check the dataset before training"
    })

    is_plain_dataset: bool = field(default=False, metadata={
        "help": "Whether or not the dataset is plain (Each sample has only two messages)"
    })

    image_process_mode: Literal['pad', 'warp', 'crop', 'no'] = field(default='pad', metadata={
        "help": "Mode to process image\n"
                "- pad: pad the image to square\n"
                "- warp: warp the image to square\n"
                "- crop: crop the image to square\n"
                "- no: process only by processor"
    })

    image_mark: str = field(default="<image>", metadata={
        "help": "The mark for image token in conversation"
    })

    dialog_key: str = field(default="conversations", metadata={
        "help": "Key for dialog"
    })

    image_key: str = field(default="image", metadata={
        "help": "Key for image"
    })

    role_key: str = field(default="from", metadata={
        "help": "Key for role"
    })

    content_key: str = field(default="value", metadata={
        "help": "Key for content"
    })

    human_key: str = field(default="human", metadata={
        "help": "Key for human"
    })

    gpt_key: str = field(default="gpt", metadata={
        "help": "Key for gpt"
    })

    def __str__(self):
        attrs_as_str = [f"{k}={v},\n" for k, v in asdict(self).items()]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__
