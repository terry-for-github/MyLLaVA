from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to json data file"})
    image_folder: Optional[str] = field(default=None, metadata={
        "help": "Path to image folder"
    })

    lazy_preprocess: bool = field(default=False, metadata={
        "help": "Whether to lazy preprocess data"
    })
    # must multimodal
    # is_multimodal: bool = field(default=False, metadata={
    # "help": "Whether the data is multimodal"})
    image_aspect_ratio: str = field(default='square', metadata={
        "help": "Aspect ratio of the image"
    })

    def __str__(self):
        attrs_as_str = [f"{k}={v},\n" for k, v in asdict(self).items()]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__
