from dataclasses import dataclass, field
from enum import auto, Enum
from typing import List, Optional


class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()
    PLAIN = auto()


@dataclass
class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = field(default="###")
    sep2: Optional[str] = field(default=None)
    version: str = field(default="unknown")
    skip_next: bool = field(default=False)

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.PLAIN:
            return self._get_prompt_plain()
        elif self.sep_stype == SeparatorStyle.TWO:
            return self._get_prompt_two()
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def _get_prompt_plain(self):
        seps = [self.sep, self.sep2]
        ret = self.system
        for i, (role, message) in enumerate(self.messages):
            if not message:
                continue
            if isinstance(message, tuple):
                message, _, _ = message
            ret += message + seps[i % 2]
        return ret

    def _get_prompt_two(self):
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(self.messages):
            ret += role + ":"
            if not message:
                continue
            if type(message) is tuple:
                message, _, _ = message
            ret += message + seps[i % 2]
        return ret


conv_llava_plain = Conversation(
    system="",
    roles=["", ""],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n"
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence "
           "assistant. The assistant gives helpful, detailed, and polite "
           "answers to the user's questions.",
    roles=["USER", "ASSISTANT"],
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>"
)

default_conversation = conv_llava_plain
conv_templates = {
    "plain": conv_llava_plain,
    "vicuna_v1": conv_vicuna_v1
}


def set_default(version: str):
    global default_conversation
    if version not in conv_templates:
        raise ValueError(f"Unknown conversation version: {version}")
    default_conversation = conv_templates[version]
