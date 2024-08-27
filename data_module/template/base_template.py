from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaseTemplate(ABC):
    system_role: str = field(default="system", metadata={"help": "System role"})
    system_prompt: str = field(default="", metadata={"help": "System description"})
    human_role: str = field(default="human", metadata={"help": "Human role"})
    gpt_role: str = field(default="gpt", metadata={"help": "GPT role"})

    @abstractmethod
    def get_template(self) -> str:
        pass

    @abstractmethod
    def add_default_system_message(self, messages):
        pass
