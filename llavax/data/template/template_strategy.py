from abc import ABC, abstractmethod


class TemplateStrategy(ABC):
    def __init__(self, num_vision_token: int, pad_token: str):
        self.num_vision_token = num_vision_token
        self.pad_token = pad_token

    @abstractmethod
    def get_chat_template(self) -> str:
        pass

    @property
    @abstractmethod
    def system_role(self) -> str:
        pass

    @property
    @abstractmethod
    def user_role(self) -> str:
        pass

    @property
    @abstractmethod
    def assistant_role(self) -> str:
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    def format_dialog(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if len(messages) == 0:
            return messages
        messages = self._format_role(messages)
        messages = self._format_content(messages)
        return messages

    def _format_role(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        for message in messages:
            assert message['role'].lower() in ['user', 'assistant', 'system']
            if message['role'].lower() == 'system':
                message['role'] = self.system_role
            elif message['role'].lower() == 'user':
                message['role'] = self.user_role
            elif message['role'].lower() == 'assistant':
                message['role'] = self.assistant_role
            else:
                raise ValueError(f'Unknown role: {message["role"]}')
        return messages

    @abstractmethod
    def _format_content(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        pass
