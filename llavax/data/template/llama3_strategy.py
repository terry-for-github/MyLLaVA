from typing import Optional
from .template_strategy import TemplateStrategy

from ...constants import IMAGE_MARK


class Llama3Strategy(TemplateStrategy):
    def get_chat_template(self) -> str:
        return (
            "{{ '<|begin_of_text|>' }}"
            "{% for message in messages %}"
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
            "{% if message['role'] == 'assistant' %}"
            "{% generation %}"
            "{{ message['content'] + '<|eot_id|>' }}"
            "{% endgeneration %}"
            "{% else %}"
            "{{ message['content'] + '<|eot_id|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
            "{% endif %}"
        )

    @property
    def system_role(self) -> str:
        return 'system'

    @property
    def user_role(self) -> str:
        return 'user'

    @property
    def assistant_role(self) -> str:
        return 'assistant'

    @property
    def system_prompt(self) -> Optional[str]:
        return (
            "You are a highly intelligent and helpful language and vision AI assistant. "
            "Whenever an image is present in the conversation, very carefully examine it "
            "and consider its content when formulating your response. You should give "
            "concise responses to very simple questions, but provide thorough responses "
            "to more complex and open-ended questions."
        )

    def _format_content(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if len(messages) == 0:
            return messages
        for message in messages:
            cnt = message['content'].count(IMAGE_MARK)
            assert cnt <= 1
            if cnt == 0:
                continue
            message['content'] = self.num_vision_token * self.pad_token + \
                message['content'].replace(IMAGE_MARK, '').strip()
        return messages
