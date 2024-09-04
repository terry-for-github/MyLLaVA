from ...constants import IMAGE_MARK

from .template_strategy import TemplateStrategy


class PlainStrategy(TemplateStrategy):
    def get_chat_template(self) -> str:
        return (
            "{{ bos_token + messages[0]['content'] }}"
            "{% generation %}"
            "{{ messages[1]['content'] + eos_token }}"
            "{% endgeneration %}"
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
    def system_prompt(self) -> str:
        return ''

    def _format_content(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        assert len(messages) == 2 and IMAGE_MARK in messages[0]['content']
        messages[0]['content'] = self.num_vision_token * self.pad_token
        messages[1]['content'] = messages[1]['content'].strip()
        return messages
