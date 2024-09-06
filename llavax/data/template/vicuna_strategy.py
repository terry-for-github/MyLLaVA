from .template_strategy import TemplateStrategy

from ...constants import IMAGE_MARK


class VicunaStrategy(TemplateStrategy):
    def get_chat_template(self) -> str:
        return (
            "{% if messages[0]['role'] == 'SYSTEM' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            f"{{% set system_message = '{self.system_prompt}' %}}"
            "{% endif %}"
            "{{ '<s>' + system_message }}"
            "{% for message in loop_messages %}"
            "{{ ' ' + message['role'] + ': ' }}"
            "{% if message['role'] == 'USER' %}"
            "{{ message['content'] }}"
            "{% elif message['role'] == 'ASSISTANT' %}"
            "{% generation %}"
            "{{ message['content'] + '</s>' }}"
            "{% endgeneration %}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ ' ASSISTANT: ' }}"
            "{% endif %}"
        )

    @property
    def system_role(self) -> str:
        return 'SYSTEM'

    @property
    def user_role(self) -> str:
        return 'USER'

    @property
    def assistant_role(self) -> str:
        return 'ASSISTANT'

    @property
    def system_prompt(self) -> str:
        return (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user\\'s "
            "questions."
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
