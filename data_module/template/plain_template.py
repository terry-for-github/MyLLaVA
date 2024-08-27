from dataclasses import dataclass

from .base_template import BaseTemplate


@dataclass(frozen=True)
class PlainTemplate(BaseTemplate):
    def get_template(self):
        return (
            "{{ bos_token + messages[0]['content']|trim + ' ' }}"
            "{% generation %}"
            "{{ messages[1]['content']|trim + eos_token }}"
            "{% endgeneration %}"
        )

    def add_default_system_message(self, messages):
        return messages
