from dataclasses import dataclass, field

from .base_template import BaseTemplate


@dataclass(frozen=True)
class VicunaTemplate(BaseTemplate):
    sys_role: str = field(default="SYSTEM")
    human_role: str = field(default="USER")
    gpt_role: str = field(default="ASSISTANT")
    system_prompt: str = field(default=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user\\'s questions."
    ))

    def get_template(self):
        return (
            f"{{% set loop_messages = messages %}}"
            f"{{% if messages[0]['role'] == '{self.sys_role}' %}}"
            f"{{% set system_message = messages[0]['content']|trim %}}"
            f"{{% set messages = messages[1:] %}}"
            f"{{% else %}}"
            f"{{% set system_message = '' %}}"
            f"{{% endif %}}"
            f"{{{{ '<s>' + system_message + '</s>' }}}}"
            f"{{% for message in loop_messages %}}"
            f"{{% if loop.index0 % 2 == 0 %}}"
            f"{{{{ '\n{self.human_role}: ' + message['content']|trim }}}}"
            f"{{% else %}}"
            f"{{{{ '\n{self.gpt_role}: ' }}}}"
            f"{{% generation %}}"
            f"{{{{ message['content']|trim + '</s>' }}}}"
            f"{{% endgeneration %}}"
            f"{{% endif %}}"
            f"{{% endfor %}}"
            f"{{% if add_generation_prompt %}}"
            f"{{{{ '\n{self.gpt_role}: ' }}}}"
            f"{{% endif %}}"
        )

    def add_default_system_message(self, messages):
        return [{'role': self.sys_role, 'content': self.system_prompt}] + messages
