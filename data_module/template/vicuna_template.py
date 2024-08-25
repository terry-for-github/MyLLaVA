from dataclasses import dataclass, field

from .base_template import BaseTemplate


@dataclass(frozen=True)
class VicunaTemplate(BaseTemplate):
    human_role: str = field(default="USER")
    gpt_role: str = field(default="ASSISTANT")
    system_prompt: str = field(default=(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user\\'s questions."
    ))

    def get_template(self):
        return (
            f"{{% set loop_messages = messages %}}"
            f"{{% for message in loop_messages %}}"
            f"{{% if loop.index0 == 0 %}}"
            f"{{{{ bos_token + '{self.system_prompt}' }}}}"
            f"{{% endif %}}"
            f"{{% if loop.index0 % 2 == 0 %}}"
            f"{{{{ '\n\n{self.human_role}: ' + message['content']|trim }}}}"
            f"{{% else %}}"
            f"{{{{ '\n\n{self.gpt_role}: ' }}}}"
            f"{{% generation %}}"
            f"{{{{ message['content']|trim + eos_token }}}}"
            f"{{% endgeneration %}}"
            f"{{% endif %}}"
            f"{{% endfor %}}"
            f"{{% if add_generation_prompt %}}"
            f"{{{{ '{self.gpt_role}' }}}}"
            f"{{% endif %}}"
        )
