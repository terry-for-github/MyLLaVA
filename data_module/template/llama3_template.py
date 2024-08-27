from dataclasses import dataclass, field

from .base_template import BaseTemplate


@dataclass(frozen=True)
class Llama3Template(BaseTemplate):
    sys_role: str = field(default="system")
    human_role: str = field(default="user")
    gpt_role: str = field(default="assistant")
    system_prompt: str = field(default=(
        "You are a highly intelligent and helpful language and vision AI assistant. "
        "Give concise responses to very simple questions, but provide thorough responses "
        "to more complex and open-ended questions."
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
            f"{{{{ '<|begin_of_text|>' }}}}"
            f"{{{{ '<|start_header_id|>' + '{self.sys_role}' + '<|end_header_id|>\n' }}}}"
            f"{{{{ system_message + '<|eot_id|>' }}}}"
            f"{{% for message in loop_messages %}}"
            f"{{% if loop.index0 % 2 == 0 %}}"
            f"{{{{ '\n<|start_header_id|>' + '{self.human_role}' + '<|end_header_id|>\n' }}}}"
            f"{{{{ message['content']|trim + '<|eot_id|>' }}}}"
            f"{{% else %}}"
            f"{{{{ '\n<|start_header_id|>' + '{self.gpt_role}' + '<|end_header_id|>\n' }}}}"
            f"{{% generation %}}"
            f"{{{{ message['content']|trim + '<|eot_id|>' }}}}"
            f"{{% endgeneration %}}"
            f"{{% endif %}}"
            f"{{% endfor %}}"
            f"{{% if add_generation_prompt %}}"
            f"{{{{ '\n<|start_header_id|>' + '{self.gpt_role}' + '<|end_header_id|>\n'}}}}"
            f"{{% endif %}}"
        )

    def add_default_system_message(self, messages):
        return [{'role': self.sys_role, 'content': self.system_prompt}] + messages
