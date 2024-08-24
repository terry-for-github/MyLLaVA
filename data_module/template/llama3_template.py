from dataclasses import dataclass, field

from .base_template import BaseTemplate


@dataclass(frozen=True)
class Llama3Template(BaseTemplate):
    sys_role: str = field(default="system")
    human_role: str = field(default="user")
    gpt_role: str = field(default="assistant")
    system_prompt: str = field(default=(
        "You are a highly intelligent and helpful language and vision AI assistant. "
        "Whenever an image is present in the conversation, very carefully examine it "
        "and consider its content when formulating your response. You should give concise "
        "responses to very simple questions, but provide thorough responses to more complex "
        "and open-ended questions."
    ))

    def get_template(self):
        return (
            f"{{% set loop_messages = messages %}}"
            f"{{% for message in loop_messages %}}"
            f"{{% if loop.index0 == 0 %}}"
            f"{{{{ bos_token }}}}"
            f"{{{{ '<|start_header_id|>' + '{self.sys_role}' + '<|end_header_id|>\n\n' }}}}"
            f"{{{{ '{self.system_prompt}' + eos_token }}}}"
            f"{{% endif %}}"
            f"{{% if loop.index0 % 2 == 0 %}}"
            f"{{{{ '<|start_header_id|>' + '{self.human_role}' + '<|end_header_id|>\n\n' }}}}"
            f"{{{{ message['content']|trim + eos_token }}}}"
            f"{{% else %}}"
            f"{{{{ '<|start_header_id|>' + '{self.gpt_role}' + '<|end_header_id|>\n\n' }}}}"
            f"{{% generation %}}"
            f"{{{{ message['content']|trim + eos_token }}}}"
            f"{{% endgeneration %}}"
            f"{{% endif %}}"
            f"{{% endfor %}}"
            f"{{% if add_generation_prompt %}}"
            f"{{{{ '<|start_header_id|>' + '{self.gpt_role}' + '<|end_header_id|>\n\n'}}}}"
            f"{{% endif %}}"
        )


@dataclass(frozen=True)
class Llama3PlainTemplate(Llama3Template):
    def get_template(self):
        return (
            f"{{% set loop_messages = messages %}}"
            f"{{% for message in loop_messages %}}"
            f"{{% if loop.index0 == 0 %}}"
            f"{{{{ bos_token }}}}"
            f"{{% endif %}}"
            f"{{% if loop.index0 % 2 == 0 %}}"
            f"{{{{ '<|start_header_id|>' + '{self.human_role}' + '<|end_header_id|>\n\n' }}}}"
            f"{{{{ message['content']|trim + eos_token }}}}"
            f"{{% else %}}"
            f"{{{{ '<|start_header_id|>' + '{self.gpt_role}' + '<|end_header_id|>\n\n' }}}}"
            f"{{% generation %}}"
            f"{{{{ message['content']|trim + eos_token }}}}"
            f"{{% endgeneration %}}"
            f"{{% endif %}}"
            f"{{% endfor %}}"
            f"{{% if add_generation_prompt %}}"
            f"{{{{ '<|start_header_id|>' + '{self.gpt_role}' + '<|end_header_id|>\n\n'}}}}"
            f"{{% endif %}}"
        )
