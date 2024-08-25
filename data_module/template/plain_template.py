from dataclasses import dataclass

from .base_template import BaseTemplate

from constants import IMAGE_MARK


@dataclass(frozen=True)
class PlainTemplate(BaseTemplate):
    def get_template(self):
        return (
            f"{{{{ bos_token + {IMAGE_MARK} + ' ' }}}}"
            f"{{% generation %}}"
            f"{{{{ messages[1]['content']|trim + eos_token }}}}"
            f"{{% endgeneration %}}"
        )
