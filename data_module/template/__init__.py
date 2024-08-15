from typing import Dict

from .base_template import BaseTemplate
from .plain_template import PlainTemplate
from .vicuna_template import VicunaTemplate

template_dict: Dict[str, BaseTemplate] = {
    "plain": PlainTemplate(),
    "vicuna": VicunaTemplate()
}

__all__ = ['template_dict']
