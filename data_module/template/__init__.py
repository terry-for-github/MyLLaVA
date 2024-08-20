from typing import Dict

from .base_template import BaseTemplate
from .plain_template import PlainTemplate
from .vicuna_template import VicunaTemplate
from .llama3_template import Llama3Template

template_dict: Dict[str, BaseTemplate] = {
    "plain": PlainTemplate(),
    "vicuna": VicunaTemplate(),
    "llama3": Llama3Template()
}

__all__ = ['template_dict']
