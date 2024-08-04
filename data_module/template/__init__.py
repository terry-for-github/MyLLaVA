from typing import Dict

from .base_template import BaseTemplate
from .plain_template import PlainTemplate
from .vicuna_template import VicunaTemplate


template_dict: Dict[str, BaseTemplate] = {
    "plain": PlainTemplate(),
    "vicuna": VicunaTemplate()
}


def get_template(version: str) -> str:
    return template_dict[version].get_template()


__all__ = ['get_template']
