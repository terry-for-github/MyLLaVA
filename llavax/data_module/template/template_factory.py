from .base_template import BaseTemplate
from .plain_template import PlainTemplate
from .vicuna_template import VicunaTemplate
from .llama3_template import Llama3Template


class TemplateFactory:
    @staticmethod
    def create_template(version: str) -> BaseTemplate:
        if version == 'plain':
            return PlainTemplate()
        elif version == 'vicuna':
            return VicunaTemplate()
        elif version == 'llama3':
            return Llama3Template()
        else:
            raise ValueError(f'Unsupported version: {version}')
