from abc import ABC, abstractmethod
from typing import Optional, Union
from functools import partial

from PIL import Image
import torch
from transformers import AutoImageProcessor, LayoutLMv3ImageProcessor

from ..constants import CACHE_DIR, IMAGE_MEAN, IMAGE_SIZE


class BaseImageLoader(ABC):
    def __init__(
        self,
        vision_tower: Union[str, list[str]],
        image_process_mode: str
    ):
        self.vision_tower = vision_tower
        self.image_process_mode = image_process_mode

    @abstractmethod
    def load_image(self, image_path: Optional[str]) -> torch.Tensor:
        '''Load and preprocess image'''
        pass


class SingleTowersImageLoader(BaseImageLoader):
    def __init__(
        self,
        vision_model_name: str,
        image_process_mode: str
    ):
        super().__init__(vision_model_name, image_process_mode)
        self.image_processor = AutoImageProcessor.from_pretrained(
            vision_model_name,
            cache_dir=CACHE_DIR
        )
        self.image_mean = IMAGE_MEAN[vision_model_name]
        self.image_size = IMAGE_SIZE[vision_model_name]
        # do not apply ocr
        if isinstance(self.image_processor, LayoutLMv3ImageProcessor):
            self.image_processor.apply_ocr = False
        self.process_func = self._get_process_func(image_process_mode)

    def _get_process_func(
        self,
        image_process_mode: str
    ) -> partial[Image.Image]:
        if image_process_mode == 'pad':
            return partial(
                self._expand_to_square,
                background_color=(
                    int(self.image_mean[0]*255),
                    int(self.image_mean[1]*255),
                    int(self.image_mean[2]*255),
                )
            )
        elif image_process_mode == 'warp':
            return partial(self._resize, size=self.image_size)
        elif image_process_mode == 'crop':
            return partial(lambda x: x)
        elif image_process_mode == 'no':
            return partial(lambda x: x)
        else:
            raise ValueError(f'Unknown image_process_mode: {image_process_mode}')

    @staticmethod
    def _expand_to_square(
        pil_image: Image.Image,
        background_color: tuple[int, int, int]
    ) -> Image.Image:
        width, height = pil_image.size
        if width == height:
            return pil_image
        if width > height:
            new_image = Image.new('RGB', (width, width), background_color)
            new_image.paste(pil_image, (0, (width - height) // 2))
        else:
            new_image = Image.new('RGB', (height, height), background_color)
            new_image.paste(pil_image, ((height - width) // 2, 0))
        return new_image

    @staticmethod
    def _resize(pil_image: Image.Image, size: int) -> Image.Image:
        return pil_image.resize((size, size))

    def load_image(self, image_path: Optional[str]) -> torch.Tensor:
        '''Load and preprocess image'''
        if image_path is None:
            return torch.randn(3, self.image_size, self.image_size)
        image = Image.open(image_path).convert('RGB')
        image = self.process_func(image)
        # image.shape == [1, 3, width, height], type(image) == Tensor
        image = self.image_processor(image, return_tensors='pt')['pixel_values']
        return image[0]


class MultiTowersImageLoader(BaseImageLoader):
    def __init__(
        self,
        vision_model_list: list[str],
        image_process_mode: str
    ):
        self.image_loader_list = [
            SingleTowersImageLoader(model_name, image_process_mode)
            for model_name in vision_model_list
        ]
        self.image_sizes = [IMAGE_SIZE[model_name] for model_name in vision_model_list]

    def load_image(self, image_path: Optional[str]) -> torch.Tensor:
        '''Load and preprocess multiple images'''
        images = [loader.load_image(image_path) for loader in self.image_loader_list]
        images = torch.cat([image.view(3, -1) for image in images], dim=1)  # type: ignore
        return images
