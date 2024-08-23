import os
from typing import List, Optional
from functools import partial

from PIL import Image
import torch
from transformers import AutoImageProcessor, LayoutLMv3ImageProcessor

from constants import CACHE_DIR, IMAGE_MEAN, IMAGE_SIZE


class ImageLoader:
    def __init__(self, image_folder: str,
                 vision_model_name: str,
                 image_mark: str,
                 image_process_mode: str):
        self.image_folder = image_folder
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name,
                                                                  cache_dir=CACHE_DIR)
        self.image_mark = image_mark
        self.image_mean = IMAGE_MEAN[vision_model_name]
        self.image_size = IMAGE_SIZE[vision_model_name]
        # do not apply ocr
        if isinstance(self.image_processor, LayoutLMv3ImageProcessor):
            self.image_processor.apply_ocr = False

        assert image_process_mode in ['pad', 'warp', 'crop'], "Invalid image_process_mode"
        if image_process_mode == 'pad':
            self.process_func = partial(
                self._expand_to_square,
                background_color=tuple(int(x*255) for x in self.image_mean)
            )
        elif image_process_mode == 'warp':
            self.process_func = partial(self._resize, size=self.image_size)
        elif image_process_mode == 'crop':
            self.process_func = lambda x: x

    @staticmethod
    def _expand_to_square(pil_image, background_color):
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
    def _resize(pil_image, size):
        return pil_image.resize((size, size))

    def __call__(self, image_file: Optional[str]) -> Optional[torch.Tensor]:
        '''Load and preprocess image'''
        if image_file is None:
            return None
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        image = self.process_func(image)
        # image.shape == [1, 3, width, height], type(image) == Tensor
        image = self.image_processor(image, return_tensors='pt')['pixel_values']
        return image[0]
