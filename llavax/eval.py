import os
import logging
from typing import Optional


import torch


class LlavaxGenerateExample:
    def __init__(self, model_pth: str):
        super().__init__()
        assert os.path.isdir(model_pth), f'Model path {model_pth} does not exist.'
        try:
            from llavax import build_image_loader, build_template_applier, build_tokenizer
            from llavax import LlavaLlamaForCausalLM
        except ImportError as e:
            print(e)
            logging.critical('Please install llavax before using LLaVAX model')
            exit(-1)
        # ============================================================================
        # Step 1: Model
        self.torch_dtype = torch.bfloat16
        self.model: LlavaLlamaForCausalLM = LlavaLlamaForCausalLM.from_pretrained(
            model_pth,
            # attn_implementation='flash_attention_2',
            torch_dtype=self.torch_dtype
        )  # type: ignore
        self.model.to(device='cuda')  # type: ignore
        # Step 2: Image Loader
        self.image_loader = build_image_loader(self.model.config.vision_tower, 'pad')
        # Step 3: Tokenizer
        self.tokenizer = build_tokenizer(model_pth)
        # Step 4: Template Applier
        self.template_applier = build_template_applier(
            strategy='llama3' if 'llama3' in model_pth.lower() else 'vicuna',
            model=self.model,
            image_loader=self.image_loader,
            tokenizer=self.tokenizer,
            is_training=False
        )
        # ============================================================================

    def generate_inner(self, messages: list[dict[str, str]], dataset: Optional[str] = None):
        # Step 1: Generation Args
        generate_kwargs = dict(
            max_new_tokens=10,
            do_sample=False,
            num_beams=3,
            pad_token_id=self.tokenizer.pad_token_id
        )
        # Step 2: Building input_tensor and image list
        image_paths = [msg['value'] for msg in messages if msg['type'] == 'image']
        prompts = [msg['value'] for msg in messages if msg['type'] == 'text']
        # FIXME Only one image and one prompt is allowed
        assert len(prompts) == 1 and len(image_paths) == 1
        dialog: list[dict[str, str]] = [
            dict(role='user', content=len(image_paths) * '<image>' + prompts[0])
        ]
        input = self.template_applier.dialog_to_input(dialog)
        batched_input = {k: v.unsqueeze(0).cuda() for k, v in input.items()}
        images = self.image_loader.load_image(image_paths[0]).unsqueeze(0).cuda()
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=batched_input['input_ids'],
                attention_mask=batched_input['attention_mask'],
                vision_token_pos=batched_input['vision_token_pos'],
                images=images.to(dtype=self.torch_dtype),
                **generate_kwargs
            )
        output: str = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return output
