from typing import Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .llava_meta import LlavaMetaModel, LlavaMetaForCausalLM, LlavaMetaConfig
from constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


# LlavaMetaConfig must be the first base class
# Otherwise, the __init__ method of it wont be called
class LlavaLlamaConfig(LlavaMetaConfig, LlamaConfig):
    model_type = "llava_llama"


# LlavaMetaModel must be the first base class
# Otherwise, the __init__ method of it wont be called
class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaLlamaConfig


# LlavaMetaForCausalLM has no __init__ method
# So it can also be the second base class
class LlavaLlamaForCausalLM(LlavaMetaForCausalLM, LlamaForCausalLM):
    config_class = LlavaLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        print('[DEBUG]', 1, '===============================================================')
        print('[DEBUG]', 1, 'LlavaLlamaForCausalLM init')
        print('[DEBUG]', 1, 'config tie weight')
        print('[DEBUG]', 1, getattr(self.config, "tie_word_embeddings", None))
        print('[DEBUG]', 1, getattr(self.config, "tie_encoder_decoder", None))
        print('[DEBUG]', 1, '_init_weights', bool(self._init_weights))
        print('[DEBUG]', 1, 'supports gradient_checkpointing')
        print('[DEBUG]', 1, self.supports_gradient_checkpointing)
        print('[DEBUG]', 1, 'gradient_checkpointing')
        print('[DEBUG]', 1, self.model.gradient_checkpointing)
        print('[DEBUG]', 1, '===============================================================')

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        assert inputs_embeds is None and position_ids is None
        inputs_embeds, attention_mask, labels = self.prepare_inputs_for_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images
        )  # type: ignore

        return super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
            **kwargs
        )

    def prepare_inputs_for_forward(
        self,
        input_ids,
        attention_mask,
        labels,
        images
    ):
        '''
        1. Embed the input_ids except the image tokens
        2. Encode the images to image_features
        3. Replace the image tokens with image_features
        4. Insert it into the right position of the input_embeds
        5. Pad the length of the attention_mask, labels
        '''
        embed_tokens = self.get_input_embeddings()  # type: ignore
        if images is None:
            return embed_tokens(input_ids), attention_mask, labels

        # input_ids: batch_size x seq_len
        # image_token_pos: (tensor(r1, r2, r3, ...), tensor(c1, c2, c3, ...))
        image_token_pos = torch.where(input_ids == IMAGE_TOKEN_INDEX)
        batch_idx, seq_idx = image_token_pos

        # Each (r, c) is the position of an image token
        assert len(batch_idx) == images.size(0)

        # Replace the image tokens with pad tokens so it can be embedded directly
        input_ids[image_token_pos] = self.config.pad_token_id  # type: ignore
        input_embeds = embed_tokens(input_ids)

        # image_features: image_num x patch_num x dim
        image_features = self.encode_images(images)
        image_token_num, llm_hidden_size = image_features[0].size()
        image_idx = 0

        # Build the require input_embeds, attention_mask, labels
        # (Because of inserting image patch tokens)
        input_embed_list = []
        attention_mask_list = []
        label_list = []
        device = self.device  # type: ignore
        for i in range(len(input_ids)):
            input_embed = input_embeds[i]
            attn_mask = attention_mask[i]
            label = labels[i]
            # If the current input has no image token, pad it with zeros
            if image_idx == len(batch_idx) or i != batch_idx[image_idx]:
                # '-1' for the IMAGE_TOKEN_INDEX
                # The padded input_embed is set to zeros
                # (zero tensor is the default padding tensor)
                zero_float = torch.zeros(image_token_num-1, llm_hidden_size, device=device)
                input_embed_list.append(torch.cat([input_embed, zero_float]))
                # The padded attention_mask is set to False
                zero_bool = torch.zeros(image_token_num-1, dtype=torch.bool, device=device)
                attention_mask_list.append(torch.cat([attn_mask, zero_bool]))
                # The padded labels are ignored
                ignore_long = torch.full((image_token_num-1,), IGNORE_INDEX,
                                         dtype=torch.long, device=device)
                label_list.append(torch.cat([label, ignore_long]))
                continue
            # Insert image features into the input_embeds
            # '+1' skips the IMAGE_TOKEN_INDEX, which is a placeholder.
            image_pos = seq_idx[image_idx]
            input_embed_list.append(torch.cat([
                input_embed[:image_pos],
                image_features[image_idx],
                input_embed[image_pos+1:]
            ]))
            attention_mask_list.append(torch.cat([
                attn_mask[:image_pos],
                torch.ones(image_token_num, dtype=torch.bool, device=device),
                attn_mask[image_pos+1:]
            ]))
            label_list.append(torch.cat([
                label[:image_pos],
                torch.full((image_token_num,), IGNORE_INDEX, dtype=torch.long, device=device),
                label[image_pos+1:]
            ]))
            image_idx += 1
            assert image_idx <= len(batch_idx)
        new_input_embeds = torch.stack(input_embed_list)
        new_attention_mask = torch.stack(attention_mask_list)
        new_labels = torch.stack(label_list)
        # We've dealed with the long dialog in the data_collator already
        # So we dont need to check the length of the input_embeds here
        return new_input_embeds, new_attention_mask, new_labels

    # def prepare_inputs_for_generation(self):
    #     pass

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        pass


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModelForCausalLM.register(LlavaLlamaConfig, LlavaLlamaForCausalLM)
