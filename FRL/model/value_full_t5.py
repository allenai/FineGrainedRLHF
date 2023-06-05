from typing import Union, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from utils.utils import logits_to_entropy, mask_pad



class Value:

    def __init__(self,
                 model_type: str,
                 model_ckpt: str,
                 model,
                 tokenizer,
                 accelerator,
                 freeze_model: bool = True,
                ):
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        
        if model is not None:
            self.model = model
            return

        # if model_ckpt is not None:
        self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
        # else:
        #     self.model = T5ForConditionalGeneration.from_pretrained(model_type)
            
        # self.linear = MLP(self.model.config.d_model, 1)
        self.linear = torch.nn.Linear(self.model.config.d_model, 1)
        # self.model.eval()
        
        # freeze all parameters
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False


    def forward_pass(self,
                     prompts_input_ids: torch.Tensor, # (B, QL)
                     prompts_attention_mask: torch.Tensor, # (B, QL)
                     generated_input_ids: torch.Tensor, # (B, KL)
                     generated_attention_mask: torch.Tensor, # (B, KL)
                    ):

        outputs = self.model(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            labels=mask_pad(generated_input_ids, generated_attention_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        logits = self.linear(outputs.decoder_hidden_states[-1]).squeeze(-1) # (B, KL)
        results = {
                'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, KL)
        }

        return results
