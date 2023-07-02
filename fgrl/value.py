# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the value functions for the fine-grained RL
#  Referenced: https://github.com/liujch1998/rainier/
#  All Rights Reserved.
#  *********************************************************


from typing import Union, List, Dict
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from typing import Optional, List, Iterable, Dict, Any, Tuple
from .utils import logits_to_entropy, mask_pad


class MLP(torch.nn.Module):
    
    def __init__(self, d_model, d_out) -> None:
        super().__init__()
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, d_out),
        )
    
    def forward(self, x):
        return self.model(x)


class T5Value:

    def __init__(self,
                 model_ckpt: str,
                 model,
                 tokenizer,
                 accelerator,
                 freeze_model: bool = False,
                ):
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        
        if model is not None:
            self.model = model
            return

        self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
            
        self.linear = MLP(self.model.config.d_model, 1)
        
        # freeze all parameters except the last layer
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False


    def forward_pass(self,
                     prompts_input_ids: torch.Tensor, # (B, input_len)
                     prompts_attention_mask: torch.Tensor, # (B, input_len)
                     generated_input_ids: torch.Tensor, # (B, output_len)
                     generated_attention_mask: torch.Tensor, # (B, output_len)
                    ):

        outputs = self.model(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            labels=mask_pad(generated_input_ids, generated_attention_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        logits = self.linear(outputs.decoder_hidden_states[-1]).squeeze(-1) # (B, output_len)
        results = {
                'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, output_len)
        }

        return results
