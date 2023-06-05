import torch
from model.t5 import T5ForTokenRegression
from utils.utils import mask_pad


class Value:

    def __init__(self,
                 model_type,
                 model_ckpt,
                 model,
                 tokenizer,
                ):
        self.tokenizer = tokenizer

        if model is not None:
            self.model = model
            return

        # self.model = T5ForTokenRegression.from_pretrained(model_type)
        
        if model_ckpt is None:
            self.model = T5ForTokenRegression.from_pretrained(model_type)
        else:
            self.model = T5ForTokenRegression.from_pretrained(model_ckpt)
        self.model.config.pad_token_id = tokenizer.pad_token_id
        
        # if model_ckpt is not None:
        #     # checkpoint = torch.load(model_ckpt, map_location='cpu')
        #     # self.model.load_state_dict(checkpoint, strict=False)
        #     # checkpoint.clear()
        #     self.model.from_pretrained(model_ckpt)
            
        self.model.eval()

    def forward_pass(self,
                     prompts_input_ids: torch.Tensor, # (B, QL)
                     prompts_attention_mask: torch.Tensor, # (B, QL)
                     generated_input_ids: torch.Tensor, # (B, KL)
                     generated_attention_mask: torch.Tensor, # (B, KL)
                    ):

        outputs = self.model.forward_cls(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            labels=mask_pad(generated_input_ids, generated_attention_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        return {
            'generated_value': mask_pad(outputs.logits, generated_attention_mask), # (B, KL)
        }
