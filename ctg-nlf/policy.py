# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the policy functions for the fine-grained RL
#  Referenced: https://github.com/liujch1998/rainier/
#  All Rights Reserved.
#  *********************************************************

from typing import Union, List, Dict
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer
from typing import Optional, List, Iterable, Dict, Any, Tuple
from utils import logits_to_entropy, mask_pad, find_last_non_masked_ids, NEGATIVE_INF


class T5Policy:

    def __init__(self,
                 model_ckpt: str,
                 device,
                 temperature: float = 1.0,
                 nlf_cond: bool = False,
                ):
        
        self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt, model_max_length=1024)
        self.tokenizer.max_input_len = 1024
        self.tokenizer.max_generated_len = 200
        self.device = device
        self.temperature = temperature
        self.nlf_cond = nlf_cond
        if self.nlf_cond:
            self.tokenizer.feedback_prefix = "feedback: "
            self.tokenizer.prompt_prefix = "input: "
        self.model = self.model.to(self.device)
        self.model.eval()


    def sample(self,
               prompts_input_ids: torch.Tensor, # (B, input_len)
               prompts_attention_mask: torch.Tensor, # (B, input_len)
               do_sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
               num_beams: int = 1,
               num_return_sequences: int = 1,
              ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        
        if temperature is None:
            temperature = self.temperature

        prompts_input_ids = prompts_input_ids.to(self.device)
        prompts_attention_mask = prompts_attention_mask.to(self.device)

        prompts_text = self.tokenizer.batch_decode(prompts_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        self.model.eval()
        if do_sample:
            generated_input_ids = self.model.generate(
                input_ids=prompts_input_ids,
                attention_mask=prompts_attention_mask,
                max_length=self.tokenizer.max_generated_len + 1,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
            ) # begins with 0 ([BOS]); ends with 1 ([EOS])
            
        else:
            generated_input_ids = self.model.generate(
                input_ids=prompts_input_ids,
                attention_mask=prompts_attention_mask,
                max_length=self.tokenizer.max_generated_len + 1,
                num_beams=num_beams,
                do_sample=False,
                num_return_sequences=num_return_sequences,
            )

        generated_input_ids = generated_input_ids[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])

        generated_input_ids = F.pad(generated_input_ids, (0, self.tokenizer.max_generated_len - generated_input_ids.size(1)), value=self.tokenizer.pad_token_id) # (B, output_len)
        generated_attention_mask = (generated_input_ids != self.tokenizer.pad_token_id).long()
        generated_text = self.tokenizer.batch_decode(generated_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # repeat input sequences for num_return_sequences times
        prompts_text = [elem for elem in prompts_text for _ in range(num_return_sequences)]
        
        return {
            'prompts_text': prompts_text,
            'prompts_input_ids': prompts_input_ids.repeat_interleave(num_return_sequences, dim=0), # (B, input_len)
            'prompts_attention_mask': prompts_attention_mask.repeat_interleave(num_return_sequences, dim=0), # (B, input_len)
            'generated_text': generated_text,
            'generated_input_ids': generated_input_ids, # (B, output_len)
            'generated_attention_mask': generated_attention_mask, # (B, output_len)
        }
    

    def forward_pass(self,
                     prompts_input_ids: torch.Tensor, # (B, input_len)
                     prompts_attention_mask: torch.Tensor, # (B, input_len)
                     generated_input_ids: torch.Tensor, # (B, output_len)
                     generated_attention_mask: torch.Tensor, # (B, output_len)
                    ):

        prompts_input_ids = prompts_input_ids.to(self.device)
        prompts_attention_mask = prompts_attention_mask.to(self.device)
        generated_input_ids = generated_input_ids.to(self.device)
        generated_attention_mask = generated_attention_mask.to(self.device)

        outputs = self.model(
            input_ids=prompts_input_ids,
            attention_mask=prompts_attention_mask,
            labels=mask_pad(generated_input_ids, generated_attention_mask, -100),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )

        generated_logits = outputs.logits # (B, output_len, V)
        logprobs = F.log_softmax(generated_logits, dim=-1)
        generated_logprobs = torch.gather(logprobs, 2, generated_input_ids[:, :, None]).squeeze(2) # (B, output_len)
        generated_entropy = logits_to_entropy(generated_logits) # (B, output_len)
        lm_loss = -1. * generated_logprobs

        results = {
            'generated_logits': generated_logits, # (B, output_len, V)
            'generated_logprobs': mask_pad(generated_logprobs, generated_attention_mask), # (B, output_len)
            'generated_entropy': mask_pad(generated_entropy, generated_attention_mask), # (B, output_len)
            'lm_loss': mask_pad(lm_loss, generated_attention_mask) # (B, output_len)
        }

        return results