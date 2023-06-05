from typing import Union, List, Dict
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration
from utils.utils import logits_to_entropy, mask_pad


class Policy:

    def __init__(self,
                 model_type: str,
                 model_ckpt: str,
                 tokenizer,
                 policy_value_sharing: bool,
                 accelerator,
                ):
        self.tokenizer = tokenizer
        self.policy_value_sharing = policy_value_sharing
        self.accelerator = accelerator

        # if policy_value_sharing:
        #     self.model = T5ForConditionalGenerationAndTokenRegression.from_pretrained(model_type)
        # else:
        self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
        
        # if model_ckpt is not None:
        #     self.model.from_pretrained(model_ckpt)
            
        self.linear = torch.nn.Linear(self.model.config.d_model, 1)    
        self.model.eval()

    def beam_decoding(self,
               prompts_input_ids: torch.Tensor, # (B, QL)
               prompts_attention_mask: torch.Tensor, # (B, QL)
               num_beams: int = 1,
              ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        
        prompts_text = self.tokenizer.batch_decode(prompts_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        generated_input_ids = unwrapped_model.generate(
                input_ids=prompts_input_ids,
                attention_mask=prompts_attention_mask,
                max_length=self.tokenizer.max_generated_len + 1,
                num_beams=num_beams,
                do_sample=False,
                synced_gpus=True,
        ) # begins with 0 ([BOS]); ends with 1 ([EOS])
        generated_input_ids = generated_input_ids[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])

        generated_input_ids = F.pad(generated_input_ids, (0, self.tokenizer.max_generated_len - generated_input_ids.size(1)), value=self.tokenizer.pad_token_id) # (B, KL)
        generated_attention_mask = (generated_input_ids != self.tokenizer.pad_token_id).long()
        generated_text = self.tokenizer.batch_decode(generated_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return {
            'prompts_text': prompts_text,
            'prompts_input_ids': prompts_input_ids, # (B, QL)
            'prompts_attention_mask': prompts_attention_mask, # (B, QL)
            'generated_text': generated_text,
            'generated_input_ids': generated_input_ids, # (B, KL)
            'generated_attention_mask': generated_attention_mask, # (B, KL)
        }
        
    def sample(self,
               prompts_input_ids: torch.Tensor, # (B, QL)
               prompts_attention_mask: torch.Tensor, # (B, QL)
               sample: bool = True,
               top_k: int = None,
               top_p: float = None,
               temperature: float = None,
               num_return_sequences: int = 1,
              ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        
        prompts_text = self.tokenizer.batch_decode(prompts_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        generated_input_ids = unwrapped_model.generate(
                input_ids=prompts_input_ids,
                attention_mask=prompts_attention_mask,
                max_length=self.tokenizer.max_generated_len + 1,
                do_sample=sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                synced_gpus=True,
        ) # begins with 0 ([BOS]); ends with 1 ([EOS])
        generated_input_ids = generated_input_ids[:, 1:].contiguous() # no beginning; ends with 1 ([EOS])

        generated_input_ids = F.pad(generated_input_ids, (0, self.tokenizer.max_generated_len - generated_input_ids.size(1)), value=self.tokenizer.pad_token_id) # (B, KL)
        generated_attention_mask = (generated_input_ids != self.tokenizer.pad_token_id).long()
        generated_text = self.tokenizer.batch_decode(generated_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # repeat input sequences for num_return_sequences times
        prompts_text = [elem for elem in prompts_text for _ in range(num_return_sequences)]
        
        return {
            'prompts_text': prompts_text,
            'prompts_input_ids': prompts_input_ids.repeat_interleave(num_return_sequences, dim=0), # (B, QL)
            'prompts_attention_mask': prompts_attention_mask.repeat_interleave(num_return_sequences, dim=0), # (B, QL)
            'generated_text': generated_text,
            'generated_input_ids': generated_input_ids, # (B, KL)
            'generated_attention_mask': generated_attention_mask, # (B, KL)
        }
    

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

        generated_logits = outputs.logits # (B, KL, V)
        logprobs = F.log_softmax(generated_logits, dim=-1)
        generated_logprobs = torch.gather(logprobs, 2, generated_input_ids[:, :, None]).squeeze(2) # (B, KL)
        generated_entropy = logits_to_entropy(generated_logits) # (B, KL)

        results = {
            'generated_logits': generated_logits, # (B, KL, V)
            'generated_logprobs': mask_pad(generated_logprobs, generated_attention_mask), # (B, KL)
            'generated_entropy': mask_pad(generated_entropy, generated_attention_mask), # (B, KL)
        }

        if self.policy_value_sharing:
            logits = self.linear(outputs.decoder_hidden_states[-1]).squeeze(-1) # (B, KL)
            results.update({
                'generated_value': mask_pad(logits, generated_attention_mask, 0), # (B, KL)
            })

        return results
