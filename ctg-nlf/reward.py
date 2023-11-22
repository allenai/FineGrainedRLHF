# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the super class for the reward functions
#  Referenced: https://github.com/liujch1998/rainier/
#  All Rights Reserved.
#  *********************************************************

import abc
import numpy as np
import torch
from typing import Optional, List, Iterable, Dict, Any, Tuple
from .utils import mask_pad


class BasicReward(metaclass=abc.ABCMeta):

    def __init__(self,
                 kl_coef,
                ):
        self.kl_coef = kl_coef
        
    @abc.abstractmethod
    def get_reward(self,
                   prompts_input_ids: torch.tensor, # (B, input_len)
                   prompts_attention_mask: torch.tensor, # (B, input_len)
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str], # [B]
                   metadata = None,
                  ) -> Dict[str, List[List[float]]]:
        
        # output format: {'rewards/raw': [[0.0, ...], [0.0, ...], ...]}
        # where each sublist is a list of token-level rewards for a single example
        
        pass
    
    def eval_metrics(self,
                   prompts_input_ids: torch.tensor, # (B, input_len)
                   prompts_attention_mask: torch.tensor, # (B, input_len)
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str], # [B]
                   metadata = None,
                  ) -> Dict[str, List[float]]:
        
        training_rewards = self.get_reward(prompts_input_ids, prompts_attention_mask, 
                                           generated_input_ids, generated_attention_mask, generated_texts, metadata)
        return {
            "eval/rewards": [np.sum(sublist) for sublist in training_rewards['rewards/raw']],
        }

    def kl_penalize_reward(self, results):
        logprobs = results['generated_logprobs']
        ref_logprobs = results['generated_ref_logprobs']
        mask = results['generated_attention_mask']
        
        # should be a list of length B to avoid gradient descent
        raw_rewards = results['rewards/raw'] 
        kl = mask_pad(logprobs - ref_logprobs, mask, pad_value=0.)
        kl_penalty = self.kl_coef * kl
        RL = logprobs.size(1)
        
        flattened_rewards = torch.tensor([
            r + [0.] * (RL-len(r))
            for r in raw_rewards
        ], device=logprobs.device) 
        
        penalized_rewards = flattened_rewards - kl_penalty

        results['rewards/raw'] = flattened_rewards 
        results['rewards/kl'] = kl 
        results['rewards/kl_penalty'] = kl_penalty 
        results['rewards/penalized'] = penalized_rewards 
        
    def aggregate_metrics(self, wandb_table, value_columns):
        # how to average over the metrics in wandb table for reporting
        # default: just average over all eval samples
        stats = {}
        for k in value_columns:
            stats[k] = np.mean([row[wandb_table.columns.index(k)] for row in wandb_table.data])
        return stats