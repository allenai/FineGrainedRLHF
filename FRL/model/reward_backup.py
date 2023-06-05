import json
import os
from typing import Optional, List, Iterable, Dict, Any, Tuple
import torch, spacy
import torch.nn.functional as F
from transformers import AutoTokenizer
from .my_longformer import LongformerForSequenceClassification, LongformerForTokenClassification
from utils.utils import reduce_mean
import abc
import numpy as np
import logging

logging.basicConfig(level=logging.ERROR)

class BasicReward(metaclass=abc.ABCMeta):

    def __init__(self,
                 kl_coef,
                 gain = 1.0,
                 bias = 0.0
                ):
        
        
        self.kl_coef = kl_coef
        self.gain, self.bias = gain, bias

        
    @abc.abstractmethod
    def get_reward(self,
                   prompts_input_ids: torch.tensor, # (B, QL)
                   prompts_attention_mask: torch.tensor, # (B, QL)
                   generated_input_ids: torch.tensor, # (B, RL)
                   generated_attention_mask: torch.tensor, # (B, RL)
                   generated_texts: List[str], # [B]
                   metadata = None,
                   override_gain = None,
                   override_bias = None,
                  ):
        pass

    def kl_penalize_reward(self, results):
        logprobs = results['generated_logprobs']
        ref_logprobs = results['generated_ref_logprobs']
        
        # should be a list of length B to avoid gradient descent
        normalized_rewards = results['rewards/normalized'] 

        kl = logprobs - ref_logprobs
        kl_penalty = self.kl_coef * kl
        RL = logprobs.size(1)
        
        flattened_rewards = torch.tensor([
            r + [0.] * (RL-len(r))
            for r in normalized_rewards
        ], device=logprobs.device) # (B, KL)
        # mask = results['generated_attention_mask']
        # flattened_rewards = torch.tensor([
        #     [0.] * (l-1) + [r] + [0.] * (RL-l)
        #     for r, l in zip(normalized_rewards, torch.sum(mask, dim=1).tolist())
        # ], device=logprobs.device) # (B, KL)
        
        penalized_rewards = flattened_rewards - kl_penalty
        # TODO: This is slightly different from the paper

        results['rewards/original'] = flattened_rewards # (B, KL)
        results['rewards/kl'] = kl # (B, KL)
        results['rewards/kl_penalty'] = kl_penalty # (B, KL)
        results['rewards/penalized'] = penalized_rewards # (B, KL)


    def write_reward_norm(self, reward_dir):
        reward_dict = {
            'gain': self.gain,
            'bias': self.bias,
        }
        with open(os.path.join(reward_dir, 'reward_normalization.json'), 'w') as f:
            json.dump(reward_dict, f, indent=4)

    def read_reward_norm(self, reward_dir):
        with open(os.path.join(reward_dir, 'reward_normalization.json')) as f:
            reward_dict = json.load(f)
        self.gain = reward_dict['gain']
        self.bias = reward_dict['bias']



class BaselineReward(BasicReward):
    
    def __init__(self,
                 tokenizer,
                 model_ckpt,
                 kl_coef,
                 batch_size=None,
                 gain = 1.0,
                 bias = 0.0
                ):
        
        
        super().__init__(kl_coef, gain, bias)
        
        # initialize model
        # self.tokenizer = LongformerTokenizerFast.from_pretrained(model_ckpt)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
        self.model = LongformerForSequenceClassification.from_pretrained(model_ckpt)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.batch_size = batch_size
        if self.batch_size is not None:
            self.batchify = lambda x: [x[i:i+batch_size] for i in range(0, len(x), batch_size)]
        
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, RL)
                   generated_attention_mask: torch.tensor, # (B, RL)
                   generated_texts: List[str],
                   metadata=None, 
                   override_gain=None, 
                   override_bias=None):
        
        reward_model_inputs = [this_meta["pred_error_reward_model_input"] + gen_text for this_meta, gen_text in zip(metadata, generated_texts)]
        
        # print(f"reward_model_inputs: {reward_model_inputs}")
        
        batched_reward_inputs = [reward_model_inputs]
        
        if self.batch_size is not None:
            batched_reward_inputs = self.batchify(reward_model_inputs)
            
        sequence_level_reward = []
        
        with torch.no_grad():
            for input_batch in batched_reward_inputs:
                inputs = self.tokenizer(input_batch, padding=True, truncation=True, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                outputs = self.model(**inputs)
                
                sequence_level_reward += outputs['logits'].squeeze(-1).tolist() 
        
        # print(f"sequence_level_reward: {sequence_level_reward}")
        
        # align with generated texts, make it fine-grained
        fine_grained_reward = [
            [0.] * (l-1) + [r]
            for r, l in zip(sequence_level_reward, torch.sum(generated_attention_mask, dim=1).tolist())
        ]
                
        # normalize the rewards        
        gain = self.gain if override_gain is None else override_gain
        bias = self.bias if override_bias is None else override_bias
        
        fine_grained_reward_normalized = [
            (np.array(r)*gain + bias).tolist()  for r in fine_grained_reward
        ]
        
        # print(f"fine_grained_reward_normalized: {fine_grained_reward_normalized}\n----------------")
        

        return {
            'rewards/raw': fine_grained_reward, # list (B)
            'rewards/normalized': fine_grained_reward_normalized, # list (B)
        }



class FineGrainedNFReward(BasicReward):
    
    def __init__(self,
                 tokenizer,
                 model_ckpt,
                 kl_coef,
                 batch_size=None,
                 gain = 1.0,
                 bias = 0.0,
                 positive_reward = 1.0,
                 negative_reward = -1.0,
                 sep = "[SEP]",
                ):
        
        
        super().__init__(kl_coef, gain, bias)
        
        # initialize model
        # self.tokenizer = LongformerTokenizerFast.from_pretrained(model_ckpt)
        
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer

        # prepare reward model and tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
        self.reward_model = LongformerForTokenClassification.from_pretrained(model_ckpt)
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()

        # prepare spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.sep = sep

        # only work for longformer for now. Change the 1:-1 for other tokenizer
        self.sep_reward_pattern = self.reward_tokenizer(self.sep, return_tensors="pt").input_ids[0][1:-1]
        self.sep_reward_pattern_len = len(self.sep_reward_pattern)

        self.error_type_to_reward = {1: negative_reward, 2: positive_reward}

        self.batch_size = batch_size
        if self.batch_size is not None:
            self.batchify = lambda x: [x[i:i+batch_size] for i in range(0, len(x), batch_size)]

    def find_sep_position(self, input_ids):
        seq_len = len(input_ids)

        indices = []

        for i in range(seq_len - self.sep_reward_pattern_len + 1):
            window = input_ids[i:i + self.sep_reward_pattern_len]
            if torch.equal(window, self.sep_reward_pattern):
                indices.append(i)

        return indices
    
    def process_one_generation(self, long_text, policy_text_len):
        doc = self.nlp(long_text)
        sentence_tokens = []

        # prepare the reward model input
    
        this_sentence_tokens = []
        for token in doc:
            if token.is_sent_start:
                if len(this_sentence_tokens) != 0:
                    sentence_tokens.append(this_sentence_tokens)
                this_sentence_tokens = []
        
            this_sentence_tokens.append(token.text)
    
        if len(this_sentence_tokens) != 0:
            sentence_tokens.append(this_sentence_tokens)
        
        reward_sentences = [f"{self.sep} {' '.join(sent)}" for sent in sentence_tokens]
    
        reward_input = ' '.join(reward_sentences)
    
        # get the indices of sentence end for t5 tokenizer
    
        # Initialize an empty list to store the end token indices of each sentence
        sentence_end_indices = []
    
        running_token_count = 0
        for sent in doc.sents:
            tokens = self.policy_tokenizer.tokenize(str(sent))
            token_count = len(tokens)
            running_token_count += token_count
            sentence_end_indices.append(running_token_count - 1)
        
        # in case overlength
    
        sentence_end_indices = [min(item, policy_text_len-1) for item in sentence_end_indices]
    
        return reward_input, sentence_end_indices

        
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, RL)
                   generated_attention_mask: torch.tensor, # (B, RL)
                   generated_texts: List[str],
                   metadata=None, 
                   override_gain=None, 
                   override_bias=None):
        
        batch_reward_inputs = []
        batch_sentence_end_indices = []

        # get the length of generated outputs
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()

        for batch_idx, (metadata, gen_text) in enumerate(zip(metadata, generated_texts)):
            reward_input, sentence_end_indices = self.process_one_generation(gen_text, 
                                                    policy_inputs_lens[batch_idx])
            question = metadata["question"]

            # formating template for the reward model
            reward_input = f"question : {' '.join([tok.text for tok in self.nlp(question)])} [SEP] answer : {reward_input}"

            batch_reward_inputs.append(reward_input)
            batch_sentence_end_indices.append(sentence_end_indices)
        
        # get the reward
        with torch.no_grad():
            inputs =self.reward_tokenizer(batch_reward_inputs, truncation=True, padding=True, return_tensors="pt")
            inputs = inputs.to(self.reward_model.device)
            batch_pred = self.reward_model(**inputs)



        fine_grained_reward = []
        for text_idx, generated_text in enumerate(generated_texts):

            # get the reward for each sentence
            this_pred = batch_pred.logits[text_idx].detach().cpu()
            generated_text_input_ids = self.reward_tokenizer(batch_reward_inputs[text_idx], return_tensors="pt", truncation=True).input_ids[0]

            sep_indices = self.find_sep_position(generated_text_input_ids)
            sentence_reward_probs = this_pred[sep_indices]

            # align back to the original sentence
            policy_sentence_end_indices = batch_sentence_end_indices[text_idx]
            policy_inputs_len = policy_inputs_lens[text_idx]

            this_reward = [0]*policy_inputs_len

            # only keep the rewards in the generated text, rather than prompt. Now work for [SEP] only
            n_sep_in_answer = len(policy_sentence_end_indices)
            sentence_reward_probs = sentence_reward_probs[-n_sep_in_answer:]

            for i, end_idx in enumerate(policy_sentence_end_indices):
                # 1 is NF-ERROR, 2 is no error
                this_reward[end_idx] = self.error_type_to_reward[torch.argmax(sentence_reward_probs[i][1:]).item() + 1]
            
            fine_grained_reward.append(this_reward)

                
        # normalize the rewards        
        gain = self.gain if override_gain is None else override_gain
        bias = self.bias if override_bias is None else override_bias
        
        fine_grained_reward_normalized = [
            (np.array(r)*gain + bias).tolist()  for r in fine_grained_reward
        ]
        
        # print(f"fine_grained_reward_normalized: {fine_grained_reward_normalized}\n----------------")
        

        return {
            'rewards/raw': fine_grained_reward, # list (B)
            'rewards/normalized': fine_grained_reward_normalized, # list (B)
        }