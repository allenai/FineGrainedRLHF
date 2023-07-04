import json
import os
from typing import Optional, List, Iterable, Dict, Any, Tuple
import torch, spacy
import torch.nn.functional as F
from transformers import AutoTokenizer
import abc
import numpy as np
import logging
import re

from fgrlhf.reward import BasicReward
from fgrlhf.reward_utils import split_text_to_subsentences, split_text_to_sentences
from fgrlhf.evaluators import get_rouge_scores

from my_longformer import LongformerForSequenceClassification, LongformerForTokenClassification

logging.basicConfig(level=logging.ERROR)


class PreferenceReward:
    def __init__(self,
                 tokenizer,
                 model_ckpt,
                 mean = 0.0,
                 std = 1.0,
                 bias = 0.0,
                 scale = 1.0,
                 ):
        
        # use mean and std to normalize the reward
        # use bias and scale to rescale the reward
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer
        
        # prepare reward tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
        self.model = LongformerForSequenceClassification.from_pretrained(model_ckpt)
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.mean = mean
        self.std = std
        
        self.bias = bias
        self.scale = scale
        
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata=None,
                   ):
        batch_reward_inputs = []

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            reward_input = f"{' '.join(meta['prompt'].split())} answer: {gen_text}"
            batch_reward_inputs.append(reward_input)
        
        # get the reward
        with torch.no_grad():
            # to align with the token classification model
            inputs =self.reward_tokenizer(batch_reward_inputs, 
                                          truncation=True, padding=True, 
                                          return_tensors="pt")
            inputs = inputs.to(self.model.device)
            outputs = self.model(**inputs)
            sequence_level_reward = outputs['logits'].squeeze(-1).tolist() 
        
        # align with generated texts, make it fine-grained
        fine_grained_reward = [
            [0.] * (l-1) + [((r-self.mean)/self.std)*self.scale + self.bias]
            for r, l in zip(sequence_level_reward, torch.sum(generated_attention_mask, dim=1).tolist())
        ]
        
        return fine_grained_reward

 
class SubSentenceVerbosityReward:
    def __init__(self,
                 tokenizer,
                 model_ckpt,
                 verbosity_positive_reward = 1.0,
                 verbosity_negative_reward = -1.0,
                 sep = "</s>",
                 ):
        
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer
        
        # prepare reward tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
        # prepare non-factual reward model
        self.nf_reward_model = LongformerForTokenClassification.from_pretrained(model_ckpt)

        for param in self.nf_reward_model.parameters():
            param.requires_grad = False
        self.nf_reward_model.eval()
        
        # prepare spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.sep = sep

        # the id corresponds to the sep token
        self.sep_id = self.reward_tokenizer.convert_tokens_to_ids(sep)
        
        # rewards
        self.verbosity_positive_reward = verbosity_positive_reward
        self.verbosity_negative_reward = verbosity_negative_reward
        
    def find_sep_position(self, input_ids):
        return torch.nonzero(input_ids == self.sep_id, as_tuple=False).squeeze().tolist()
    
    def process_one_generation(self, long_text, policy_text_len):
        
        sentence_end_char_idxs = split_text_to_subsentences(long_text, self.nlp)
           
        sentences = [long_text[sentence_end_char_idxs[i]:sentence_end_char_idxs[i+1]] for i in range(len(sentence_end_char_idxs)-1)]
        
        # Initialize an empty list to store the end token indices of each sentence
        sentence_end_indices = []
    
        for sent_idx in range(len(sentences)):
            tokens = self.policy_tokenizer.tokenize(long_text[:sentence_end_char_idxs[sent_idx+1]])
            token_count = len(tokens)
            sentence_end_indices.append(token_count - 1)
        
        reward_sentences = [f"{self.sep} {sent}" for sent in sentences]
    
        reward_input = ' '.join(reward_sentences)
        
        # in case overlength    
        sentence_end_indices = [min(item, policy_text_len-1) for item in sentence_end_indices]
    
        return reward_input, sentence_end_indices
    
    
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata=None
                   ):
        
        batch_nf_reward_inputs = []
        batch_sentence_end_indices = []
        
        # get the length of generated outputs
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            reward_input, sentence_end_indices = self.process_one_generation(gen_text, 
                                                    policy_inputs_lens[batch_idx])

            # input for the verbosity (non-factual) reward model
            nf_reward_input = f"question: {meta['question']} answer: {reward_input}"
            batch_nf_reward_inputs.append(nf_reward_input)
            
            # the indices of sentence ends for the policy model output
            batch_sentence_end_indices.append(sentence_end_indices)
            
        # get the reward
        with torch.no_grad():
            # to align with the token classification model
            inputs =self.reward_tokenizer([s.split() for s in batch_nf_reward_inputs], 
                                          truncation=True, padding=True, 
                                          is_split_into_words=True,
                                          return_tensors="pt")
            inputs = inputs.to(self.nf_reward_model.device)
            
            # verbosity reward model
            batch_nf_pred = self.nf_reward_model(**inputs)
            
        verbosity_rewards = []
        n_corrects = []
        
        for text_idx, generated_text in enumerate(generated_texts):
            
            # extract the rewards from verbosity reward model output
            this_nf_pred = batch_nf_pred.logits[text_idx].detach().cpu()
            
            generated_text_input_ids = self.reward_tokenizer(
                batch_nf_reward_inputs[text_idx].split(), 
                return_tensors="pt", 
                is_split_into_words=True,
                truncation=True).input_ids[0]
            
            
            # get the indices of </s>
            sep_indices = self.find_sep_position(generated_text_input_ids)
            
            sentence_nf_reward_probs = this_nf_pred[sep_indices]
            
            # align back to the original sentence
            policy_sentence_end_indices = batch_sentence_end_indices[text_idx]
            policy_inputs_len = policy_inputs_lens[text_idx]

            this_verbosity_reward = [0]*policy_inputs_len
            
            this_n_correct = 0

            for i, end_idx in enumerate(policy_sentence_end_indices):
                nf_error_type = torch.argmax(sentence_nf_reward_probs[i][[1,2]]).item()
                verbosity_reward = self.verbosity_positive_reward if nf_error_type == 1 else self.verbosity_negative_reward
                this_verbosity_reward[end_idx] = verbosity_reward
                
                if nf_error_type == 1:
                    this_n_correct += 1
            n_corrects.append(this_n_correct)
                
            verbosity_rewards.append(this_verbosity_reward)
            
        return {"verbosity_rewards": verbosity_rewards, 
                "n_sub_sentences": [len(item) for item in batch_sentence_end_indices], 
                "n_corrects": n_corrects}


class FactualityReward:
    def __init__(self,
                 tokenizer,
                 model_ckpt,
                 factuality_positive_reward = 1.0,
                 factuality_negative_reward = -1.0,
                 sep = "</s>",
                 ):
        
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer

        # prepare reward tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
        # prepare factual reward model
        self.f_reward_model = LongformerForTokenClassification.from_pretrained(model_ckpt)
        
        for param in self.f_reward_model.parameters():
            param.requires_grad = False
        self.f_reward_model.eval()
        
        # prepare spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.sep = sep
        
        # the id corresponds to the sep token
        self.sep_id = self.reward_tokenizer.convert_tokens_to_ids(sep)

        # rewards
        self.factuality_positive_reward = factuality_positive_reward
        self.factuality_negative_reward = factuality_negative_reward

    def find_sep_position(self, input_ids):
        return torch.nonzero(input_ids == self.sep_id, as_tuple=False).squeeze().tolist()
    
    def process_one_generation(self, long_text, policy_text_len):
        
        sentence_end_char_idxs= split_text_to_sentences(long_text, self.nlp)
           
        sentences = [long_text[sentence_end_char_idxs[i]:sentence_end_char_idxs[i+1]] for i in range(len(sentence_end_char_idxs)-1)]
        
        # Initialize an empty list to store the end token indices of each sentence
        sentence_end_indices = []
    
        for sent_idx in range(len(sentences)):
            tokens = self.policy_tokenizer.tokenize(long_text[:sentence_end_char_idxs[sent_idx+1]])
            token_count = len(tokens)
            sentence_end_indices.append(token_count - 1)
        
        reward_sentences = [f"{self.sep} {sent}" for sent in sentences]
    
        reward_input = ' '.join(reward_sentences)
        
        # in case overlength    
        sentence_end_indices = [min(item, policy_text_len-1) for item in sentence_end_indices]
    
        return reward_input, sentence_end_indices
    
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata=None,
                   ):
        
        batch_f_reward_inputs = []
        batch_sentence_end_indices = []
        
        # get the length of generated outputs
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            reward_input, sentence_end_indices = self.process_one_generation(gen_text, 
                                                    policy_inputs_lens[batch_idx])

            # input for the factual reward model
            f_reward_input = f"{meta['prompt']} answer: {reward_input}"
            batch_f_reward_inputs.append(f_reward_input)
            
            # the indices of sentence ends for the policy model output
            batch_sentence_end_indices.append(sentence_end_indices)
        
        # get the reward
        with torch.no_grad():
            
            # to align with the token classification model
            inputs =self.reward_tokenizer([s.split() for s in batch_f_reward_inputs], 
                                          truncation=True, padding=True, 
                                          is_split_into_words=True,
                                          return_tensors="pt")
            inputs = inputs.to(self.f_reward_model.device)
            
            # factual reward model
            batch_f_pred = self.f_reward_model(**inputs)
            
        factuality_rewards = []
        n_corrects = []
        
        for text_idx, generated_text in enumerate(generated_texts):
            
            # extract the rewards from factual reward model output
            this_f_pred = batch_f_pred.logits[text_idx].detach().cpu()
            
            generated_text_input_ids = self.reward_tokenizer(
                batch_f_reward_inputs[text_idx].split(), 
                return_tensors="pt", 
                is_split_into_words=True,
                truncation=True).input_ids[0]
            
            # get the indices of </s>
            sep_indices = self.find_sep_position(generated_text_input_ids)
            sentence_f_reward_probs = this_f_pred[sep_indices]
            
            # align back to the original sentence
            policy_sentence_end_indices = batch_sentence_end_indices[text_idx]
            policy_inputs_len = policy_inputs_lens[text_idx]
            
            this_factuality_reward = [0]*policy_inputs_len
            
            this_n_correct = 0
            
            for i, end_idx in enumerate(policy_sentence_end_indices):
                
                # 0 is has error, 1 is no error
                f_error_type = torch.argmax(sentence_f_reward_probs[i][[0,2]]).item()
                factuality_reward = self.factuality_positive_reward if f_error_type == 1 else self.factuality_negative_reward
                
                # aggregate the rewards
                this_factuality_reward[end_idx] = factuality_reward
                
                if f_error_type == 1:
                    this_n_correct += 1
                    
            n_corrects.append(this_n_correct)
                
            factuality_rewards.append(this_factuality_reward)
            
        return {"factuality_rewards": factuality_rewards,
                "n_sentences": [len(item) for item in batch_sentence_end_indices],
                "n_corrects": n_corrects}

 
class FineGrainedReward(BasicReward):
    
    def __init__(self,
                 tokenizer,
                 non_factual_model_ckpt,
                 factual_model_ckpt,
                 completeness_model_ckpt,
                 kl_coef,
                 verbosity_positive_reward = 1.0,
                 verbosity_negative_reward = -1.0,
                 factuality_positive_reward = 1.0,
                 factuality_negative_reward = -1.0,
                 completeness_reward_mean = 0.0,
                 completeness_reward_std = 1.0,
                 completeness_reward_bias = 0.0,
                 completeness_reward_scale = 1.0,
                 sep = "</s>"
                ):
        
        super().__init__(kl_coef)
        
        self.completeness_reward_bias = completeness_reward_bias
        self.completeness_reward_scale = completeness_reward_scale
        
        self.verbosity_reward = SubSentenceVerbosityReward(tokenizer,
            non_factual_model_ckpt,
            verbosity_positive_reward,
            verbosity_negative_reward,
            sep = sep)
        
        self.factuality_reward = FactualityReward(tokenizer,
            factual_model_ckpt,
            factuality_positive_reward,
            factuality_negative_reward,
            sep = sep)
        
        self.completeness_reward = PreferenceReward(tokenizer,
            completeness_model_ckpt,
            mean=completeness_reward_mean,
            std=completeness_reward_std,
            bias=completeness_reward_bias,
            scale=completeness_reward_scale)
        
        self.nlp = spacy.load("en_core_web_sm")
    
    def get_finegrained_reward(self, prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata):
        
        fine_grained_rewards = []
        n_sub_sentences = []
        n_sentences = []
        
        verbosity = self.verbosity_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                            generated_input_ids, generated_attention_mask, 
                                                            generated_texts, metadata)
            
        n_sub_sentences = verbosity['n_sub_sentences']
        verbosity_rewards = verbosity['verbosity_rewards']
        n_verbosity_correct = verbosity['n_corrects']
        
        factuality = self.factuality_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                    generated_input_ids, generated_attention_mask, 
                                                    generated_texts, metadata)

        n_sentences = factuality['n_sentences']
        factuality_rewards = factuality['factuality_rewards']
        n_factuality_correct = factuality['n_corrects']
            
        completeness_rewards = self.completeness_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                    generated_input_ids, generated_attention_mask, 
                                                    generated_texts, metadata)
        
        # combine the rewards
        for text_idx, generated_text in enumerate(generated_texts):
            
            fine_grained_reward = [a+b+c for a,b,c in zip(verbosity_rewards[text_idx], 
                                                         factuality_rewards[text_idx], 
                                                         completeness_rewards[text_idx])]
            
            fine_grained_rewards.append(fine_grained_reward)
            
        return {"rewards": fine_grained_rewards, 
                "n_sub_sentences": n_sub_sentences, 
                "n_sentences": n_sentences,
                "verbosity_rewards": verbosity_rewards,
                "n_verbosity_correct": n_verbosity_correct,
                "factuality_rewards": factuality_rewards,
                "n_factuality_correct": n_factuality_correct,
                "completeness_rewards": completeness_rewards}
        
        

    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata=None, 
                   ):
        
        rewards_output = self.get_finegrained_reward(prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata)
        
        return {'rewards/raw': rewards_output['rewards']}
            
        
    def eval_metrics(self, 
                prompts_input_ids: torch.tensor, 
                prompts_attention_mask: torch.tensor, 
                generated_input_ids: torch.tensor, # (B, output_len)
                generated_attention_mask: torch.tensor, # (B, output_len)
                generated_texts: List[str],
                metadata=None, 
                ):
        
        output = {}
        
        finegrained_rewards_output = self.get_finegrained_reward(prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata)
        
        # convert finegrained rewards to portions
        n_sub_sentences = finegrained_rewards_output['n_sub_sentences']
        n_sentences = finegrained_rewards_output['n_sentences']
        
        relevance_ratios = []
        factuality_ratios = []
        completeness_rewards = []
        
        for text_idx, generated_text in enumerate(generated_texts):
            # verbosity reward
            n_sub_sentence = n_sub_sentences[text_idx]
            n_verbosity_correct = finegrained_rewards_output['n_verbosity_correct'][text_idx]
            relevance_ratios.append(n_verbosity_correct/ n_sub_sentence)
            
            # factuality reward
            n_sentence = n_sentences[text_idx]
            n_factuality_correct = finegrained_rewards_output['n_factuality_correct'][text_idx]
            factuality_ratios.append(n_factuality_correct / n_sentence)
            
            # completeness reward
            completeness_rewards.append((finegrained_rewards_output['completeness_rewards'][text_idx][-1]-self.completeness_reward_bias)/self.completeness_reward_scale)
        
        # compute rouge scores
        rouge_scores = get_rouge_scores(generated_texts, [m['references'] for m in metadata])
        
        # lens of generations
        generation_lens = torch.sum(generated_attention_mask, dim=-1).tolist()
        
        output.update({
            "eval/rouge": rouge_scores,
            "eval/rewards": [np.sum(sublist) for sublist in finegrained_rewards_output['rewards']],
            "eval/relevance_ratios": relevance_ratios,
            "eval/factuality_ratios": factuality_ratios,
            "eval/completeness_rewards": completeness_rewards,
            "eval/n_sub_sentences": n_sub_sentences,
            "eval/n_sentences": n_sentences,
            "eval/lengths": generation_lens
        })
        
        return output
    
    
    def aggregate_metrics(self, wandb_table, value_columns):
        # how to average over the metrics in wandb table for reporting
        stats = {}
        for k in value_columns:
            stats[k] = np.mean([row[wandb_table.columns.index(k)] for row in wandb_table.data])
        
        # relevance ratios and factual ratios are weighted by the number of (sub)sentences
        
        stats['eval/relevance_ratios'] = (np.sum([row[wandb_table.columns.index('eval/relevance_ratios')] 
                                                  * row[wandb_table.columns.index('eval/n_sub_sentences')] 
                                                  for row in wandb_table.data]) 
                                          / np.sum([row[wandb_table.columns.index('eval/n_sub_sentences')] 
                                                    for row in wandb_table.data]))
        
        stats['eval/factuality_ratios'] = (np.sum([row[wandb_table.columns.index('eval/factuality_ratios')]
                                                   * row[wandb_table.columns.index('eval/n_sentences')]
                                                   for row in wandb_table.data])
                                           / np.sum([row[wandb_table.columns.index('eval/n_sentences')]
                                                     for row in wandb_table.data]))
        
        return stats
    
    

class BaselineReward(BasicReward):
    
    def __init__(self,
                 tokenizer,
                 baseline_model_ckpt,
                 kl_coef,
                 baseline_reward_mean = 0.0,
                 baseline_reward_std = 1.0,
                 baseline_reward_bias = 0.0,
                 baseline_reward_scale = 1.0,
                ):
        
        super().__init__(kl_coef)

        self.baseline_reward = PreferenceReward(tokenizer, 
            baseline_model_ckpt, 
            mean=baseline_reward_mean,
            std=baseline_reward_std,
            bias=baseline_reward_bias,
            scale=baseline_reward_scale)
        
        self.nlp = spacy.load("en_core_web_sm")
        
   
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   metadata=None, 
                   override_gain=None, 
                   override_bias=None):
        
        rewards = self.baseline_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                generated_input_ids, generated_attention_mask, 
                                                generated_texts, metadata)
        
        return {'rewards/raw': rewards}
            
        
    def eval_metrics(self, 
                prompts_input_ids: torch.tensor, 
                prompts_attention_mask: torch.tensor, 
                generated_input_ids: torch.tensor, # (B, output_len)
                generated_attention_mask: torch.tensor, # (B, output_len)
                generated_texts: List[str],
                metadata=None, 
                override_gain=None, 
                override_bias=None):
        
        output = {}
        baseline_rewards_output = self.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                generated_input_ids, generated_attention_mask, 
                                                generated_texts, metadata)['rewards/raw']
        # compute rouge scores
        rouge_scores = get_rouge_scores(generated_texts, [m['references'] for m in metadata])
        
        # lens of generations
        generation_lens = torch.sum(generated_attention_mask, dim=-1).tolist()
        
        output.update({
            "eval/rouge": rouge_scores,
            "eval/rewards": [np.sum(sublist) for sublist in baseline_rewards_output],
            "eval/lengths": generation_lens
        })
        
        return output