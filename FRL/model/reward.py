import json
import os
from typing import Optional, List, Iterable, Dict, Any, Tuple
import torch, spacy
import torch.nn.functional as F
from transformers import AutoTokenizer
from .my_longformer import LongformerForSequenceClassification, LongformerForTokenClassification
from utils.utils import reduce_mean, mask_pad
import abc
import numpy as np
import logging
import re


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
    
    def eval_reward(self,
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
        mask = results['generated_attention_mask']
        
        # should be a list of length B to avoid gradient descent
        normalized_rewards = results['rewards/normalized'] 
        kl = mask_pad(logprobs - ref_logprobs, mask, pad_value=0.)
        kl_penalty = self.kl_coef * kl
        RL = logprobs.size(1)
        
        flattened_rewards = torch.tensor([
            r + [0.] * (RL-len(r))
            for r in normalized_rewards
        ], device=logprobs.device) # (B, KL)
        
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
                   generated_input_ids: torch.tensor, # (B, RL)
                   generated_attention_mask: torch.tensor, # (B, RL)
                   generated_texts: List[str],
                   metadata=None, 
                   override_gain=None, 
                   override_bias=None):
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

        
class VerbosityReward:
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
        doc = self.nlp(long_text)
        
        sentence_end_char_idxs= [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0]
           
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
                   generated_input_ids: torch.tensor, # (B, RL)
                   generated_attention_mask: torch.tensor, # (B, RL)
                   generated_texts: List[str],
                   metadata=None, 
                   override_gain=None, 
                   override_bias=None):
        
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

            for i, end_idx in enumerate(policy_sentence_end_indices):
                nf_error_type = torch.argmax(sentence_nf_reward_probs[i][[1,2]]).item()
                verbosity_reward = self.verbosity_positive_reward if nf_error_type == 1 else self.verbosity_negative_reward
                this_verbosity_reward[end_idx] = verbosity_reward
                
            verbosity_rewards.append(this_verbosity_reward)
            
        return verbosity_rewards


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
    
    
        # split long text into subsentences
    def split_text_to_subsentences(self, long_text):
        def get_sub_sentence_starts(tokens, min_subsent_words=5):

            def _is_tok_end_of_subsent(tok):
                if re.match('[,;!?]', tok[-1]) is not None:
                    return True
                return False

            # assert len(tokens) > 0
            is_subsent_starts = [True]
            prev_tok = tokens[0]
            prev_subsent_start_idx = 0
            for i, tok in enumerate(tokens[1:]):
                tok_id = i + 1
                if _is_tok_end_of_subsent(prev_tok) and tok_id + min_subsent_words < len(tokens):
                    if tok_id - prev_subsent_start_idx < min_subsent_words:
                        if prev_subsent_start_idx > 0:
                            is_subsent_starts += [True]
                            is_subsent_starts[prev_subsent_start_idx] = False
                            prev_subsent_start_idx = tok_id
                        else:
                            is_subsent_starts += [False]
                    else:
                        is_subsent_starts += [True]
                        prev_subsent_start_idx = tok_id
                else:
                    is_subsent_starts += [False]
                prev_tok = tok

            return is_subsent_starts


        def tokenize_with_indices(text):
            tokens = text.split()
            token_indices = []

            current_index = 0
            for token in tokens:
                start_index = text.find(token, current_index)
                token_indices.append((token, start_index))
                current_index = start_index + len(token)

            return token_indices
        
        doc = self.nlp(long_text)
        sentence_start_char_idxs= [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0]
        
        char_starts = []
        
        for sentence_idx, sentence_start_char_idx in enumerate(sentence_start_char_idxs[:-1]):
            
            sentence = long_text[sentence_start_char_idx: sentence_start_char_idxs[sentence_idx+1]]
            
            tokens_with_indices = tokenize_with_indices(sentence)
            
            tokens = [i[0] for i in tokens_with_indices]
            is_sub_starts = get_sub_sentence_starts(tokens, min_subsent_words=5)
            
            for token_with_idx, is_sub_start in zip(tokens_with_indices, is_sub_starts):
                if is_sub_start:
                    char_starts.append(sentence_start_char_idx + token_with_idx[1])
        
        return char_starts + [len(long_text)]
    
    
    def process_one_generation(self, long_text, policy_text_len):
        
        sentence_end_char_idxs = self.split_text_to_subsentences(long_text)
           
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
                   generated_input_ids: torch.tensor, # (B, RL)
                   generated_attention_mask: torch.tensor, # (B, RL)
                   generated_texts: List[str],
                   metadata=None, 
                   override_gain=None, 
                   override_bias=None):
        
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
        doc = self.nlp(long_text)
        
        sentence_end_char_idxs= [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0]
           
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
                   generated_input_ids: torch.tensor, # (B, RL)
                   generated_attention_mask: torch.tensor, # (B, RL)
                   generated_texts: List[str],
                   metadata=None, 
                   override_gain=None, 
                   override_bias=None):
        
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

                
class CompletenessReward:
    
    def __init__(self,
                 tokenizer,
                 model_ckpt,
                 completeness_positive_reward = 1.0,
                 completeness_negative_reward = -1.0,
                 sep = "</s>",
                 split_by_sentence = False,
                 ):
    
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer
        
        # whether to average the rewards by sentence
        self.split_by_sentence = split_by_sentence

        # prepare reward tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        
        # prepare completeness reward model
        self.completeness_reward_model = LongformerForTokenClassification.from_pretrained(model_ckpt)
        
        for param in self.completeness_reward_model.parameters():
            param.requires_grad = False
        self.completeness_reward_model.eval()
        
        # for sentence split
        self.nlp = spacy.load("en_core_web_sm")
        
        # the id corresponds to the sep token
        self.sep_id = self.reward_tokenizer.convert_tokens_to_ids(sep)
        
        # rewards
        self.completeness_positive_reward = completeness_positive_reward
        self.completeness_negative_reward = completeness_negative_reward
        
    def find_sep_position(self, input_ids):
        return torch.nonzero(input_ids == self.sep_id, as_tuple=False).squeeze().tolist()
    
    def process_one_generation(self, long_text, policy_text_len):
        doc = self.nlp(long_text)
        
        sentence_end_char_idxs= [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0]
           
        sentences = [long_text[sentence_end_char_idxs[i]:sentence_end_char_idxs[i+1]] for i in range(len(sentence_end_char_idxs)-1)]
        
        # Initialize an empty list to store the end token indices of each sentence
        sentence_end_indices = []
    
        for sent_idx in range(len(sentences)):
            tokens = self.policy_tokenizer.tokenize(long_text[:sentence_end_char_idxs[sent_idx+1]])
            token_count = len(tokens)
            sentence_end_indices.append(token_count - 1)
        
        # in case overlength    
        sentence_end_indices = [min(item, policy_text_len-1) for item in sentence_end_indices]
    
        return sentence_end_indices
    
    def process_completeness_inputs(self, generated_text, metadata):
        missing_answer_inputs = []
        
        question = metadata["question"]
        
        for passage in metadata["passages"]:
            this_entry = f"question: {question} context: wikipage: {passage['wikipage']} text: "
            this_entry += f"{passage['content']} answer: </s> {generated_text}"
            missing_answer_inputs.append(this_entry)
            
        return missing_answer_inputs
    
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, RL)
                   generated_attention_mask: torch.tensor, # (B, RL)
                   generated_texts: List[str],
                   metadata=None, 
                   override_gain=None, 
                   override_bias=None):
        
        batch_missing_answer_inputs = []
        batch_n_passages = []
        batch_passage_id = 0
        batch_passage_ids = []
        
        # get the length of generated outputs
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            
            # process the input for completeness reward model (one passage each)
            missing_answer_inputs = self.process_completeness_inputs(gen_text, meta)
            batch_missing_answer_inputs.extend(missing_answer_inputs)
            batch_n_passages.append(len(missing_answer_inputs))
            
            this_passages = list(range(batch_passage_id, batch_passage_id+len(missing_answer_inputs)))
            batch_passage_ids.append(this_passages)
            batch_passage_id += len(missing_answer_inputs)
            
        # get the reward
        with torch.no_grad():
            
            # to align with the token classification model
            inputs =self.reward_tokenizer([s.split() for s in batch_missing_answer_inputs], 
                                          truncation=True, padding=True, 
                                          is_split_into_words=True,
                                          return_tensors="pt")
            inputs = inputs.to(self.completeness_reward_model.device)
            
            # completeness reward model
            batch_missing_answer_pred = self.completeness_reward_model(**inputs)
            
        completeness_rewards = []
        
        for text_idx, generated_text in enumerate(generated_texts):
            
            policy_inputs_len = policy_inputs_lens[text_idx]
            
            # process the sequence level completeness reward first
            this_passage_ids = batch_passage_ids[text_idx]
            
            completeness_reward = 0
            
            sequence_error_flag = False
            
            for this_passage_id in this_passage_ids:

                this_passage_input = batch_missing_answer_inputs[this_passage_id]
                this_passage_input_ids = self.reward_tokenizer(this_passage_input.split(), 
                                          return_tensors="pt", truncation=True,
                                          is_split_into_words=True).input_ids[0]
                reward_index = self.find_sep_position(this_passage_input_ids)[0]
                
                this_passage_reward = batch_missing_answer_pred.logits[this_passage_id].detach().cpu()
                
                # 0 is has error, 1 is no error
                complete_error_type = torch.argmax(this_passage_reward[reward_index][[0,2]]).item()
                
                if complete_error_type == 0:
                    sequence_error_flag = True
                # completeness_reward += self.completeness_positive_reward if complete_error_type == 1 else self.completeness_negative_reward
            
            if not self.split_by_sentence:
                completeness_reward = self.completeness_positive_reward if not sequence_error_flag else self.completeness_negative_reward
                this_completeness_reward = [0]*policy_inputs_len
                this_completeness_reward[-1] = completeness_reward
                completeness_rewards.append(this_completeness_reward)
                
            else:
                # split the reward to each sentence end
                completeness_reward = self.completeness_positive_reward if not sequence_error_flag else self.completeness_negative_reward
                sentence_end_indices = self.process_one_generation(generated_text, policy_inputs_len)
                n_sentences = len(sentence_end_indices)
                this_completeness_reward = [0]*policy_inputs_len
                
                for sentence_end_idx in sentence_end_indices:
                    this_completeness_reward[sentence_end_idx] = completeness_reward/n_sentences
                    
                completeness_rewards.append(this_completeness_reward)
            
        return completeness_rewards


class FineGrainedReward(BasicReward):
    
    def __init__(self,
                 tokenizer,
                 non_factual_model_ckpt,
                 factual_model_ckpt,
                 completeness_model_ckpt,
                 kl_coef,
                 batch_size=None,
                 gain = 1.0,
                 bias = 0.0,
                 verbosity_positive_reward = 1.0,
                 verbosity_negative_reward = -1.0,
                 factuality_positive_reward = 1.0,
                 factuality_negative_reward = -1.0,
                 completeness_positive_reward = 1.0,
                 completeness_negative_reward = -1.0,
                 sep = "</s>",
                 reward_mode = 0,
                ):
        
        super().__init__(kl_coef, gain, bias)

        self.reward_mode = reward_mode
        
        self.verbosity_reward = VerbosityReward(tokenizer,
            non_factual_model_ckpt,
            verbosity_positive_reward,
            verbosity_negative_reward,
            sep = sep)
        
        self.factuality_reward = FactualityReward(tokenizer,
            factual_model_ckpt,
            factuality_positive_reward,
            factuality_negative_reward,
            sep = sep)
        
        self.completeness_reward = CompletenessReward(tokenizer,
            completeness_model_ckpt,
            completeness_positive_reward,
            completeness_negative_reward,
            sep = sep)
        
        self.nlp = spacy.load("en_core_web_sm")
                                                  
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

        fine_grained_rewards = []
        n_sentences = []
        
        verbosity_rewards = self.verbosity_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                            generated_input_ids, generated_attention_mask, 
                                                            generated_texts, metadata)
        
        factuality_rewards = self.factuality_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                    generated_input_ids, generated_attention_mask, 
                                                    generated_texts, metadata)
        
        completeness_rewards = self.completeness_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                    generated_input_ids, generated_attention_mask, 
                                                    generated_texts, metadata)
        
        # combine the rewards
        for text_idx, generated_text in enumerate(generated_texts):
            doc = self.nlp(generated_text)
            n_sentence = len([sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0])
            n_sentences.append(n_sentence)
            
            fine_grained_reward = [a+b+c for a,b,c in zip(verbosity_rewards[text_idx], 
                                                         factuality_rewards[text_idx], 
                                                         completeness_rewards[text_idx])]
            
            fine_grained_rewards.append(fine_grained_reward)
            
                
        # normalize the rewards        
        gain = self.gain if override_gain is None else override_gain
        bias = self.bias if override_bias is None else override_bias
        
        fine_grained_reward_normalized = [
            (np.array(r)*gain + bias).tolist()  for r in fine_grained_rewards
        ]
        
        # print(f"fine_grained_reward_normalized: {fine_grained_reward_normalized}\n----------------")
        

        return {
            'rewards/raw': fine_grained_rewards, # list (B)
            'rewards/normalized': fine_grained_reward_normalized, # list (B)
            'rewards/verbosity': verbosity_rewards, # list (B)
            'rewards/factuality': factuality_rewards, # list (B)
            'rewards/completeness': completeness_rewards, # list (B)
            'rewards/n_sentences': n_sentences, # list (B)
        }
    
    
    
class BaselineReward(BasicReward):
    
    def __init__(self,
                 tokenizer,
                 baseline_model_ckpt,
                 kl_coef,
                 rm_mean = 0.0,
                 rm_std = 1.0,
                 scale = 1.0,
                 bias = 0.0,
                ):
        
        super().__init__(kl_coef, 1.0, 0.0)

        self.baseline_reward = PreferenceReward(tokenizer, baseline_model_ckpt, 
                                                mean=rm_mean, std=rm_std, 
                                                bias=bias, scale=scale)
        
        self.nlp = spacy.load("en_core_web_sm")
                                                  
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, RL)
                   generated_attention_mask: torch.tensor, # (B, RL)
                   generated_texts: List[str],
                   metadata=None, 
                   override_gain=None, 
                   override_bias=None):

        fine_grained_rewards = []
        n_sentences = []
        
        rewards = self.baseline_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                generated_input_ids, generated_attention_mask, 
                                                generated_texts, metadata)
        
        # combine the rewards
        for text_idx, generated_text in enumerate(generated_texts):
            doc = self.nlp(generated_text)
            n_sentence = len([sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0])
            n_sentences.append(n_sentence)
            
            this_reward = rewards[text_idx]
            
            fine_grained_rewards.append(this_reward)
            

        return {
            'rewards/raw': fine_grained_rewards, # list (B)
            'rewards/normalized': fine_grained_rewards, # list (B)
            'rewards/n_sentences': n_sentences, # list (B)
        }    


class AllReward(BasicReward):
    
    def __init__(self,
                 tokenizer,
                 baseline_model_ckpt,
                 non_factual_model_ckpt,
                 factual_model_ckpt,
                 completeness_model_ckpt,
                 kl_coef,
                 fine_grained = False,
                 batch_size=None,
                 baseline_reward_mean = 0.0,
                 baseline_reward_std = 1.0,
                 baseline_reward_bias = 0.0,
                 baseline_reward_scale = 1.0,
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
        
        super().__init__(kl_coef, 1.0, 0.0)
        
        self.fine_grained = fine_grained
        
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
        
        if not fine_grained:
            self.baseline_reward = PreferenceReward(tokenizer, 
                baseline_model_ckpt, 
                mean=baseline_reward_mean,
                std=baseline_reward_std,
                bias=baseline_reward_bias,
                scale=baseline_reward_scale)
        
        self.nlp = spacy.load("en_core_web_sm")
        
    def get_baseline_reward(self, prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata):
        
        rewards = self.baseline_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                generated_input_ids, generated_attention_mask, 
                                                generated_texts, metadata)
            
        return {"rewards": rewards}
    
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
                   generated_input_ids: torch.tensor, # (B, RL)
                   generated_attention_mask: torch.tensor, # (B, RL)
                   generated_texts: List[str],
                   metadata=None, 
                   override_gain=None, 
                   override_bias=None):
        
        # get the baseline reward
        if not self.fine_grained:
            
            rewards_output = self.get_baseline_reward(prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata)
            
            return {'rewards/normalized': rewards_output['rewards']}
            
        # our fine-grained reward
        else:
            rewards_output = self.get_finegrained_reward(prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata)
        
            return {'rewards/normalized': rewards_output['rewards']}
            
        
    def eval_reward(self, 
                prompts_input_ids: torch.tensor, 
                prompts_attention_mask: torch.tensor, 
                generated_input_ids: torch.tensor, # (B, RL)
                generated_attention_mask: torch.tensor, # (B, RL)
                generated_texts: List[str],
                metadata=None, 
                override_gain=None, 
                override_bias=None):
        
        output = {}
        output["eval/baseline_rewards"] = [[0]] * len(generated_texts)
        
        if not self.fine_grained:
            baseline_rewards_output = self.get_baseline_reward(prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, metadata)
            
            output["eval/baseline_rewards"] = baseline_rewards_output['rewards']
            
        
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
            if self.completeness_reward_scale == 0.:
                completeness_rewards.append(0.)
            else:
                completeness_rewards.append((finegrained_rewards_output['completeness_rewards'][text_idx][-1]-self.completeness_reward_bias)/self.completeness_reward_scale)
        
        output.update({
            "eval/finegrained_rewards": finegrained_rewards_output['rewards'],
            "eval/relevance_ratios": relevance_ratios,
            "eval/factuality_ratios": factuality_ratios,
            "eval/completeness_rewards": completeness_rewards,
            "eval/n_sub_sentences": n_sub_sentences,
            "eval/n_sentences": n_sentences,
        })
        
        return output