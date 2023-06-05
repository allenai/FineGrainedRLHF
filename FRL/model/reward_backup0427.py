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
        
        # initialize model
        # self.tokenizer = LongformerTokenizerFast.from_pretrained(model_ckpt)
        
        self.reward_mode = reward_mode
        
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer

        # prepare reward tokenizer
        self.reward_tokenizer = AutoTokenizer.from_pretrained(non_factual_model_ckpt)
        
        # prepare non-factual reward model
        self.nf_reward_model = LongformerForTokenClassification.from_pretrained(non_factual_model_ckpt)
        
        for param in self.nf_reward_model.parameters():
            param.requires_grad = False
        self.nf_reward_model.eval()
        
        # prepare factual reward model
        self.f_reward_model = LongformerForTokenClassification.from_pretrained(factual_model_ckpt)
        
        for param in self.f_reward_model.parameters():
            param.requires_grad = False
        self.f_reward_model.eval()
        
        # prepare completeness reward model
        self.completeness_reward_model = LongformerForTokenClassification.from_pretrained(completeness_model_ckpt)
        
        for param in self.completeness_reward_model.parameters():
            param.requires_grad = False
        self.completeness_reward_model.eval()
        
        # prepare spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.sep = sep

        # the id corresponds to the sep token
        self.sep_id = self.reward_tokenizer.convert_tokens_to_ids(self.sep)

        # rewards
        self.verbosity_positive_reward = verbosity_positive_reward
        self.verbosity_negative_reward = verbosity_negative_reward
        self.factuality_positive_reward = factuality_positive_reward
        self.factuality_negative_reward = factuality_negative_reward
        self.completeness_positive_reward = completeness_positive_reward
        self.completeness_negative_reward = completeness_negative_reward

        self.batch_size = batch_size
        if self.batch_size is not None:
            self.batchify = lambda x: [x[i:i+batch_size] for i in range(0, len(x), batch_size)]

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
        
        batch_nf_reward_inputs = []
        batch_f_reward_inputs = []
        batch_sentence_end_indices = []
        
        batch_missing_answer_inputs = []
        batch_n_passages = []
        batch_passage_id = 0
        batch_passage_ids = []

        # get the length of generated outputs
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            reward_input, sentence_end_indices = self.process_one_generation(gen_text, 
                                                    policy_inputs_lens[batch_idx])

            # input for the verbosity (non-factual) reward model
            nf_reward_input = f"question: {meta['question']} answer: {reward_input}"
            batch_nf_reward_inputs.append(nf_reward_input)
            
            # input for the factual reward model
            f_reward_input = f"{meta['prompt']} answer: {reward_input}"
            batch_f_reward_inputs.append(f_reward_input)
            
            # the indices of sentence ends for the policy model output
            batch_sentence_end_indices.append(sentence_end_indices)
            
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
            inputs =self.reward_tokenizer([s.split() for s in batch_nf_reward_inputs], 
                                          truncation=True, padding=True, 
                                          is_split_into_words=True,
                                          return_tensors="pt")
            inputs = inputs.to(self.nf_reward_model.device)
            
            # verbosity reward model
            batch_nf_pred = self.nf_reward_model(**inputs)
            
            # to align with the token classification model
            inputs =self.reward_tokenizer([s.split() for s in batch_f_reward_inputs], 
                                          truncation=True, padding=True, 
                                          is_split_into_words=True,
                                          return_tensors="pt")
            inputs = inputs.to(self.nf_reward_model.device)
            
            # factual reward model
            batch_f_pred = self.f_reward_model(**inputs)

            # to align with the token classification model
            inputs =self.reward_tokenizer([s.split() for s in batch_missing_answer_inputs], 
                                          truncation=True, padding=True, 
                                          is_split_into_words=True,
                                          return_tensors="pt")
            inputs = inputs.to(self.completeness_reward_model.device)
            
            # completeness reward model
            batch_missing_answer_pred = self.completeness_reward_model(**inputs)

        fine_grained_reward = []
        verbosity_rewards = []
        factuality_rewards = []
        completeness_rewards = []
        n_sentences = []
        
        
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
            
            # print("in reward model")
            # print(f"generated text: {generated_text}")
            # print(f"reward input: {batch_reward_inputs[text_idx]}")
            # print(f"sep indices: {sep_indices}")
            # print("----------------------------------------------")

            # align back to the original sentence
            policy_sentence_end_indices = batch_sentence_end_indices[text_idx]
            policy_inputs_len = policy_inputs_lens[text_idx]

            this_reward = [0]*policy_inputs_len
            this_factuality_reward = [0]*policy_inputs_len
            this_verbosity_reward = [0]*policy_inputs_len
            
            # process the sequence level completeness reward first
            this_passage_ids = batch_passage_ids[text_idx]
            
            completeness_reward = 0
            
            for this_passage_id in this_passage_ids:

                this_passage_input = batch_missing_answer_inputs[this_passage_id]
                this_passage_input_ids = self.reward_tokenizer(this_passage_input.split(), 
                                          return_tensors="pt", truncation=True,
                                          is_split_into_words=True).input_ids[0]
                reward_index = self.find_sep_position(this_passage_input_ids)[0]
                
                
                this_passage_reward = batch_missing_answer_pred.logits[this_passage_id].detach().cpu()
                
                # 0 is has error, 1 is no error
                complete_error_type = torch.argmax(this_passage_reward[reward_index][[0,2]]).item()
                
                completeness_reward += self.completeness_positive_reward if complete_error_type == 1 else self.completeness_negative_reward
            
            this_completeness_reward = [0]*policy_inputs_len
            this_completeness_reward[-1] = completeness_reward

            for i, end_idx in enumerate(policy_sentence_end_indices):
                
                # 0 is has error, 1 is no error
                f_error_type = torch.argmax(sentence_f_reward_probs[i][[0,2]]).item()
                nf_error_type = torch.argmax(sentence_nf_reward_probs[i][[1,2]]).item()
                
                factuality_reward = self.factuality_positive_reward if f_error_type == 1 else self.factuality_negative_reward
                verbosity_reward = self.verbosity_positive_reward if nf_error_type == 1 else self.verbosity_negative_reward
                
                # aggregate the rewards
                this_reward[end_idx] = factuality_reward + verbosity_reward
                this_factuality_reward[end_idx] = factuality_reward
                this_verbosity_reward[end_idx] = verbosity_reward
            
            this_reward[-1] += completeness_reward
            
            fine_grained_reward.append(this_reward)
            verbosity_rewards.append(this_verbosity_reward)
            factuality_rewards.append(this_factuality_reward)
            completeness_rewards.append(this_completeness_reward)

            n_sentences.append(len(policy_sentence_end_indices))
                
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
            'rewards/verbosity': verbosity_rewards, # list (B)
            'rewards/factuality': factuality_rewards, # list (B)
            'rewards/completeness': completeness_rewards, # list (B)
            'rewards/n_sentences': n_sentences, # list (B)
        }