import argparse
from collections import defaultdict
from itertools import chain
import json
import logging
import numpy as np
import os
import random
import shutil
from tqdm import tqdm
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
import transformers
import accelerate
import wandb
import evaluate


from utils.utils import ensure_dir, set_seed, reduce_mean, reduce_sum, ceil_div, whiten, clamp
from args import get_args
from model.policy import Policy
from model.value import Value
from model.reward import AllReward


logging.basicConfig(level=logging.ERROR)

accelerator = accelerate.Accelerator()
device = accelerator.device
logging.basicConfig(level=logging.INFO)
log = accelerate.logging.get_logger(__name__, log_level='INFO')
def log_info(s):
    if accelerator.is_main_process:
        log.info(s)
        
import evaluate
import nltk

args = get_args()
metric = evaluate.load("rouge", experiment_id=args.run_name)

def postprocess_text(preds, list_of_labels):
    
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    list_of_labels = [["\n".join(nltk.sent_tokenize(label.strip())) for label in labels] 
                      for labels in list_of_labels]
    return preds, list_of_labels

def rouge_score(preds, labels):

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(preds, labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=False)
    result = [round(v * 100, 4) for v in result['rougeLsum']]
    return result


# This does not lowercase the data, by default
class QADataset(Dataset):
    def __init__(self, split, tokenizer, accelerator=None, length_limit=None):
        super().__init__()
        
        self.split = split
        self.dataset_fns = {
            "train": "../asqa_new/processed_train_oracle.json",
            "dev": "../asqa_new/processed_test_oracle.json",
            "test": "../asqa_new/processed_test_oracle.json"
        }
        
        # self.dataset_fns = {
        #     "train": "../asqa_new/processed_train_oracle.json",
        #     "dev": "../asqa_new/processed_dev_oracle.json",
        #     "test": "../asqa_new/processed_test_oracle.json"
        # }
        
        self.n_card = 1
        if accelerator is not None:
            self.n_card = accelerator.num_processes
        
        
        self.tokenizer = tokenizer

        self.instances = self.load_datasets()
        
        if length_limit is not None:
            self.instances = self.instances[:length_limit]

        if split == 'train':
            random.shuffle(self.instances)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def load_datasets(self): 
        instances = []
        
        task_data = None
        with open(self.dataset_fns[self.split], 'r') as f:
            task_data = json.load(f)
            
        for task_instance in task_data:
            instances.append({
                "prompt": task_instance["text"],
                "references": task_instance["answer"],
                "metadata": {
                    "prompt": task_instance['text'],
                    "passages": task_instance['passages'],
                    "question": task_instance['question'],}
            })
        
        log_info(f'Loaded split {self.split} with {len(instances)} total instances')
        
        instances = instances[:len(instances)//self.n_card*self.n_card]  # or Trainer will stuck
        return instances

    # Make a collate function to fix dataloader weird list batching
    def collate_fn(self, batch):
        
        # process input prompts
        prompts = [item['prompt'] for item in batch]
        prompts_tok = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors='pt', 
            padding='max_length', 
            truncation=True,
            max_length=self.tokenizer.max_input_len,
            # padding_side=self.tokenizer.padding_side, # YUSHI: change later, now Ellen pad defaultly
            )
        
        prompts_input_ids = prompts_tok.input_ids
        prompts_attention_mask = prompts_tok.attention_mask
        
        # process references
        references = [item['references'] for item in batch]
        # references_tok = self.tokenizer.batch_encode_plus(
        #     references,
        #     return_tensors='pt', padding='max_length', truncation='longest_first', max_length=self.tokenizer.max_generated_len)
        
        # references_input_ids = references_tok.input_ids
        # references_attention_mask = references_tok.attention_mask
        # references_labels = references_input_ids.clone()
        # references_labels[references_attention_mask == 0] = -100
        
        # process metadata
        metadata = [item['metadata'] for item in batch]
        

        result = {
            'prompts_input_ids': prompts_input_ids,
            'prompts_attention_mask': prompts_attention_mask,
            'references': references,
            'metadata': metadata
        }
        return result


class PPOTrainer:
    def __init__(self,
                 args: argparse.Namespace,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 ref_policy_model: Policy,
                 policy_model: Policy,
                 value_model: Value,
                 reward_model: AllReward,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 init_step: int,
                 eval_accs: Dict,
                ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.ref_policy_model = ref_policy_model
        self.policy_model = policy_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # early stopping if KL too big
        self.should_early_stop = False
        self.huge_kl_count = 0
        
        self.kl_negative_count = 0

        self.batchify = lambda x, n: [x[i:i + n] for i in range(0, len(x), n)]
        
        if self.args.mode == 'train':
            if not args.nolog and accelerator.is_main_process:
                wandb.init(entity="uwnlp", project='asqa_eval', name=args.run_name, config=args)
                wandb.define_metric('train/step')
                wandb.define_metric('eval/step')
                wandb.define_metric('train/*', step_metric='train/step')
                wandb.define_metric('eval/*', step_metric='eval/step', summary='max')

            self.train_sampler = iter(self.train_dataloader)
            for _ in range(init_step % len(self.train_dataloader)):
                next(self.train_sampler)

            self.eval_accs = eval_accs

        elif self.args.mode == 'eval':
            if not args.nolog and accelerator.is_main_process:
                wandb.init(project='rainier_eval', name=args.run_name, config=args)
                wandb.define_metric('eval/step')
                wandb.define_metric('eval/*', step_metric='eval/step')


    def compute_advantages(self, results, num_samples):
        
        # the advantages should be normalized within the samples with same input
        
        old_values = results['generated_value']
        rewards = results['rewards/penalized']
        mask = results['generated_attention_mask'] # (B, KL)
        
        with torch.no_grad():
            # if accelerator.is_main_process:
            #     log.info(f'original rewards: {rewards}')
            if self.args.whiten_rewards:
                whitened_rewards = whiten(rewards, mask, shift_mean=False, accelerator=accelerator)
            else:
                whitened_rewards = rewards
            
            lastgaelam = 0
            advantages_reversed = []
            # gen_length = whitened_rewards.size(1)
            gen_length = mask.sum(dim=1).max().item() # to match the original implementation in V1
            for t in reversed(range(gen_length)):
                nextvalues = old_values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = whitened_rewards[:, t] + self.args.gamma * nextvalues - old_values[:, t]
                lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            advantages = F.pad(advantages, (0, whitened_rewards.size(1) - gen_length), value=0.0)
            returns = advantages + old_values
            
            whitened_advantages = advantages.detach()
            # whiten advantage by same input
            # notice: no accelerator here! 
            # Only normalize inside the samples from same prompt in the batch
            # for i in range(0, advantages.size(0), num_samples):
            #     whitened_advantages[i:i+num_samples] = whiten(advantages[i:i+num_samples], 
            #                                                   mask[i:i+num_samples]).detach()

            # if accelerator.is_main_process:
            #     log.info(f'original advantages: {advantages}')
            whitened_advantages = whiten(advantages, mask, accelerator=accelerator).detach()
            # if accelerator.is_main_process:
            #     log.info(f'whitened advantages: {whitened_advantages}')
            
        results['whitened_advantages'] = whitened_advantages
        results['returns'] = returns
                

    def loss(self, results, all_mask_weight):
        
        old_values = results['generated_value']
        old_logprobs = results['generated_logprobs']
        mask = results['generated_attention_mask'] # (B, KL)
        
        whitened_advantages = results['whitened_advantages']
        returns = results['returns']

        # all_mask = accelerator.gather(mask) # (num_gpus * B, KL)
        weight = mask.sum(dim=1).float().mean().item() / all_mask_weight

        forward_inputs = {
            'prompts_input_ids': results['prompts_input_ids'],
            'prompts_attention_mask': results['prompts_attention_mask'],
            'generated_input_ids': results['generated_input_ids'],
            'generated_attention_mask': results['generated_attention_mask'],
        }

        policy_forward = self.policy_model.forward_pass(**forward_inputs)
        new_logprobs = policy_forward['generated_logprobs']

        ratio = torch.exp(new_logprobs - old_logprobs)
        pg_losses1 = -whitened_advantages * ratio
        pg_losses2 = -whitened_advantages * torch.clamp(ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange)
        pg_loss = reduce_mean(torch.max(pg_losses1, pg_losses2), mask)
        pg_loss = pg_loss * weight
        # if pg_loss < -4.0:
        #     print(f'process_index: {accelerator.process_index}')
        #     print(f'pg_loss: {pg_loss}')
        #     print(f'advantages: {whitened_advantages}')
        #     print(f'ratio: {ratio}')
        #     print(f'mask: {mask}')
        # pg_clipfrac = reduce_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        if self.args.policy_value_sharing:
            new_values = policy_forward['generated_value']
        else:
            value_forward = self.value_model.forward_pass(**forward_inputs)
            new_values = value_forward['generated_value']
            new_values *= mask  # TODO: I doubt if this line is necessary

        new_values_clipped = clamp(new_values, old_values - self.args.cliprange_value, old_values + self.args.cliprange_value)
        vf_losses1 = torch.square(new_values - returns)
        vf_losses2 = torch.square(new_values_clipped - returns)
        vf_loss = .5 * reduce_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_loss = vf_loss * weight
        # vf_clipfrac = reduce_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        loss = self.args.pg_coef * pg_loss + self.args.vf_coef * vf_loss

        results['loss/total'] = loss
        results['loss/policy'] = pg_loss
        results['loss/value'] = vf_loss

    def train(self, step):
        self.valid(step=step)

        accelerator.wait_for_everyone()
        try:
            batch = next(self.train_sampler)
        except StopIteration:
            self.train_sampler = iter(self.train_dataloader)
            batch = next(self.train_sampler)
        
        self.ref_policy_model.model.eval()
        
        self.policy_model.model.eval()
        # self.policy_model.model.train()
        
        self.value_model.model.eval()
        self.value_model.linear.eval()
        # self.value_model.model.train()
        
        # Rollout from current policy
        with torch.no_grad():
            results = self.policy_model.sample(
                prompts_input_ids=batch['prompts_input_ids'],
                prompts_attention_mask=batch['prompts_attention_mask'],
                temperature=self.args.temperature,
                top_k = self.args.top_k,
                top_p = self.args.top_p,
                num_return_sequences=self.args.num_samples,
            )
    
        forward_inputs = {
            'prompts_input_ids': results['prompts_input_ids'],
            'prompts_attention_mask': results['prompts_attention_mask'],
            'generated_input_ids': results['generated_input_ids'],
            'generated_attention_mask': results['generated_attention_mask'],
        }
        
        with torch.no_grad():
            policy_forward = self.policy_model.forward_pass(**forward_inputs)
            results.update(policy_forward)
        
        # Run value network
        if not self.args.policy_value_sharing:
            with torch.no_grad(): # treat the values at beginning of step as ground-truth
                value_forward = self.value_model.forward_pass(**forward_inputs)
                results['generated_value'] = value_forward['generated_value']
                results['generated_value'] *= results['generated_attention_mask']  # TODO: I doubt if this line is necessary

        # Run ref policy
        with torch.no_grad():
            ref_policy_forward = self.ref_policy_model.forward_pass(**forward_inputs)
            results['generated_ref_logits'] = ref_policy_forward['generated_logits']
            results['generated_ref_logprobs'] = ref_policy_forward['generated_logprobs']
        
        # Get reward
        with torch.no_grad():
            reward_results = self.reward_model.get_reward(
                prompts_input_ids=results['prompts_input_ids'],
                prompts_attention_mask=results['prompts_attention_mask'],
                generated_input_ids=results['generated_input_ids'],
                generated_attention_mask=results['generated_attention_mask'],
                generated_texts=results['generated_text'],
                metadata = [elem for elem in batch['metadata'] for _ in range(self.args.num_samples)],
            )
            results.update(reward_results)
            self.reward_model.kl_penalize_reward(results)

        # Get advantages
        self.compute_advantages(results, self.args.num_samples)
        
        # if accelerator.is_main_process:
        #     for k, v in results.items():
        #         log.info(f'{k}: {v}')
        
        n_results = len(results['generated_input_ids'])
        
        loss_totals, loss_policies, loss_values =  [], [], []
        reward_penalizeds, reward_kls, reward_normalizeds = [], [], []
        
        # Train
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        
        self.policy_model.model.train()
        self.value_model.model.train()
        self.value_model.linear.train()
        
        for ppo_epoch_idx in range(self.args.noptepochs):
            self.optimizer.zero_grad()
            
            # get the weight for each sub-batch
            mask = results['generated_attention_mask']
            all_mask = accelerator.gather(mask)
            all_mask_weight = all_mask.sum(dim=1).float().mean().item()
            
            for batch_idx in range(0,n_results, self.args.training_batch_size):
                batch_results = {}
                
                for k, v in results.items():
                    batch_results[k] = v[batch_idx:batch_idx+self.args.training_batch_size]
            
                self.loss(batch_results, all_mask_weight)
                # gradient accumulation weight
                accelerator.backward(batch_results['loss/total'])
                
                # logging
                if ppo_epoch_idx == self.args.noptepochs - 1:
                    loss_total = batch_results['loss/total'].unsqueeze(0) # (1)
                    loss_policy = batch_results['loss/policy'].unsqueeze(0) # (1)
                    loss_value = batch_results['loss/value'].unsqueeze(0) # (1)
                    reward_penalized = torch.mean(reduce_sum(batch_results['rewards/penalized'], batch_results['generated_attention_mask'], axis=1)).unsqueeze(0) # (1)
                    reward_kl = torch.mean(reduce_sum(batch_results['rewards/kl'], batch_results['generated_attention_mask'], axis=1)).unsqueeze(0) # (1)
                    reward_normalized =  torch.mean(reduce_sum(batch_results['rewards/original'], batch_results['generated_attention_mask'], axis=1)).unsqueeze(0) # (1)

                    loss_totals.append(loss_total)
                    loss_policies.append(loss_policy)
                    loss_values.append(loss_value)
                    reward_penalizeds.append(reward_penalized)
                    reward_kls.append(reward_kl)
                    reward_normalizeds.append(reward_normalized)
                    
                
            if self.args.clip_grad:
                accelerator.clip_grad_norm_(
                    chain(self.policy_model.model.parameters(),
                        self.policy_model.linear.parameters()),
                    self.args.max_grad_norm)
                accelerator.clip_grad_norm_(
                    chain(self.value_model.model.parameters(),
                        self.value_model.linear.parameters()),
                    self.args.max_grad_norm)
                
            self.optimizer.step()
            self.scheduler.step()

        loss_total = torch.cat(loss_totals, dim=0)
        loss_policy = torch.cat(loss_policies, dim=0)
        loss_value = torch.cat(loss_values, dim=0)
        reward_penalized = torch.cat(reward_penalizeds, dim=0)
        reward_kl = torch.cat(reward_kls, dim=0)
        reward_normalized = torch.cat(reward_normalizeds, dim=0)

        losses_total = accelerator.gather(loss_total) # (num_gpus)
        losses_policy = accelerator.gather(loss_policy) # (num_gpus)
        # if accelerator.is_main_process:
        #     log.info(f'losses_policy: {losses_policy}')
        losses_value = accelerator.gather(loss_value) # (num_gpus)
        rewards_penalized = accelerator.gather(reward_penalized) # (num_gpus)
        rewards_kl = accelerator.gather(reward_kl) # (num_gpus)
        rewards_normalized = accelerator.gather(reward_normalized) # (num_gpus)

        loss_total = losses_total.mean().item()
        loss_policy = losses_policy.mean().item()
        # if accelerator.is_main_process:
        #     log.info(f'loss_policy: {loss_policy}')
        loss_value = losses_value.mean().item()
        reward_penalized = rewards_penalized.mean().item()
        reward_kl = rewards_kl.mean().item()
        reward_normalized = rewards_normalized.mean().item()
    
        # Logging
        if not self.args.nolog and accelerator.is_main_process:

            this_batch_kl = np.mean(reward_kl)

            if step % self.args.log_interval == 0:
                wandb.log({
                    'train/step': step,
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/loss/total': np.mean(loss_total),
                    'train/loss/policy': np.mean(loss_policy),
                    'train/loss/value': np.mean(loss_value),
                    'train/reward/penalized': np.mean(reward_penalized),
                    'train/reward/KL': this_batch_kl,
                    'train/reward/normalized': np.mean(reward_normalized),
                })
                
            if this_batch_kl > self.args.kl_threshold:
                log.info(f"KL divergence {this_batch_kl} exceeds threshold {self.args.kl_threshold}")
                self.huge_kl_count += 1
                if self.huge_kl_count >= 5:
                    self.should_early_stop = True

    def valid(self, step):
        if self.args.eval_loop_cap is not None and self.args.eval_loop_cap == 0:
            return
        if step % self.args.eval_interval != 0:
            return
        log_info(f'Evaluating [step {step}] ...')

        accelerator.wait_for_everyone()
        
        self.policy_model.model.eval()
        if not self.args.policy_value_sharing:
            self.value_model.model.eval()
        
        columns=["step", "inputs", "gold_answer", "outputs", 
                 "baseline_reward", "finegrained_reward", "rouge",
                 "length", "n_sentence", "n_sub_sentence",
                 "relevance_ratio", "factuality_ratio", "completeness_reward"]

        wandb_table = wandb.Table(columns=columns)
        
        n_entries = 0
        reward_normalizeds = 0
        rouge_scores = 0
        generated_lengths = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.eval_dataloader) if accelerator.is_main_process else self.eval_dataloader):
                if self.args.eval_loop_cap is not None and i == self.args.eval_loop_cap:
                    break
                
                # process references into tokens
                references_tok = self.policy_model.tokenizer.batch_encode_plus(
                    [r[0] for r in batch['references']],
                    return_tensors='pt', padding='max_length', 
                    truncation=True, 
                    max_length=self.policy_model.tokenizer.max_generated_len*2)
                 
                batch['references_input_ids'] = references_tok.input_ids.to(batch['prompts_input_ids'].device)

                results = self.policy_model.beam_decoding(
                    prompts_input_ids=batch['prompts_input_ids'],
                    prompts_attention_mask=batch['prompts_attention_mask'],
                    num_beams=1
                )

                eval_results = self.reward_model.eval_reward(
                    prompts_input_ids=results['prompts_input_ids'],
                    prompts_attention_mask=results['prompts_attention_mask'],
                    generated_input_ids=results['generated_input_ids'],
                    generated_attention_mask=results['generated_attention_mask'],
                    generated_texts=results['generated_text'],
                    metadata = batch['metadata'],
                )
                
                this_baseline_rewards = torch.tensor([np.sum(sublist) for sublist in eval_results['eval/baseline_rewards']],
                                            device=results['generated_input_ids'].device)
                this_finegrained_rewards = torch.tensor([np.sum(sublist) for sublist in eval_results['eval/finegrained_rewards']],
                                            device=results['generated_input_ids'].device)
                
                this_relevance_ratios = torch.tensor(eval_results['eval/relevance_ratios'], device=results['generated_input_ids'].device)
                this_factuality_ratios = torch.tensor(eval_results['eval/factuality_ratios'], device=results['generated_input_ids'].device)
                
                this_completeness_rewards = torch.tensor(eval_results['eval/completeness_rewards'], device=results['generated_input_ids'].device)
                
                this_n_sentences = torch.tensor(eval_results['eval/n_sentences'], device=results['generated_input_ids'].device)
                this_n_sub_sentences = torch.tensor(eval_results['eval/n_sub_sentences'], device=results['generated_input_ids'].device)
                
                # compute reward
                labels = batch['references']
                rouge_scores = rouge_score(results['generated_text'], labels)
                
                rouge_scores = torch.tensor(rouge_scores, device=results['generated_input_ids'].device)
                
                
                results_prompts_input_ids = accelerator.gather_for_metrics(results['prompts_input_ids'])
                results_generated_input_ids = accelerator.gather_for_metrics(results['generated_input_ids'])
                results_generated_attention_mask = accelerator.gather_for_metrics(results['generated_attention_mask'])
                batch_references_input_ids = accelerator.gather_for_metrics(batch['references_input_ids'])
                
                this_baseline_rewards = accelerator.gather_for_metrics(this_baseline_rewards)
                this_finegrained_rewards = accelerator.gather_for_metrics(this_finegrained_rewards)
                this_relevance_ratios = accelerator.gather_for_metrics(this_relevance_ratios)
                this_factuality_ratios = accelerator.gather_for_metrics(this_factuality_ratios)
                this_completeness_rewards = accelerator.gather_for_metrics(this_completeness_rewards)
                this_n_sentences = accelerator.gather_for_metrics(this_n_sentences)
                this_n_sub_sentences = accelerator.gather_for_metrics(this_n_sub_sentences)
                rouge_scores = accelerator.gather_for_metrics(rouge_scores)                
                
                if accelerator.is_main_process:
                    
                    prompt_inputs = self.policy_model.tokenizer.batch_decode(results_prompts_input_ids,
                                                                  skip_special_tokens=True, 
                                                                  clean_up_tokenization_spaces=True)
                    
                    generated_texts = self.policy_model.tokenizer.batch_decode(results_generated_input_ids,
                                                                  skip_special_tokens=True, 
                                                                  clean_up_tokenization_spaces=True)
                    references = self.policy_model.tokenizer.batch_decode(batch_references_input_ids,
                                                                  skip_special_tokens=True, 
                                                                  clean_up_tokenization_spaces=True)
                    
                    this_data_batch_size = results_prompts_input_ids.shape[0]
                    this_lens = torch.sum(results_generated_attention_mask, dim=-1)
                    
                    for batch_i in range(this_data_batch_size):
                        
                        # if n_entries >= self.eval_dataloader.number_of_samples:
                        #     break
                
                        wandb_table.add_data(step, prompt_inputs[batch_i], 
                                             references[batch_i], generated_texts[batch_i],
                                             this_baseline_rewards[batch_i].item(), 
                                             this_finegrained_rewards[batch_i].item(),
                                             rouge_scores[batch_i].item(), 
                                             this_lens[batch_i].item(),
                                             this_n_sentences[batch_i].item(),
                                             this_n_sub_sentences[batch_i].item(),
                                             this_relevance_ratios[batch_i].item(),
                                             this_factuality_ratios[batch_i].item(),
                                             this_completeness_rewards[batch_i].item())
                        
                        n_entries += 1

        if not self.args.nolog and accelerator.is_main_process:
    
            # do statistics        
            n_dev_samples = len(wandb_table.data)
            
            mean_baseline_reward = sum(row[wandb_table.columns.index('baseline_reward')] for row in wandb_table.data) / n_dev_samples
            mean_finegrained_reward = sum(row[wandb_table.columns.index('finegrained_reward')] for row in wandb_table.data) / n_dev_samples
            mean_rouge = sum(row[wandb_table.columns.index('rouge')] for row in wandb_table.data) / n_dev_samples
            mean_length = sum(row[wandb_table.columns.index('length')] for row in wandb_table.data) / n_dev_samples
            
            ns = np.array([row[wandb_table.columns.index('n_sentence')] for row in wandb_table.data])
            n_subs = np.array([row[wandb_table.columns.index('n_sub_sentence')] for row in wandb_table.data])
            relative_ratios = np.array([row[wandb_table.columns.index('relevance_ratio')] for row in wandb_table.data])
            factual_ratios = np.array([row[wandb_table.columns.index('factuality_ratio')] for row in wandb_table.data])
            
            mean_n_sentence = np.mean(ns)
            mean_n_sub_sentence = np.mean(n_subs)
            mean_relevance_ratio = np.sum(relative_ratios*n_subs) / np.sum(n_subs)
            mean_factuality_ratio = np.sum(factual_ratios*ns) / np.sum(ns)
            mean_completeness_reward = sum(row[wandb_table.columns.index('completeness_reward')] for row in wandb_table.data) / n_dev_samples
           
            
            stats = {
                'eval/step': step,
                'eval/finegrained_reward': mean_finegrained_reward,
                'eval/baseline_reward': mean_baseline_reward,
                'eval/rouge': mean_rouge,
                'eval/length': mean_length,
                'eval/n_sentence': mean_n_sentence,
                'eval/n_sub_sentence': mean_n_sub_sentence,
                'eval/relevance_ratio': mean_relevance_ratio,
                'eval/factuality_ratio': mean_factuality_ratio,
                'eval/completeness_reward': mean_completeness_reward,
                f'eval_generation/step_{step}': wandb_table,
            }
            wandb.log(stats)

            if self.args.fine_grained:
                reward_normalizeds = mean_finegrained_reward
            else:
                reward_normalizeds = mean_baseline_reward
        
            log_info(f'Evaluated [step {step}] rewards = {reward_normalizeds:.4f}, rouge = {mean_rouge:.4f}, length = {mean_length:.4f}')
            
        self.eval_accs[step] = reward_normalizeds


    """
    Internally set bias and gain terms based on the data from the dataloader
    """
    def set_reward_norm(self):
        accelerator.wait_for_everyone()

        with torch.no_grad():
            rewards = []
            for i, batch in enumerate(tqdm(self.train_dataloader) if accelerator.is_main_process else self.train_dataloader):
                if self.args.eval_loop_cap is not None and i == self.args.eval_loop_cap:
                    break
                results = self.policy_model.sample(
                    prompts_input_ids=batch['prompts_input_ids'],
                    prompts_attention_mask=batch['prompts_attention_mask'],
                    temperature=self.args.temperature,
                    top_k = self.args.top_k,
                    top_p = self.args.top_p,
                )
                results = self.reward_model.get_reward(
                    prompts_input_ids=batch['prompts_input_ids'],
                    prompts_attention_mask=batch['prompts_attention_mask'],
                    generated_input_ids=results['generated_input_ids'],
                    generated_attention_mask=results['generated_attention_mask'],
                    generated_texts=results['generated_text'],
                    metadata = batch['metadata'],
                    override_bias=0,
                    override_gain=1,
                )
                rewards += results['rewards/raw']

        rewards = torch.tensor(rewards, device=accelerator.device) # (N)
        rewards = accelerator.gather(rewards) # (num_gpus * N)
        # rewards = rewards[:len(self.train_dataloader.dataset)] # remove padding

        old_mean, old_std = rewards.mean().item(), rewards.std().item()
        new_mean, new_std = 0.0, 1.0
        self.reward_model.gain = new_std / old_std
        self.reward_model.bias = new_mean - self.reward_model.gain * old_mean

        log_info(f'Reward normalization coefficients set to: gain = {self.reward_model.gain:.4f} | bias = {self.reward_model.bias:.4f}')


def main():
    

    set_seed(args.seed, args.cuda_deterministic)

    # Set up save directories
    if not args.nosave:
        if args.mode == 'train':
            args.output_dir = '../eval'
            if args.load_from_ckpt is not None:
                args.save_dir = os.path.dirname(os.path.dirname(args.load_from_ckpt))
                args.run_name = args.save_dir.split('/')[-1]
            else:
                args.save_dir = os.path.join(args.output_dir, args.run_name)
            args.reward_dir = os.path.join(args.save_dir, 'reward')
            args.model_dir = os.path.join(args.save_dir, 'model')
            args.generated_dir = os.path.join(args.save_dir, 'knowledge')
            if accelerator.is_main_process:
                for d in [args.save_dir, args.reward_dir, args.model_dir, args.generated_dir]:
                    ensure_dir(d)

        log_info(f'Write to output directory: {args.save_dir}')
        if accelerator.is_main_process:
            with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    # Load data
    log_info(f'Loading data ...')

    tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_ckpt, model_max_length=args.max_input_len)
    # tokenizer.padding_side = args.input_padding_side   # Yushi: set padding side directly. Default is right.
    tokenizer.max_input_len = args.max_input_len
    tokenizer.max_generated_len = args.max_generated_len
    

    if args.mode == 'train':
        train_dataset = QADataset( 'train', tokenizer, accelerator=accelerator)
        # train ds is shuffled in its constructor
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=train_dataset.collate_fn)

        eval_dataset = QADataset( 'dev',  tokenizer, accelerator=accelerator, length_limit=None)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=eval_dataset.collate_fn)

        train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    elif args.mode == 'eval':
        train_dataset = None
        train_dataloader = None

        eval_dataset = QADataset(args.eval_split, tokenizer)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)

        eval_dataloader = accelerator.prepare(eval_dataloader)

    # Initialize models and optimizer
    log_info(f'Initializing models ...')
    if args.mode == 'train':
        ref_policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            tokenizer=tokenizer,
            policy_value_sharing=args.policy_value_sharing,
            accelerator=accelerator,
        )
        ref_policy.model, ref_policy.linear = accelerator.prepare(ref_policy.model, ref_policy.linear)
        policy = Policy(
            model_type=args.model_type,
            model_ckpt=args.model_ckpt,
            tokenizer=tokenizer,
            policy_value_sharing=args.policy_value_sharing,
            accelerator=accelerator,
        )
        
        # load policy model from run results
        if args.run_model_ckpt:
            policy.model.load_state_dict(torch.load(args.run_model_ckpt)['model'])
            log_info(f'Loaded policy model from {args.run_model_ckpt}')
        
        policy.model, policy.linear = accelerator.prepare(policy.model, policy.linear)
        
        value = Value(
            model_type=args.model_type,
            model_ckpt=args.value_model_ckpt,
            model=policy.model if args.policy_value_sharing else None,
            tokenizer=tokenizer,
            accelerator=accelerator,
            freeze_model=False if args.policy_value_sharing else args.freeze_value_model,
            # freeze_model=False if args.policy_value_sharing else True,
            )
        if not args.policy_value_sharing:
            value.model, value.linear = accelerator.prepare(value.model, value.linear)
        
        reward = AllReward(
            tokenizer=tokenizer,
            baseline_model_ckpt=args.baseline_model_ckpt,
            non_factual_model_ckpt=args.non_factual_model_ckpt,
            factual_model_ckpt=args.factual_model_ckpt,
            completeness_model_ckpt=args.completeness_model_ckpt,
            kl_coef=args.kl_coef,
            fine_grained = args.fine_grained,
            batch_size=None,
            baseline_reward_mean = args.baseline_reward_mean,
            baseline_reward_std = args.baseline_reward_std,
            baseline_reward_bias = args.baseline_reward_bias,
            baseline_reward_scale = args.baseline_reward_scale,
            verbosity_positive_reward = args.verbosity_positive_reward,
            verbosity_negative_reward = args.verbosity_negative_reward,
            factuality_positive_reward = args.factuality_positive_reward,
            factuality_negative_reward = args.factuality_negative_reward,
            completeness_reward_mean = args.completeness_reward_mean,
            completeness_reward_std = args.completeness_reward_std,
            completeness_reward_bias = args.completeness_reward_bias,
            completeness_reward_scale = args.completeness_reward_scale,
            sep = "</s>"
        )
        
        # prepare reward models
        reward.verbosity_reward.nf_reward_model = accelerator.prepare(reward.verbosity_reward.nf_reward_model)
        reward.factuality_reward.f_reward_model = accelerator.prepare(reward.factuality_reward.f_reward_model)
        reward.completeness_reward.model = accelerator.prepare(reward.completeness_reward.model)
        
        if not args.fine_grained:
            reward.baseline_reward.model = accelerator.prepare(reward.baseline_reward.model)
        
        # We never need to optimize the reward model's parameters separately!
        if args.policy_value_sharing:
            parameters = chain(policy.model.parameters(), policy.linear.parameters())
        else:
            parameters = chain(policy.model.parameters(), policy.linear.parameters(), value.model.parameters(), value.linear.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr, eps=1e-5)
        # args.total_steps = ceil_div(args.total_episodes, args.batch_size * int(os.environ['SLURM_GPUS_ON_NODE']) * int(os.environ['SLURM_JOB_NUM_NODES']))
        args.total_steps = ceil_div(args.total_episodes, args.batch_size * accelerator.num_processes * args.num_samples)
        
        scheduler = transformers.get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=100*args.noptepochs*accelerator.num_processes,
            num_training_steps=args.total_steps*args.noptepochs*accelerator.num_processes,
        )
        
        
        init_step = 0
        eval_accs = {}
        
            
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    # Set up trainer
    trainer = PPOTrainer(
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        ref_policy_model=ref_policy,
        policy_model=policy,
        value_model=value,
        reward_model=reward,
        optimizer=optimizer,
        scheduler=scheduler,
        init_step=init_step,
        eval_accs=eval_accs,
    )

    # Evaluate baseline (no knowledge)
    if args.eval_baseline:
        trainer.eval(step=-1)

    # Evaluation
    trainer.valid(step=0)


if __name__ == '__main__':
    main()
