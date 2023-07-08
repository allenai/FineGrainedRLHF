# *********************************************************
#  Version 1
#  Author: Yushi Hu
#  Date: 2023-06-20
#  Description: the PPO trainer
#  Referenced: https://github.com/liujch1998/rainier/
#  All Rights Reserved.
#  *********************************************************

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
import yaml
import nltk
from typing import Optional, List, Iterable, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
import accelerate
import wandb

from .utils import ensure_dir, set_seed, reduce_mean, reduce_sum, ceil_div, whiten, clamp

logging.basicConfig(level=logging.ERROR)

class PPOTrainer:
    def __init__(self,
                 args: argparse.Namespace,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 ref_policy_model,
                 policy_model,
                 value_model,
                 reward_model,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 accelerator: accelerate.Accelerator,
                 log_info,
                ):
        
        self.accelerator = accelerator
        self.log_info = log_info
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

        self.batchify = lambda x, n: [x[i:i + n] for i in range(0, len(x), n)]
        
        if self.accelerator.is_main_process:
            if args['logging']['wandb_log']:
                wandb.init(entity=args["logging"]["wandb_entity"], project=args["logging"]["wandb_project"], name=args['logging']['run_name'], config=args)
            else:
                wandb.init(config=args, mode='disabled')
            
            wandb.define_metric('train/step')
            wandb.define_metric('eval/step')
            wandb.define_metric('train/*', step_metric='train/step')
            wandb.define_metric('eval/*', step_metric='eval/step', summary='max')

        self.train_sampler = iter(self.train_dataloader)
        for _ in range(len(self.train_dataloader)):
            next(self.train_sampler)

        self.eval_accs = {}
        

    def compute_advantages(self, results, num_samples):
        
        old_values = results['generated_value']
        rewards = results['rewards/penalized']
        mask = results['generated_attention_mask'] # (B, KL)
        
        with torch.no_grad():
            if self.args['ppo']['whiten_rewards']:
                whitened_rewards = whiten(rewards, mask, shift_mean=False, accelerator=self.accelerator)
            else:
                whitened_rewards = rewards
            
            lastgaelam = 0
            advantages_reversed = []
            gen_length = mask.sum(dim=1).max().item()
            for t in reversed(range(gen_length)):
                nextvalues = old_values[:, t + 1] if t < gen_length - 1 else 0.0
                delta = whitened_rewards[:, t] + self.args['ppo']['gamma'] * nextvalues - old_values[:, t]
                lastgaelam = delta + self.args['ppo']['gamma'] * self.args['ppo']['lam'] * lastgaelam
                advantages_reversed.append(lastgaelam)
            advantages = torch.stack(advantages_reversed[::-1], dim=1)
            advantages = F.pad(advantages, (0, whitened_rewards.size(1) - gen_length), value=0.0)
            returns = advantages + old_values
            
            whitened_advantages = advantages.detach()
            whitened_advantages = whiten(advantages, mask, accelerator=self.accelerator).detach()

            
        results['whitened_advantages'] = whitened_advantages
        results['returns'] = returns
                

    def loss(self, results, all_mask_weight):
        
        old_values = results['generated_value']
        old_logprobs = results['generated_logprobs']
        mask = results['generated_attention_mask'] # (B, KL)
        
        whitened_advantages = results['whitened_advantages']
        returns = results['returns']

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
        pg_losses2 = -whitened_advantages * torch.clamp(ratio, min=1.0 - self.args['ppo']['cliprange'], max=1.0 + self.args['ppo']['cliprange'])
        pg_loss = reduce_mean(torch.max(pg_losses1, pg_losses2), mask)
        pg_loss = pg_loss * weight

        if self.args['model']['value_model']['policy_value_sharing']:
            new_values = policy_forward['generated_value']
        else:
            value_forward = self.value_model.forward_pass(**forward_inputs)
            new_values = value_forward['generated_value']
            new_values *= mask

        new_values_clipped = clamp(new_values, old_values - self.args['ppo']['cliprange_value'], old_values + self.args['ppo']['cliprange_value'])
        vf_losses1 = torch.square(new_values - returns)
        vf_losses2 = torch.square(new_values_clipped - returns)
        vf_loss = .5 * reduce_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_loss = vf_loss * weight

        loss = self.args['ppo']['pg_coef'] * pg_loss + self.args['ppo']['vf_coef'] * vf_loss

        results['loss/total'] = loss
        results['loss/policy'] = pg_loss
        results['loss/value'] = vf_loss

    def train(self, step):
        if step % self.args['train']['eval_interval'] == 0:
            self.save(step=step)
            self.valid(step=step)

        self.accelerator.wait_for_everyone()
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
                num_return_sequences=self.args['env']['train_num_samples_per_input'],
                **self.args['model']['policy_model']['train_generation_kwargs'],
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
        if not self.args['model']['value_model']['policy_value_sharing']:
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
                metadata = [elem for elem in batch['metadata'] for _ in range(self.args['env']['train_num_samples_per_input'])],
            )
            results.update(reward_results)
            self.reward_model.kl_penalize_reward(results)

        # Get advantages
        self.compute_advantages(results, self.args['env']['train_num_samples_per_input'])
        
        n_results = len(results['generated_input_ids'])
        
        loss_totals, loss_policies, loss_values =  [], [], []
        reward_penalizeds, reward_kls, reward_raws = [], [], []
        
        # Train
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
        
        self.policy_model.model.train()
        self.value_model.model.train()
        self.value_model.linear.train()
        
        for ppo_epoch_idx in range(self.args['train']['n_ppo_epoch_per_rollout']):
            self.optimizer.zero_grad()
            
            # get the weight for each sub-batch
            mask = results['generated_attention_mask']
            all_mask = self.accelerator.gather(mask)
            all_mask_weight = all_mask.sum(dim=1).float().mean().item()
            
            for batch_idx in range(0,n_results, self.args['train']['training_batch_size_per_card']):
                batch_results = {}
                
                for k, v in results.items():
                    batch_results[k] = v[batch_idx:batch_idx+self.args['train']['training_batch_size_per_card']]
            
                self.loss(batch_results, all_mask_weight)
                # gradient accumulation weight
                self.accelerator.backward(batch_results['loss/total'])
                
                # logging
                if ppo_epoch_idx == self.args['train']['n_ppo_epoch_per_rollout'] - 1:
                    loss_total = batch_results['loss/total'].unsqueeze(0) # (1)
                    loss_policy = batch_results['loss/policy'].unsqueeze(0) # (1)
                    loss_value = batch_results['loss/value'].unsqueeze(0) # (1)
                    reward_penalized = torch.mean(reduce_sum(batch_results['rewards/penalized'], batch_results['generated_attention_mask'], axis=1)).unsqueeze(0) # (1)
                    reward_kl = torch.mean(reduce_sum(batch_results['rewards/kl'], batch_results['generated_attention_mask'], axis=1)).unsqueeze(0) # (1)
                    reward_raw =  torch.mean(reduce_sum(batch_results['rewards/raw'], batch_results['generated_attention_mask'], axis=1)).unsqueeze(0) # (1)

                    loss_totals.append(loss_total)
                    loss_policies.append(loss_policy)
                    loss_values.append(loss_value)
                    reward_penalizeds.append(reward_penalized)
                    reward_kls.append(reward_kl)
                    reward_raws.append(reward_raw)
                    
                
            if self.args['train']['clip_grad']:
                self.accelerator.clip_grad_norm_(
                    chain(self.policy_model.model.parameters(),
                        self.policy_model.linear.parameters(),
                        self.value_model.model.parameters(),
                        self.value_model.linear.parameters()
                        ),
                    self.args['train']['max_grad_norm'])
                
            self.optimizer.step()
            self.scheduler.step()

        loss_total = torch.cat(loss_totals, dim=0)
        loss_policy = torch.cat(loss_policies, dim=0)
        loss_value = torch.cat(loss_values, dim=0)
        reward_penalized = torch.cat(reward_penalizeds, dim=0)
        reward_kl = torch.cat(reward_kls, dim=0)
        reward_raw = torch.cat(reward_raws, dim=0)

        losses_total = self.accelerator.gather(loss_total) # (num_gpus)
        losses_policy = self.accelerator.gather(loss_policy) # (num_gpus)

        losses_value = self.accelerator.gather(loss_value) # (num_gpus)
        rewards_penalized = self.accelerator.gather(reward_penalized) # (num_gpus)
        rewards_kl = self.accelerator.gather(reward_kl) # (num_gpus)
        rewards_raw = self.accelerator.gather(reward_raw) # (num_gpus)

        loss_total = losses_total.mean().item()
        loss_policy = losses_policy.mean().item()

        loss_value = losses_value.mean().item()
        reward_penalized = rewards_penalized.mean().item()
        reward_kl = rewards_kl.mean().item()
        reward_raw = rewards_raw.mean().item()
    
        # Logging
        if self.args['logging']['wandb_log'] and self.accelerator.is_main_process:

            this_batch_kl = np.mean(reward_kl)

            if step % self.args['logging']['log_interval'] == 0:
                wandb.log({
                    'train/step': step,
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/loss/total': np.mean(loss_total),
                    'train/loss/policy': np.mean(loss_policy),
                    'train/loss/value': np.mean(loss_value),
                    'train/reward/penalized': np.mean(reward_penalized),
                    'train/reward/KL': this_batch_kl,
                    'train/reward/raw': np.mean(reward_raw),
                })
                
            if this_batch_kl > self.args['train']['kl_threshold']:
                self.log_info(f"KL divergence {this_batch_kl} exceeds threshold {self.args['train']['kl_threshold']}")
                self.huge_kl_count += 1
                if self.huge_kl_count >= 5:
                    self.should_early_stop = True

    def valid(self, step):
        self.log_info(f'Evaluating [step {step}] ...')

        self.accelerator.wait_for_everyone()
        
        self.policy_model.model.eval()
        if not self.args['model']['value_model']['policy_value_sharing']:
            self.value_model.model.eval()
            
        columns=["step", "inputs", "outputs"]
        wandb_table = None
        
        n_entries = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.eval_dataloader) if self.accelerator.is_main_process else self.eval_dataloader):

                results = self.policy_model.sample(
                    prompts_input_ids=batch['prompts_input_ids'],
                    prompts_attention_mask=batch['prompts_attention_mask'],
                    **self.args['model']['policy_model']['eval_generation_kwargs'],
                )

                eval_results = self.reward_model.eval_metrics(
                    prompts_input_ids=results['prompts_input_ids'],
                    prompts_attention_mask=results['prompts_attention_mask'],
                    generated_input_ids=results['generated_input_ids'],
                    generated_attention_mask=results['generated_attention_mask'],
                    generated_texts=results['generated_text'],
                    metadata = batch['metadata'],
                )
                
                # gather all results
                batch = self.accelerator.gather_for_metrics(batch)
                results = self.accelerator.gather_for_metrics(results)
                
                for eval_k, eval_v in eval_results.items():
                    eval_results[eval_k] = self.accelerator.gather(
                        torch.tensor(eval_v, device=results['generated_input_ids'].device))
                    
                # initialize wandb table if it does not exist
                if wandb_table is None:
                    columns.extend(list(eval_results.keys())) 
                    wandb_table = wandb.Table(columns=columns)
                
                if self.accelerator.is_main_process:
                    
                    prompt_inputs = self.policy_model.tokenizer.batch_decode(results['prompts_input_ids'],
                                                                  skip_special_tokens=True, 
                                                                  clean_up_tokenization_spaces=True)
                    
                    generated_texts = self.policy_model.tokenizer.batch_decode(results['generated_input_ids'],
                                                                  skip_special_tokens=True, 
                                                                  clean_up_tokenization_spaces=True)
                    
                    this_data_batch_size = results['prompts_input_ids'].shape[0]
                    this_lens = torch.sum(results['generated_attention_mask'], dim=-1)
                    
                    for batch_i in range(this_data_batch_size):
                        
                        this_entry = [step, prompt_inputs[batch_i], generated_texts[batch_i]]
                        
                        for eval_v in eval_results.values():
                            this_entry.append(eval_v[batch_i].item())
                        
                        wandb_table.add_data(*this_entry)
                        n_entries += 1

        if self.accelerator.is_main_process:
    
            # do statistics        
            n_dev_samples = len(wandb_table.data)
            
            stats = {'eval/step': step,
                     f'eval_generation/step_{step}': wandb_table}
            
            value_columns = columns[3:] # the first three are steps, inputs, outputs
            stats.update(self.reward_model.aggregate_metrics(wandb_table, value_columns))
            
            
            if self.args['logging']['wandb_log']:
                wandb.log(stats)

            mean_rewards = stats["eval/rewards"]
        
            self.log_info(f'Evaluated [step {step}] rewards = {mean_rewards:.4f}')
            
            prev_best_step = None if len(self.eval_accs) == 0 else max(self.eval_accs, key=self.eval_accs.get)
            self.eval_accs[step] = mean_rewards
            if prev_best_step is None or mean_rewards > self.eval_accs[prev_best_step]:
                if prev_best_step is not None:
                    try:
                        os.remove(f"{self.args['logging']['save_dir']}/ckp_{prev_best_step}.pth")
                    except:
                        self.log_info(f'Cannot remove previous best ckpt!')
                shutil.copy(f"{self.args['logging']['save_dir']}/last.pth", f"{self.args['logging']['save_dir']}/ckp_{step}.pth")
                self.log_info(f'Best ckpt updated to [step {step}]')
                
                # save best policy again
                self.accelerator.wait_for_everyone()
                policy_model_state_dict = self.accelerator.unwrap_model(self.policy_model.model).state_dict()
                self.accelerator.save(policy_model_state_dict, f"{self.args['logging']['save_dir']}/best_policy.pth")


    def save(self, step):
        # this will overwrite an existing ckpt with the save filename!
        self.accelerator.wait_for_everyone()
        policy_model_state_dict = self.accelerator.unwrap_model(self.policy_model.model).state_dict()
        policy_linear_state_dict = self.accelerator.unwrap_model(self.policy_model.linear).state_dict()
        if not self.args['model']['value_model']['policy_value_sharing']:
            value_model_state_dict = self.accelerator.unwrap_model(self.value_model.model).state_dict()
            value_linear_state_dict = self.accelerator.unwrap_model(self.value_model.linear).state_dict()

        result = {
            'model': policy_model_state_dict,
            'linear': policy_linear_state_dict,
            'step': step,
            'eval_accs': self.eval_accs,
             # 'optimizer': optimizer_state_dict,
        }
        if not self.args['model']['value_model']['policy_value_sharing']:
            result['value_model'] = value_model_state_dict
            result['value_linear'] = value_linear_state_dict
        self.accelerator.wait_for_everyone()
        self.accelerator.save(result, f"{self.args['logging']['save_dir']}/last.pth")
        self.log_info(f'[step {step}] model checkpoint saved')


