# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, Set
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn, BufferedDataLoader
import jsonlines
import json
from verl.utils.aws_utils import aws_copy, aws_check_file_exists
from collections import Counter
from verl.custom_pytorch.serialization import load
from verl.utils.model import compute_position_id_with_mask


WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, config=None):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        use_std=config.algorithm.grpo.use_std)
        data.batch['advantages'] = advantages

        w = getattr(config.scale_reasoning, 'plain_adv_weight', 1.0)
        if 'on_policy_mask' in data.batch and w != 1.0:
            # advantages: [B, T], is_plain: [B]
            adv = data.batch['advantages']
            mask = data.batch['on_policy_mask'].float().unsqueeze(-1)  # [B,1]

            # sanity checks
            assert adv.ndim == 2, f"expected advantages shape [B,T], got {adv.shape}"
            assert mask.ndim == 2, f"expected mask shape [B,1],    got {mask.shape}"
            assert mask.size(0) == adv.size(0), \
                f"batch‐size mismatch: mask {mask.size(0)} vs adv {adv.size(0)}"
            assert mask.size(1) == 1, f"mask must have shape [B,1], got {mask.shape}"

            # scale only the plain‐prompt advantages:
            print(f"scaling advantages by {w}")
            data.batch['advantages'] = adv * (1 + (w - 1) * mask)
        else:
            if 'on_policy_mask' not in data.batch:
                print(f"no on_policy_mask found, skipping scaling")
            else:
                print(f"no scaling because w = {w}")
        
        data.batch['returns'] = returns
    elif adv_estimator == 'rloo':
        response_length = data.batch['responses'].size(-1)
        response_mask = data.batch['attention_mask'][:, -response_length:]
        advantages, returns = core_algos.compute_rloo_returns(data=data, eos_mask=response_mask, n_samples=config.actor_rollout_ref.rollout.n, config=config)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    sequence_verify_score = batch.batch['verify_scores']
    sequence_format_score = batch.batch['format_scores']

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        
        'train/filter_accuracy':
            torch.mean(sequence_verify_score).detach().item(),
        'train/format_accuracy':
            torch.mean(sequence_format_score).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }

def compute_global_metrics(batch, global_metrics: dict):
    response_info = _compute_response_info(batch)
    num_response_tokens = torch.sum(response_info['response_length']).item()
    correct_response_tokens = torch.sum(response_info['response_length'] * batch.batch['verify_scores'].squeeze(-1)).item()
    incorrect_response_tokens = torch.sum(response_info['response_length'] * (1 - batch.batch['verify_scores'].squeeze(-1))).item()
    
    global_metrics['global_metrics/total_train_samples'] += len(batch)
    global_metrics['global_metrics/total_train_tokens'] += num_response_tokens
    global_metrics['global_metrics/total_train_correct_tokens'] += correct_response_tokens
    global_metrics['global_metrics/total_train_incorrect_tokens'] += incorrect_response_tokens
    global_metrics['global_metrics/total_train_correct_responses'] += torch.sum(batch.batch['verify_scores']).item()
    global_metrics['global_metrics/total_train_incorrect_responses'] += torch.sum(1 - batch.batch['verify_scores']).item()

    return global_metrics

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, text="{name}: {seconds:.1f} seconds") as timer:
        yield

    # if name is already in timing_raw, update it
    if name in timing_raw:
        timing_raw[name] += timer.last
    else:
        timing_raw[name] = timer.last


def save_step_data(batch, tokenizer, step, local_dir, s3_dir=None):
    """
    Save the prompts, responses, and grades from the current batch to a jsonl file.
    Data is grouped by unique prompts, with each prompt having a list of responses and scores.
    
    Args:
        batch: The current batch of data
        tokenizer: Tokenizer to decode token IDs
        step: Current global step
        local_dir: Directory to save local files
        s3_dir: Optional S3 directory to upload data
    """
    # Create step directory
    step_dir = os.path.join(local_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Extract prompts, responses and grades
    prompts_idx = batch.batch['prompts'] if 'prompts' in batch.batch else batch.batch['input_ids']
    responses_idx = batch.batch['responses']
    scores = batch.batch['verify_scores'].squeeze(-1).cpu().tolist() if 'verify_scores' in batch.batch else []
    
    # Decode tokens to text
    prompt_texts = [tokenizer.decode(prompt, skip_special_tokens=True) for prompt in prompts_idx]
    response_texts = [tokenizer.decode(response, skip_special_tokens=True) for response in responses_idx]
    
    # Group by prompts
    prompt_data = {}
    
    for i, (prompt, response) in enumerate(zip(prompt_texts, response_texts)):
        score = scores[i] if i < len(scores) else None
        
        if prompt not in prompt_data:
            prompt_data[prompt] = {
                'prompt': prompt,
                'responses': [],
                'scores': [],
                'step': step
            }
        
        prompt_data[prompt]['responses'].append(response)
        if score is not None:
            prompt_data[prompt]['scores'].append(score)
    
    # Save to jsonl file
    output_file = os.path.join(step_dir, "step_data.jsonl")
    with jsonlines.open(output_file, mode='w') as writer:
        for data in prompt_data.values():
            writer.write(data)
    
    # Upload to S3 if path provided
    if s3_dir:
        s3_step_dir = os.path.join(s3_dir, f"step_{step}")
        aws_copy(step_dir, s3_step_dir, recursive=True)
    
    return list(prompt_data.keys())


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'remax':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'rloo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # Track unique prompts for metrics
        self.unique_prompts: Set[str] = set()

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'. But config is: {config}")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated). But config is: {config}.")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        nnodes = self.config.trainer.nnodes
        n_gpus = self.config.trainer.n_gpus_per_node
        dp_size = nnodes * n_gpus
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         apply_chat_template=self.config.data.get('apply_chat_template', True),
                                         use_hint=self.config.scale_reasoning.get('use_hint', False),
                                         use_cot=self.config.scale_reasoning.get('use_cot', False),
                                         use_ref_prompt=self.config.scale_reasoning.get('use_ref_prompt', False),
                                         custom_chat_template=self.config.scale_reasoning.get('custom_chat_template', None))
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        
        self.test_dataset = RLHFDataset(parquet_files=self.config.data.test_files,
                                        tokenizer=self.tokenizer,
                                        prompt_key=self.config.data.prompt_key,
                                        max_prompt_length=self.config.data.max_prompt_length,
                                        filter_prompts=True,
                                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        truncation='error',
                                        apply_chat_template=self.config.data.get('apply_chat_template', True),
                                        use_hint=self.config.scale_reasoning.get('use_hint', False),
                                        use_cot=self.config.scale_reasoning.get('use_cot', False),
                                        use_ref_prompt=self.config.scale_reasoning.get('use_ref_prompt', False),
                                        custom_chat_template=self.config.scale_reasoning.get('custom_chat_template', None))
        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                         batch_size=(len(self.test_dataset) // dp_size) * dp_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn) 

        self.train_dataloader = BufferedDataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error',
                                       apply_chat_template=self.config.data.get('apply_chat_template', True),
                                       use_hint=self.config.scale_reasoning.get('use_hint', False),
                                       use_cot=self.config.scale_reasoning.get('use_cot', False),
                                       use_ref_prompt=self.config.scale_reasoning.get('use_ref_prompt', False),
                                       custom_chat_template=self.config.scale_reasoning.get('custom_chat_template', None))
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=(len(self.val_dataset) // dp_size) * dp_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)                    

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1
        assert len(self.test_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        print(f'Size of test dataloader: {len(self.test_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores, format_scores, raw_prompt, answers, prefix):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np

        # Create tuples of (input, output, score, format_score) and sort by input text
        samples = list(zip(inputs, raw_prompt, answers, outputs, scores, format_scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step", "example_id", "full_prompt", "raw_prompt", "answer", "output", "score", "format_score"]

        if prefix == 'val':
            if not hasattr(self, 'validation_table'):
                # Initialize the table on first call
                self.validation_table = wandb.Table(columns=columns)
        elif prefix == 'test':
            if not hasattr(self, 'test_table'):
                # Initialize the table on first call
                self.test_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        if prefix == 'val':
            new_table = wandb.Table(columns=columns, data=self.validation_table.data)
        elif prefix == 'test':
            new_table = wandb.Table(columns=columns, data=self.test_table.data)

        # Add new row with all data
        # row_data = []
        for i, sample in enumerate(samples):
            row_data = [self.global_steps, i] + list(sample)
            new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({f"{prefix}/generations": new_table}, step=self.global_steps)
        
        if prefix == 'val':
            self.validation_table = new_table
        elif prefix == 'test':
            self.test_table = new_table

    def save_validation_info(self, test_batch_lst, reward_tensor_lst, data_source_lst, metric_dict, ground_truth_lst):
        data_source_info_map = {}
        for test_batch, reward_tensor_batch, data_source_batch, ground_truth_batch in zip(test_batch_lst, reward_tensor_lst, data_source_lst, ground_truth_lst):
            test_batch_dict = test_batch.pop(['prompts', 'responses'])
            tokenizer = self.tokenizer
            
            prompts_idx = test_batch_dict.batch['prompts']
            responses_idx = test_batch_dict.batch['responses']
            
            prompt_strs = [tokenizer.decode(prompt, skip_special_tokens=True) for prompt in prompts_idx]
            response_strs = [tokenizer.decode(response, skip_special_tokens=True) for response in responses_idx]
            
            for prompt, response, reward, data_source, ground_truth in zip(prompt_strs, response_strs, reward_tensor_batch, data_source_batch, ground_truth_batch):
                if data_source not in data_source_info_map:
                    data_source_info_map[data_source] = []
                data_source_info_map[data_source].append({
                    'prompt': prompt,
                    'response': response,
                    'reward': reward.sum(-1).item(),
                    'ground_truth': ground_truth
                })

        local_dir = os.path.join(self.config.trainer.default_local_dir, 'validation', f"global_step_{self.global_steps}")
        os.makedirs(local_dir, exist_ok=True)

        for data_source, info_lst in data_source_info_map.items():
            # Create a safe filename by replacing slashes with underscores
            safe_filename = data_source.replace('/', '_')
            
            # save to local and to s3
            local_path = os.path.join(local_dir, f"{safe_filename}.jsonl")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with jsonlines.open(local_path, mode='w') as writer:
                for info in info_lst:
                    # Store original data source in the info
                    info['data_source'] = data_source
                    writer.write(info)
        
        local_metrics_path = os.path.join(local_dir, f"metrics.json")
        with open(local_metrics_path, 'w') as f:
            json.dump(metric_dict, f)
        
        if self.config.trainer.s3_path is not None:
            s3_dir = os.path.join(self.config.trainer.s3_path, 'validation', f"global_step_{self.global_steps}")
            aws_copy(local_dir, s3_dir, recursive=True)

    def _validate(self):
        dataloaders = [self.val_dataloader, self.test_dataloader]
        prefix_names = ["val", "test"]
        metric_dict = {}

        for dataloader, prefix_name in zip(dataloaders, prefix_names):
            reward_tensor_lst = []
            format_reward_tensor_lst = []
            data_source_lst = []
            ground_truth_lst = []
            test_batch_lst = []

            # Lists to collect samples for the table
            sample_inputs = []
            sample_outputs = []
            sample_scores = []
            sample_answers = []
            sample_raw_prompt = []
            sample_format_scores = []
            for test_data in dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if (
                    self.config.reward_model.enable
                    and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
                ):
                    return {}

                # Store original inputs
                input_ids = test_batch.batch["input_ids"]
                input_texts = [
                    self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
                ]
                sample_inputs.extend(input_texts)

                test_gen_batch = test_batch.pop(["input_ids", "attention_mask", "position_ids"])
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": False,
                    "validate": True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                    test_gen_batch, self.actor_rollout_wg.world_size
                )
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                    test_gen_batch_padded
                )
                # unpad
                test_output_gen_batch = unpad_dataproto(
                    test_output_gen_batch_padded, pad_size=pad_size
                )
                print("validation generation end")

                # Store generated outputs
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [
                    self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
                ]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)
                test_batch_lst.append(test_batch)
                # evaluate using reward_function
                reward_tensor, format_reward_tensor = self.val_reward_fn(test_batch)
                format_reward_tensor = format_reward_tensor.squeeze(-1)

                # Store scores
                scores = reward_tensor.sum(-1).cpu().tolist()
                format_scores = format_reward_tensor.cpu().tolist()
                sample_scores.extend(scores)
                sample_format_scores.extend(format_scores)
                sample_raw_prompt.extend([data_item.non_tensor_batch['raw_prompt'][-1]['content'] for data_item in test_batch])
                sample_answers.extend([data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in test_batch])

                reward_tensor_lst.append(reward_tensor)
                format_reward_tensor_lst.append(format_reward_tensor)
                data_source_lst.append(
                    test_batch.non_tensor_batch.get(
                        "data_source", ["unknown"] * reward_tensor.shape[0]
                    )
                )
                ground_truth = [
                    data_item.non_tensor_batch["reward_model"]["ground_truth"]
                    for data_item in test_batch
                ]
                ground_truth_lst.append(ground_truth)

            self._maybe_log_val_generations_to_wandb(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                format_scores=sample_format_scores,
                raw_prompt=sample_raw_prompt,
                answers=sample_answers,
                prefix=prefix_name,
            )

            reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
            format_reward_tensor = torch.cat(format_reward_tensor_lst, dim=0).cpu()  # (batch_size,)
            data_sources = np.concatenate(data_source_lst, axis=0)

            # evaluate test_score based on data source
            data_source_metrics = {}
            for i in range(reward_tensor.shape[0]):
                data_source = data_sources[i]
                if data_source not in data_source_metrics:
                    data_source_metrics[data_source] = {
                        'reward': [],
                        'format': []
                    }
                data_source_metrics[data_source]['reward'].append(reward_tensor[i].item())
                data_source_metrics[data_source]['format'].append(format_reward_tensor[i].item())

            # Process metrics
            domain_metrics = {}
            source_metrics = {}
            subcategory_metrics = {}

            # Calculate metrics for each level
            for data_source, reward_dict in data_source_metrics.items():
                rewards = reward_dict['reward']
                format_scores = reward_dict['format']
                parts = data_source.split("/")

                # Domain level (first component) - required
                domain = parts[0]
                if domain not in domain_metrics:
                    domain_metrics[domain] = {
                        'reward': [],
                        'format': []
                    }
                domain_metrics[domain]['reward'].extend(rewards)
                domain_metrics[domain]['format'].extend(format_scores)

                # Source level (second component) - optional
                if len(parts) > 1:
                    source = parts[1]
                    if source not in source_metrics:
                        source_metrics[source] = {
                            'reward': [],
                            'format': []
                        }
                    source_metrics[source]['reward'].extend(rewards)
                    source_metrics[source]['format'].extend(format_scores)

                # Subcategory level (third component) - optional
                if len(parts) > 2:
                    subcategory = parts[2]
                    if subcategory not in subcategory_metrics:
                        subcategory_metrics[subcategory] = {
                            'reward': [],
                            'format': []
                        }
                    subcategory_metrics[subcategory]['reward'].extend(rewards)
                    subcategory_metrics[subcategory]['format'].extend(format_scores)

            # Log metrics for each level (only if metrics exist)
            for domain, reward_dict in domain_metrics.items():
                metric_dict[f"{prefix_name}/domain/{domain}/score"] = np.mean(reward_dict['reward'])
                metric_dict[f"{prefix_name}/domain/{domain}/format_score"] = np.mean(reward_dict['format'])

            if source_metrics:  # Only log source metrics if we have any
                for source, reward_dict in source_metrics.items():
                    metric_dict[f"{prefix_name}/source/{source}/score"] = np.mean(reward_dict['reward'])
                    metric_dict[f"{prefix_name}/source/{source}/format_score"] = np.mean(reward_dict['format'])

            if subcategory_metrics:  # Only log subcategory metrics if we have any
                for subcategory, reward_dict in subcategory_metrics.items():
                    metric_dict[f"{prefix_name}/subcategory/{subcategory}/score"] = np.mean(reward_dict['reward'])
                    metric_dict[f"{prefix_name}/subcategory/{subcategory}/format_score"] = np.mean(reward_dict['format'])

            # Keep the overall metric
            metric_dict[f"{prefix_name}/test_score/all"] = np.mean(reward_tensor.cpu().tolist())
            metric_dict[f"{prefix_name}/test_format_score/all"] = np.mean(format_reward_tensor.cpu().tolist())

            self.save_validation_info(
                test_batch_lst, reward_tensor_lst, data_source_lst, metric_dict, ground_truth_lst
            )

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        s3_actor_path = None if self.config.trainer.s3_path is None else os.path.join(
            self.config.trainer.s3_path, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, self.global_steps)
        self.actor_rollout_wg.save_huggingface_checkpoint(actor_local_path, s3_actor_path)
       
        if self.config.trainer.s3_path is not None:
            self.actor_rollout_wg.upload_to_s3(actor_local_path, s3_actor_path, recursive=True)
            self.actor_rollout_wg.upload_to_s3(os.path.join(actor_local_path, 'huggingface'), os.path.join(s3_actor_path, 'huggingface'), recursive=True, node_1_only=True)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            s3_critic_path = None if self.config.trainer.s3_path is None else os.path.join(
                self.config.trainer.s3_path, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, self.global_steps)
            if self.config.trainer.s3_path is not None:
                self.critic_wg.upload_to_s3(critic_local_path, s3_critic_path, recursive=True)

        if self.use_rm:
            rm_local_path = os.path.join(local_global_step_folder, 'rm')
            s3_rm_path = None if self.config.trainer.s3_path is None else os.path.join(
                self.config.trainer.s3_path, f'global_step_{self.global_steps}', 'rm')
            self.rm_wg.save_checkpoint(rm_local_path, self.global_steps)
            if self.config.trainer.s3_path is not None:
                self.rm_wg.upload_to_s3(rm_local_path, s3_rm_path, recursive=True)

        # Save unique prompts set
        unique_prompts_local_path = os.path.join(local_global_step_folder, 'unique_prompts.json')
        try:
            with open(unique_prompts_local_path, 'w') as f:
                json.dump(list(self.unique_prompts), f)
            print(f"Successfully saved {len(self.unique_prompts)} unique prompts to {unique_prompts_local_path}")
        except Exception as e:
            print(f"Warning: Failed to save unique prompts: {str(e)}")

        # save dataloader with more robust approach
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        try:
            import dill
            # Save with explicit binary mode indication and protocol version
            torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill, pickle_protocol=4)
            print(f"Successfully saved dataloader to {dataloader_local_path}")
        except Exception as e:
            print(f"Warning: Failed to save dataloader: {str(e)}")
            # Consider fallback to saving just the essential state

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

        # Ensure S3 copy uses binary mode for data.pt file
        if self.config.trainer.s3_path is not None:
            try:
                # First copy the dataloader file with explicit content-type for binary data
                s3_dataloader_path = os.path.join(self.config.trainer.s3_path, 
                                               f'global_step_{self.global_steps}', 'data.pt')
                import boto3
                s3_client = boto3.client('s3')
                bucket, key = s3_dataloader_path.replace('s3://', '').split('/', 1)
                s3_client.upload_file(
                    dataloader_local_path, 
                    bucket, 
                    key,
                    ExtraArgs={'ContentType': 'application/octet-stream'}
                )
                print(f"Successfully uploaded dataloader to S3 with binary content type")

                # Upload unique prompts to S3
                s3_unique_prompts_path = os.path.join(self.config.trainer.s3_path, 
                                                    f'global_step_{self.global_steps}', 'unique_prompts.json')
                bucket, key = s3_unique_prompts_path.replace('s3://', '').split('/', 1)
                s3_client.upload_file(
                    unique_prompts_local_path,
                    bucket,
                    key,
                    ExtraArgs={'ContentType': 'application/json'}
                )
                print(f"Successfully uploaded unique prompts to S3")

                s3_latest_checkpointed_iteration = os.path.join(self.config.trainer.s3_path, 'latest_checkpointed_iteration.txt')
                bucket, key = s3_latest_checkpointed_iteration.replace('s3://', '').split('/', 1)
                s3_client.upload_file(
                    local_latest_checkpointed_iteration,
                    bucket,
                    key,
                    ExtraArgs={'ContentType': 'text/plain'}
                )
                print(f"Successfully uploaded latest_checkpointed_iteration.txt to S3 with text content type")
            except Exception as e:
                print(f"Warning: S3 upload with content type failed, falling back to standard copy: {str(e)}")
                if self.config.trainer.s3_path is not None:
                    aws_copy(self.config.trainer.default_local_dir, self.config.trainer.s3_path, recursive=True)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        if self.config.trainer.s3_path is not None:
            remote_tracker_file_path = get_checkpoint_tracker_filename(self.config.trainer.s3_path)
            local_tracker_file_path = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
            if aws_check_file_exists(remote_tracker_file_path):
                try:
                    if self.config.trainer.s3_path is not None:
                        aws_copy(remote_tracker_file_path, local_tracker_file_path)
                except Exception as e:
                    print(f'Error copying tracker file')
                    raise e

                # check if local_tracker_file_path exists
                if not os.path.exists(local_tracker_file_path):
                    global_step_folder = None
                else:
                    try:
                        # overwrite iteration number if self.config.trainer.resume_from_iteration is not None
                        if self.config.trainer.resume_from_iteration is not None:
                            iteration_number = int(self.config.trainer.resume_from_iteration)
                        else:
                            iteration_number = int(open(local_tracker_file_path, 'r').read())   
                        latest_global_step_remote_path = os.path.join(self.config.trainer.s3_path, f'global_step_{iteration_number}')
                        latest_global_step_local_path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{iteration_number}')
                        # download the dataloader
                        if self.config.trainer.s3_path is not None:
                            aws_copy(os.path.join(latest_global_step_remote_path, 'data.pt'), os.path.join(latest_global_step_local_path, 'data.pt'), recursive=False)
                        
                        # download the unique prompts file
                        unique_prompts_remote_path = os.path.join(latest_global_step_remote_path, 'unique_prompts.json')
                        unique_prompts_local_path = os.path.join(latest_global_step_local_path, 'unique_prompts.json')
                        if self.config.trainer.s3_path is not None and aws_check_file_exists(unique_prompts_remote_path):
                            aws_copy(unique_prompts_remote_path, unique_prompts_local_path, recursive=False)
                    except Exception as e:
                        print(f'Error copying dataloader or unique prompts file')
                        raise e
                    
                    checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
                    if not os.path.isabs(checkpoint_folder):
                        working_dir = os.getcwd()
                        checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
                    global_step_folder = find_latest_ckpt_path(checkpoint_folder, resume_from_iteration=iteration_number)  # None if no latest
            else:
                global_step_folder = None
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        rm_path = os.path.join(global_step_folder, 'rm')

        s3_actor_path = None if self.config.trainer.s3_path is None else os.path.join(
            self.config.trainer.s3_path, f'global_step_{self.global_steps}', 'actor')
        s3_critic_path = None if self.config.trainer.s3_path is None else os.path.join(
            self.config.trainer.s3_path, f'global_step_{self.global_steps}', 'critic')
        s3_rm_path = None if self.config.trainer.s3_path is None else os.path.join(
            self.config.trainer.s3_path, f'global_step_{self.global_steps}', 'rm')

        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, s3_actor_path)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, s3_critic_path)
        # load reward model
        if self.use_rm:
            self.rm_wg.load_checkpoint(rm_path, s3_rm_path)

        # Load unique prompts set
        unique_prompts_path = os.path.join(global_step_folder, 'unique_prompts.json')
        if os.path.exists(unique_prompts_path):
            try:
                with open(unique_prompts_path, 'r') as f:
                    loaded_prompts = json.load(f)
                    self.unique_prompts = set(loaded_prompts)
                print(f"Successfully loaded {len(self.unique_prompts)} unique prompts from checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load unique prompts: {str(e)}")
                self.unique_prompts = set()

        # load dataloader, with improved error handling
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        try:
            import dill
            # Explicitly specify map_location to CPU and use dill for pickle
            self.train_dataloader = load(
                dataloader_local_path, 
                map_location=torch.device('cpu'),
                pickle_module=dill
            )
            print(f"Successfully loaded dataloader from {dataloader_local_path}, current position: {self.train_dataloader.current_position}")
            # Ensure dataloader is properly initialized
            if isinstance(self.train_dataloader.dataset, RLHFDataset):
                self.train_dataloader.dataset.resume_dataset_state()
        except (AttributeError, ModuleNotFoundError, TypeError) as e:
            print(f"Warning: Failed to load dataloader checkpoint: {str(e)}")
            print("Creating a new dataloader from scratch")
            self._create_dataloader()  # Recreate the dataloader
            # Fast-forward the dataloader to approximately the right position
            # This is approximate since we can't know the exact state
            samples_processed = self.global_steps * self.config.data.train_batch_size
            for _ in range(min(samples_processed // self.config.data.train_batch_size, 
                              len(self.train_dataloader))):
                try:
                    self.train_dataloader.get_next_batch()
                except StopIteration:
                    break

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _roll_with_balanced_batch(self, batch, batch_of_keys, timing_raw, acc_lower_bound=0.001, acc_upper_bound=0.999):

        number_of_sampling_distribution = len(batch_of_keys)
        n_samples = self.config.actor_rollout_ref.rollout.n
        number_per_sampling_distribution = n_samples // number_of_sampling_distribution

        flattened_keys = [key for key_group in batch_of_keys for key in key_group]
        gen_batch = batch.select(flattened_keys, deepcopy=True)
        meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'do_sample': True,
            'n_samples': number_per_sampling_distribution
        }

        # construct input_ids, attention_mask, position_ids
        input_ids = []
        attention_mask = []
        position_ids = []
        on_policy_mask = []
        for i in range(len(gen_batch)):
            for k, key_group in enumerate(batch_of_keys):
                input_ids.append(gen_batch.batch[key_group[0]][i, :])
                attention_mask.append(gen_batch.batch[key_group[1]][i, :])
                position_ids.append(gen_batch.batch[key_group[2]][i, :])
                on_policy_mask.extend([k == 0] * number_per_sampling_distribution) # HACK: first key group is on policy
        
        gen_batch_balanced = {
            'input_ids': torch.stack(input_ids, dim=0),
            'attention_mask': torch.stack(attention_mask, dim=0),
            'position_ids': torch.stack(position_ids, dim=0)
        }

        gen_batch_balanced = DataProto.from_dict(tensors=gen_batch_balanced)
        gen_batch_balanced.meta_info = meta_info

        gen_output = self.actor_rollout_wg.generate_sequences(gen_batch_balanced)
        gen_output.rename(['input_ids', 'attention_mask', 'position_ids'], ['prompt_variant_input_ids', 'prompt_variant_attention_mask', 'prompt_variant_position_ids'])
        gen_output.batch['on_policy_mask'] = torch.tensor(on_policy_mask)
        
        # reconstruct the input_ids dropping the prompt variant and restoring orignal input_ids
        full_batch = batch.repeat(n_samples, interleave=True)
        full_batch_input_ids = torch.cat([full_batch.batch['original_input_ids'], gen_output.batch['responses']], dim=1)
        response_length = gen_output.batch['responses'].size(1)
        full_batch_attention_mask = torch.cat([full_batch.batch['original_attention_mask'], gen_output.batch['prompt_variant_attention_mask'][:, -response_length:]], dim=1)
        full_batch_position_ids = compute_position_id_with_mask(full_batch_attention_mask)

        full_batch.batch['input_ids'] = full_batch_input_ids
        full_batch.batch['attention_mask'] = full_batch_attention_mask
        full_batch.batch['position_ids'] = full_batch_position_ids  

        full_batch = full_batch.union(gen_output)

        with _timer('verify', timing_raw):
            reward_tensor, format_reward_tensor = self.reward_fn(full_batch)

        full_batch.batch['verify_scores'] = reward_tensor.sum(-1, keepdim=True)
        full_batch.batch['gt_scores'] = reward_tensor
        full_batch.batch['token_level_scores'] = reward_tensor
        full_batch.batch['format_scores'] = format_reward_tensor
        accuracy = torch.mean(full_batch.batch['verify_scores']).item()
        
        if self.config.data.get('filter_accuracy') or self.config.data.get('filter_truncated'):
            filtered_batch, reroll_batch_unique, reroll_batch_full, filtered_indices = self._apply_filters(full_batch, acc_lower_bound, acc_upper_bound)
        else:
            filtered_batch = full_batch
            reroll_batch_unique = None
            reroll_batch_full = None  

        return filtered_batch, reroll_batch_unique, reroll_batch_full, accuracy

    def _roll_with_prompt_variant(self, batch, keys, timing_raw, old_full_batch=None, acc_lower_bound=0.001, acc_upper_bound=0.999, is_on_policy=True):
        """Reroll the batch with hints"""
        n_samples = self.config.actor_rollout_ref.rollout.n
        batch_size = len(batch) // n_samples
        
        gen_batch = batch.select(keys, deepcopy=True)
        
        gen_batch.meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'do_sample': True,
            'n_samples': n_samples
        }

        gen_batch.rename(keys, ['input_ids', 'attention_mask', 'position_ids'])
        gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        gen_output.rename(['input_ids', 'attention_mask', 'position_ids'], ['prompt_variant_input_ids', 'prompt_variant_attention_mask', 'prompt_variant_position_ids'])

        # reconstruct the input_ids dropping the prompt variant and restoring orignal input_ids
        full_batch = batch.repeat(n_samples, interleave=True)
        full_batch_input_ids = torch.cat([full_batch.batch['original_input_ids'], gen_output.batch['responses']], dim=1)
        response_length = gen_output.batch['responses'].size(1)
        full_batch_attention_mask = torch.cat([full_batch.batch['original_attention_mask'], gen_output.batch['prompt_variant_attention_mask'][:, -response_length:]], dim=1)
        full_batch_position_ids = compute_position_id_with_mask(full_batch_attention_mask)

        # test if the generated input_ids and attention_mask are the same as the reconstructed input_ids and attention_mask when keys are original
        if keys == ['original_input_ids', 'original_attention_mask', 'original_position_ids']:
            assert torch.all(full_batch_input_ids == gen_output.batch['prompt_variant_input_ids']), "input_ids are not the same as the original"
            assert torch.all(full_batch_attention_mask == gen_output.batch['prompt_variant_attention_mask']), "attention_mask are not the same as the original"

        full_batch.batch['input_ids'] = full_batch_input_ids
        full_batch.batch['attention_mask'] = full_batch_attention_mask
        full_batch.batch['position_ids'] = full_batch_position_ids

        # drop prompt variant from full_batch if it exists from previous variant generation
        if 'prompt_variant_input_ids' in full_batch.batch:
            full_batch.batch.pop('prompt_variant_input_ids')
            full_batch.batch.pop('prompt_variant_attention_mask')
            full_batch.batch.pop('prompt_variant_position_ids')

        # drop old responses
        if 'responses' in full_batch.batch:
            full_batch.batch.pop('responses')
            full_batch.batch.pop('prompts')
        
        full_batch = full_batch.union(gen_output)

        # print non variant examples
        print(f"Non variant examples: {self.tokenizer.decode(full_batch.batch['input_ids'][0], skip_special_tokens=True)}")
        # print variant examples
        print(f"Variant examples: {self.tokenizer.decode(full_batch.batch['prompt_variant_input_ids'][0], skip_special_tokens=True)}")

        with _timer('verify', timing_raw):
            reward_tensor, format_reward_tensor = self.reward_fn(full_batch)

        full_batch.batch['verify_scores'] = reward_tensor.sum(-1, keepdim=True)
        full_batch.batch['gt_scores'] = reward_tensor
        full_batch.batch['token_level_scores'] = reward_tensor
        full_batch.batch['format_scores'] = format_reward_tensor
        accuracy = torch.mean(full_batch.batch['verify_scores']).item()
        
        if self.config.data.get('filter_accuracy') or self.config.data.get('filter_truncated'):
            filtered_batch, reroll_batch_unique, reroll_batch_full, filtered_indices = self._apply_filters(full_batch, acc_lower_bound, acc_upper_bound)
        else:
            filtered_batch = full_batch
            reroll_batch_unique = None
            reroll_batch_full = None

        # if config mixin in on, then merge the filtered_batch with the old batch on the filtered_indices, keeping the entries in filtered_batch where verify_scores is 1 and the rest from the old batch
        if self.config.scale_reasoning.get('mixin', False) and old_full_batch is not None and filtered_indices is not None:
            old_full_batch.reorder(filtered_indices)
            for i in range(0, len(filtered_batch), n_samples):
                filtered_batch_slice = filtered_batch.select(deepcopy=True).slice(torch.arange(i, i + n_samples))
                old_batch_verify_scores = old_full_batch.batch['verify_scores'][i:i + n_samples].squeeze(-1)
                keep_slices = torch.arange(i, i + n_samples)[~old_batch_verify_scores.bool()]
                keep_slices = keep_slices.int()
                old_batch_slice_wrong = old_full_batch.select(deepcopy=True).slice(keep_slices)
                wrong_ptr = 0
                for j in range(n_samples):
                    if filtered_batch.batch['verify_scores'][i + j].item() == 1:
                        continue
                    else:
                        if wrong_ptr < len(old_batch_slice_wrong):
                            for key in filtered_batch.batch.keys():
                                filtered_batch.batch[key][i + j] = old_batch_slice_wrong.batch[key][wrong_ptr]
                            for key in filtered_batch.non_tensor_batch.keys():
                                filtered_batch.non_tensor_batch[key][i + j] = old_batch_slice_wrong.non_tensor_batch[key][wrong_ptr]
                            wrong_ptr += 1

        if filtered_batch is not None:
            on_policy_mask = [is_on_policy] * len(filtered_batch)
            filtered_batch.batch['on_policy_mask'] = torch.tensor(on_policy_mask)
            
        return filtered_batch, reroll_batch_unique, reroll_batch_full, accuracy

    def rollout(self):
        """Generate rollouts with filtering, maintaining proper batch sizes for distributed training"""

        batch_size = self.config.data.train_batch_size
        n_samples = self.config.actor_rollout_ref.rollout.n
        world_size = self.actor_rollout_wg.world_size
        target_size = batch_size * n_samples
        metrics = {}
        timing_raw = {}

        valid_samples = []
        train_accuracy = []
        train_accuracy_hint = []
        train_accuracy_cot = []
        hint_count = 0
        cot_count = 0
        
        # First check buffer for any existing valid samples
        if self.train_dataloader.buffer_size() > 0:
            buffer_samples = self.train_dataloader.get_from_buffer(
                count=target_size, 
                dp_size=world_size
            )
            print(f"Using {len(buffer_samples)} samples from buffer")
            valid_samples.append(buffer_samples)
        
        # Keep collecting samples until we have enough or exhaust the dataset
        while DataProto.list_length(valid_samples) < target_size:
            try:
                with _timer('gen', timing_raw):
                    # Get next batch from dataloader

                    current_batch = self.train_dataloader.get_next_batch()

                    # save original input_ids, attention_mask, position_ids
                    current_batch.batch['original_input_ids'] = current_batch.batch['input_ids']
                    current_batch.batch['original_attention_mask'] = current_batch.batch['attention_mask']
                    current_batch.batch['original_position_ids'] = current_batch.batch['position_ids']
                    
                    if self.config.scale_reasoning.roll_with_balanced_batch:
                        # HACK: hardcoded 2 different sampling distributions for now
                        filtered_batch, reroll_batch_unique, reroll_batch_full, accuracy = self._roll_with_balanced_batch(
                            current_batch, 
                            [['original_input_ids', 'original_attention_mask', 'original_position_ids'], ['hint_input_ids', 'hint_attention_mask', 'hint_position_ids']], 
                            timing_raw, 
                            acc_lower_bound=self.config.data.accuracy_lower_bound, 
                            acc_upper_bound=self.config.data.accuracy_upper_bound)
                    else:
                        filtered_batch, reroll_batch_unique, reroll_batch_full, accuracy = self._roll_with_prompt_variant(
                            current_batch, 
                            ['original_input_ids', 'original_attention_mask', 'original_position_ids'], 
                            timing_raw, 
                            acc_lower_bound=self.config.data.accuracy_lower_bound, 
                            acc_upper_bound=self.config.data.accuracy_upper_bound,
                            is_on_policy=True)
                    train_accuracy.append(accuracy)

                # Apply accuracy/truncation filtering if configured
                if self.config.data.get('filter_accuracy') or self.config.data.get('filter_truncated'):
                    if self.config.scale_reasoning.reroll_with_hints and reroll_batch_unique is not None and len(reroll_batch_unique) > 0:
                        hint_filtered_batch, hint_reroll_batch_unique, hint_reroll_batch_full, hint_accuracy = self._roll_with_prompt_variant(
                            reroll_batch_unique, 
                            ['hint_input_ids', 'hint_attention_mask', 'hint_position_ids'], 
                            timing_raw, 
                            reroll_batch_full, 
                            acc_lower_bound=self.config.data.hint_accuracy_lower_bound, 
                            acc_upper_bound=self.config.data.hint_accuracy_upper_bound,
                            is_on_policy=False)
                        if filtered_batch is None:
                            filtered_batch = hint_filtered_batch
                        else:
                            if hint_filtered_batch is not None:
                                filtered_batch = DataProto.concat([filtered_batch, hint_filtered_batch])
                        train_accuracy_hint.append(hint_accuracy)
                        hint_count += len(reroll_batch_unique)
                        if self.config.scale_reasoning.reroll_with_cot and hint_reroll_batch_unique is not None and len(hint_reroll_batch_unique) > 0:
                            cot_filtered_batch, cot_reroll_batch_unique, cot_reroll_batch_full, cot_accuracy = self._roll_with_prompt_variant(
                                hint_reroll_batch_unique, 
                                ['cot_input_ids', 'cot_attention_mask', 'cot_position_ids'], 
                                timing_raw, 
                                hint_reroll_batch_full,
                                is_on_policy=False)
                            if filtered_batch is None:
                                filtered_batch = cot_filtered_batch
                            else:
                                if cot_filtered_batch is not None:
                                    filtered_batch = DataProto.concat([filtered_batch, cot_filtered_batch])
                            train_accuracy_cot.append(cot_accuracy)
                            cot_count += len(hint_reroll_batch_unique)

                if filtered_batch is not None and len(filtered_batch) > 0:
                    valid_samples.append(filtered_batch)
            
                print(f"Collected {DataProto.list_length(valid_samples)}/{target_size} samples")

            except StopIteration:
                # End of dataset reached
                # add current valid samples to buffer
                if len(valid_samples) > 0:  
                    valid_samples_data_proto = DataProto.concat(valid_samples)
                    self.train_dataloader.add_to_buffer(valid_samples_data_proto)
                
                self.train_dataloader.current_position = 0
                raise StopIteration

        # Ensure we have proper batch size for distributed training
        # HACK: to make things backward compatible if any of the valid samples don't have original_input_ids, original_attention_mask, original_position_ids, we need to drop them in all of them
        for valid_sample in valid_samples:
            if 'original_input_ids' not in valid_sample.batch:
                for valid_sample_ in valid_samples:
                    if 'original_input_ids' in valid_sample_.batch:
                        valid_sample_.pop(batch_keys=['original_input_ids', 'original_attention_mask', 'original_position_ids', 'prompt_variant_input_ids', 'prompt_variant_attention_mask', 'prompt_variant_position_ids'])
                break

        valid_samples_data_proto = DataProto.concat(valid_samples)
        valid_size = (len(valid_samples_data_proto) // world_size) * world_size

        # shuffle the batch with n_samples rows as one unit -> to prevent hints and cot rollouts all at the end of the batch
        valid_samples_data_proto = valid_samples_data_proto.shuffle_n_samples(n_samples)
        
        if valid_size > target_size:
            # Store excess samples in buffer
            excess = valid_samples_data_proto[target_size:valid_size]
            self.train_dataloader.add_to_buffer(excess)
            
        valid_samples_data_proto.reorder(torch.arange(target_size))

        # Combine timing metrics
        metrics.update(compute_timing_metrics(valid_samples_data_proto, timing_raw))
        metrics['train/raw_accuracy'] = torch.tensor(train_accuracy).mean().item()
        if len(train_accuracy_hint) > 0:
            metrics['train/hint_accuracy'] = torch.tensor(train_accuracy_hint).mean().item()
            metrics['train/hint_count'] = hint_count
        if len(train_accuracy_cot) > 0:
            metrics['train/cot_accuracy'] = torch.tensor(train_accuracy_cot).mean().item()
            metrics['train/cot_count'] = cot_count

        # apply one uuid so that each consecutive n_samples have the same uid
        uid = np.array(sum([[str(uuid.uuid4())] * n_samples for _ in range(len(valid_samples_data_proto) // n_samples)], []), dtype=object)
        valid_samples_data_proto.non_tensor_batch['uid'] = uid

        if self.config.scale_reasoning.use_ref_prompt:
            # append responses to end of ref_prompt input_ids
            input_ids = torch.cat([valid_samples_data_proto.batch['ref_prompt_input_ids'], valid_samples_data_proto.batch['responses']], dim=-1)
            response_length = valid_samples_data_proto.batch['responses'].size(1)
            attention_mask = torch.cat([valid_samples_data_proto.batch['ref_prompt_attention_mask'], valid_samples_data_proto.batch['attention_mask'][:, -response_length:]], dim=-1)
            position_ids = compute_position_id_with_mask(attention_mask)

            valid_samples_data_proto.batch['ref_prompt_input_ids'] = input_ids
            valid_samples_data_proto.batch['ref_prompt_attention_mask'] = attention_mask
            valid_samples_data_proto.batch['ref_prompt_position_ids'] = position_ids

        return valid_samples_data_proto, metrics

    def _apply_filters(self, batch, acc_lower_bound=0.001, acc_upper_bound=0.999):
        """Apply accuracy and truncation filtering to batch"""
        world_size = self.actor_rollout_wg.world_size
        n_samples = self.config.actor_rollout_ref.rollout.n
        reward_tensor = batch.batch['verify_scores']
        
        # First do accuracy filtering if enabled
        if self.config.data.filter_accuracy:
            reward_matrix = reward_tensor.reshape(-1, n_samples)
            acc_tensor = torch.mean(reward_matrix, dim=-1)
            counts = Counter(acc_tensor.tolist())
            print("Accuracy distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(counts.items())))

            acc_mask = (acc_tensor >= acc_lower_bound) & (
                        acc_tensor <= acc_upper_bound)
            acc_mask_reroll = (acc_tensor <= self.config.scale_reasoning.reroll_threshold)
        else:
            # If accuracy filtering disabled, keep all samples
            acc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)

        # Then do truncation filtering if enabled
        if self.config.data.filter_truncated:
            responses = batch.batch['responses']
            attention_mask = batch.batch['attention_mask']
            response_mask = attention_mask[:, -responses.size(1):]

            # Calculate response lengths
            response_lengths = response_mask.sum(-1)  # (batch_size,)
            response_lengths = response_lengths.reshape(-1, n_samples)  # (num_prompts, n_samples)

            # Get max possible length from config
            max_len = self.config.data.max_response_length

            # Check if any response in the group hits max length (indicating possible truncation)
            has_truncated = (response_lengths >= max_len).any(dim=-1)

            # Print distribution of truncated vs non-truncated
            truncated_counts = Counter(has_truncated.tolist())
            print("Truncation distribution:", 
                f"Truncated: {truncated_counts[True] if True in truncated_counts else 0}, "
                f"Non-truncated: {truncated_counts[False] if False in truncated_counts else 0}")
            
            # Keep only prompts where no response was truncated
            trunc_mask = ~has_truncated
        else:
            # If truncation filtering disabled, keep all samples
            trunc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)

        # Combine both masks
        combined_mask = acc_mask & trunc_mask

        # Expand mask to cover all samples for each prompt
        final_mask = combined_mask.repeat_interleave(n_samples)
        all_reroll_mask = acc_mask_reroll.repeat_interleave(n_samples)

        # Apply the mask to the batch using reorder instead of slice
        filtered_indices = torch.nonzero(final_mask).squeeze(-1)
        all_reroll_indices = torch.nonzero(all_reroll_mask).squeeze(-1)
        
        # If no samples pass the filter, return the original batch

        original_batch_len = len(batch)

        if len(filtered_indices) == 0:
            print("Warning: No samples passed the filtering criteria. Returning original batch.")
            
            # reduce from n_samples to 1 per prompt
            batch.reorder(torch.arange(0, len(batch), n_samples))
            return None, batch, None, None
        else:
            reroll_batch = batch.select(deepcopy=True)
            batch.reorder(filtered_indices)

            # NOTE: we need to have atleast world_size samples to process
            if (len(all_reroll_indices) // n_samples) >= world_size:
                reroll_batch.reorder(all_reroll_indices)
                # reduce from n_samples to 1 per prompt and truncate to world_size
                reroll_batch_unique = reroll_batch.select(deepcopy=True)
                reroll_batch_unique.reorder(torch.arange(0, (((len(all_reroll_indices) // n_samples) // world_size) * world_size) * n_samples, n_samples))
            else:
                reroll_batch = None
                reroll_batch_unique = None
            
        print(f"Filtered batch size: {len(batch)} (from original size: {original_batch_len})")
        return batch, reroll_batch_unique, reroll_batch, filtered_indices

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        global_metrics = {
            'global_metrics/total_train_samples': 0,
            'global_metrics/total_train_tokens': 0,
            'global_metrics/total_train_correct_tokens': 0,
            'global_metrics/total_train_incorrect_tokens': 0,
            'global_metrics/total_train_correct_responses': 0,
            'global_metrics/total_train_incorrect_responses': 0,
            'global_metrics/unique_prompts': 0,  # Track unique prompts
        }

        # Create directory for step data
        steps_data_dir = os.path.join(self.config.trainer.default_local_dir, 'training_data_per_step')
        os.makedirs(steps_data_dir, exist_ok=True)
        s3_steps_data_dir = os.path.join(self.config.trainer.s3_path, 'training_data_per_step') if self.config.trainer.s3_path else None

        for epoch in range(self.config.trainer.total_epochs):
            self.train_dataloader.start_new_epoch()
            while True:
                metrics = {}
                timing_raw = {}

                with _timer('step', timing_raw):
                    # generate a batch

                    try:
                        gen_batch_output, gen_batch_metrics = self.rollout()
                    except StopIteration:
                        print("Reached end of dataset, ending training.")
                        break

                    metrics.update(gen_batch_metrics)

                    if self.config.algorithm.adv_estimator == 'remax':
                        with _timer('gen_max', timing_raw):
                            gen_batch = gen_batch_output.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor, format_reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output


                    batch = gen_batch_output

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.scale_reasoning.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  config=self.config)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                
                # Save step data to jsonl
                with _timer('save_step_data', timing_raw):
                    prompt_texts = save_step_data(
                        batch=batch, 
                        tokenizer=self.tokenizer,
                        step=self.global_steps,
                        local_dir=steps_data_dir,
                        s3_dir=s3_steps_data_dir
                    )
                    
                    # Update unique prompts set and counter
                    self.unique_prompts.update(prompt_texts)
                    global_metrics['global_metrics/unique_prompts'] = len(self.unique_prompts)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # update global metrics
                global_metrics = compute_global_metrics(batch=batch, global_metrics=global_metrics)
                metrics.update(global_metrics)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return