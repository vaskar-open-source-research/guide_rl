# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio

import torch

from verl.protocol import DataProto
from graders.verl_grader.verl_grader import VERLGrader
from verl.workers.reward_manager.format_grader import verify_luffy_math_format, verify_math_format, verify_code_format, verify_qwen_simple_format, verify_llama_simple_format
import os


class RiftRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, response_format="default", do_verify_format=True) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.response_format = response_format
        self.grader_cls = VERLGrader()
        self.do_verify_format = do_verify_format
        

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        correct_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # batch size x 1
        format_reward_tensor = torch.zeros(data.batch['responses'].shape[0], dtype=torch.float32)
        format_reward_tensor = format_reward_tensor.unsqueeze(-1)

        already_print_data_sources = {}

        # batched scoring
        response_ids = data.batch['responses']
        response_length = response_ids.shape[-1]
        valid_response_length = data.batch['attention_mask'][:, -response_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        abilities = data.non_tensor_batch['ability']
        
        assert len(sequences_str) == len(ground_truth) == len(data_sources)
        try:
            if self.do_verify_format:
                sequences_str_verified = []
                for i, (seq, ability) in enumerate(zip(sequences_str, abilities)):
                    if ability == "math" or ability == "stem" or ability == "llm":
                        if self.response_format == "luffy_format":
                            if verify_luffy_math_format(seq):
                                sequences_str_verified.append(seq)
                                format_reward_tensor[i] = 1.0
                            else:
                                sequences_str_verified.append("")
                                format_reward_tensor[i] = 0.0
                        elif self.response_format == "qwen_format":
                            if verify_qwen_simple_format(seq):
                                sequences_str_verified.append(seq)
                                format_reward_tensor[i] = 1.0
                            else:
                                sequences_str_verified.append("")
                                format_reward_tensor[i] = 0.0
                        elif self.response_format == "llama_format":
                            if verify_llama_simple_format(seq):
                                sequences_str_verified.append(seq)
                                format_reward_tensor[i] = 1.0
                            else:
                                sequences_str_verified.append("")
                                format_reward_tensor[i] = 0.0
                        else:
                            if verify_math_format(seq):
                                sequences_str_verified.append(seq)
                                format_reward_tensor[i] = 1.0
                            else:
                                sequences_str_verified.append("")
                                format_reward_tensor[i] = 0.0
                    elif ability == "code":
                        if verify_code_format(seq):
                            sequences_str_verified.append(seq)
                            format_reward_tensor[i] = 1.0
                        else:
                            sequences_str_verified.append("")
                            format_reward_tensor[i] = 0.0
                    else:
                        sequences_str_verified.append(seq)
                        format_reward_tensor[i] = 1.0
            else:
                sequences_str_verified = sequences_str
            
            timeouts = []
            for ability in abilities:
                if ability == "math" or ability == "stem":
                    timeouts.append(5)
                elif ability == "code":
                    timeouts.append(5)
                elif ability == "llm":
                    timeouts.append(30)
                else:
                    raise ValueError(f"Ability {ability} not supported")
        
            scores = self.grader_cls.grade_batch(sequences_str_verified, ground_truth, abilities, timeouts=timeouts, num_workers=32)

        except asyncio.TimeoutError as e:
            print('Global timeout in reward computing! Setting all as 0.')
            scores = [0. for _ in range(len(sequences_str))]
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [0. for _ in range(len(sequences_str))]

        for i in range(len(data)):
            data_source = data_sources[i]
            correct_reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str[i])  # Print only the specific sequence, not the entire list
        
        # HACK: remove all the of core dump files
        print("Removing all core dump files")
        exec_cmd = "rm -rf /core.ray\:\:main_task.*"
        os.system(exec_cmd)
        return correct_reward_tensor, format_reward_tensor

if __name__ == "__main__":
    # test that reward manager works
    from transformers import AutoTokenizer
    from verl.protocol import DataProto
    import gc

    model_output = """
    <think> blah blah </think> \
    <answer> \\boxed{1} </answer>
    """.strip()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    model_output_input_ids = tokenizer(model_output, return_tensors="pt")["input_ids"]
    reward_manager = RiftRewardManager(tokenizer=tokenizer, num_examine=10)
    data_proto = DataProto.from_dict(tensors={
            "prompts": model_output_input_ids.repeat(2, 1),
            "responses": model_output_input_ids.repeat(2, 1),
            "attention_mask": torch.ones_like(model_output_input_ids).repeat(2, 1)
    }, non_tensors={
        "data_source": ["open_source"] * 2,
        "ability": ["all_math"] * 2,
        "reward_model": [{"ground_truth": "1"}, {"ground_truth": "2"}]
    })

    result = reward_manager(data_proto)
    print(result)
    