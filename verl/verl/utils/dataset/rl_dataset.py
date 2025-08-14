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

from omegaconf import ListConfig
import os
from typing import List, Union
import copy
import pandas as pd
import polars as pl
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.dataset.templates import qwen_math_chat_template, llama_math_chat_template

def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 apply_chat_template=True,
                 use_hint=False,
                 use_cot=False,
                 use_ref_prompt=False,
                 custom_chat_template=None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.hint_key = 'hint'
        self.cot_key = 'cot'
        self.ref_prompt_key = 'ref_prompt'
        self.use_hint = use_hint
        self.use_cot = use_cot
        self.use_ref_prompt = use_ref_prompt
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.apply_chat_template = apply_chat_template
        if custom_chat_template is not None:
            if custom_chat_template == 'qwen_math_chat_template':
                self.custom_chat_template = qwen_math_chat_template
            elif custom_chat_template == 'llama_math_chat_template':
                self.custom_chat_template = llama_math_chat_template
            else:
                raise ValueError(f'Invalid custom chat template: {custom_chat_template}')
        else:
            self.custom_chat_template = None
        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_local_path_from_hdfs
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pl.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pl.concat(dataframes)
        self.dataframe = self.dataframe.to_pandas()

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        hint_key = self.hint_key
        cot_key = self.cot_key

        if self.custom_chat_template:

            self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer(self.custom_chat_template.format(prompt=doc[prompt_key][-1]["content"]))["input_ids"]) <= self.max_prompt_length,
                                                             axis=1)]

            if self.use_ref_prompt and self.ref_prompt_key in self.dataframe.columns:
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                    tokenizer(self.custom_chat_template.format(prompt=doc[self.ref_prompt_key][-1]["content"]))["input_ids"]) <= self.max_prompt_length,
                                                             axis=1)]
            
            if self.use_hint and hint_key in self.dataframe.columns:  
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                    tokenizer(self.custom_chat_template.format(prompt=doc[hint_key][-1]["content"]))["input_ids"]) <= self.max_prompt_length,
                                                               axis=1)]
            if self.use_cot and cot_key in self.dataframe.columns:
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                    tokenizer(self.custom_chat_template.format(prompt=doc[cot_key][-1]["content"]))["input_ids"]) <= self.max_prompt_length,
                                                             axis=1)]
                
        elif self.apply_chat_template:
            self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                             axis=1)]

            if self.use_ref_prompt and self.ref_prompt_key in self.dataframe.columns:
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                    tokenizer.apply_chat_template(doc[self.ref_prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                             axis=1)]

            if self.use_hint and hint_key in self.dataframe.columns:
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                    tokenizer.apply_chat_template(doc[hint_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                             axis=1)]
            if self.use_cot and cot_key in self.dataframe.columns:
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                    tokenizer.apply_chat_template(doc[cot_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                             axis=1)]
        else:
            self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer(doc[prompt_key][0]["content"])["input_ids"]) <= self.max_prompt_length,
                                                             axis=1)]

            if self.use_ref_prompt and self.ref_prompt_key in self.dataframe.columns:
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                    tokenizer(doc[self.ref_prompt_key][0]["content"])["input_ids"]) <= self.max_prompt_length,
                                                             axis=1)]
            
            if self.use_hint and hint_key in self.dataframe.columns:  
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                    tokenizer(doc[hint_key][0]["content"])["input_ids"]) <= self.max_prompt_length,
                                                               axis=1)]
            if self.use_cot and cot_key in self.dataframe.columns:
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                    tokenizer(doc[cot_key][0]["content"])["input_ids"]) <= self.max_prompt_length,
                                                             axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # check if use_hint is an attribute
        if not hasattr(self, 'use_hint'):
            self.use_hint = False
            self.hint_key = None
        
        if not hasattr(self, 'use_cot'):
            self.use_cot = False
            self.cot_key = None

        if not hasattr(self, 'use_ref_prompt'):
            self.use_ref_prompt = False
            self.ref_prompt_key = None
        
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        if self.custom_chat_template:
            prompt_with_chat_template = self.custom_chat_template.format(prompt=chat[-1]["content"])
        elif self.apply_chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        else:
            prompt_with_chat_template = chat[0]["content"]

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        if self.use_ref_prompt and self.ref_prompt_key in row_dict:
            ref_prompt_chat = row_dict.pop(self.ref_prompt_key)
            if self.custom_chat_template:
                ref_prompt_prompt_with_chat_template = self.custom_chat_template.format(prompt=ref_prompt_chat[-1]["content"])
            elif self.apply_chat_template:
                ref_prompt_prompt_with_chat_template = self.tokenizer.apply_chat_template(ref_prompt_chat, add_generation_prompt=True, tokenize=False)
            else:
                ref_prompt_prompt_with_chat_template = ref_prompt_chat[0]["content"]
            ref_prompt_input_ids, ref_prompt_attention_mask = verl_F.tokenize_and_postprocess_data(prompt=ref_prompt_prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)
            ref_prompt_position_ids = compute_position_id_with_mask(ref_prompt_attention_mask)
            row_dict['ref_prompt_input_ids'] = ref_prompt_input_ids[0]
            row_dict['ref_prompt_attention_mask'] = ref_prompt_attention_mask[0]
            row_dict['ref_prompt_position_ids'] = ref_prompt_position_ids[0]

        if self.use_hint and self.hint_key in row_dict:
            hint_chat = row_dict.pop(self.hint_key)
            if self.custom_chat_template:
                hint_prompt_with_chat_template = self.custom_chat_template.format(prompt=hint_chat[-1]["content"])
            elif self.apply_chat_template:
                hint_prompt_with_chat_template = self.tokenizer.apply_chat_template(hint_chat, add_generation_prompt=True, tokenize=False)
            else:
                hint_prompt_with_chat_template = hint_chat[0]["content"]
            hint_input_ids, hint_attention_mask = verl_F.tokenize_and_postprocess_data(prompt=hint_prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)
            hint_position_ids = compute_position_id_with_mask(hint_attention_mask)
            row_dict['hint_input_ids'] = hint_input_ids[0]
            row_dict['hint_attention_mask'] = hint_attention_mask[0]
            row_dict['hint_position_ids'] = hint_position_ids[0]

        if self.use_cot and self.cot_key in row_dict:
            cot_chat = row_dict.pop(self.cot_key)
            if self.custom_chat_template:
                cot_prompt_with_chat_template = self.custom_chat_template.format(prompt=cot_chat[-1]["content"])
            elif self.apply_chat_template:
                cot_prompt_with_chat_template = self.tokenizer.apply_chat_template(cot_chat, add_generation_prompt=True, tokenize=False)
            else:
                cot_prompt_with_chat_template = cot_chat[0]["content"]

            cot_input_ids, cot_attention_mask = verl_F.tokenize_and_postprocess_data(prompt=cot_prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)
            cot_position_ids = compute_position_id_with_mask(cot_attention_mask)
            row_dict['cot_input_ids'] = cot_input_ids[0]
            row_dict['cot_attention_mask'] = cot_attention_mask[0]
            row_dict['cot_position_ids'] = cot_position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()


class BufferedDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 generator=None, **kwargs):
        # Initialize parent DataLoader with all possible parameters
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            **kwargs
        )
        self.buffer = None
        self.dataloader_iter = None
        self.current_position = 0  # Track current position in dataset

    def start_new_epoch(self):
        """Reset iterator for new epoch"""
        self.dataloader_iter = super().__iter__()
        # Skip to the current position if needed
        for _ in range(self.current_position):
            try:
                next(self.dataloader_iter)
            except StopIteration:
                # If we reach the end, reset position and iterator
                self.current_position = 0
                self.dataloader_iter = super().__iter__()
                break

    def get_next_batch(self):
        """Get next batch from the iterator"""
        try:
            batch = next(self.dataloader_iter)
            self.current_position += 1
            return DataProto.from_single_dict(batch)
        except StopIteration:
            raise StopIteration

    def add_to_buffer(self, samples):
        """Add samples to buffer"""
        if self.buffer is None:
            self.buffer = samples
        else:
            self.buffer = DataProto.concat([self.buffer, samples])

    def get_from_buffer(self, count, dp_size):
        """Get samples from buffer"""
        if count > self.buffer_size():
            count = (self.buffer_size() // dp_size) * dp_size
        samples = self.buffer.slice(range(0, count))
        self.buffer = self.buffer.slice(range(count, self.buffer_size()))
        return samples

    def buffer_size(self):
        """Get current buffer size"""
        if self.buffer is None:
            return 0
        return len(self.buffer)

    def __iter__(self):
        """Override iterator to use our custom iteration"""
        self.start_new_epoch()
        return self
    
    def __next__(self):
        """Implementation for iterator protocol"""
        return self.get_next_batch()
    
    def __getstate__(self):
        """Custom state getter for pickling support"""
        state = self.__dict__.copy()
        # Remove the dataloader iterator since it can't be pickled
        if 'dataloader_iter' in state:
            del state['dataloader_iter']
        # Keep current_position to resume from same point
        return state
    
    def __setstate__(self, state):
        """Custom state setter to restore after unpickling"""
        self.__dict__.update(state)
        # Iterator will be recreated when needed with start_new_epoch()
        self.dataloader_iter = None
    