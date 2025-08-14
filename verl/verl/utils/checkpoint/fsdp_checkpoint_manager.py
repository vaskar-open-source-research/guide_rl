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

import ray
import os
import time
import warnings

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig

from verl.utils.fs import copy_local_path_from_hdfs, is_non_local
from verl.utils.aws_utils import aws_copy, aws_check_file_exists

from transformers import PreTrainedTokenizer

from .checkpoint_manager import BaseCheckpointManager
from .rlxf_checkpoint_utils import save_hf_checkpoint


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save 
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(self, model: FSDP, optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        super().__init__(model, optimizer, lr_scheduler, tokenizer)

    def load_checkpoint(self, path=None, del_local_after_load=True, s3_path=None, global_step=0, *args, **kwargs):
        if path is None:
            return
        
        # only need to copy from s3 if worker is rank 0 for this machine
        if self.rank % 8 == 0:
            aws_copy(s3_path, path, recursive=True, exclude="*huggingface*")
        
        self._file_based_barrier(path, message=f"load_checkpoint", s3_path=None, target_rank=(self.rank // 8) * 8)
        # torch.distributed.barrier()

        # every rank download its own checkpoint
        remote_model_path = os.path.join(path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt')
        remote_optim_path = os.path.join(path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt')
        remote_extra_state_path = os.path.join(path, f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt')
        print(
            f'[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}'
        )
        local_model_path = copy_local_path_from_hdfs(remote_model_path)
        local_optim_path = copy_local_path_from_hdfs(remote_optim_path)
        local_extra_state_path = copy_local_path_from_hdfs(remote_extra_state_path)

        try:
            model_state_dict = torch.load(local_model_path)
            optimizer_state_dict = torch.load(local_optim_path)
            extra_state_dict = torch.load(local_extra_state_path)
        except Exception as e:
            print(f"[rank-{self.rank}]: Failed to load model state dict from {local_model_path}, make sure world_size={self.world_size}, nnodes={self.world_size // 8} matches the checkpoint")
            raise e
            
        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(
                    f'[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored'
                )

        lr_scheduler_state_dict = extra_state_dict['lr_scheduler']

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
        # recover random state
        if 'rng' in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict['rng'])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
    
    def _file_based_barrier(self, local_path, message="barrier", s3_path=None, target_rank=0):
        """
        Custom barrier implementation using S3 file system signals to avoid any torch.distributed timeout issues.
        
        Args:
            local_path (str): Directory where local signal files will be created
            message (str, optional): A message identifier for this barrier. Defaults to "barrier"
            s3_path (str, optional): S3 path where signals will be stored. If None, only use local filesystem.
        """
        barrier_dir = os.path.join(local_path, '.barrier_signals')
        os.makedirs(barrier_dir, exist_ok=True)
        local_signal_file = os.path.join(barrier_dir, f"{message}_done")
        
        # If s3_path is provided, use it for coordination across nodes
        if s3_path:
            s3_barrier_path = os.path.join(s3_path, '.barrier_signals')
            s3_signal_file = os.path.join(s3_barrier_path, f"{message}_done")
        
            if self.rank == target_rank:
                # Create local signal file
                with open(local_signal_file, 'w') as f:
                    f.write(f"Rank {target_rank} completed {message} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Upload signal file to S3
                try:
                    print(f"[rank-{self.rank}]: Uploading signal file to S3: {s3_signal_file}")
                    aws_copy(local_signal_file, s3_signal_file)
                    print(f"[rank-{self.rank}]: Created signal file for {message} at {s3_signal_file}")
                except Exception as e:
                    print(f"[rank-{self.rank}]: Failed to upload signal file to S3: {e}")
            else:
                # Non-zero ranks poll for the signal file on S3
                print(f"[rank-{self.rank}]: Waiting for S3 signal file: {s3_signal_file}")
                max_wait_time = 3600  # 1 hour max wait in seconds
                poll_interval = 10  # Check every 10 seconds
                waited_time = 0
                
                while waited_time <= max_wait_time:
                    try:
                        # Try to download the signal file from S3
                        if aws_check_file_exists(s3_signal_file):
                            aws_copy(s3_signal_file, local_signal_file)
                            if os.path.exists(local_signal_file):
                                print(f"[rank-{self.rank}]: Received signal for {message} after {waited_time} seconds")
                                break
                    except Exception:
                        # File might not exist yet, continue polling
                        pass
                    
                    time.sleep(poll_interval)
                    waited_time += poll_interval
                    if waited_time % 60 == 0:  # Log every minute
                        print(f"[rank-{self.rank}]: Still waiting for {message} signal, {waited_time} seconds elapsed")
                
                if waited_time > max_wait_time:
                    print(f"[rank-{self.rank}]: WARNING - Waited over {max_wait_time/3600} hour for {message}, proceeding anyway")
        else:
            # Fall back to local filesystem barrier if no S3 path provided
            if self.rank == target_rank:
                # Rank 0 does its work and then signals completion by creating a file
                with open(local_signal_file, 'w') as f:
                    f.write(f"Rank 0 completed {message} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"[rank-{self.rank}]: Created signal file for {message} at {local_signal_file}")
            else:
                # Non-zero ranks poll for the signal file
                print(f"[rank-{self.rank}]: Waiting for signal file: {local_signal_file}")
                max_wait_time = 3600  # 1 hour max wait in seconds
                poll_interval = 10  # Check every 10 seconds
                waited_time = 0
                
                while not os.path.exists(local_signal_file):
                    time.sleep(poll_interval)
                    waited_time += poll_interval
                    if waited_time % 60 == 0:  # Log every minute
                        print(f"[rank-{self.rank}]: Still waiting for {message} signal, {waited_time} seconds elapsed")
                    if waited_time > max_wait_time:
                        print(f"[rank-{self.rank}]: WARNING - Waited over {max_wait_time/3600} hour for {message}, proceeding anyway")
                        break
                
                if os.path.exists(local_signal_file):
                    print(f"[rank-{self.rank}]: Received signal for {message} after {waited_time} seconds")

    def save_huggingface_checkpoint(self, local_path: str, s3_path: str, *args, **kwargs):
        
        local_path = self.local_mkdir(local_path)
        
        print(f"[rank-{self.rank}]: Gathering full model weights")
        try:
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
                full_state_dict = self.model.state_dict()
                print(f"[rank-{self.rank}]: Finished gathering full model weights.")

                if self.rank == 0:
                    hf_local_path = os.path.join(local_path, 'huggingface')
                    os.makedirs(hf_local_path, exist_ok=True)
                    
                    print(f"[rank-{self.rank}]: Saving model to {hf_local_path}")
                    self.model.save_pretrained(hf_local_path, state_dict=full_state_dict)
                    # save_hf_checkpoint(full_state_dict, hf_local_path, is_main_process=True)
                    self.tokenizer.save_pretrained(hf_local_path)
                    print(f"[rank-{self.rank}]: Saved model to {hf_local_path}")

                    # HACK: transformers saving has weird behavior, temporary workaround
                    # delete the model.safetensors file from the output as it corrupts .from_pretrained() loading
                    file_path_to_remove = f"{hf_local_path}/model.safetensors"
                    if os.path.exists(file_path_to_remove):
                        # Check if there are more than one .safetensors files in the directory
                        safetensors_files = [
                            file for file in os.listdir(hf_local_path) if file.endswith(".safetensors")
                        ]
                        if len(safetensors_files) > 1:
                            os.remove(file_path_to_remove)

        except Exception as e:
            print(f'[rank-{self.rank}]: Error gathering full model weights: {e}')

        # Replace the torch.distributed.barrier() with our file-based barrier to avoid timeout issues
        self._file_based_barrier(local_path, message="huggingface_checkpoint_save", s3_path=s3_path)

        import gc
        gc.collect()
        torch.cuda.empty_cache()


    def save_checkpoint(self, local_path: str, global_step: int, remove_previous_ckpt=False, *args, **kwargs):
        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        # TODO: shall we remove previous ckpt every save?
        if remove_previous_ckpt:
            self.remove_previous_save_local_path()
        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.model.state_dict()
                if self.optimizer is not None:
                    optimizer_state_dict = self.optimizer.state_dict()
                else:
                    optimizer_state_dict = None
                if self.lr_scheduler is not None:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict()
                else:
                    lr_scheduler_state_dict = None

                extra_state_dict = {
                    'lr_scheduler': lr_scheduler_state_dict,
                    'rng': self.get_rng_state(),
                }
                model_path = os.path.join(local_path, f'model_world_size_{self.world_size}_rank_{self.rank}.pt')
                optim_path = os.path.join(local_path, f'optim_world_size_{self.world_size}_rank_{self.rank}.pt')
                extra_path = os.path.join(local_path, f'extra_state_world_size_{self.world_size}_rank_{self.rank}.pt')

                print(f'[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}')
                print(f'[rank-{self.rank}]: Saving optimizer to {os.path.abspath(optim_path)}')
                print(f'[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}')
                torch.save(model_state_dict, model_path)
                torch.save(optimizer_state_dict, optim_path)
                torch.save(extra_state_dict, extra_path)

        # set timeout to 30 minutes
        torch.distributed.barrier()

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        self.previous_save_local_path = local_path
