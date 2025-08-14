from unittest.mock import Mock, patch, MagicMock
import torch
import pandas as pd
from verl.protocol import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, BufferedDataLoader
from omegaconf import OmegaConf
import os

FSDP_CONFIG = 'ppo_trainer.yaml'
MEGATRON_CONFIG = 'ppo_megatron_trainer.yaml'

def load_ppo_config(fsdp_or_megatron='fsdp'):
    """Load the PPO FSDP configuration from the yaml file"""

    # Get the absolute path to the config file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = FSDP_CONFIG if fsdp_or_megatron == 'fsdp' else MEGATRON_CONFIG
    config_path = os.path.join(current_dir, f'../verl/trainer/config/{config_file}')
    
    # Load and return the config
    return OmegaConf.load(config_path)

def override_config(base_config, override_dict):
    """
    Override values in a configuration with values from a dictionary.
    Handles nested dictionaries properly.
    
    Args:
        base_config: The base OmegaConf configuration
        override_dict: Dictionary containing override values
        
    Returns:
        Updated OmegaConf configuration
    """
    # Create an OmegaConf object from the override dictionary
    override_conf = OmegaConf.create(override_dict)
    
    # Merge the base config with the override config
    # The merge will recursively update nested values
    merged_config = OmegaConf.merge(base_config, override_conf)
    
    return merged_config

def create_mock_dataset():
    """Create a mock RLHFDataset with basic test data"""
    # Create a mock DataFrame with proper data
    mock_df = pd.DataFrame({
        'prompt': ['Test prompt 1', 'Test prompt 2'] * 5,
        'response': ['Test response 1', 'Test response 2'] * 5,
    })

    # Mock dataset
    mock_dataset = MagicMock(spec=RLHFDataset)
    type(mock_dataset).__len__ = Mock(return_value=1000)
    mock_dataset.dataframe = mock_df
    
    # Mock dataset methods
    mock_dataset._read_files_and_tokenize = Mock()
    mock_dataset.get_item_by_idx = Mock(return_value={
        'input_ids': torch.randint(0, 1000, (50,)),
        'attention_mask': torch.ones(50),
        'position_ids': torch.arange(50),
    })

    return mock_dataset

def create_mock_dataloader(sample_batch, mock_dataset):
    """Create a mock BufferedDataLoader with the given sample batch"""
    dataloader = Mock(spec=BufferedDataLoader)
    dataloader.buffer = []
    dataloader.buffer_size = lambda: len(dataloader.buffer)
    
    def get_next_batch():
        # Always include the base tensors in the batch
        batch = sample_batch.batch.copy()
        return DataProto.from_single_dict(batch)
        
    dataloader.get_next_batch = Mock(side_effect=get_next_batch)
    dataloader.dataset = mock_dataset
    return dataloader

def create_sample_batch(batch_size, input_len=50):
    """Create a sample batch with given batch size and input length"""
    base_tensors = {
        'input_ids': torch.randint(0, 1000, (batch_size, input_len)),
        'attention_mask': torch.ones(batch_size, input_len),
        'position_ids': torch.arange(input_len).unsqueeze(0).expand(batch_size, -1),
    }
    return DataProto.from_single_dict(base_tensors)

def create_sample_buffer_batch(batch_size=4, input_len=50, response_len=20):
    """Create a sample batch with buffer-specific fields"""
    reward_tensor = torch.zeros(batch_size, response_len)
    verify_scores = reward_tensor.clone()
    verify_scores[:, -1] = 1.0
    verify_scores = verify_scores.sum(-1, keepdim=True)
    
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, input_len + response_len)),
        'attention_mask': torch.ones(batch_size, input_len + response_len),
        'position_ids': torch.arange(input_len + response_len).unsqueeze(0).expand(batch_size, -1),
        'responses': torch.randint(0, 1000, (batch_size, response_len)),
        'verify_scores': verify_scores,
        'gt_scores': torch.rand(batch_size, response_len),
        'prompts': torch.randint(0, 1000, (batch_size, input_len)),
        'token_level_scores': reward_tensor,
    }
    
    return DataProto.from_single_dict(batch)

def setup_generation_mock(sample_response_batch, mock_actor_rollout_wg, mock_reward_fn, n_samples, batch_size, seq_len=20):
    """Setup mock for generate_sequences with configurable parameters"""
    total_size = batch_size * n_samples
    
    response_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    response_attention = torch.ones(batch_size, seq_len)
    response_position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    response_input_ids = response_input_ids.repeat_interleave(n_samples, dim=0)
    response_attention = response_attention.repeat_interleave(n_samples, dim=0)
    response_position_ids = response_position_ids.repeat_interleave(n_samples, dim=0)

    idx = sample_response_batch.batch['input_ids'].repeat_interleave(n_samples, dim=0)
    attention_mask = sample_response_batch.batch['attention_mask'].repeat_interleave(n_samples, dim=0)
    position_ids = sample_response_batch.batch['position_ids'].repeat_interleave(n_samples, dim=0)

    gen_output_dict = {
        'prompts': idx,
        'responses': response_input_ids,
        'input_ids': torch.cat([idx, response_input_ids], dim=-1),
        'attention_mask': torch.cat([attention_mask, response_attention], dim=-1),
        'position_ids': torch.cat([position_ids, response_position_ids], dim=-1),
    }
    
    gen_output = DataProto.from_single_dict(gen_output_dict)
    
    mock_actor_rollout_wg.generate_sequences = Mock(return_value=gen_output)
    reward_tensor = torch.zeros(total_size, seq_len)
    reward_tensor[:, -1] = 1.0
    mock_reward_fn.return_value = reward_tensor
    
    return gen_output
