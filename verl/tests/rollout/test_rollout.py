import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
import pandas as pd
from collections import defaultdict, Counter

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.utils.dataset.rl_dataset import BufferedDataLoader, RLHFDataset
from ..test_utils import (
    load_ppo_config, create_mock_dataset, create_mock_dataloader,
    create_sample_batch, create_sample_buffer_batch, setup_generation_mock, override_config
)

class TestRollout(unittest.TestCase):
    def setUp(self):
        # Load config from yaml file
        self.config = load_ppo_config()

        override_configs = {
            'actor_rollout_ref': {'rollout' : {'n' : 4}},
            'data' : {'train_batch_size': 8}
        }

        self.config = override_config(self.config, override_configs)

        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.eos_token_id = 2
        self.tokenizer.pad_token_id = 0

        # Create mock dataset
        self.mock_dataset = create_mock_dataset()

        # Create patches
        self.patches = [
            # Mock dataset initialization to return our prepared mock_dataset
            patch('verl.utils.dataset.rl_dataset.RLHFDataset', return_value=self.mock_dataset),
            # Mock file system operations
            patch('verl.utils.fs.copy_local_path_from_hdfs', return_value='/mock/local/path'),
            # Mock pandas read_parquet to return our mock DataFrame
            patch('pandas.read_parquet', return_value=self.mock_dataset.dataframe),
            # Mock tokenizer loading
            patch('verl.utils.hf_tokenizer', return_value=self.tokenizer),
            # Patch _create_dataloader to use our mock
            patch.object(RayPPOTrainer, '_create_dataloader', return_value=None)
        ]

        # Start all patches
        for p in self.patches:
            p.start()

        # Mock worker classes
        mock_actor_worker = MagicMock()
        mock_critic_worker = MagicMock()
        mock_reward_model_worker = MagicMock()

        # Mock resource pool manager
        resource_pool_manager = Mock()
        resource_pool = Mock()
        resource_pool.create_worker_group.return_value = Mock()
        resource_pool_manager.get_resource_pool.return_value = resource_pool
        resource_pool_manager.create_resource_pool = Mock()

        # Create role worker mapping
        role_worker_mapping = {
            Role.ActorRollout: mock_actor_worker,
            Role.Critic: mock_critic_worker,
            Role.RewardModel: mock_reward_model_worker,
            Role.RefPolicy: mock_actor_worker
        }

        # Mock reward functions
        self.reward_fn = Mock()
        reward_tensor = torch.zeros(8, 10)
        reward_tensor[:, -1] = 1.0
        self.reward_fn.return_value = reward_tensor
        self.val_reward_fn = Mock()

        # Initialize trainer with mocks
        self.trainer = RayPPOTrainer(
            config=self.config,
            tokenizer=self.tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=Mock(),
            reward_fn=self.reward_fn,
            val_reward_fn=self.val_reward_fn
        )

        # Mock worker groups
        self.actor_rollout_wg = Mock()
        self.actor_rollout_wg.world_size = 2
        self.critic_wg = Mock()
        self.rm_wg = Mock()
        
        # Attach worker groups to trainer
        self.trainer.actor_rollout_wg = self.actor_rollout_wg
        self.trainer.critic_wg = self.critic_wg
        self.trainer.rm_wg = self.rm_wg

        # Create sample batch for testing
        self.sample_batch = create_sample_batch(self.config.data.train_batch_size)
        
        # Set the train_dataloader directly after initialization
        self.trainer.train_dataloader = create_mock_dataloader(self.sample_batch, self.mock_dataset)

        # Mock KL controller
        self.trainer.kl_ctrl = Mock()
        self.trainer.kl_ctrl.value = 0.1

    def tearDown(self):
        # Stop all patches
        for p in self.patches:
            p.stop()

    def test_rollout_basic_flow(self):
        """Test basic rollout flow without buffer"""
        # Setup mocks
        self.trainer.train_dataloader.get_next_batch.return_value = self.sample_batch
        setup_generation_mock(self.sample_batch, self.actor_rollout_wg, self.reward_fn, 
                             self.config.actor_rollout_ref.rollout.n, 
                             self.config.data.train_batch_size)
       
        # Run rollout
        result, metrics = self.trainer.rollout()
        
        # Basic assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, DataProto)  # Check it's a DataProto object
        self.assertIsInstance(metrics, dict)

    def test_rollout_with_buffer(self):
        """Test rollout with existing buffer content"""
        # Setup buffer with some samples
        buffer_samples = create_sample_buffer_batch(batch_size=4)
        self.trainer.train_dataloader.buffer = buffer_samples
        self.trainer.train_dataloader.get_from_buffer = Mock(return_value=buffer_samples)
        self.trainer.train_dataloader.buffer_size = lambda: len(buffer_samples)
        
        # Setup other mocks
        self.trainer.train_dataloader.get_next_batch.return_value = self.sample_batch
        setup_generation_mock(self.sample_batch, self.actor_rollout_wg, self.reward_fn, 
                             self.config.actor_rollout_ref.rollout.n, 
                             self.config.data.train_batch_size)

        # Run rollout
        result, metrics = self.trainer.rollout()

        self.assertEqual(len(result), self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n)
        # Verify buffer was used
        self.trainer.train_dataloader.get_from_buffer.assert_called_once()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, DataProto)

    def test_rollout_filtering(self):
        """Test filtering logic in rollout"""
        # Setup mocks with specific reward values for testing filtering
        self.trainer.train_dataloader.get_next_batch.return_value = self.sample_batch
        gen_output = setup_generation_mock(self.sample_batch, self.actor_rollout_wg, self.reward_fn, 
                                         self.config.actor_rollout_ref.rollout.n, 
                                         self.config.data.train_batch_size)
        
        # Set reward values that should trigger filtering
        batch_size = self.config.data.train_batch_size
        n_samples = self.config.actor_rollout_ref.rollout.n
        seq_len = 20
        
        # Create reward tensor with shape [batch_size * n_samples, seq_len]
        # need to set the last token to 1.0 to pass the filter
        rewards = torch.zeros(batch_size * n_samples, seq_len)
        # set the last token of the first n_samples to 1.0
        rewards[:4 * n_samples, -1] = 1.0
        # set the last token of the last n_samples to 0.0
        rewards[4 * n_samples:, -1] = 0.0

        self.reward_fn.return_value = rewards

        # Run rollout
        result, metrics = self.trainer.rollout()

        # Verify filtering occurred but some samples remained
        self.assertIsNotNone(result)
        self.assertIsInstance(result, DataProto)
        self.assertGreater(len(result), 0)

        self.assertEqual(len(result), len(self.sample_batch) * n_samples)

        # verify consecutive n_samples have the same response
        for i in range(len(result) // n_samples):
            for j in range(n_samples-1):
                self.assertTrue(torch.equal(result.batch['responses'][i * n_samples + j], result.batch['responses'][i * n_samples + j + 1]))

        # verify the uid is the same for consecutive n_samples
        self.assertEqual(len(result.non_tensor_batch['uid']), len(result))
        for i in range(len(result) // n_samples):
            for j in range(n_samples-1):
                self.assertEqual(result.non_tensor_batch['uid'][i * n_samples + j], result.non_tensor_batch['uid'][i * n_samples + j + 1])

        self.assertEqual(metrics['train/raw_accuracy'], 0.5)

    def test_rollout_batch_size_handling(self):
        """Test proper handling of batch sizes for distributed training"""
        # Setup mocks
        self.trainer.train_dataloader.get_next_batch.return_value = self.sample_batch
        setup_generation_mock(self.sample_batch, self.actor_rollout_wg, self.reward_fn, 
                             self.config.actor_rollout_ref.rollout.n, 
                             self.config.data.train_batch_size)

        # Run rollout
        result, metrics = self.trainer.rollout()

        # Verify batch size is divisible by world_size
        self.assertEqual(len(result) % self.actor_rollout_wg.world_size, 0)

        # check the raw accuracy
        self.assertEqual(metrics['train/raw_accuracy'], 1.0)



if __name__ == '__main__':
    unittest.main()
