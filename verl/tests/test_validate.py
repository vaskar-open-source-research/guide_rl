from unittest.mock import MagicMock, patch

import numpy as np
import torch
from tests.test_utils import (
    create_mock_dataloader,
    create_mock_dataset,
    create_sample_batch,
    setup_generation_mock,
)

from verl import DataProto


@patch("verl.protocol.pad_dataproto_to_divisor")
@patch("verl.protocol.unpad_dataproto")
def test_validate(mock_unpad, mock_pad):
    # Create mock config
    mock_config = MagicMock()
    mock_config.reward_model.enable = False

    # Create mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "mock decoded text"

    # Create mock worker group
    mock_worker_group = MagicMock()
    mock_worker_group.world_size = 2

    # Create mock reward function that returns different scores for different sources
    def mock_reward_function(batch):
        data_sources = batch.non_tensor_batch.get("data_source", [])
        scores = []
        for source in data_sources:
            if "health_AI" in source:
                scores.append(0.9)  # 90% score for health_AI
            elif "general_reasoning" in source:
                scores.append(0.7)  # 70% score for general_reasoning
            elif "math" in source:
                scores.append(0.8)  # 80% score for math domain
            else:
                scores.append(0.5)
        # Return a tensor of shape (batch_size, 1)
        return torch.tensor(scores, dtype=torch.float).unsqueeze(-1)

    mock_reward_fn = MagicMock(side_effect=mock_reward_function)

    # Create base sample batch
    batch_size = 4
    sample_batch = create_sample_batch(batch_size)

    # Setup generation mock
    gen_output = setup_generation_mock(
        sample_batch,
        mock_worker_group,
        mock_reward_fn=mock_reward_fn,  # Pass the mock reward function we created
        n_samples=1,  # For validation we use n_samples=1
        batch_size=batch_size,
    )

    # Setup pad/unpad mocks
    mock_pad.return_value = (sample_batch, 0)
    mock_unpad.side_effect = lambda x, pad_size: x

    # Modify mock_worker_group to return consistent prompts
    def mock_generate_sequences(batch):
        # Get input sequence length
        input_len = batch.batch["input_ids"].size(1)
        response_len = 20  # Fixed response length for testing

        # Create output that matches input prompts
        gen_output_dict = {
            "input_ids": torch.cat(
                [batch.batch["input_ids"], torch.randint(0, 1000, (batch_size, response_len))],
                dim=1,
            ),
            "attention_mask": torch.cat(
                [batch.batch["attention_mask"], torch.ones(batch_size, response_len)], dim=1
            ),
            "position_ids": torch.cat(
                [
                    batch.batch["position_ids"],
                    torch.arange(input_len, input_len + response_len).expand(batch_size, -1),
                ],
                dim=1,
            ),
            "responses": torch.randint(0, 1000, (batch_size, response_len)),
        }

        return DataProto.from_single_dict(gen_output_dict)

    mock_worker_group.generate_sequences = MagicMock(side_effect=mock_generate_sequences)

    def create_test_dataloader(size=4):
        # Use create_sample_buffer_batch from test_utils which creates proper batch structure
        from tests.test_utils import create_sample_buffer_batch

        # Create base batch with proper structure
        base_batch = create_sample_buffer_batch(batch_size=size)

        # Create non-tensor batch data as numpy arrays
        data_source = np.array(
            [
                "math/general_reasoning/health_AI",
                "math/general_reasoning/numina_cn_k12",
                "math/PRIME-RL/health_AI",
                "math/PRIME-RL/numina_cn_k12",
            ],
            dtype=object,
        )

        # Create reward_model data as a regular numpy array of dictionaries
        reward_model = np.array([{"style": "rule", "ground_truth": "correct"}] * size, dtype=object)

        # Convert TensorDict to dictionary of tensors, excluding 'responses'
        tensor_dict = {}
        for key in base_batch.batch.keys():
            if key != "responses":  # Skip responses key
                tensor_dict[key] = base_batch.batch[key]

        # Create a dictionary that matches what DataProto.from_single_dict expects
        batch_dict = {
            **tensor_dict,  # Add all tensor data except responses
            "data_source": data_source,  # Add non-tensor data
            "reward_model": reward_model,
        }

        # Create mock dataset and dataloader
        mock_dataset = create_mock_dataset()
        mock_dataloader = create_mock_dataloader(batch_dict, mock_dataset)

        # Make the dataloader iterable return the dictionary
        mock_dataloader.__iter__ = MagicMock(return_value=iter([batch_dict]))

        return mock_dataloader

    # Create mock trainer instance
    class MockTrainer:
        def __init__(self):
            self.config = mock_config
            self.tokenizer = mock_tokenizer
            self.actor_rollout_wg = mock_worker_group
            self.val_reward_fn = mock_reward_fn
            self.val_dataloader = create_test_dataloader()
            self.test_dataloader = create_test_dataloader()

        def _maybe_log_val_generations_to_wandb(self, *args, **kwargs):
            pass

        def save_validation_info(self, *args, **kwargs):
            pass

        def _validate(self):
            from verl.trainer.ppo.ray_trainer import RayPPOTrainer

            # Create a new instance of RayPPOTrainer._validate bound to self
            validate_method = RayPPOTrainer._validate.__get__(self, RayPPOTrainer)
            return validate_method()

    # Run the test
    trainer = MockTrainer()
    metric_dict = trainer._validate()

    # Print and verify the results
    print("\nValidation Metrics:")
    for key, value in sorted(metric_dict.items()):
        print(f"{key}: {value}")

    # Verify expected metrics are present and reasonable
    expected_metrics = [
        # Domain metrics
        "val/domain/math/score",
        # Source metrics
        "val/source/general_reasoning/score",
        # Subcategory metrics
        "val/subcategory/health_AI/score",
        # Test metrics
        "test/domain/math/score",
        "test/source/general_reasoning/score",
        "test/subcategory/health_AI/score",
        "test/test_score/all",
    ]

    for metric in expected_metrics:
        assert metric in metric_dict, f"Missing metric: {metric}"
        assert isinstance(metric_dict[metric], float), f"Metric {metric} should be float"
        assert 0 <= metric_dict[metric] <= 1, f"Metric {metric} should be between 0 and 1"
    assert (
        abs(metric_dict["val/domain/math/score"] - 0.825) < 1e-5
    ), "Math domain should score 0.825 (average of 0.9, 0.7, 0.9, 0.8)"
    assert (
        abs(metric_dict["val/source/general_reasoning/score"] - 0.8) < 1e-5
    ), "general_reasoning source should score 0.8 (average of 0.9, 0.7)"
    assert (
        abs(metric_dict["val/subcategory/health_AI/score"] - 0.9) < 1e-5
    ), "health_AI subcategory should score 0.9"

    print("\nAll assertions passed!")


if __name__ == "__main__":
    test_validate()
