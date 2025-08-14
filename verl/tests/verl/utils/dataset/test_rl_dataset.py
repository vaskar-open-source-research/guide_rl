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
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from verl.custom_pytorch.serialization import load
from verl.utils.dataset.rl_dataset import RLHFDataset, BufferedDataLoader, collate_fn
from verl.utils import hf_tokenizer
import tempfile
import dill
import boto3
import botocore
import pickle


def get_gsm8k_data():
    # prepare test dataset
    # url = "https://github.com/eric-haibin-lin/verl-data/raw/refs/heads/main/gsm8k/train.parquet"
    local_folder = os.path.expanduser('/mnt/efs/vaskarnath/workspace/models/verl/experiments/math_reasoning/data/gsm8k_prime_format')
    local_path = os.path.join(local_folder, 'train.parquet')
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def test_rl_dataset():
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer('deepseek-ai/deepseek-coder-1.3b-instruct')
    local_path = get_gsm8k_data()
    dataset = RLHFDataset(parquet_files=local_path, tokenizer=tokenizer, prompt_key='prompt', max_prompt_length=256)

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    from verl import DataProto

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    data = dataset[0]['input_ids']
    output = tokenizer.batch_decode([data])[0]
    print(f'type: type{output}')
    print(f'\n\noutput: {output}')


def test_buffered_dataloader():
    
    # Initialize dataset
    tokenizer = hf_tokenizer('deepseek-ai/deepseek-coder-1.3b-instruct')
    local_path = get_gsm8k_data()
    dataset = RLHFDataset(parquet_files=local_path, tokenizer=tokenizer, prompt_key='prompt', max_prompt_length=4096)
    
    # Create a BufferedDataLoader
    batch_size = 8
    buffered_loader = BufferedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    buffered_loader.start_new_epoch()   
    
    # Collect some batches
    batches = []
    iterations = 3
    for i in range(iterations):
        batch = next(buffered_loader)
        batches.append(batch)
        print(f"Batch {i+1} position: {buffered_loader.current_position}")
    
    # Save the dataloader state
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        # Save current position and other state
        loader_state = {
            'current_position': buffered_loader.current_position,
            'buffer': buffered_loader.buffer,
            'dataset': dataset
        }
        pickle.dump(loader_state, temp_file)
    
    # Create a new dataloader and load the saved state
    new_dataset = RLHFDataset(parquet_files=local_path, tokenizer=tokenizer, prompt_key='prompt', max_prompt_length=4096)
    new_loader = BufferedDataLoader(
        dataset=new_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    # Load the saved state
    with open(temp_path, 'rb') as temp_file:
        saved_state = pickle.load(temp_file)
        new_loader.current_position = saved_state['current_position']
        new_loader.buffer = saved_state['buffer']
    
    # Start new iteration from saved position
    new_loader.start_new_epoch()
    
    # Get next batch and verify continuation
    next_batch = next(new_loader)
    print(f"Restored loader position: {new_loader.current_position}")
    
    # Clean up
    import os
    os.unlink(temp_path)
    
    # Verify new_loader continues from where the old one left off
    assert new_loader.current_position == iterations + 1, "Dataloader position not properly restored"
    
    print("BufferedDataLoader save/load test passed successfully")

def test_buffered_dataloader_with_torch_save_and_load():
    """Test saving and loading the dataloader using torch.save with dill."""

    # Initialize dataset
    tokenizer = hf_tokenizer('deepseek-ai/deepseek-coder-1.3b-instruct')
    local_path = get_gsm8k_data()
    dataset = RLHFDataset(parquet_files=local_path, tokenizer=tokenizer, prompt_key='prompt', max_prompt_length=4096)
    
    # Create a BufferedDataLoader
    batch_size = 4
    buffered_loader = BufferedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    buffered_loader.start_new_epoch()
    
    # Collect some batches to advance the position
    iterations = 3
    for i in range(iterations):
        batch = next(buffered_loader)
        print(f"Batch {i+1} position: {buffered_loader.current_position}")
    
    # Save the dataloader using torch.save with dill (mimicking ray_trainer)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        print(f"Saving dataloader to {temp_path}")
        torch.save(buffered_loader, temp_path, pickle_module=dill)
    
    # Load the dataloader using torch.load with dill
    loaded_loader = load(temp_path, map_location=torch.device('cpu'), pickle_module=dill)
    
    # Resume dataset state as done in ray_trainer
    if isinstance(loaded_loader.dataset, RLHFDataset):
        loaded_loader.dataset.resume_dataset_state()
    
    # Verify the current position was maintained
    print(f"Original position: {buffered_loader.current_position}")
    print(f"Loaded position: {loaded_loader.current_position}")
    assert loaded_loader.current_position == buffered_loader.current_position, "Position not properly restored"
    
    # Start new iteration cycle and verify we can continue from where we left off
    loaded_loader.start_new_epoch()
    
    # Get next batch to verify continuation
    next_batch = next(loaded_loader)
    print(f"Advanced position after loading: {loaded_loader.current_position}")
    
    # Verify that after getting the next batch, the position is one more than before
    assert loaded_loader.current_position == iterations + 1, "Dataloader did not correctly advance after loading"
    
    # Clean up
    import os
    os.unlink(temp_path)
    
    print("BufferedDataLoader torch save/load test passed successfully")


def test_dataloader_checkpoint_integration():
    """Real integration test for saving dataloader locally and reloading it."""
    
    # Initialize dataset and dataloader
    tokenizer = hf_tokenizer('deepseek-ai/deepseek-coder-1.3b-instruct')
    local_path = get_gsm8k_data()
    dataset = RLHFDataset(parquet_files=local_path, tokenizer=tokenizer, prompt_key='prompt', max_prompt_length=4096)
    
    batch_size = 4
    buffered_loader = BufferedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    # Advance position to mimic training progress
    buffered_loader.start_new_epoch()
    iterations = 3
    for i in range(iterations):
        batch = next(buffered_loader)
    
    original_position = buffered_loader.current_position
    print(f"Original position: {original_position}")
    
    # Create temp dir for local storage
    with tempfile.TemporaryDirectory() as local_dir:
        # Set up paths
        checkpoint_dir = os.path.join(local_dir, f'global_step_3')
        os.makedirs(checkpoint_dir, exist_ok=True)
        dataloader_path = os.path.join(checkpoint_dir, 'data.pt')
        
        # Save dataloader using the same method as in ray_trainer.py
        print(f"Saving dataloader to {dataloader_path}")
        torch.save(buffered_loader, dataloader_path, pickle_module=dill, pickle_protocol=4)
        
        # First test: load directly from local file
        print("Test 1: Direct local file loading")
        loaded_loader = load(
            dataloader_path,
            map_location=torch.device('cpu'),
            pickle_module=dill
        )
        
        # Ensure dataset is properly initialized
        if isinstance(loaded_loader.dataset, RLHFDataset):
            loaded_loader.dataset.resume_dataset_state()
            
        # Verify position was maintained
        print(f"Loaded position: {loaded_loader.current_position}")
        assert loaded_loader.current_position == original_position, \
            f"Position not preserved: {loaded_loader.current_position} != {original_position}"
        
        # Continue iteration to verify functionality
        loaded_loader.start_new_epoch()
        next_batch = next(loaded_loader)
        print(f"Advanced position after loading: {loaded_loader.current_position}")
        assert loaded_loader.current_position == original_position + 1, \
            "Dataloader did not correctly advance after loading"
        
        # Second test: Check if we're running in an environment with S3 access
        try:
            print("Test 2: Testing with S3 if credentials available")
            # Check if we have AWS credentials configured
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials is None:
                print("No AWS credentials found, skipping S3 test")
            else:
                # Create a unique bucket name for testing
                bucket_name = f"scale-ml"
                s3_key = "genai/test-checkpoint/global_step_3/data.pt"
                s3_client = boto3.client('s3')
                
                try:
                    # Try to create a bucket                    
                    # Upload the dataloader file
                    s3_client.upload_file(
                        dataloader_path,
                        bucket_name,
                        s3_key,
                        ExtraArgs={'ContentType': 'application/octet-stream'}
                    )
                    print(f"Uploaded dataloader to s3://{bucket_name}/{s3_key}")
                    
                    # Download to a new location
                    with tempfile.TemporaryDirectory() as download_dir:
                        download_path = os.path.join(download_dir, 'data.pt')
                        s3_client.download_file(bucket_name, s3_key, download_path)
                        print(f"Downloaded dataloader from S3 to {download_path}")
                        
                        # Load the downloaded file
                        s3_loaded_loader = load(
                            download_path,
                            map_location=torch.device('cpu'),
                            pickle_module=dill
                        )
                        
                        # Ensure dataset is properly initialized
                        if isinstance(s3_loaded_loader.dataset, RLHFDataset):
                            s3_loaded_loader.dataset.resume_dataset_state()
                        
                        # Verify position was maintained
                        print(f"S3 loaded position: {s3_loaded_loader.current_position}")
                        assert s3_loaded_loader.current_position == original_position, \
                            f"Position not preserved after S3: {s3_loaded_loader.current_position} != {original_position}"
                        
                        # Continue iteration to verify functionality
                        s3_loaded_loader.start_new_epoch()
                        next_batch = next(s3_loaded_loader)
                        print(f"Advanced position after S3 loading: {s3_loaded_loader.current_position}")
                        assert s3_loaded_loader.current_position == original_position + 1, \
                            "Dataloader did not correctly advance after S3 loading"
                
                finally:
                    # Clean up test bucket
                    try:
                        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
                        s3_client.delete_bucket(Bucket=bucket_name)
                        print(f"Cleaned up test S3 bucket: {bucket_name}")
                    except Exception as e:
                        print(f"Error cleaning up S3 resources: {e}")
        
        except (botocore.exceptions.ClientError, botocore.exceptions.NoCredentialsError) as e:
            print(f"S3 operations failed: {e}. Skipping S3 test.")
    
    print("Dataloader checkpoint integration test completed!")

if __name__ == "__main__":
    test_buffered_dataloader()
    test_buffered_dataloader_with_torch_save_and_load()
    test_dataloader_checkpoint_integration()
