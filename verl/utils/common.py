import logging
import os
import random
import subprocess
import tempfile
from importlib import import_module
from math import ceil
from typing import Any, Dict, List, Union

import jsonlines
import numpy as np
import ray
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from omegaconf import OmegaConf

import utils.globals as rift_globals
from utils.st_utils import get_job_info

####################################################################################
#  OS utils
####################################################################################


def setup_logging():
    job_info = get_job_info()
    logging.basicConfig(
        format=f"[node_rank: {job_info.node_rank}/{job_info.num_nodes - 1}] %(levelname)s %(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%Y/%m/%d %H:%M:%S",
        force=True,
    )


def subprocess_run(cmd: List[str], env_vars: Dict[str, str] = None):
    """
    Run an os command using subprocess.run with given os env variables in a new copy of the environment
    """
    logging.info(f"subprocess_run: {cmd} with env_vars: {env_vars}")
    env_copy = os.environ.copy()
    if env_vars:
        for key, value in env_vars.items():
            if isinstance(value, list):
                value = ",".join(value)
            else:
                value = str(value)
            env_copy[key] = value
    subprocess.run(cmd, check=True, env=env_copy)
    del env_copy


def set_os_env_vars(env_vars: dict, print_values: bool = False):
    """
    Set os env variables for the current process
    """
    for key, value in env_vars.items():
        logging.info(f"setting os env var {key}={value if print_values else '****'}")
        if isinstance(value, list):
            value = ",".join(value)
        else:
            value = str(value)
        os.environ[key] = value


def unset_os_env_vars(env_vars: set):
    """
    Unset os env variables for the current process
    """
    for env_var in env_vars:
        logging.info(f"unsetting os env var {env_var}")
        os.environ.pop(env_var, None)


def aws_copy(
    src: str,
    dst: str,
    recursive: bool = False,
    use_s5cmd: bool = rift_globals.DEFAULT_S5CMD_FOR_AWS_COPY,
    exclude: str = "",
):
    """
    Copy files from src to dst using 'aws s3 cp' OR 's5cmd cp' (at least one of src or dst must be S3 path)
    """
    if use_s5cmd:
        cmd = ["s5cmd", "cp"]
        if len(exclude) > 0:
            cmd += ["--exclude", exclude]

        if recursive:
            src = src.rstrip("/") + "/*"
            dst = dst.rstrip("/") + "/"
        cmd += [src, dst]
    else:
        cmd = ["aws", "s3", "cp"]
        if len(exclude) > 0:
            cmd += ["--exclude", exclude]

        if recursive:
            cmd += ["--recursive"]
        cmd += [src, dst]

    try:
        logging.info("trying aws_copy with AWS_PROFILE=ml-worker")
        subprocess_run(cmd, env_vars={"AWS_PROFILE": "ml-worker"})
    except:
        logging.info("trying aws_copy again! without AWS_PROFILE")
        subprocess_run(cmd)


def aws_remove(
    s3_dir: str,
    recursive: bool = False,
    use_s5cmd: bool = rift_globals.DEFAULT_S5CMD_FOR_AWS_COPY,
):
    """
    Delete files from s3 using 'aws s3 rm' OR 's5cmd rm'
    """
    if use_s5cmd:
        cmd = ["s5cmd", "rm", f"{s3_dir}/*"]
    else:
        cmd = ["aws", "s3", "rm", "--recursive", f"{s3_dir}/"]

    try:
        logging.info("trying aws_delete with AWS_PROFILE=ml-worker")
        subprocess_run(cmd, env_vars={"AWS_PROFILE": "ml-worker"})
    except:
        logging.info("trying aws_delete again! without AWS_PROFILE")
        subprocess_run(cmd)


def universal_copy(
    src: str,
    dst: str,
    recursive: bool = False,
    use_s5cmd: bool = rift_globals.DEFAULT_S5CMD_FOR_AWS_COPY,
    exclude: str = "",
):
    """
    Copy files from src to dst using 'aws s3 cp' OR 's5cmd cp' OR 'cp'
    """
    if src.startswith("s3://") or dst.startswith("s3://"):
        aws_copy(src, dst, recursive, use_s5cmd, exclude)
    else:
        cmd = ["cp"]
        if recursive:
            cmd += ["-a"]
            src = src.rstrip("/") + "/."
            dst = dst.rstrip("/") + "/"
            if not os.path.exists(dst):  # create dst directory if it doesn't exist
                os.makedirs(dst, exist_ok=True)
        # TODO: add exclusion back - `cp` doesn't support it however
        # if len(exclude) > 0:
        #     cmd += ["--exclude", exclude]
        cmd += [src, dst]
        subprocess_run(cmd)


def get_gpu_memory_utilization():
    """
    Get the GPU memory utilization
    """
    mem_usage = []
    for device in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(device)
        mem_usage.append((total - free) // 1024**2)
    return mem_usage


def ray_init():
    # Check if Ray is already initialized
    if ray.is_initialized():
        logging.info("Ray is already initialized. Using existing Ray instance.")
        return

    ray.init()

    # job_info = get_job_info()
    # leader_port = job_info.leader_port
    # leader_addr = job_info.leader_addr

    # # ray.init(address=f"{leader_addr}:{leader_port}", ignore_reinit_error=True)
    # if job_info.node_rank == 0:
    #     cmd = ["ray", "start", "--head", "--node-ip-address", str(leader_addr), "--port", str(leader_port)]
    # else:
    #     current_ip = subprocess.check_output(["hostname", "-I"]).decode("utf-8").split()[0]
    #     cmd = ["ray", "start", "--address", f"{leader_addr}:{leader_port}", "--node-ip-address", str(current_ip)]

    # subprocess_run(cmd)

    # runtime_env = {
    #     "env_vars": {
    #         "NCCL_SOCKET_IFNAME": "eth0",  # Specify network interface for NCCL
    #         "NCCL_DEBUG": "INFO",  # Enable NCCL debugging
    #         "NCCL_IB_DISABLE": "1",  # Disable InfiniBand if not properly configured
    #         "NCCL_SOCKET_NTHREADS": "4",
    #         "NCCL_NSOCKS_PERTHREAD": "4",
    #         "NCCL_P2P_DISABLE": "1",  # Disable peer-to-peer memory transfers
    #         "NCCL_SHM_DISABLE": "1",  # Disable shared memory operations
    #         "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    #         "RAY_memory_monitor_refresh_ms": "0",  # Disable Ray's memory monitor which can be flaky
    #         # Add GKE-specific settings
    #         "NCCL_ASYNC_ERROR_HANDLING": "1",
    #         "NCCL_BLOCKING_WAIT": "1",
    #         "RAY_BACKEND_LOG_LEVEL": "debug",  # More detailed Ray logs for debugging
    #     }
    # }

    # ray.init(
    #     address=f"{ray_addr}:{ray_port}",
    #     ignore_reinit_error=True,
    #     logging_level=logging.INFO,
    #     # runtime_env=runtime_env,
    #     # _system_config={
    #     #     "object_timeout_milliseconds": 10000,  # Increased from 5000
    #     #     # "num_heartbeats_timeout": 600,  # Increased from 300
    #     #     # "raylet_heartbeat_timeout_milliseconds": 20000,  # Increased from 10000
    #     #     "task_retry_delay_ms": 5000,  # Increased from 2000
    #     #     "worker_register_timeout_seconds": 120,  # Increased from 60
    #     #     # "object_store_full_max_retries": 10,
    #     #     "object_store_full_delay_ms": 1000,
    #     #     "kill_idle_workers_interval_ms": 0,  # Prevent Ray from killing idle workers
    #     #     # "worker_lease_timeout_milliseconds": 120000,
    #     #     # Disable cross-node operations
    #     #     "scheduler_spread_threshold": 0.0,  # Prevent tasks from being scheduled on other nodes
    #     #     # "scheduler_score_local_node_resources": 1e9,  # Heavily prefer local node scheduling
    #     # }
    # )


def ray_shutdown():
    # cmd = ["ray", "stop"]
    # cmd += ["--force"]
    # subprocess_run(cmd)

    ray.shutdown()


####################################################################################
#  Common utils
####################################################################################


def load_config(config_path: str):
    """
    Load a config from a yaml file from local or s3 path
    """
    assert config_path.endswith(".yaml")
    logging.info("-" * 60)
    logging.info(f"loading config from {config_path}")
    logging.info("-" * 60)
    if config_path.startswith("s3://"):  # load from s3
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, config_path.split("/")[-1])
            aws_copy(config_path, tmp_path)
            logging.info(f"local_config_path: {tmp_path}")
            config = OmegaConf.load(tmp_path)
    else:  # load from local
        config = OmegaConf.load(config_path)

    OmegaConf.resolve(config)  # resolve all references/interpolations in config

    ### set frequently used utility global variables
    rift_globals.local_exp_dir = os.path.join(
        config.local_project_dir, config.exp_name
    )  # local experiment dir
    rift_globals.s3_exp_dir = os.path.join(
        config.s3_project_dir, config.exp_name
    )  # s3 experiment dir

    os.makedirs(rift_globals.local_exp_dir, exist_ok=True)  # create local experiment directory

    job_info = get_job_info()
    rift_globals.job_env = job_info.job_env
    rift_globals.job_id = job_info.job_id
    rift_globals.num_nodes = job_info.num_nodes
    rift_globals.node_rank = job_info.node_rank

    return config


def set_random_seed(seed: int, torch_deterministic_backend: bool = False):
    """
    Set random seed for the current job wherever possible
    """
    ### initialize a job random seed which is common across all nodes
    if seed > 0:  # user provided seed
        job_random_seed = seed
    else:  # no user provided seed
        hash_value = abs(hash(rift_globals.job_id))  # hash job_id to get a random seed
        job_random_seed = (hash_value % 10000) + 1  # range 1-10000

    node_random_seed = job_random_seed + rift_globals.node_rank  # unique seed for each node

    ### set global variables
    rift_globals.job_random_seed = job_random_seed
    rift_globals.node_random_seed = node_random_seed

    logging.info(
        f"setting job_random_seed: {job_random_seed}, node_random_seed: {node_random_seed}"
    )
    random.seed(node_random_seed)
    np.random.seed(node_random_seed)
    torch.manual_seed(node_random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(node_random_seed)
        torch.backends.cudnn.deterministic = torch_deterministic_backend


def get_class_from_string(class_name_str: str):
    """
    Get a class definition from a string
    """
    module_path, class_name = class_name_str.rsplit(".", 1)
    module = import_module(module_path)
    class_def = getattr(module, class_name)
    return class_def


def repeat_list_elements(data: List[Any], n: int) -> List[Any]:
    """
    repeats each element in the data n times.
    example:
        n = 2, data = ["a", "b", "c"]
        returns ["a", "a", "b", "b", "c", "c"]
    """
    assert n > 0, "number of samples must be greater than 0"
    assert len(data) > 0, "data must be a non-empty list"
    return [d for data_sample in data for d in [data_sample] * n]


def group_list_elements(data: List[Any], n: int) -> List[List[Any]]:
    """
    groups the data by grouping n samples at a time.
    example:
        n = 2, data = ["a1", "a2", "b1", "b2", "c1", "c2"]
        returns [["a1", "a2"], ["b1", "b2"], ["c1", "c2"]]
    """
    assert n > 0, "number of samples must be greater than 0"
    assert len(data) > 0, "data must be a non-empty list"
    assert len(data) % n == 0, "number of data samples must be divisible by n"
    return [data[i : i + n] for i in range(0, len(data), n)]


def flatten_list_strict(data: List[List[Any]], n: int) -> List[Any]:
    """
    flattens a list of lists into a single list, but only if all elements of the list are of length n.
    example:
        n = 2, data = [["a1", "a2"], ["b1", "b2"], ["c1", "c2"]]
        returns ["a1", "a2", "b1", "b2", "c1", "c2"]
    """
    assert len(data) > 0, "data must be a non-empty list"
    result = []
    for sublist in data:
        assert isinstance(sublist, list), f"element should be a list, but got {type(sublist)}"
        assert len(sublist) == n, f"length of sublist ({len(sublist)}) must be equal to n ({n})"
        result.extend(sublist)
    return result


def list_to_dict(data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Convert a list of dicts (~jsonl) to a dict of lists (~dataframe)
    """
    assert isinstance(data, list) and len(data) > 0, "data must be a non-empty list"
    assert isinstance(data[0], dict), "data must be a list of dicts"

    result = {k: [d[k] for d in data] for k in data[0].keys()}
    # check if all lists are of the same length
    assert all(
        len(result[k]) == len(result[list(result.keys())[0]]) for k in result.keys()
    ), "all lists are not of the same length"
    return result


def dict_to_list(data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Convert a dict of lists (~dataframe) to a list of dicts (~jsonl)
    """
    assert isinstance(data, dict) and len(data) > 0, "data must be a non-empty dict"
    for key in data.keys():
        assert isinstance(data[key], list) and len(data[key]) == len(
            data[list(data.keys())[0]]
        ), f"value for all keys must be a list of the same length, but key {key} is not!"

    result = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
    # check if all dicts are of the same length
    assert all(
        len(result[i]) == len(result[0]) for i in range(len(result))
    ), "all dicts are not of the same length"
    return result


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a jsonl objects from one or more files
    """
    result = []
    logging.info(f"loading jsonl from: {file_path}")
    with jsonlines.open(file_path, "r") as reader:
        result.extend([obj for obj in reader])
    return result


def save_jsonl(data: List[Dict[str, Any]], output_file: str):
    """
    Save a list of dictionaries to a jsonl file
    """
    assert output_file.endswith(".jsonl")
    os.makedirs(
        os.path.dirname(output_file), exist_ok=True
    )  # create output directory if it doesn't exist
    logging.info(f"saving jsonl to: {output_file}")
    with jsonlines.open(output_file, "w") as writer:
        writer.write_all(data)


def get_dataset_to_local(
    name: str,
    dataset_path: str,
    use_s5cmd: bool = rift_globals.DEFAULT_S5CMD_FOR_AWS_COPY,
    hf_data_split: Union[str, None] = None,
):
    """
    Save the dataset to the local base directory; can be access using load_dataset_from_local()
    """
    save_path = f"{get_base_dir()}/datasets/{name}"
    logging.info(f"saving dataset from {dataset_path} to {save_path}")

    file_ext = dataset_path.split("/")[-1].split(".")[-1]  # get file extension

    if file_ext == "jsonl":  # jsonl file
        save_path = f"{save_path}.{file_ext}"
        universal_copy(dataset_path, save_path, use_s5cmd=use_s5cmd)
    else:  # assuming dataset_path points to HF dataset dir or is valid HF dataset name
        try:
            if dataset_path.startswith("s3://") or os.path.exists(
                dataset_path
            ):  # HF dataset on s3 or local
                universal_copy(dataset_path, save_path, recursive=True, use_s5cmd=use_s5cmd)
            else:  # download HF dataset from HF hub
                dataset = load_dataset(dataset_path, split=hf_data_split)
                dataset.save_to_disk(save_path)
        except:
            raise ValueError(
                f"Error saving dataset {name}, dataset_path should point to one of the following: .jsonl file, HF dataset dir or valid HF Hub name"
            )


def load_dataset_from_local(name: str, hf_data_split: str = "train") -> List[Dict[str, Any]]:
    """
    Load the dataset from the local base directory which were copied using copy_dataset_to_local()
    returns a list of dicts
    """
    load_path = f"{get_base_dir()}/datasets/{name}"
    logging.info(f"loading dataset from {load_path}")

    if os.path.isfile(f"{load_path}.jsonl"):  # jsonl file
        data = load_jsonl(f"{load_path}.jsonl")
    else:  # HF dataset
        hf_dataset = load_from_disk(load_path)
        if isinstance(hf_dataset, DatasetDict):  # if HF DatasetDict
            logging.info(f"loading HF dataset split: {hf_data_split}")
            data = hf_dataset[hf_data_split].to_list()
        if isinstance(hf_dataset, Dataset):  # if HF Dataset
            data = hf_dataset.to_list()

    # data = list_to_dict(data)

    return data


def find_key_in_data_dict(data_dict: Dict[str, Any], key_list: List[str]) -> str:
    """check if any of the keys from key_list are present in data_dict and return the first key found"""
    for key in key_list:
        if key in data_dict:
            return key
    raise ValueError(f"none of the keys from {key_list} were found in data_dict")


def get_data_parallel_chunks(data: List[Any], dp_size: int) -> List[List[Any]]:
    """Divide data into equal chunks for each data parallel rank"""
    chunk_size = ceil(len(data) / dp_size)
    data_chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    return data_chunks


def select_cycled_prompts_from_data(data: List[Any], i_iter: int, num_prompts: int) -> List[Any]:
    """Cycle through dataset to select num_prompts from data based on the current iteration index"""
    if len(data) < num_prompts:
        logging.warning(f"len(data) ({len(data)}) is less than the num_prompts ({num_prompts})")

    max_cycle_iter = len(data) // num_prompts
    cycle_iter = i_iter % max_cycle_iter if max_cycle_iter > 0 else 0
    return data[cycle_iter * num_prompts : (cycle_iter + 1) * num_prompts]


####################################################################################
# RIFT Directory utils
# use these functions to keep output directories consistent and organized
####################################################################################


def get_exp_dir(get_s3_dir: bool = False):
    """
    get the experiment directory (local or s3)
    """
    return rift_globals.local_exp_dir if not get_s3_dir else rift_globals.s3_exp_dir


def get_base_dir(get_s3_dir: bool = False):
    """
    save all base models, train and eval datasets, etc. in this directory
    example: local_exp_dir/base
    example: s3_exp_dir/base if get_s3_dir=True
    """
    exp_dir = rift_globals.local_exp_dir if not get_s3_dir else rift_globals.s3_exp_dir
    base_dir = f"{exp_dir}/base"
    if not get_s3_dir:
        os.makedirs(base_dir, exist_ok=True)
    return base_dir


def get_datagen_dir(i_iter: int, get_s3_dir: bool = False):
    """
    save all datagen data in this directory
    example: local_exp_dir/iteration_0/datagen
    example: s3_exp_dir/iteration_0/datagen if get_s3_dir=True
    """
    exp_dir = rift_globals.local_exp_dir if not get_s3_dir else rift_globals.s3_exp_dir
    datagen_dir = f"{exp_dir}/iteration_{i_iter}/datagen"
    if not get_s3_dir:
        os.makedirs(datagen_dir, exist_ok=True)
    return datagen_dir


def get_model_training_dir(i_iter: int, get_s3_dir: bool = False):
    """
    save all model training data in this directory
    example: local_exp_dir/iteration_0/model_training
    example: s3_exp_dir/iteration_0/model_training if get_s3_dir=True
    """
    exp_dir = rift_globals.local_exp_dir if not get_s3_dir else rift_globals.s3_exp_dir
    model_training_dir = f"{exp_dir}/iteration_{i_iter}/model_training"
    if not get_s3_dir:
        os.makedirs(model_training_dir, exist_ok=True)
    return model_training_dir


def get_eval_dir(i_iter: int, get_s3_dir: bool = False):
    """
    save all eval data in this directory
    example: local_exp_dir/iteration_0/eval
    example: s3_exp_dir/iteration_0/eval if get_s3_dir=True
    """
    exp_dir = rift_globals.local_exp_dir if not get_s3_dir else rift_globals.s3_exp_dir
    eval_dir = f"{exp_dir}/iteration_{i_iter}/eval"
    if not get_s3_dir:
        os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def get_models_repo_dir():
    """
    get local path for scale's models repository
    """
    if "MODELS_DIR" in os.environ:
        return os.environ["MODELS_DIR"]

    user_path = os.path.expanduser("~")

    if os.path.exists("/workspace/rift"):
        models_repo_dir = "/workspace"
    elif os.path.exists("/models"):
        models_repo_dir = "/models"
    elif os.path.exists(f"{user_path}/src/models"):
        models_repo_dir = f"{user_path}/src/models"
    elif os.path.exists(f"{user_path}/models"):
        models_repo_dir = f"{user_path}/models"
    else:
        raise ValueError("could not find models repo directory!")

    set_os_env_vars({"MODELS_DIR": models_repo_dir}, print_values=True)
    return models_repo_dir


####################################################################################
# Misc utils
####################################################################################


####################################################################################
#  TESTING CODE
####################################################################################

if __name__ == "__main__":

    setup_logging()

    config = load_config("projects/project_zero/configs/config.yaml")

    # test_multinode_barrier()
    # test_aws_copy()

    get_dataset_to_local(
        "train/base_dataset", "ScaleFrontierData/gpqa_diamond"
    )  # aime24 / HLE / gpqa_diamond
    data = load_dataset_from_local("train/base_dataset")

    print(type(data))
    for i in range(5):
        keys = data[i].keys()
        print(keys)
        for key in keys:
            print(f">>> {key}:\n{data[i][key]}")
        print("-" * 60)

    # get_dataset_to_local(
    #     "train/base_dataset2",
    #     "s3://scale-ml/genai/rift/benchmarks/math500/test.jsonl",
    #     use_s5cmd=False,
    # )
    # data = load_dataset_from_local("train/base_dataset2")
