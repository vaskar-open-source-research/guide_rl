import gc
import json
import os
import re
from typing import Dict, Optional, Union

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file as safe_save_file
from torch import Tensor
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME


DeviceType = Union[str, torch.device]


def state_dict_to_dtype(
    state_dict: Dict[str, Tensor], dtype: Optional[torch.dtype] = None, inplace: bool = False
) -> Dict[str, Tensor]:
    if dtype is not None and not dtype.is_floating_point:
        raise ValueError(f"dtype must be a floating point type, but got {dtype}")

    elif dtype is None:
        return state_dict

    elif inplace:
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if v.is_floating_point():
                new_v = v.to(dtype=dtype)
                del v
                gc.collect()
            else:
                new_v = v

            state_dict[k] = new_v
        return state_dict

    else:
        return {k: v.to(dtype=dtype) if v.is_floating_point() else v for k, v in state_dict.items()}


def save_hf_checkpoint(
    state_dict: dict,
    save_directory: Union[str, os.PathLike],
    is_main_process: bool,
    max_shard_size: Union[int, str] = "5GB",
    safe_serialization: bool = True,
) -> None:
    """
    Save state dictionary in HF format. Adapted from `transformers.modeling_utils.PreTrainedModel.save_pretrained`;
        see https://github.com/huggingface/transformers/blob/d5bdac3db7c779cc7d8e53808926b9a3237318f4/src/transformers/modeling_utils.py#L2477

    Args:
        state_dict: The state dictionary to save. Can be used to only save parts of the model or if
            special precautions need to be taken when recovering the state dictionary of a model
            (like when using model parallelism).
        save_directory: Directory to which to save. Will be created if it doesn't exist.
        is_main_process: Whether the process calling this is the main process or not.
            Useful when in distributed training and need to call this function on all processes.
            In this case, set `is_main_process=True` only on the main process to avoid race conditions.
        max_shard_size: The maximum size for a checkpoint before being sharded. Checkpoints shard will
            then be each of size lower than this size. If expressed as a string, needs to be digits
            followed by a unit (like `"5MB"`). We default it to 5GB in order for models to be able to
            run easily on free-tier google colab instances without CPU OOM issues.

            Notice that if a single weight of the model is bigger than `max_shard_size`, it will be
            in its own checkpoint shard which will be bigger than `max_shard_size`.
        safe_serialization:
            Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).

    Returns:
        None

    """
    if not safe_serialization:
        raise NotImplementedError("Only safe serialization is supported for now.")

    weights_name = SAFE_WEIGHTS_NAME

    # Shard the model if it is too big.
    filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(
        ".safetensors", "{suffix}.safetensors"
    )
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )
    # Save index if sharded
    index = None
    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }

    # Clean the folder from a previous save
    for filename in os.listdir(save_directory):
        full_filename = os.path.join(save_directory, filename)
        # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
        # in distributed settings to avoid race conditions.
        weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

        # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
        filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
        reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

        if (
            filename.startswith(weights_no_suffix)
            and os.path.isfile(full_filename)
            and filename not in state_dict_split.filename_to_tensors.keys()
            and is_main_process
            and reg.fullmatch(filename_no_suffix) is not None
        ):
            os.remove(full_filename)

    # Save the model
    filename_to_tensors = state_dict_split.filename_to_tensors.items()
    for shard_file, tensors in tqdm(filename_to_tensors, desc="Saving model weights"):
        shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}

        safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})

    if index is not None:
        save_index_file = SAFE_WEIGHTS_INDEX_NAME
        save_index_file = os.path.join(save_directory, save_index_file)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)