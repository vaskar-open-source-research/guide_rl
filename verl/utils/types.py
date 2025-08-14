from typing import TypedDict


class ModelConfig(TypedDict, total=False):  # total=False makes all keys optional
    model: str  # model path
    dtype: str
    max_model_len: int
    max_num_seqs: int
    max_batch_size: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    data_parallel_size: int
