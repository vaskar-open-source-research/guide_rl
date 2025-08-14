import os
from megatron.fused_kernels import load


class FakeArgs:
    rank = 0


# 7.0 for V100
# 8.0 for A100/A800
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0+PTX;8.0+PTX"

load(FakeArgs)