# Default Environment Variables
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=random
export PYTHONPATH="${PYTHONPATH}:/workspace"
export NVCR=1

# Build-time Environment Variables
export PIP_NO_CACHE_DIR=off
export PIP_DISABLE_PIP_VERSION_CHECK=on
export PIP_DEFAULT_TIMEOUT=100
export DEBIAN_FRONTEND=noninteractive
export DEBCONF_NONINTERACTIVE_SEEN=true

# set up uv venv
uv venv
source .venv/bin/activate

# Install UV using pip
uv pip install --upgrade pip --no-cache-dir
uv pip install uv
uv pip install --upgrade pip setuptools wheel

uv pip install -r verl/requirements.txt
uv pip install -e verl/pyext-0.7

# uninstall nv-pytorch fork
uv pip uninstall pytorch-quantization \
     pytorch-triton \
     torch \
     torch-tensorrt \
     torchvision \
     xgboost transformer_engine flash_attn \
     apex megatron-core

uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index https://download.pytorch.org/whl/cu118
uv pip install --no-cache-dir vllm==0.6.3
MAX_JOBS=64 uv pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation git+https://github.com/NVIDIA/apex
MAX_JOBS=64 NINJA_FLAGS="-j64" uv pip install flash-attn==2.5.8 --no-cache-dir --no-build-isolation
MAX_JOBS=64 NINJA_FLAGS="-j64" uv pip install git+https://github.com/NVIDIA/TransformerEngine.git@v1.7
uv pip install -e verl/Megatron-LM
uv pip install ray==2.38

export FORCE_CUDA="1"
export DD_SERVICE=verl 
export PYTHONPATH="${PYTHONPATH}:verl"

source .venv/bin/activate