set -x

export NCCL_DEBUG=WARN
export WANDB_API_KEY=''
export WANDB_HOST=''
export WANDB_ENTITY=''
export WANDB_PROJECT=''
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/workspace:/workspace/verl:$PYTHONPATH"

source .venv/bin/activate

PROJECT_NAME='guide-math-open-source'
EXPERIMENT_NAME='llama-3.1-baseline'
DATA_PATH='verl/experiments/math_reasoning/data/guidance_data_full'
LOCAL_SFT_MODEL_PATH='meta-llama/Llama-3.1-8B-Instruct'
CKPT_PATH='/workspace/verl/ckpt'
S3_PATH='s3://scale-ml/genai/maple/guide-math-luffy'

nnodes=$1
node_rank=$JOB_COMPLETION_INDEX

base_dir=$(pwd)

export PYTHONPATH="$base_dir:$base_dir/verl:$PYTHONPATH"

read -r -d '' training_command <<EOF
uv run --active --directory $base_dir python -m verl.trainer.main_ppo --config-name=ppo_trainer\
    algorithm.adv_estimator=grpo \
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/validation.parquet"] \
    data.test_files=["$DATA_PATH/test.parquet"] \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.apply_chat_template=True \
    actor_rollout_ref.model.path=$LOCAL_SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH \
    trainer.s3_path="$S3_PATH/$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$nnodes \
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.total_epochs=2 \
    scale_reasoning.response_format=llama_format \
    trainer.val_before_train=True \
    data.filter_accuracy=True \
    data.filter_truncated=False \
    data.accuracy_lower_bound=0.001 \
    data.accuracy_upper_bound=0.999 \
    data.hint_accuracy_lower_bound=0.001 \
    data.hint_accuracy_upper_bound=0.999 \
    scale_reasoning.reroll_with_hints=False \
    scale_reasoning.reroll_with_cot=False \
    actor_rollout_ref.actor.importance_weighted_reroll=False \
    scale_reasoning.do_verify_format=False \
    scale_reasoning.reroll_threshold=0.0 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    scale_reasoning.custom_chat_template=luffy_chat_template \
    trainer.val_generations_to_log_to_wandb=50
EOF

echo "$training_command"

if [ $nnodes -eq 1 ]; then
    $training_command
else
    if [ $node_rank -eq 0 ]; then

        current_ip=$(hostname -I | awk '{print $1}')
        ray start --head --node-ip-address=$LEADER_ADDR --port=$LEADER_PORT --dashboard-host 0.0.0.0 --dashboard-port 8265
        # wait until all nodes are ready
        while true; do
            ray status
            nodes_ready=$(ray status | grep -c "1 node")
            if [ $nodes_ready -eq $nnodes ]; then
                break
            fi
            echo "Only $nodes_ready nodes are ready. Waiting for all nodes to be ready..."
            sleep 10
        done

        $training_command

    else
        current_ip=$(hostname -I | awk '{print $1}')
        ray start --address="$LEADER_ADDR:$LEADER_PORT" --node-ip-address=$current_ip
        while true; do
            echo "Running mulit-node ray cluster..."
            sleep 10
        done
    fi
fi
