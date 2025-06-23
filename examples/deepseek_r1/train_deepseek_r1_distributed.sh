#!/bin/bash

# Runs DeepSeek R1 671B model SFT

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8 # Adjust as needed
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=$1 # e.g., /path/to/your/checkpoints/deepseek_r1
TOKENIZER_PATH=$2 # e.g., /path/to/your/tokenizer/folder_or_file
DATA_PATH=$3 # e.g., /path/to/your/sft_data

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# Model arguments from DeepSeek R1 config.json
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 16384 # From config: "max_position_embeddings": 163840, but typically seq-length is smaller for training
    --max-position-embeddings 163840 # From config
    --num-layers 61 # From config
    --hidden-size 7168 # From config
    --ffn-hidden-size 18432 # From config: "intermediate_size"
    --num-attention-heads 128 # From config
    --kv-channels 56 # hidden_size / num_attention_heads = 7168 / 128 = 56. Or v_head_dim from config if applicable
    --num-key-value-heads 128 # From config
    --init-method-std 0.02 # From config: "initializer_range"
    --attention-dropout 0.0 # From config
    --hidden-dropout 0.0 # Assuming 0.0, not specified directly for this in config
    --normalization RMSNorm # From config: "rms_norm_eps" implies RMSNorm
    --norm-epsilon 1e-06 # From config: "rms_norm_eps"
    --position-embedding-type rope # From config: "rope_theta", "rope_scaling"
    --swiglu # From config: "hidden_act": "silu" (SiLU/SwiGLU are often used together or imply SwiGLU FFN)
    --untie-embeddings-and-output-weights # From config: "tie_word_embeddings": false
    # --group-query-attention # Not explicitly in config, but common with num_key_value_heads != num_attention_heads. Here they are equal.
    # --num-query-groups 8 # Only if group_query_attention is used and num_key_value_heads is a divisor
    --no-masked-softmax-fusion # Keep as per Mixtral, can be tuned
    --no-position-embedding # For RoPE, explicit position embeddings are often disabled
    --rotary-base 10000 # From config: "rope_theta"
    # Potentially add args for YARN RoPE scaling if Megatron supports them:
    # --rotary-scaling-factor 40 # From config: "rope_scaling": {"factor": 40}
    # --rotary-beta-fast 32 # From config
    # --rotary-beta-slow 1 # From config
    # --rotary-original-max-position-embeddings 4096 # From config
)

# MoE arguments from DeepSeek R1 config.json
MOE_ARGS=(
    --num-experts 256 # From config: "n_routed_experts" (n_shared_experts is 1, usually total experts = n_routed_experts + n_shared_experts)
    --moe-router-topk 8 # From config: "num_experts_per_tok"
    --moe-router-load-balancing-type aux_loss # Common choice, Mixtral uses this. Deepseek might have specific ("noaux_tc"?)
    --moe-aux-loss-coeff 1e-2 # Default, tune as needed
    --moe-grouped-gemm # Usually beneficial for performance
    --moe-token-dispatcher-type alltoall # Common choice
    # --moe-expert-model-parallel-size may be part of MODEL_PARALLEL_ARGS
    # The following are from Mixtral and might be relevant or need adjustment
    --overlap-param-gather
    --overlap-grad-reduce
    # Deepseek specific MoE params from config that might not have direct flags:
    # "first_k_dense_replace": 3
    # "moe_intermediate_size": 2048 (per expert FFN intermediate size)
    # "norm_topk_prob": true
    # "routed_scaling_factor": 2.5
    # "scoring_func": "sigmoid"
    # "topk_group": 4
    # "topk_method": "noaux_tc"
)

DATA_ARGS=(
    --tokenizer-type DeepseekV3Tokenizer # Needs to match your tokenizer implementation in Megatron
    --tokenizer-model ${TOKENIZER_PATH} # Path to the tokenizer model/files
    --data-path $DATA_PATH
    --split 99990,8,2 # Example split, adjust for your SFT dataset
    --vocab-size 129280 # From config
)

TRAINING_ARGS=(
    --micro-batch-size 1 # Adjust based on GPU memory
    --global-batch-size 256 # Adjust based on your setup (e.g., 8 nodes * 8 GPUs * 1 mbs * 4 acc_grad = 256)
    --lr 1e-5 # Typical starting SFT LR, tune this
    --train-iters 100000 # Adjust for your SFT length
    --lr-decay-iters 80000 # Adjust accordingly
    --lr-decay-style cosine
    --min-lr 1.0e-6 # Typical for SFT
    --weight-decay 0.1
    --lr-warmup-iters 200 # Adjust
    --clip-grad 1.0
    --bf16 # From config: "torch_dtype": "bfloat16"
    --use-flash-attn # Recommended for performance with modern GPUs and long sequences
)

# Parallelism settings for a 671B model will be critical and need careful tuning
# This is a starting point and likely needs significant adjustment based on your specific hardware (number of nodes, GPUs per node)
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8 # Example: For 8 GPUs per node
    --pipeline-model-parallel-size 8 # Example: For multi-node or if a single node cannot hold a layer
    --expert-model-parallel-size 1 # From config "ep_size": 1. If this means experts are not split, set to 1.
                                   # If you have 256 experts and want to parallelize them, this should be higher (e.g., 8 if TP=8, meaning each TP group handles 32 experts)
                                   # This interacts heavily with --num-experts. If ep_size=1 means each expert is on one rank,
                                   # then with TP=8, each rank would handle 256/8 = 32 experts.
                                   # If n_routed_experts refers to total experts in the system, and you want to distribute them,
                                   # then expert-model-parallel-size should be a factor of n_routed_experts.
                                   # For instance, if you have 64 GPUs (8 nodes * 8 GPUs), you might try TP=8, PP=8, EP=1 (if ep_size=1 means each rank has all experts)
                                   # OR TP=8, PP=1, EP=8 (distributing 256 experts over 8 expert parallel groups).
                                   # This is the most complex part to configure for a large MoE.
    --use-distributed-optimizer
    --sequence-parallel # Usually beneficial for long sequences
)

LOGGING_ARGS=(
    --log-interval 10
    --save-interval 500 # Adjust as needed
    --eval-interval 100 # Adjust as needed
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH # Specify if you are resuming SFT or loading pretrained weights
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    # --no-load-optim # Uncomment if starting SFT from a pretrained, non-SFT checkpoint
    # --no-load-rng # Uncomment if starting SFT from a pretrained, non-SFT checkpoint
    --finetune # Important for SFT to load all layers strictly
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"DeepSeek_R1_SFT"}
        --wandb-exp-name ${WANDB_NAME:-"deepseek_r1_671b_sft"}
    )
fi

# The main script for SFT might be different from pretrain_gpt.py
# It could be pretrain_gpt.py with --finetune, or a dedicated script like finetune_gpt.py if available.
# Assuming pretrain_gpt.py handles SFT via --finetune based on common Megatron practice.
# If you have a specific SFT script (e.g., from `tasks/`), use that.
# For example, if there's `tasks/main.py` or `finetune_gpt.py`.
# Using pretrain_gpt.py as a placeholder for now.
# You might need to create or use a specific SFT script.
# For SFT, you often use `megatron.training.training.py` or a wrapper like `pretrain_gpt.py`
# with appropriate dataset and finetuning flags.

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}

echo "SFT script finished."
