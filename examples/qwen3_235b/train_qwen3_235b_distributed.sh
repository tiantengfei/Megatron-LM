#!/bin/bash

# Runs Qwen3-235B-A22B model SFT

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8 # Adjust as needed
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=$1 # e.g., /path/to/your/checkpoints/qwen3_235b
TOKENIZER_PATH=$2 # e.g., /path/to/your/qwen_tokenizer_folder_or_file
DATA_PATH=$3 # e.g., /path/to/your/sft_data

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

# Model arguments from Qwen3-235B-A22B config.json
MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 8192 # Recommended to be less than or equal to max_position_embeddings, adjust based on memory
    --max-position-embeddings 40960 # From config
    --num-layers 94 # From config
    --hidden-size 4096 # From config
    --ffn-hidden-size 12288 # From config: "intermediate_size"
    --num-attention-heads 64 # From config
    --kv-channels 64 # hidden_size / num_attention_heads = 4096 / 64 = 64. Also head_dim from config is 128, but that might be for GQA files. Standard calculation is H/A.
    --num-key-value-heads 4 # From config
    --init-method-std 0.02 # From config: "initializer_range"
    --attention-dropout 0.0 # From config
    --hidden-dropout 0.0 # Assuming 0.0, standard for large models
    --normalization RMSNorm # From config: "rms_norm_eps" implies RMSNorm
    --norm-epsilon 1e-06 # From config: "rms_norm_eps"
    --position-embedding-type rope # From config: "rope_theta"
    --swiglu # From config: "hidden_act": "silu"
    --untie-embeddings-and-output-weights # From config: "tie_word_embeddings": false
    --group-query-attention # num_key_value_heads (4) != num_attention_heads (64)
    --num-query-groups 4 # Should be num_key_value_heads
    --no-masked-softmax-fusion # Keep as per Mixtral/previous, can be tuned
    --no-position-embedding # For RoPE
    --rotary-base 1000000.0 # From config: "rope_theta"
    # Qwen specific: "decoder_sparse_step": 1, "max_window_layers": 94 might need custom model code if not supported by generic Megatron GPT.
    # "mlp_only_layers": [] - Megatron likely doesn't have a direct flag, implies all layers are standard transformer layers.
)

# MoE arguments from Qwen3-235B-A22B config.json
MOE_ARGS=(
    --num-experts 128 # From config
    --moe-router-topk 8 # From config: "num_experts_per_tok"
    --moe-router-load-balancing-type aux_loss # Common choice, Qwen config has "router_aux_loss_coef"
    --moe-aux-loss-coeff 0.001 # From config: "router_aux_loss_coef"
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall # Common choice
    --overlap-param-gather
    --overlap-grad-reduce
    # Qwen specific MoE params:
    # "moe_intermediate_size": 1536 (per expert FFN intermediate size) - this might be implicitly handled by ffn-hidden-size in expert definition or require specific expert FFN sizing.
    # "norm_topk_prob": true - May require custom MoE router code if not a standard Megatron feature.
    # "output_router_logits": false - May require custom MoE router code.
)

DATA_ARGS=(
    --tokenizer-type Qwen3Tokenizer # Needs to match your Qwen tokenizer implementation in Megatron
    --tokenizer-model ${TOKENIZER_PATH} # Path to the tokenizer model/files (e.g., .tiktoken file for Qwen)
    --data-path $DATA_PATH
    --split 99990,8,2 # Example split, adjust for your SFT dataset
    --vocab-size 151936 # From config
    # BOS/EOS tokens if needed by data processing or generation, not typically for training flags here
    # --bos-token-id 151643
    # --eos-token-id 151645
)

TRAINING_ARGS=(
    --micro-batch-size 1 # Adjust based on GPU memory
    --global-batch-size 128 # Adjust (e.g., 4 nodes * 8 GPUs * 1 mbs * 4 acc_grad = 128)
    --lr 1e-5 # Typical starting SFT LR, tune this
    --train-iters 100000 # Adjust for your SFT length
    --lr-decay-iters 80000 # Adjust accordingly
    --lr-decay-style cosine
    --min-lr 1.0e-6 # Typical for SFT
    --weight-decay 0.1
    --lr-warmup-iters 200 # Adjust
    --clip-grad 1.0
    --bf16 # From config: "torch_dtype": "bfloat16"
    --use-flash-attn # Recommended
)

# Parallelism settings for a 235B MoE model.
# These will need careful tuning based on your specific hardware.
# A 235B model is smaller than 671B but still very large.
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8  # Example for 8 GPUs per node
    --pipeline-model-parallel-size 4  # Example, adjust based on number of nodes and memory
    --expert-model-parallel-size 2  # Example: 128 experts / 2 = 64 experts per EP group.
                                    # If TP=8, PP=4, EP=2, total GPUs = 8*4*2 = 64 GPUs needed.
                                    # Adjust EP so that (num_experts / EP) is manageable per rank,
                                    # and TP * PP * EP fits your total GPU count.
                                    # E.g., if you have 32 GPUs (4 nodes * 8 GPUs): TP=8, PP=2, EP=2 might work.
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 10
    --save-interval 500
    --eval-interval 100
    --eval-iters 10
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH # Specify if resuming or loading pretrained weights
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    # --no-load-optim
    # --no-load-rng
    --finetune
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Qwen3_235B_SFT"}
        --wandb-exp-name ${WANDB_NAME:-"qwen3_235b_sft"}
    )
fi

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}

echo "SFT script for Qwen3-235B finished."
