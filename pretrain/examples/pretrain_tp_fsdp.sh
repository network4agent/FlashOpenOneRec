#!/bin/bash
# Single-node 8-GPU training with TP+FSDP (tp_size=2, dp_size=4)
# Launch with: bash examples/pretrain_tp_fsdp.sh

MODEL_DIR=/code/hf_models/Qwen3-1.7B_itemic
OUTPUT_DIR=/code/onerec_pretrain/model_output/stg1_tp_fsdp
mkdir -p $OUTPUT_DIR

set -x

SCRIPT_FILE=$(readlink -f $0)
echo $(date '+%Y-%m-%d %H:%M:%S') >> $OUTPUT_DIR/task_info.log
echo "script: ${SCRIPT_FILE}" >> $OUTPUT_DIR/task_info.log
echo "=========================" >> $OUTPUT_DIR/task_info.log

echo "Output: $OUTPUT_DIR"

export PYTHONPATH=$PWD:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE=8
TP_SIZE=2
MASTER_PORT=${MASTER_PORT:-29500}

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    recipes/train_qwen3.py \
        --tp_size $TP_SIZE \
        --model_dir $MODEL_DIR \
        --output_dir $OUTPUT_DIR \
        --dataset_config examples/dataset_config/pretrain.json \
        --freeze_llm \
        --use_tie_weights \
        --start_optimize_embedding_index 151669 \
        --model_class Qwen3ForCausalLM \
        --monitor_datasource_loss \
        --monitor_datasource_cnt \
        --max_length 32768 \
        --learning_rate 2e-4 \
        --min_lr 1e-4 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 200 \
        --num_training_steps 2000 \
        --save_checkpoint_per_step 50 \
        --minibatch_size 16384 \
        --logging_per_step 5 \
        --use_fp32_weight \
        --seed 19260817 \
        --enable_profiler \
        --enable_gradient_checkpointing \
        --use_chunked_loss_computer \
    2>&1 | tee $OUTPUT_DIR/stdout.log
