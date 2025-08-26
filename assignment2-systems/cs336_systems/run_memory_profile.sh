#!/bin/bash

# 定义要运行的 Python 脚本名称
SCRIPT_NAME="cs336_systems/memory_analysis_fp32.py"

# 定义模型尺寸和批量大小
MODEL_SIZE="xl"
BATCH_SIZE=1

echo "开始为 ${MODEL_SIZE} 模型生成内存快照..."

# 1. 前向传播模式
echo "--- 正在运行前向传播 (forward pass) ---"
echo "--- 上下文长度: 128"
python3 ${SCRIPT_NAME} --model_size ${MODEL_SIZE} --mode forward --block_size 128 --batch_size ${BATCH_SIZE}

echo "--- 上下文长度: 256"
python3 ${SCRIPT_NAME} --model_size ${MODEL_SIZE} --mode forward --block_size 256 --batch_size ${BATCH_SIZE}

echo "--- 上下文长度: 512"
python3 ${SCRIPT_NAME} --model_size ${MODEL_SIZE} --mode forward --block_size 512 --batch_size ${BATCH_SIZE}

echo "前向传播分析完成。"
echo "-----------------------------------"

# 2. 完整训练步骤模式
echo "--- 正在运行完整训练步骤 (full training step) ---"
echo "--- 上下文长度: 128"
python3 ${SCRIPT_NAME} --model_size ${MODEL_SIZE} --mode forward_backward --block_size 128 --batch_size ${BATCH_SIZE}

echo "--- 上下文长度: 256"
python3 ${SCRIPT_NAME} --model_size ${MODEL_SIZE} --mode forward_backward --block_size 256 --batch_size ${BATCH_SIZE}

echo "--- 上下文长度: 512"
python3 ${SCRIPT_NAME} --model_size ${MODEL_SIZE} --mode forward_backward --block_size 512 --batch_size ${BATCH_SIZE}

echo "完整训练步骤分析完成。"
echo "-----------------------------------"

echo "所有任务已完成。请检查生成的 .pickle 文件。"