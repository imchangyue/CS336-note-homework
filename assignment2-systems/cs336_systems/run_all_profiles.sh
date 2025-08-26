#!/bin/bash

# --- 脚本配置 ---

# 根据你的实际路径修改此变量
PYTHON_SCRIPT="/home/code_backup/code/cs336/assignment2-systems/cs336_systems/benchmark_profile.py"

# nsys可执行文件路径
NSYS_PATH="/opt/nvidia/nsight-systems/2025.5.1/target-linux-x64/nsys"

# 定义模型尺寸和上下文长度
MODEL_SIZES=("small" "medium" "large" "xl" "2.7B")
BLOCK_SIZES=(128 256 512 1024)

# CSV结果文件配置
CSV_RESULTS_FILE="benchmark_results.csv"

# --- 创建结果文件夹 ---

RESULT_DIR="result"
echo "正在创建结果文件夹：${RESULT_DIR}"
mkdir -p "${RESULT_DIR}"

# --- 清空或备份现有的CSV结果文件 ---
if [ -f "${CSV_RESULTS_FILE}" ]; then
    BACKUP_FILE="${CSV_RESULTS_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "备份现有结果文件为：${BACKUP_FILE}"
    cp "${CSV_RESULTS_FILE}" "${BACKUP_FILE}"
    # 清空现有文件以准备新的测试结果
    > "${CSV_RESULTS_FILE}"
fi

# --- 运行前向传播剖析 ---

echo "--- 开始运行前向传播剖析 ---"
for model in "${MODEL_SIZES[@]}"; do
    for block_size in "${BLOCK_SIZES[@]}"; do
        echo "正在运行：${model} 模型，上下文长度 ${block_size}，模式：forward"
        
        # 确定批量大小
        batch_size=4
        if [[ "${model}" == "xl" || "${model}" == "2.7B" ]]; then
            batch_size=1
        fi

        # 生成nsys配置文件输出路径
        nsys_output="${RESULT_DIR}/forward_${model}_${block_size}"
        
        echo "  - nsys输出：${nsys_output}.nsys-rep"
        echo "  - CSV结果追加到：${CSV_RESULTS_FILE}"
        
        # 运行带性能分析的基准测试
        ${NSYS_PATH} profile --trace=cuda,cudnn,cublas,nvtx \
            -o "${nsys_output}" \
            python3 "${PYTHON_SCRIPT}" \
                --model_size "${model}" \
                --mode forward \
                --block_size "${block_size}" \
                --batch_size "${batch_size}" \
                --output_file "${CSV_RESULTS_FILE}" \
                --append \
                --write_per_result
        
        # 检查上一个命令的退出状态
        if [ $? -ne 0 ]; then
            echo "警告：${model} 模型，上下文长度 ${block_size} 的前向传播剖析可能因显存不足而失败。"
            # 即使失败也尝试记录OOM结果到CSV
            echo "尝试记录OOM结果..."
            python3 "${PYTHON_SCRIPT}" \
                --model_size "${model}" \
                --mode forward \
                --block_size "${block_size}" \
                --batch_size "${batch_size}" \
                --output_file "${CSV_RESULTS_FILE}" \
                --append \
                --write_per_result 2>/dev/null || true
        fi
        echo "--------------------------------------------------"
    done
done

# --- 运行完整训练步骤剖析 ---

echo "--- 开始运行完整训练步骤剖析 ---"
for model in "${MODEL_SIZES[@]}"; do
    for block_size in "${BLOCK_SIZES[@]}"; do
        echo "正在运行：${model} 模型，上下文长度 ${block_size}，模式：full_training_step"

        # 确定批量大小
        batch_size=4
        if [[ "${model}" == "xl" || "${model}" == "2.7B" ]]; then
            batch_size=1
        fi

        # 生成nsys配置文件输出路径
        nsys_output="${RESULT_DIR}/train_step_${model}_${block_size}"
        
        echo "  - nsys输出：${nsys_output}.nsys-rep"
        echo "  - CSV结果追加到：${CSV_RESULTS_FILE}"

        # 运行带性能分析的基准测试
        ${NSYS_PATH} profile --trace=cuda,cudnn,cublas,nvtx \
            -o "${nsys_output}" \
            python3 "${PYTHON_SCRIPT}" \
                --model_size "${model}" \
                --mode full_training_step \
                --block_size "${block_size}" \
                --batch_size "${batch_size}" \
                --output_file "${CSV_RESULTS_FILE}" \
                --append \
                --write_per_result

        if [ $? -ne 0 ]; then
            echo "警告：${model} 模型，上下文长度 ${block_size} 的完整训练步骤剖析可能因显存不足而失败。"
            # 即使失败也尝试记录OOM结果到CSV
            echo "尝试记录OOM结果..."
            python3 "${PYTHON_SCRIPT}" \
                --model_size "${model}" \
                --mode full_training_step \
                --block_size "${block_size}" \
                --batch_size "${batch_size}" \
                --output_file "${CSV_RESULTS_FILE}" \
                --append \
                --write_per_result 2>/dev/null || true
        fi
        echo "--------------------------------------------------"
    done
done

# --- 总结 ---
echo ""
echo "=========================================="
echo "所有剖析任务已完成！"
echo "- .nsys-rep 文件已存放在 '${RESULT_DIR}' 文件夹中"
echo "- 基准测试结果已保存到 '${CSV_RESULTS_FILE}'"
echo ""

# 显示CSV结果文件的基本统计信息
if [ -f "${CSV_RESULTS_FILE}" ]; then
    echo "CSV结果文件统计："
    echo "- 总行数：$(wc -l < "${CSV_RESULTS_FILE}")"
    echo "- 文件大小：$(ls -lh "${CSV_RESULTS_FILE}" | awk '{print $5}')"
    echo ""
    echo "最近几行结果："
    tail -n 5 "${CSV_RESULTS_FILE}"
fi

echo "=========================================="