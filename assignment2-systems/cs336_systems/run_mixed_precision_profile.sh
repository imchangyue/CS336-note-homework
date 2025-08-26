#!/bin/bash

# --- 脚本配置 ---

# 根据你的实际路径修改此变量
PYTHON_SCRIPT="/home/code_backup/code/cs336/assignment2-systems/cs336_systems/mixed_precision_benchmark.py"

# 定义模型尺寸和上下文长度
MODEL_SIZES=("small" "medium" "large" "xl" "2.7B")
BLOCK_SIZES=(128 256 512 1024)

# CSV结果文件配置
CSV_RESULTS_FILE="mixed_precision_results.csv"

# --- 创建结果文件夹 ---

RESULT_DIR="mixed_precision_results"
echo "正在创建结果文件夹：${RESULT_DIR}"
mkdir -p "${RESULT_DIR}"

# --- 清空或备份现有的CSV结果文件 ---
if [ -f "${CSV_RESULTS_FILE}" ]; then
    BACKUP_FILE="${CSV_RESULTS_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "备份现有结果文件为：${BACKUP_FILE}"
    cp "${CSV_RESULTS_FILE}" "${BACKUP_FILE}"
fi

# 创建CSV文件头
echo "model_size,block_size,batch_size,mode,precision,mean_time_ms,std_dev_ms,speedup" > "${CSV_RESULTS_FILE}"
echo "CSV结果文件已初始化：${CSV_RESULTS_FILE}"

# --- 运行混合精度基准测试函数 ---

run_benchmark_and_log() {
    local model=$1
    local block_size=$2
    local batch_size=$3
    local mode=$4
    
    echo "正在运行：${model} 模型，上下文长度 ${block_size}，批量大小 ${batch_size}，模式：${mode}"
    
    # 运行基准测试并捕获输出
    output_file="${RESULT_DIR}/output_${model}_${block_size}_${mode}.log"
    
    python3 "${PYTHON_SCRIPT}" \
        --model_size "${model}" \
        --mode "${mode}" \
        --block_size "${block_size}" \
        --batch_size "${batch_size}" \
        --warmup_steps 3 \
        --measure_steps 5 \
        2>&1 | tee "${output_file}"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "成功完成：${model} 模型，上下文长度 ${block_size}，模式：${mode}"
        
        # 解析输出并写入CSV
        parse_and_append_results "${output_file}" "${model}" "${block_size}" "${batch_size}" "${mode}"
    else
        echo "警告：${model} 模型，上下文长度 ${block_size}，模式：${mode} 执行失败（可能因显存不足）"
        # 记录失败结果
        echo "${model},${block_size},${batch_size},${mode},FP32,OOM,OOM,N/A" >> "${CSV_RESULTS_FILE}"
        echo "${model},${block_size},${batch_size},${mode},BF16,OOM,OOM,N/A" >> "${CSV_RESULTS_FILE}"
    fi
    
    echo "--------------------------------------------------"
}

# --- 解析结果并追加到CSV ---
parse_and_append_results() {
    local log_file=$1
    local model=$2
    local block_size=$3
    local batch_size=$4
    local mode=$5
    
    echo "解析日志文件: $log_file"
    
    # 提取FP32结果 - 查找"FP32 (Full Precision)"之后的结果
    fp32_mean=$(grep -A 10 "FP32 (Full Precision)" "$log_file" | grep "平均每步耗时" | head -1 | grep -oE '[0-9]+\.[0-9]+')
    fp32_std=$(grep -A 10 "FP32 (Full Precision)" "$log_file" | grep "标准差" | head -1 | grep -oE '[0-9]+\.[0-9]+')
    
    # 提取BF16结果 - 查找"BF16 (Mixed Precision)"之后的结果
    bf16_mean=$(grep -A 10 "BF16 (Mixed Precision)" "$log_file" | grep "平均每步耗时" | head -1 | grep -oE '[0-9]+\.[0-9]+')
    bf16_std=$(grep -A 10 "BF16 (Mixed Precision)" "$log_file" | grep "标准差" | head -1 | grep -oE '[0-9]+\.[0-9]+')
    
    # 提取加速比
    speedup=$(grep "混合精度加速比" "$log_file" | grep -oE '[0-9]+\.[0-9]+')
    
    # 检查是否支持BF16
    bf16_supported=$(grep -q "BF16 (Mixed Precision)" "$log_file" && echo "yes" || echo "no")
    
    echo "解析结果: FP32_mean=$fp32_mean, FP32_std=$fp32_std, BF16_mean=$bf16_mean, BF16_std=$bf16_std, speedup=$speedup, bf16_supported=$bf16_supported"
    
    # 写入CSV - FP32结果
    if [ -n "$fp32_mean" ] && [ -n "$fp32_std" ]; then
        echo "${model},${block_size},${batch_size},${mode},FP32,${fp32_mean},${fp32_std},1.00" >> "${CSV_RESULTS_FILE}"
        echo "已记录FP32结果"
    else
        echo "${model},${block_size},${batch_size},${mode},FP32,ERROR,ERROR,N/A" >> "${CSV_RESULTS_FILE}"
        echo "FP32结果记录失败"
    fi
    
    # 写入CSV - BF16结果
    if [ "$bf16_supported" = "yes" ]; then
        if [ -n "$bf16_mean" ] && [ -n "$bf16_std" ]; then
            if [ -n "$speedup" ]; then
                echo "${model},${block_size},${batch_size},${mode},BF16,${bf16_mean},${bf16_std},${speedup}" >> "${CSV_RESULTS_FILE}"
                echo "已记录BF16结果，加速比: ${speedup}x"
            else
                # 如果没有找到加速比，手动计算
                if [ -n "$fp32_mean" ]; then
                    calc_speedup=$(python3 -c "print(f'{float('$fp32_mean')/float('$bf16_mean'):.2f}')" 2>/dev/null || echo "N/A")
                    echo "${model},${block_size},${batch_size},${mode},BF16,${bf16_mean},${bf16_std},${calc_speedup}" >> "${CSV_RESULTS_FILE}"
                    echo "已记录BF16结果，计算的加速比: ${calc_speedup}x"
                else
                    echo "${model},${block_size},${batch_size},${mode},BF16,${bf16_mean},${bf16_std},N/A" >> "${CSV_RESULTS_FILE}"
                    echo "已记录BF16结果，无法计算加速比"
                fi
            fi
        else
            echo "${model},${block_size},${batch_size},${mode},BF16,ERROR,ERROR,N/A" >> "${CSV_RESULTS_FILE}"
            echo "BF16结果解析失败"
        fi
    else
        echo "${model},${block_size},${batch_size},${mode},BF16,UNSUPPORTED,UNSUPPORTED,N/A" >> "${CSV_RESULTS_FILE}"
        echo "BF16不支持"
    fi
}

# --- 检查Python脚本是否存在 ---
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "错误：找不到Python脚本 '${PYTHON_SCRIPT}'"
    echo "请确保脚本存在并修改PYTHON_SCRIPT变量指向正确路径"
    exit 1
fi

# --- 检查CUDA和BF16支持 ---
echo "正在检查CUDA和BF16支持..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
else:
    print('警告：未检测到CUDA，将只运行FP32测试')
"

echo ""

# --- 运行仅前向传播基准测试 ---

echo "=========================================="
echo "--- 开始运行仅前向传播混合精度基准测试 ---"
echo "=========================================="

for model in "${MODEL_SIZES[@]}"; do
    for block_size in "${BLOCK_SIZES[@]}"; do
        # 确定批量大小（大模型使用较小批量大小避免OOM）
        batch_size=4
        if [[ "${model}" == "xl" || "${model}" == "2.7B" ]]; then
            batch_size=1
        fi
        
        run_benchmark_and_log "${model}" "${block_size}" "${batch_size}" "forward"
    done
done

# --- 运行前向+后向传播基准测试 ---

echo ""
echo "=========================================="
echo "--- 开始运行前向+后向传播混合精度基准测试 ---"
echo "=========================================="

for model in "${MODEL_SIZES[@]}"; do
    for block_size in "${BLOCK_SIZES[@]}"; do
        # 确定批量大小
        batch_size=4
        if [[ "${model}" == "xl" || "${model}" == "2.7B" ]]; then
            batch_size=1
        fi
        
        run_benchmark_and_log "${model}" "${block_size}" "${batch_size}" "forward_backward"
    done
done

# --- 生成简单的分析报告 ---

echo ""
echo "=========================================="
echo "正在生成分析报告..."
echo "=========================================="

REPORT_FILE="${RESULT_DIR}/mixed_precision_report.txt"

{
    echo "混合精度基准测试报告"
    echo "生成时间: $(date)"
    echo "========================================"
    echo ""
    
    echo "测试配置："
    echo "- 模型尺寸: ${MODEL_SIZES[*]}"
    echo "- 上下文长度: ${BLOCK_SIZES[*]}"
    echo "- 测试模式: forward, forward_backward"
    echo ""
    
    echo "平均加速比统计（按模型尺寸）："
    for model in "${MODEL_SIZES[@]}"; do
        avg_speedup=$(awk -F, -v model="$model" '$1==model && $5=="BF16" && $8!="N/A" && $8!="UNSUPPORTED" {sum+=$8; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' "${CSV_RESULTS_FILE}")
        printf "%-10s: %s\n" "$model" "$avg_speedup"
    done
    
    echo ""
    echo "详细结果请查看: ${CSV_RESULTS_FILE}"
    
} > "${REPORT_FILE}"

cat "${REPORT_FILE}"

# --- 总结 ---
echo ""
echo "=========================================="
echo "所有混合精度基准测试已完成！"
echo "=========================================="
echo "结果文件："
echo "- CSV结果: '${CSV_RESULTS_FILE}'"
echo "- 日志文件: '${RESULT_DIR}/' 目录中"
echo "- 分析报告: '${REPORT_FILE}'"
echo ""

# 显示CSV结果文件的基本统计信息
if [ -f "${CSV_RESULTS_FILE}" ]; then
    echo "CSV结果文件统计："
    echo "- 总行数：$(wc -l < "${CSV_RESULTS_FILE}")"
    echo "- 文件大小：$(ls -lh "${CSV_RESULTS_FILE}" | awk '{print $5}')"
    echo ""
    echo "样例结果（前10行）："
    head -n 10 "${CSV_RESULTS_FILE}"
    echo ""
    echo "成功测试数量：$(awk -F, '$6!="OOM" && $6!="ERROR" && $6!="UNSUPPORTED"' "${CSV_RESULTS_FILE}" | wc -l)"
    echo "失败测试数量：$(awk -F, '$6=="OOM" || $6=="ERROR"' "${CSV_RESULTS_FILE}" | wc -l)"
fi

echo "=========================================="