import json
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm # 导入字体管理器

# --- 在此添加 Matplotlib 字体配置 ---
# 这是一个在Linux环境下常用的免费开源中文字体，你可以从网上下载
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc' # 假设字体文件在此路径
# 如果你的系统没有这个字体，请自行下载并更新路径
# 字体下载链接示例: https://mirrors.tuna.tsinghua.edu.cn/ubuntu/pool/universe/w/wqy-zenhei/wqy-zenhei_0.9.45-6_all.deb
# 下载后解压出 .ttc 文件，并放在一个方便的位置

# 检查字体文件是否存在，如果不存在则跳过字体配置
if os.path.exists(font_path):
    # 将字体添加到 Matplotlib 的缓存中
    fm.fontManager.addfont(font_path)
    # 配置全局字体
    plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
    print("Matplotlib 中文字体配置成功。")
else:
    print("警告：未找到中文字体文件，图表中的中文可能无法正常显示。")
# ----------------------------------

# 拟合的幂律函数模型： y = A * x^b
def power_law(x, A, b):
    return A * np.power(x, b)

# 拟合的对数线性模型： log(y) = log(A) + b * log(x)
def log_linear_fit(x, log_A, b):
    return log_A + b * x

def main():
    # 1. 创建结果文件夹
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"创建了 '{result_dir}' 文件夹。")

    # 2. 加载数据
    try:
        with open('data/isoflops_curves.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: data/isoflops_curves.json not found.")
        return

    df = pd.DataFrame(data)

    # 3. 按计算预算分组并找到每组中的最优模型
    optimal_runs = df.loc[df.groupby('compute_budget')['final_loss'].idxmin()]
    
    C_opt_values = optimal_runs['compute_budget'].values
    N_opt_values = optimal_runs['parameters'].values
    D_opt_values = C_opt_values / (6 * N_opt_values)

    print("找到的最优 IsoFLOPs 数据点:")
    print("---------------------------------")
    print(optimal_runs[['compute_budget', 'parameters', 'final_loss']])
    print("---------------------------------")
    
    optimal_runs.to_csv(os.path.join(result_dir, 'optimal_data_points.csv'), index=False)
    print(f"最优数据点已保存到 '{result_dir}/optimal_data_points.csv'。")

    # ---
    # Part 1: 模型大小 (N) 缩放法则拟合与预测
    # ---
    print("正在拟合模型大小的缩放法则...")
    log_C = np.log(C_opt_values)
    log_N = np.log(N_opt_values)
    params_n, _ = curve_fit(log_linear_fit, log_C, log_N)
    log_A_n, b_n = params_n
    A_n = np.exp(log_A_n)

    plt.figure(figsize=(10, 6))
    plt.scatter(C_opt_values, N_opt_values, color='blue', label='数据点: N_opt vs C_i')
    C_extrapolate = np.logspace(np.log10(C_opt_values.min()), 24, 100)
    N_predicted = power_law(C_extrapolate, A_n, b_n)
    plt.plot(C_extrapolate, N_predicted, color='red', linestyle='--', label=f'拟合曲线: $N \\propto C^{{{b_n:.4f}}}$')
    
    plt.title('计算预算与最优模型大小的缩放法则')
    plt.xlabel('计算预算 C (FLOPs)')
    plt.ylabel('最优模型大小 N (参数量)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    
    plt.savefig(os.path.join(result_dir, 'model_size_scaling_law.png'))
    plt.close()
    print(f"图表已保存到 '{result_dir}/model_size_scaling_law.png'。")

    N_pred_23 = power_law(1e23, A_n, b_n)
    N_pred_24 = power_law(1e24, A_n, b_n)
    
    with open(os.path.join(result_dir, 'predictions.txt'), 'w') as f:
        f.write("--- 模型大小预测结果 ---\n")
        f.write(f"对于 10^23 FLOPs 预算, 预测最优模型大小: {N_pred_23:.2e}\n")
        f.write(f"对于 10^24 FLOPs 预算, 预测最优模型大小: {N_pred_24:.2e}\n")

    # ---
    # Part 2: 数据集大小 (D) 缩放法则拟合与预测
    # ---
    print("正在拟合数据集大小的缩放法则...")
    log_D = np.log(D_opt_values)
    params_d, _ = curve_fit(log_linear_fit, log_C, log_D)
    log_A_d, b_d = params_d
    A_d = np.exp(log_A_d)

    plt.figure(figsize=(10, 6))
    plt.scatter(C_opt_values, D_opt_values, color='green', label='数据点: D_opt vs C_i')
    D_predicted = power_law(C_extrapolate, A_d, b_d)
    plt.plot(C_extrapolate, D_predicted, color='red', linestyle='--', label=f'拟合曲线: $D \\propto C^{{{b_d:.4f}}}$')

    plt.title('计算预算与最优数据集大小的缩放法则')
    plt.xlabel('计算预算 C (FLOPs)')
    plt.ylabel('最优数据集大小 D (Token)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    
    plt.savefig(os.path.join(result_dir, 'dataset_size_scaling_law.png'))
    plt.close()
    print(f"图表已保存到 '{result_dir}/dataset_size_scaling_law.png'。")

    D_pred_23 = power_law(1e23, A_d, b_d)
    D_pred_24 = power_law(1e24, A_d, b_d)

    with open(os.path.join(result_dir, 'predictions.txt'), 'a') as f:
        f.write("\n--- 数据集大小预测结果 ---\n")
        f.write(f"对于 10^23 FLOPs 预算, 预测最优数据集大小: {D_pred_23:.2e}\n")
        f.write(f"对于 10^24 FLOPs 预算, 预测最优数据集大小: {D_pred_24:.2e}\n")
    
    print(f"所有预测结果已追加到 '{result_dir}/predictions.txt'。")

if __name__ == "__main__":
    main()