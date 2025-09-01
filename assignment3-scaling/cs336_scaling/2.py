import os
import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- 1. 模拟API和辅助函数 ---

def simulate_chinchilla_loss(C, N, lr):
    """
    模拟一个对计算量、模型大小和学习率都敏感的损失函数。
    这仅用于演示，实际任务中你会调用一个真实的API。
    """
    # 假设的最优关系（基于Chinchilla论文）
    N_true_opt = 3e8 * (C / 1e18)**0.5
    lr_true_opt = 3e-4 
    
    # 偏离最优模型大小的惩罚
    log_dist_penalty = (np.log(N) - np.log(N_true_opt))**2 * 0.2
    # 偏离最优学习率的惩罚
    lr_penalty = (np.log10(lr) - np.log10(lr_true_opt))**2 * 0.1

    # 基础损失随计算量下降
    base_loss = 4.5 * (C / 1e18)**-0.1
    # 增加随机性
    noise = np.random.normal(0, 0.02)
    
    return base_loss + log_dist_penalty + lr_penalty + noise

def call_training_api(model_hyperparams, training_flops):
    """
    模拟调用外部训练API的函数。
    在真实作业中，你需要在这里实现HTTP请求等逻辑。
    """
    n_layer = model_hyperparams['n_layer']
    d_model = model_hyperparams['d_model']
    lr = model_hyperparams['learning_rate']
    
    # 根据作业中的公式估算非嵌入参数数量
    num_params = 12 * n_layer * d_model**2
    
    # 模拟API返回损失
    loss = simulate_chinchilla_loss(training_flops, num_params, lr)
    
    return {
        "compute_budget": training_flops,
        "parameters": num_params,
        "final_loss": loss,
        "hyperparams": model_hyperparams
    }

def power_law(x, A, a):
    """幂律函数 f(x) = A * x^a"""
    return A * np.power(x, a)

def loss_power_law(x, B, b):
    """损失的幂律函数 f(x) = B * x^-b"""
    return B * np.power(x, -b)


# --- 2. 核心功能模块 ---

def find_optimal_learning_rate(total_budget, scan_params):
    """使用一小部分预算来扫描并确定一个较优的学习率。"""
    print("🔬 步骤 1: 开始进行学习率扫描...")
    
    lr_scan_budget = scan_params['scan_budget']
    lr_scan_config = scan_params['model_config']
    learning_rates_to_try = scan_params['lr_candidates']

    num_scans = len(learning_rates_to_try)
    flops_for_lr_scan = num_scans * lr_scan_budget

    if flops_for_lr_scan > total_budget:
        print("错误：用于学习率扫描的预算不足！")
        return None, 0

    best_lr, min_loss = None, float('inf')
    
    print(f"将使用 {flops_for_lr_scan:.1e} FLOPs ({num_scans} 次运行 @ {lr_scan_budget:.1e} FLOPs/次)")
    
    for lr in learning_rates_to_try:
        hyperparams = {**lr_scan_config, 'learning_rate': lr, 'batch_size': 256}
        result = call_training_api(hyperparams, lr_scan_budget)
        print(f"  测试 LR = {lr:.1e}, 得到 Loss = {result['final_loss']:.4f}")
        
        if result['final_loss'] < min_loss:
            min_loss = result['final_loss']
            best_lr = lr
            
    print(f"✅ 学习率扫描完成. 最佳学习率: {best_lr:.1e} (Loss: {min_loss:.4f})")
    return best_lr, flops_for_lr_scan

def collect_isoflops_data(total_budget, flops_already_used, best_lr, compute_levels, model_configs):
    """使用剩余预算收集IsoFLOPs数据。"""
    print("\n🚀 步骤 2: 开始查询API，进行IsoFLOPs数据收集...")
    
    api_results = []
    total_flops_used = flops_already_used

    for C in compute_levels:
        print(f"\n探索计算预算 C = {C:.1e} FLOPs")
        for config in model_configs.get(C, []):
            if total_flops_used + C > total_budget:
                print("⚠️ 预算不足，停止在该计算水平的进一步查询。")
                break
            
            hyperparams = {
                'n_layer': config['n_layer'],
                'd_model': config['d_model'],
                'learning_rate': best_lr,
                'batch_size': 256
            }
            
            result = call_training_api(hyperparams, C)
            api_results.append(result)
            total_flops_used += C
            
            print(f"  查询: N={result['parameters']:.2e}, Loss={result['final_loss']:.4f}")
    
    print(f"\n✅ API查询完成. 总计使用 FLOPs: {total_flops_used:.2e}")
    return pd.DataFrame(api_results)

def analyze_and_fit(df):
    """分析数据，找到最优点，并拟合缩放法则。"""
    print("\n📊 步骤 3: 分析数据并拟合缩放法则...")
    if df.empty:
        print("错误：没有收集到数据，无法进行分析。")
        return None, None

    # 1. 找到每个IsoFLOPs曲线的最低点
    optimal_points = df.loc[df.groupby('compute_budget')['final_loss'].idxmin()]
    optimal_points = optimal_points.sort_values('compute_budget').reset_index(drop=True)
    
    # 2. 计算最优数据量 D_opt = C / (6 * N_opt)
    optimal_points['tokens'] = optimal_points['compute_budget'] / (6 * optimal_points['parameters'])
    
    print("找到的最优 IsoFLOPs 数据点:")
    print(optimal_points[['compute_budget', 'parameters', 'tokens', 'final_loss']])

    C, N, D, L = (
        optimal_points['compute_budget'].values,
        optimal_points['parameters'].values,
        optimal_points['tokens'].values,
        optimal_points['final_loss'].values,
    )

    # 3. 拟合三个缩放法则
    try:
        popt_n, _ = curve_fit(power_law, C, N, p0=[1e9, 0.5])
        popt_d, _ = curve_fit(power_law, C, D, p0=[1e12, 0.5])
        popt_l, _ = curve_fit(loss_power_law, L, C, p0=[4.0, 0.1]) # 注意这里L和C位置反了以拟合L(C)
        popt_l, _ = curve_fit(loss_power_law, C, L, p0=[4.0, 0.1])
    except RuntimeError as e:
        print(f"拟合失败: {e}. 请检查数据点是否足够或形态是否正确。")
        return optimal_points, None

    fit_params = {'N': popt_n, 'D': popt_d, 'L': popt_l}
    print("✅ 缩放法则拟合完成。")
    return optimal_points, fit_params

def plot_scaling_law(data, fit_params, C_pred_range, result_dir):
    """可视化缩放法则拟合效果。"""
    print("\n📈 步骤 4: 生成并保存可视化图表...")
    
    # Model Size Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data['compute_budget'], data['parameters'], label='Optimal Points (Data)')
    plt.plot(C_pred_range, power_law(C_pred_range, *fit_params['N']), 
             label=f'Fit: N(C) = {fit_params["N"][0]:.2e} * C^{fit_params["N"][1]:.2f}', color='r')
    plt.xscale('log'); plt.yscale('log'); plt.title('Model Size Scaling Law');
    plt.xlabel('Compute Budget (FLOPs)'); plt.ylabel('Optimal Model Size (Parameters)');
    plt.grid(True); plt.legend();
    plt.savefig(os.path.join(result_dir, 'model_size_scaling.png'))

    # Dataset Size Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data['compute_budget'], data['tokens'], label='Optimal Points (Data)')
    plt.plot(C_pred_range, power_law(C_pred_range, *fit_params['D']), 
             label=f'Fit: D(C) = {fit_params["D"][0]:.2e} * C^{fit_params["D"][1]:.2f}', color='b')
    plt.xscale('log'); plt.yscale('log'); plt.title('Dataset Size Scaling Law');
    plt.xlabel('Compute Budget (FLOPs)'); plt.ylabel('Optimal Dataset Size (Tokens)');
    plt.grid(True); plt.legend();
    plt.savefig(os.path.join(result_dir, 'dataset_size_scaling.png'))

    # Loss Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data['compute_budget'], data['final_loss'], label='Minimum Loss (Data)')
    plt.plot(C_pred_range, loss_power_law(C_pred_range, *fit_params['L']), 
             label=f'Fit: L(C) = {fit_params["L"][0]:.2f} * C^{-fit_params["L"][1]:.2f}', color='g')
    plt.xscale('log'); plt.yscale('log'); plt.title('Loss Scaling Law');
    plt.xlabel('Compute Budget (FLOPs)'); plt.ylabel('Minimum Loss');
    plt.grid(True); plt.legend();
    plt.savefig(os.path.join(result_dir, 'loss_scaling.png'))
    print("✅ 图表已保存。")

def predict_and_report(C_target, fit_params):
    """使用拟合的法则进行外推预测并推荐超参数。"""
    print("\n📋 步骤 5: 外推预测并生成最终报告...")

    N_pred = power_law(C_target, *fit_params['N'])
    D_pred = power_law(C_target, *fit_params['D'])
    L_pred = loss_power_law(C_target, *fit_params['L'])

    print(f"\n--- 最终预测 (目标计算量 C = {C_target:.0e} FLOPs) ---")
    print(f"  预测的最优模型大小 (N_pred): {N_pred:.3e} (~{N_pred/1e9:.2f}B parameters)")
    print(f"  预测的最优数据量   (D_pred): {D_pred:.3e} (~{D_pred/1e12:.2f}T tokens)")
    print(f"  预测的最低训练损失 (L_pred): {L_pred:.4f}")

    # 推荐最终超参数
    target_n_layer = 80  # 基于 Llama 等模型的常见选择
    pred_d_model = math.sqrt(N_pred / (12 * target_n_layer))
    d_model_final = round(pred_d_model / 64) * 64
    n_head_final = d_model_final // 64
    final_params = 12 * target_n_layer * d_model_final**2

    print("\n--- 建议的最终超参数 ---")
    print(f"  n_layer: {target_n_layer}")
    print(f"  d_model: {d_model_final}")
    print(f"  n_head: {n_head_final}")
    print(f"  (对应的模型大小约为: {final_params:.3e}, 与预测值 {N_pred:.3e} 接近。)")
    

# --- 3. 主执行流程 ---

def main():
    # --- 作业参数定义 ---
    TOTAL_BUDGET = 2e18
    TARGET_COMPUTE = 1e19
    RESULT_DIR = 'full_assignment_results'
    os.makedirs(RESULT_DIR, exist_ok=True)

    # --- LR 扫描配置 ---
    lr_scan_parameters = {
        'scan_budget': 1e17,
        'model_config': {'n_layer': 24, 'd_model': 1024},
        'lr_candidates': [1e-3, 3e-4, 1e-4, 3e-5]
    }

    # --- IsoFLOPs 扫描配置 ---
    compute_levels_to_scan = [1e17, 3e17, 6e17, 1e18]
    model_configs_to_scan = {
        1e17: [{'n_layer': 24, 'd_model': d} for d in [512, 768, 1024, 1280, 1536]],
        3e17: [{'n_layer': 32, 'd_model': d} for d in [768, 1024, 1280, 1536, 2048]],
        6e17: [{'n_layer': 40, 'd_model': d} for d in [1024, 1280, 1536, 2048, 2560]],
        1e18: [{'n_layer': 48, 'd_model': d} for d in [1280, 1536, 2048, 2560, 3072]]
    }

    # 步骤 1: 确定最优学习率
    best_lr, flops_used_for_lr = find_optimal_learning_rate(TOTAL_BUDGET, lr_scan_parameters)
    if best_lr is None: return

    # 步骤 2: 收集IsoFLOPs数据
    all_runs_df = collect_isoflops_data(
        TOTAL_BUDGET, flops_used_for_lr, best_lr,
        compute_levels_to_scan, model_configs_to_scan
    )
    all_runs_df.to_csv(os.path.join(RESULT_DIR, 'all_api_runs.csv'), index=False)

    # 步骤 3: 分析数据并拟合
    optimal_points_df, fit_parameters = analyze_and_fit(all_runs_df)
    if optimal_points_df is None or fit_parameters is None: return
    optimal_points_df.to_csv(os.path.join(RESULT_DIR, 'optimal_points.csv'), index=False)

    # 步骤 4: 可视化
    C_plot_range = np.logspace(
        np.log10(optimal_points_df['compute_budget'].min()), 
        np.log10(TARGET_COMPUTE * 10), 100
    )
    plot_scaling_law(optimal_points_df, fit_parameters, C_plot_range, RESULT_DIR)

    # 步骤 5: 预测并报告
    predict_and_report(TARGET_COMPUTE, fit_parameters)
    print(f"\n🎉 所有流程执行完毕！结果保存在 '{RESULT_DIR}' 文件夹中。")

if __name__ == '__main__':
    main()