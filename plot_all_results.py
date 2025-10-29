#!/usr/bin/env python3
"""
绘制所有实验结果的loss曲线
支持多种可视化方式和对比分析
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np
from datetime import datetime

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_all_csv_data(csv_dir="csv_logs"):
    """加载所有CSV文件的数据"""
    csv_files = glob.glob(f"{csv_dir}/*.csv")
    
    if not csv_files:
        print(f"❌ 在 {csv_dir} 目录中没有找到CSV文件")
        return None
    
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                # 从文件名提取实验信息
                filename = Path(csv_file).stem
                parts = filename.split('_')
                
                # 解析文件名格式: MODE_SCOPE_bsBATCH_qQUERY_lrLR
                mode = parts[0]
                scope = parts[1]
                batch_size = int(parts[2].replace('bs', ''))
                q = parts[3].replace('q', '') if 'q' in parts[3] else 'N/A'
                lr = parts[4].replace('lr', '')
                
                # 添加实验标识
                df['mode'] = mode
                df['scope'] = scope
                df['batch_size'] = batch_size
                df['q'] = q
                df['lr'] = lr
                df['experiment'] = filename
                
                all_data.append(df)
                print(f"✅ 加载: {filename} ({len(df)} 行)")
            else:
                print(f"⚠️  空文件: {csv_file}")
        except Exception as e:
            print(f"❌ 加载失败 {csv_file}: {e}")
    
    if not all_data:
        print("❌ 没有成功加载任何数据")
        return None
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 总共加载了 {len(combined_df)} 行数据，来自 {len(all_data)} 个实验")
    
    return combined_df

def plot_loss_curves(df, output_dir="plots", figsize=(15, 10)):
    """绘制所有loss曲线"""
    if df is None or df.empty:
        print("❌ 没有数据可以绘制")
        return
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 1. 按模式分组的loss曲线
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Loss Curves Comparison', fontsize=16, fontweight='bold')
    
    # 子图1: 所有实验的loss曲线
    ax1 = axes[0, 0]
    for exp in df['experiment'].unique():
        exp_data = df[df['experiment'] == exp]
        ax1.plot(exp_data['step'], exp_data['loss'], 
                label=exp, alpha=0.7, linewidth=1)
    ax1.set_title('All Experiments')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 按模式分组
    ax2 = axes[0, 1]
    for mode in df['mode'].unique():
        mode_data = df[df['mode'] == mode]
        for exp in mode_data['experiment'].unique():
            exp_data = mode_data[mode_data['experiment'] == exp]
            ax2.plot(exp_data['step'], exp_data['loss'], 
                    label=f"{mode}_{exp.split('_')[1]}", alpha=0.7)
    ax2.set_title('By Mode (FO vs ZO)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 按scope分组
    ax3 = axes[1, 0]
    for scope in df['scope'].unique():
        scope_data = df[df['scope'] == scope]
        for exp in scope_data['experiment'].unique():
            exp_data = scope_data[scope_data['experiment'] == exp]
            ax3.plot(exp_data['step'], exp_data['loss'], 
                    label=f"{scope}_{exp.split('_')[0]}", alpha=0.7)
    ax3.set_title('By Scope (Reduced vs Full)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Loss')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 按batch size分组
    ax4 = axes[1, 1]
    for bs in sorted(df['batch_size'].unique()):
        bs_data = df[df['batch_size'] == bs]
        for exp in bs_data['experiment'].unique():
            exp_data = bs_data[bs_data['experiment'] == exp]
            ax4.plot(exp_data['step'], exp_data['loss'], 
                    label=f"bs{bs}_{exp.split('_')[0]}", alpha=0.7)
    ax4.set_title('By Batch Size')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Loss')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curves_overview.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 保存概览图: {output_dir}/loss_curves_overview.png")

def plot_zo_analysis(df, output_dir="plots", figsize=(15, 12)):
    """专门分析ZO实验的结果"""
    if df is None or df.empty:
        return
    
    zo_data = df[df['mode'] == 'ZO'].copy()
    if zo_data.empty:
        print("⚠️  没有ZO实验数据")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('ZO Experiments Analysis', fontsize=16, fontweight='bold')
    
    # 子图1: 不同q值的loss曲线
    ax1 = axes[0, 0]
    for q in sorted(zo_data['q'].unique(), key=lambda x: int(x) if x != 'N/A' else 0):
        if q == 'N/A':
            continue
        q_data = zo_data[zo_data['q'] == q]
        for exp in q_data['experiment'].unique():
            exp_data = q_data[q_data['experiment'] == exp]
            ax1.plot(exp_data['step'], exp_data['loss'], 
                    label=f"q={q}", alpha=0.7)
    ax1.set_title('Loss by Query Budget (q)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 不同batch size的loss曲线
    ax2 = axes[0, 1]
    for bs in sorted(zo_data['batch_size'].unique()):
        bs_data = zo_data[zo_data['batch_size'] == bs]
        for exp in bs_data['experiment'].unique():
            exp_data = bs_data[bs_data['experiment'] == exp]
            ax2.plot(exp_data['step'], exp_data['loss'], 
                    label=f"bs={bs}", alpha=0.7)
    ax2.set_title('Loss by Batch Size')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 不同scope的loss曲线
    ax3 = axes[0, 2]
    for scope in zo_data['scope'].unique():
        scope_data = zo_data[zo_data['scope'] == scope]
        for exp in scope_data['experiment'].unique():
            exp_data = scope_data[scope_data['experiment'] == exp]
            ax3.plot(exp_data['step'], exp_data['loss'], 
                    label=f"scope={scope}", alpha=0.7)
    ax3.set_title('Loss by Scope')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 最终loss vs q值
    ax4 = axes[1, 0]
    final_losses = zo_data.groupby(['q', 'batch_size', 'scope'])['loss'].last().reset_index()
    for scope in final_losses['scope'].unique():
        scope_data = final_losses[final_losses['scope'] == scope]
        for bs in scope_data['batch_size'].unique():
            bs_data = scope_data[scope_data['batch_size'] == bs]
            q_values = [int(q) for q in bs_data['q'] if q != 'N/A']
            losses = bs_data[bs_data['q'] != 'N/A']['loss'].values
            if len(q_values) == len(losses):
                ax4.plot(q_values, losses, 'o-', label=f"bs={bs}, scope={scope}")
    ax4.set_title('Final Loss vs Query Budget')
    ax4.set_xlabel('Query Budget (q)')
    ax4.set_ylabel('Final Loss')
    ax4.legend()
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 梯度范数变化
    ax5 = axes[1, 1]
    for exp in zo_data['experiment'].unique():
        exp_data = zo_data[zo_data['experiment'] == exp]
        if 'grad_norm' in exp_data.columns:
            ax5.plot(exp_data['step'], exp_data['grad_norm'], 
                    label=exp, alpha=0.7)
    ax5.set_title('Gradient Norm Evolution')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Gradient Norm')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # 子图6: 收敛速度分析
    ax6 = axes[1, 2]
    for exp in zo_data['experiment'].unique():
        exp_data = zo_data[zo_data['experiment'] == exp]
        if len(exp_data) > 1:
            initial_loss = exp_data['loss'].iloc[0]
            final_loss = exp_data['loss'].iloc[-1]
            reduction = (initial_loss - final_loss) / initial_loss * 100
            ax6.bar(exp, reduction, alpha=0.7)
    ax6.set_title('Loss Reduction Percentage')
    ax6.set_xlabel('Experiment')
    ax6.set_ylabel('Loss Reduction (%)')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/zo_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 保存ZO分析图: {output_dir}/zo_analysis.png")

def plot_fo_vs_zo_comparison(df, output_dir="plots", figsize=(15, 8)):
    """对比FO和ZO方法"""
    if df is None or df.empty:
        return
    
    # 分离FO和ZO数据
    fo_data = df[df['mode'] == 'FO'].copy()
    zo_data = df[df['mode'] == 'ZO'].copy()
    
    if fo_data.empty or zo_data.empty:
        print("⚠️  需要FO和ZO数据才能进行对比")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('FO vs ZO Comparison', fontsize=16, fontweight='bold')
    
    # 子图1: 平均loss曲线对比
    ax1 = axes[0, 0]
    
    # FO平均曲线
    fo_avg = fo_data.groupby('step')['loss'].mean()
    ax1.plot(fo_avg.index, fo_avg.values, 'b-', label='FO (Average)', linewidth=2)
    
    # ZO平均曲线
    zo_avg = zo_data.groupby('step')['loss'].mean()
    ax1.plot(zo_avg.index, zo_avg.values, 'r-', label='ZO (Average)', linewidth=2)
    
    ax1.set_title('Average Loss Comparison')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 最终loss分布
    ax2 = axes[0, 1]
    fo_final = fo_data.groupby('experiment')['loss'].last()
    zo_final = zo_data.groupby('experiment')['loss'].last()
    
    ax2.hist(fo_final, bins=10, alpha=0.7, label='FO', color='blue')
    ax2.hist(zo_final, bins=10, alpha=0.7, label='ZO', color='red')
    ax2.set_title('Final Loss Distribution')
    ax2.set_xlabel('Final Loss')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 收敛速度对比
    ax3 = axes[1, 0]
    
    # 计算每个实验的收敛速度
    fo_speed = []
    zo_speed = []
    
    for exp in fo_data['experiment'].unique():
        exp_data = fo_data[fo_data['experiment'] == exp]
        if len(exp_data) > 1:
            initial_loss = exp_data['loss'].iloc[0]
            final_loss = exp_data['loss'].iloc[-1]
            reduction = (initial_loss - final_loss) / initial_loss * 100
            fo_speed.append(reduction)
    
    for exp in zo_data['experiment'].unique():
        exp_data = zo_data[zo_data['experiment'] == exp]
        if len(exp_data) > 1:
            initial_loss = exp_data['loss'].iloc[0]
            final_loss = exp_data['loss'].iloc[-1]
            reduction = (initial_loss - final_loss) / initial_loss * 100
            zo_speed.append(reduction)
    
    ax3.boxplot([fo_speed, zo_speed], labels=['FO', 'ZO'])
    ax3.set_title('Convergence Speed Comparison')
    ax3.set_ylabel('Loss Reduction (%)')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 训练稳定性
    ax4 = axes[1, 1]
    
    # 计算loss的标准差（稳定性指标）
    fo_std = fo_data.groupby('step')['loss'].std()
    zo_std = zo_data.groupby('step')['loss'].std()
    
    ax4.plot(fo_std.index, fo_std.values, 'b-', label='FO', linewidth=2)
    ax4.plot(zo_std.index, zo_std.values, 'r-', label='ZO', linewidth=2)
    ax4.set_title('Training Stability (Loss Std)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Loss Standard Deviation')
    ax4.legend()
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fo_vs_zo_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ 保存FO vs ZO对比图: {output_dir}/fo_vs_zo_comparison.png")

def generate_summary_report(df, output_dir="plots"):
    """生成总结报告"""
    if df is None or df.empty:
        return
    
    report_file = f"{output_dir}/experiment_summary.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("实验总结报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 基本统计
        f.write("基本统计:\n")
        f.write(f"  总实验数: {df['experiment'].nunique()}\n")
        f.write(f"  总数据点: {len(df)}\n")
        f.write(f"  FO实验数: {len(df[df['mode'] == 'FO']['experiment'].unique())}\n")
        f.write(f"  ZO实验数: {len(df[df['mode'] == 'ZO']['experiment'].unique())}\n\n")
        
        # 按模式统计
        f.write("按模式统计:\n")
        mode_stats = df.groupby('mode').agg({
            'loss': ['mean', 'std', 'min', 'max'],
            'step': 'max'
        }).round(4)
        f.write(str(mode_stats) + "\n\n")
        
        # 按scope统计
        f.write("按Scope统计:\n")
        scope_stats = df.groupby('scope').agg({
            'loss': ['mean', 'std', 'min', 'max'],
            'step': 'max'
        }).round(4)
        f.write(str(scope_stats) + "\n\n")
        
        # ZO实验的q值分析
        if 'ZO' in df['mode'].values:
            f.write("ZO实验Query Budget分析:\n")
            zo_data = df[df['mode'] == 'ZO']
            q_stats = zo_data.groupby('q').agg({
                'loss': ['mean', 'std', 'min', 'max'],
                'step': 'max'
            }).round(4)
            f.write(str(q_stats) + "\n\n")
        
        # 最佳实验
        f.write("最佳实验 (按最终loss):\n")
        best_experiments = df.groupby('experiment')['loss'].last().sort_values().head(5)
        for i, (exp, loss) in enumerate(best_experiments.items(), 1):
            f.write(f"  {i}. {exp}: {loss:.4f}\n")
    
    print(f"✅ 保存总结报告: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="绘制所有实验结果的loss曲线")
    parser.add_argument("--csv-dir", default="csv_logs", help="CSV文件目录")
    parser.add_argument("--output-dir", default="plots", help="输出图片目录")
    parser.add_argument("--figsize", nargs=2, type=int, default=[15, 10], help="图片大小")
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--overview", action="store_true", help="生成概览图")
    parser.add_argument("--zo-analysis", action="store_true", help="生成ZO分析图")
    parser.add_argument("--comparison", action="store_true", help="生成FO vs ZO对比图")
    parser.add_argument("--summary", action="store_true", help="生成总结报告")
    
    args = parser.parse_args()
    
    # 如果没有指定任何选项，默认生成所有图表
    if not any([args.all, args.overview, args.zo_analysis, args.comparison, args.summary]):
        args.all = True
    
    print("🚀 开始绘制实验结果...")
    
    # 加载数据
    df = load_all_csv_data(args.csv_dir)
    if df is None:
        return
    
    # 创建输出目录
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # 生成图表
    if args.all or args.overview:
        print("\n📊 生成概览图...")
        plot_loss_curves(df, args.output_dir, tuple(args.figsize))
    
    if args.all or args.zo_analysis:
        print("\n📊 生成ZO分析图...")
        plot_zo_analysis(df, args.output_dir, tuple(args.figsize))
    
    if args.all or args.comparison:
        print("\n📊 生成FO vs ZO对比图...")
        plot_fo_vs_zo_comparison(df, args.output_dir, tuple(args.figsize))
    
    if args.all or args.summary:
        print("\n📊 生成总结报告...")
        generate_summary_report(df, args.output_dir)
    
    print(f"\n✅ 所有图表已保存到: {args.output_dir}")
    print("📁 生成的文件:")
    for file in Path(args.output_dir).glob("*.png"):
        print(f"  - {file}")
    for file in Path(args.output_dir).glob("*.txt"):
        print(f"  - {file}")

if __name__ == "__main__":
    main()
