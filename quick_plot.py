#!/usr/bin/env python3
"""
快速绘制所有实验结果的loss曲线
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_plot():
    """加载并绘制所有CSV数据"""
    csv_files = glob.glob("csv_logs_mezo_epochs10/*.csv")
    
    if not csv_files:
        print("❌ 在 csv_logs 目录中没有找到CSV文件")
        return
    
    print(f"📊 找到 {len(csv_files)} 个CSV文件")
    
    # 创建输出目录
    Path("plots").mkdir(exist_ok=True)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('All Experiments Loss Curves', fontsize=16, fontweight='bold')
    
    all_data = []
    
    # 加载所有数据
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                # 从文件名提取信息
                filename = Path(csv_file).stem
                parts = filename.split('_')
                
                mode = parts[0]
                scope = parts[1]
                batch_size = int(parts[2].replace('bs', ''))
                q = parts[3].replace('q', '') if 'q' in parts[3] else 'N/A'
                lr = parts[4].replace('lr', '')
                
                df['mode'] = mode
                df['scope'] = scope
                df['batch_size'] = batch_size
                df['q'] = q
                df['lr'] = lr
                df['experiment'] = filename
                
                all_data.append(df)
                print(f"✅ 加载: {filename}")
        except Exception as e:
            print(f"❌ 加载失败 {csv_file}: {e}")
    
    if not all_data:
        print("❌ 没有成功加载任何数据")
        return
    
    # 合并数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 子图1: 所有实验的loss曲线
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(range(len(combined_df['experiment'].unique())))
    for i, exp in enumerate(combined_df['experiment'].unique()):
        exp_data = combined_df[combined_df['experiment'] == exp]
        ax1.plot(exp_data['step'], exp_data['loss'], 
                label=exp, alpha=0.7, color=colors[i])
    ax1.set_title('All Experiments')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 按模式分组
    ax2 = axes[0, 1]
    for mode in combined_df['mode'].unique():
        mode_data = combined_df[combined_df['mode'] == mode]
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
    for scope in combined_df['scope'].unique():
        scope_data = combined_df[combined_df['scope'] == scope]
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
    for bs in sorted(combined_df['batch_size'].unique()):
        bs_data = combined_df[combined_df['batch_size'] == bs]
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
    plt.savefig("plots/all_loss_curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 保存图片: plots/all_loss_curves.png")
    
    # 打印统计信息
    print(f"\n📊 统计信息:")
    print(f"  总实验数: {combined_df['experiment'].nunique()}")
    print(f"  总数据点: {len(combined_df)}")
    print(f"  FO实验数: {len(combined_df[combined_df['mode'] == 'FO']['experiment'].unique())}")
    print(f"  ZO实验数: {len(combined_df[combined_df['mode'] == 'ZO']['experiment'].unique())}")
    
    # 最佳实验
    print(f"\n🏆 最佳实验 (按最终loss):")
    best_experiments = combined_df.groupby('experiment')['loss'].last().sort_values().head(5)
    for i, (exp, loss) in enumerate(best_experiments.items(), 1):
        print(f"  {i}. {exp}: {loss:.4f}")

if __name__ == "__main__":
    load_and_plot()

