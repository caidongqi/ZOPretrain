#!/usr/bin/env python3
"""
快速测试ZO vs FO在SST-2上的表现
"""

import subprocess
import time

def run_experiment(mode, lr, epochs=1, batch_size=16, q=2, epsilon=1e-4):
    """运行单个实验"""
    cmd = [
        "python", "zo_sst_finetune.py",
        "--mode", mode,
        "--lr", str(lr),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--freeze_encoder",
        "--csv_file", f"results/sst2_{mode}_test.csv",
        "--log_interval", "50"
    ]
    
    if mode == "ZO":
        cmd.extend(["--q", str(q), "--epsilon", str(epsilon)])
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {mode} completed in {duration:.1f}s")
            # 提取最终准确率
            lines = result.stdout.split('\n')
            for line in lines:
                if "Final validation accuracy:" in line:
                    acc = float(line.split(":")[1].strip())
                    print(f"   Final accuracy: {acc:.4f}")
                    return acc
        else:
            print(f"❌ {mode} failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"⏰ {mode} timed out")
        return None

def main():
    print("🚀 Testing ZO vs FO on SST-2")
    print("=" * 50)
    
    # 测试参数
    experiments = [
        ("FO", 5e-4, "First-Order baseline"),
        ("ZO", 1e-3, "ZO with q=2, ε=1e-4"),
        ("ZO", 2e-3, "ZO with q=2, ε=1e-4 (higher LR)"),
        ("ZO", 1e-3, "ZO with q=4, ε=1e-4"),
    ]
    
    results = []
    
    for mode, lr, desc in experiments:
        print(f"\n📊 {desc}")
        print("-" * 30)
        
        q = 4 if "q=4" in desc else 2
        acc = run_experiment(mode, lr, q=q)
        
        if acc is not None:
            results.append((mode, lr, q, acc, desc))
    
    # 总结结果
    print(f"\n📋 Results Summary")
    print("=" * 50)
    print(f"{'Mode':<6} {'LR':<8} {'Q':<3} {'Accuracy':<10} {'Description'}")
    print("-" * 50)
    
    for mode, lr, q, acc, desc in results:
        print(f"{mode:<6} {lr:<8.1e} {q:<3} {acc:<10.4f} {desc}")
    
    # 找出最佳结果
    if results:
        best = max(results, key=lambda x: x[3])
        print(f"\n🏆 Best result: {best[0]} with LR={best[1]:.1e}, Q={best[2]}, Acc={best[3]:.4f}")

if __name__ == "__main__":
    main()
