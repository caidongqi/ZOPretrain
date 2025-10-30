# ZO 优化器扫描 - 项目结构

## 📁 项目文件组织

### 🎯 新增/修改的核心文件

```
zo-test/
├── 🔧 CORE SCRIPTS (核心脚本)
│   ├── parallel_sweep.sh                  ✅ 修改：添加优化器扫描支持
│   ├── run_zo_optimizer_sweep.sh           ✅ 新增：一键启动脚本
│   └── test_optimizer_sweep.sh             ✅ 新增：测试脚本
│
├── 📖 DOCUMENTATION (文档)
│   ├── QUICK_START.md                      ✅ 新增：快速入门指南
│   ├── SETUP_SUMMARY.md                    ✅ 新增：完整设置说明
│   ├── OPTIMIZER_SWEEP_README.md           ✅ 新增：详细使用文档
│   ├── PROJECT_STRUCTURE.md                ✅ 新增：本文件
│   └── README_*.md                         📦 现有文档
│
└── 📊 OUTPUT DIRECTORIES (输出目录)
    ├── results/                           # PNG 损失曲线
    ├── csv_logs_mezo_epochs10_large/      # CSV 训练日志
    ├── job_logs_*/                        # 各任务详细日志
    └── cache/                             # 数据集缓存
```

## 🔄 工作流程图

```
┌─────────────────────────────────────────────────────────┐
│                   开始优化器扫描                          │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────v───────────┐
         │  执行扫描脚本          │
         ├───────────────────────┤
         │ bash run_zo_          │
         │ optimizer_sweep.sh    │
         └───────────┬───────────┘
                     │
      ┌──────────────┼──────────────┐
      │              │              │
      v              v              v
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ ZO+Adam │  │ZO+Muon  │  │ FO+Adam │
  │ q=1     │  │ q=1     │  │ (BP)    │
  └────┬────┘  └────┬────┘  └────┬────┘
       │            │            │
       │            │      ┌─────────────┐
       │            │      │ FO+Muon (BP)│
       │            │      └──────┬──────┘
       │            │             │
       └────────┬───┴──────┬──────┘
                │          │
       ┌────────v────────┬─v──────────┐
       │   GPU 2         │  GPU 3     │
       │ (轮换分配)      │ (轮换分配) │
       └────────┬────────┴────────┬──┘
                │                 │
                └────────┬────────┘
                         │
              ┌──────────v──────────┐
              │  生成输出文件       │
              ├────────────────────┤
              │ - PNG 图表         │
              │ - CSV 数据         │
              │ - 详细日志         │
              └────────┬───────────┘
                       │
              ┌────────v──────────┐
              │  分析结果         │
              ├────────────────────┤
              │ - 对比损失曲线    │
              │ - 性能分析        │
              │ - 生成报告        │
              └────────────────────┘
```

## 📊 实验设计结构

```
Experiment Grid (4 个并行实验)
│
├── ZO Mode + Adam Optimizer
│   └── Config: bs=2, lr=1e-6, q=1, epochs=10
│       ├── Output: ZO_full_bs2_q1_lr1e-6_optadam.png
│       ├── Data:   ZO_full_bs2_q1_lr1e-6_optadam.csv
│       └── Log:    job_logs_*/ZO_full_bs2_q1_lr1e-6_optadam.log
│
├── ZO Mode + Muon Optimizer
│   └── Config: bs=2, lr=1e-6, q=1, epochs=10
│       ├── Output: ZO_full_bs2_q1_lr1e-6_optmuon.png
│       ├── Data:   ZO_full_bs2_q1_lr1e-6_optmuon.csv
│       └── Log:    job_logs_*/ZO_full_bs2_q1_lr1e-6_optmuon.log
│
├── FO Mode + Adam Optimizer (BP Baseline)
│   └── Config: bs=2, lr=1e-6, epochs=10
│       ├── Output: FO_full_bs2_q1_lr1e-6_optadam.png
│       ├── Data:   FO_full_bs2_q1_lr1e-6_optadam.csv
│       └── Log:    job_logs_*/FO_full_bs2_q1_lr1e-6_optadam.log
│
└── FO Mode + Muon Optimizer (BP Baseline)
    └── Config: bs=2, lr=1e-6, epochs=10
        ├── Output: FO_full_bs2_q1_lr1e-6_optmuon.png
        ├── Data:   FO_full_bs2_q1_lr1e-6_optmuon.csv
        └── Log:    job_logs_*/FO_full_bs2_q1_lr1e-6_optmuon.log
```

## 📝 文件修改详情

### 1. `parallel_sweep.sh` 主要改动

| 行号 | 类型 | 改动 | 影响 |
|------|------|------|------|
| 26 | 新增 | `OPTIMIZERS=("adam" "muon")` | 支持优化器维度 |
| 56-59 | 新增 | `--optimizers` 参数处理 | 命令行配置 |
| 151-155 | 修改 | 添加 `for opt in` 循环 | 生成优化器组合 |
| 160-161 | 修改 | 添加 `for opt in` 循环 | FO 模式也支持 |
| 178 | 修改 | `read -r ... opt` | 解析优化器参数 |
| 180 | 修改 | 在名称中添加 `_opt${opt}` | 结果文件标识 |
| 192 | 新增 | `cmd="$cmd --optimizer $opt"` | 传递给 Python 脚本 |

### 2. `reproduce_zo_paper.py` 兼容性

已有现成支持：
- ✅ `--optimizer` 参数（第 363 行）
- ✅ `--zo_use_optimizer` 标志（第 370 行）
- ✅ Adam 和 Muon 实现（第 242-255 行）

**无需修改！**

## 🔄 参数流向图

```
Command Line Arguments
        │
        ├── --modes ZO,FO
        │           │
        │           └─> Mode Loop (FO/ZO)
        │
        ├── --batch-sizes 2
        │           │
        │           └─> Batch Loop
        │
        ├── --query-budgets 1
        │           │
        │           └─> Q Loop (仅 ZO)
        │
        ├── --learning-rates 1e-6
        │           │
        │           └─> LR Loop
        │
        ├── --optimizers adam,muon
        │           │
        │           └─> Optimizer Loop ⭐ NEW
        │
        └── 其他参数 (--epochs, --gpus 等)
                    │
                    └─> Global Config

    ⬇ (所有组合的笛卡尔积)

    Experiment Config: exp_id:mode:scope:batch_size:q:lr:opt
    │
    ├─> ZO:full:2:1:1e-6:adam
    ├─> ZO:full:2:1:1e-6:muon
    ├─> FO:full:2:N/A:1e-6:adam
    └─> FO:full:2:N/A:1e-6:muon

    ⬇ (每个通过 GPU 轮换)

    GPU Assignment (GPU 2 or 3)
    │
    └─> Python Script Call:
        python reproduce_zo_paper.py \
            --mode ZO \
            --optimizer adam \
            --learning_rate 1e-6 \
            --batch_size 2 \
            --query_budget_q 1 \
            ... (其他参数)
```

## 📈 并行执行策略

```
Timeline Example (2 GPUs, 2 Parallel Jobs):

Time │ GPU 2              │ GPU 3
─────┼────────────────────┼────────────────────
  0  │ Exp 1: ZO+Adam     │ Exp 2: ZO+Muon
     │ [=========>        │ [=========>
─────┼────────────────────┼────────────────────
 10m │ Exp 3: FO+Adam     │ Exp 4: FO+Muon
     │ [=======>          │ [=======>
─────┼────────────────────┼────────────────────
 20m │ [COMPLETE]         │ [COMPLETE]
     │                    │
```

## 🎯 使用场景

### 场景 1：快速测试
```bash
# ⚡ 1 分钟内完成
bash test_optimizer_sweep.sh
```
生成：4 个实验，1 epoch 各

### 场景 2：标准扫描
```bash
# ⏱️ 30-60 分钟
bash run_zo_optimizer_sweep.sh
```
生成：4 个实验，10 epoch 各

### 场景 3：自定义扫描
```bash
# 🎨 灵活配置
bash parallel_sweep.sh \
    --modes "ZO" \
    --query-budgets "1,2,4,8" \
    --optimizers "adam" \
    --epochs 5
```
生成：8 个 ZO+Adam 实验（不同的 q 值）

## 📊 输出数据结构

### CSV 文件格式 (csv_logs_mezo_epochs10_large/*)

```csv
timestamp,epoch,step,mode,scope,q,lr,batch_size,loss,grad_norm
2025-10-29 23:00:01,1,0,ZO,full,1,1e-6,2,4.5234,1.2345
2025-10-29 23:00:05,1,1,ZO,full,1,1e-6,2,4.3456,1.0987
...
```

**列说明：**
- `timestamp`: 记录时间
- `epoch`: 当前 epoch
- `step`: 训练步数（全局）
- `mode`: ZO 或 FO
- `scope`: full 或 reduced
- `q`: 查询预算（ZO）或 N/A（FO）
- `lr`: 学习率
- `batch_size`: 批次大小
- `loss`: 交叉熵损失
- `grad_norm`: 梯度范数（仅 ZO）

### 日志文件位置

```
job_logs_20251029_230000/
├── ZO_full_bs2_q1_lr1e-6_optadam.log
│   ├── Command: python reproduce_zo_paper.py ...
│   ├── GPU: 2
│   ├── Start time: ...
│   ├── [Training Output]
│   ├── End time: ...
│   └── SUCCESS/FAILED
└── ... (其他 3 个实验)
```

## 🔗 脚本依赖关系

```
run_zo_optimizer_sweep.sh (一键启动)
        │
        └─> calls
        │
        v
parallel_sweep.sh (主要逻辑)
        │
        ├─> calls generate_experiments()
        ├─> calls run_single_experiment() × 4
        │
        └─> each calls
            │
            v
        python reproduce_zo_paper.py (核心训练)
            │
            ├─> loads data
            ├─> creates model
            ├─> initializes optimizer (Adam/Muon)
            ├─> trains with ZO/FO gradient estimation
            └─> generates output (PNG + CSV)
```

## ⚙️ 配置覆盖优先级

```
Default Values (in script)
        ↓
Command Line Arguments (--optimizers, --batch-sizes 等)
        ↓
Final Configuration (used in experiment loop)
```

**示例：**
```bash
# 默认：--optimizers "adam,muon"
bash run_zo_optimizer_sweep.sh

# 覆盖：仅 Adam
bash parallel_sweep.sh --optimizers "adam" ...

# 等效于上述第二个命令
OPTIMIZERS=("adam") bash parallel_sweep.sh ...
```

## 📋 快速参考

| 任务 | 命令 | 时间 |
|------|------|------|
| 测试配置 | `bash test_optimizer_sweep.sh` | ~5 分钟 |
| 完整扫描 | `bash run_zo_optimizer_sweep.sh` | ~45 分钟 |
| 自定义扫描 | `bash parallel_sweep.sh --...` | 可变 |
| 查看结果 | `ls results/*.png` | 即时 |
| 分析数据 | 见 Python 脚本示例 | 可变 |

## 🎓 核心概念对应

| 概念 | 实现位置 | 作用 |
|------|--------|------|
| ZO 梯度估计 | `reproduce_zo_paper.py:124-183` | 核心算法 |
| Optimizer 选择 | `reproduce_zo_paper.py:234-257` | 优化器加载 |
| 参数扫描 | `parallel_sweep.sh:141-171` | 实验组合生成 |
| 并行执行 | `parallel_sweep.sh:215-309` | 任务调度 |
| 结果记录 | `parallel_sweep.sh:320-326` | CSV 日志 |

## ✅ 验证清单

- [x] 语法检查通过 (`bash -n`)
- [x] 脚本有执行权限 (`chmod +x`)
- [x] 参数传递正确
- [x] 输出目录创建逻辑
- [x] 文档完整
- [x] 示例代码可运行

---

**文档版本：** 1.0  
**最后更新：** 2025-10-29  
**维护者：** AI Assistant
