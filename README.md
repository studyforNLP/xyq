# 面向多里程碑多技能可中断项目调度的 Rollout-Q-learning

本项目实现了一个面向多里程碑、多技能人员、任务可中断项目调度问题的实验代码框架。核心方法为 Rollout-Q-learning，简称 RQL，即通过 Q-learning 学习动作偏好，并用 rollout cost-to-go 前瞻评价辅助训练和最终构解。

代码当前主要服务于论文实验，包括正式对比实验、RQL 参数正交实验、RQL 收敛性实验、cost-to-go 耗时诊断，以及多 Sheet Excel 结果输出。

## 问题描述

项目由带紧前紧后约束的任务网络构成。每个任务可由具备相应技能的服务人员执行，不同人员执行同一任务的处理时间可以不同。若人员不具备任务所需技能，则不能执行该任务。

任务允许被中断。任务中断后会记录原执行人员和剩余处理时间，后续必须由原执行人员从原进度恢复执行。任一服务人员在同一时刻最多执行一项任务，一个任务同一时刻只由一个人员执行。

项目中设置多个里程碑事件，每个里程碑具有计划完成时间和惩罚权重。若里程碑实际完成时间晚于计划完成时间，则产生加权延迟惩罚。算法目标是最小化总延迟惩罚。

目标函数口径为：

```text
penalty = sum(weight_m * max(0, actual_finish_m - due_time_m))
```

## 当前算法

当前主算法为 `RQL = Q-learning policy + rollout cost-to-go assisted decoding`。

主要特点：

- 使用 6 维离散状态表示：`NLF, SBI, MUR, RUR, CRT, ITN`。
- 使用完整动作三元组：`(action_type, task_id, person_id)`。
- 显式动作包括：`assign`、`resume`、`continue`。
- `interrupt` 不是独立学习动作，而是在人员冲突时自动插入。
- 训练阶段将局部奖励、全局惩罚和 cost-to-go 前瞻收益结合。
- 最终构解阶段使用 Q 值、cost-to-go fallback 和 beam search 共同选择动作集。
- 正式 RQL 默认采用 scale-specific 收敛策略：J10 使用 checkpoint，J20/J30 及更大规模使用 combined profile。

## 对比方法

正式实验支持以下方法：

| 方法 | 含义 |
|---|---|
| `MPV-SLK-EFT` | 里程碑优先、松弛时间、最早完成时间组合规则 |
| `MXS+MF` | 后继任务数最多优先，再选预计完成时间最早人员 |
| `SLK-EFT` | 松弛时间优先，再选预计完成时间最早人员 |
| `SPT` | 最短处理时间规则 |
| `ECT` | 最早完成时间规则 |
| `FIFO` | 先到先服务规则 |
| `QL` | 不使用 rollout cost-to-go 的普通 Q-learning |
| `RH` | 纯启发式 rollout，不训练 Q 表 |
| `RQL` | Q-learning + rollout cost-to-go 前瞻评价 |

## 数据目录

正式案例位于 `cases/`：

```text
cases/
  j10.mm/
  j20.mm/
  j30.mm/
  j60.mm/
  j90.mm/
  j120.mm/
```

规模含义：

- `J10` 表示 10 个真实任务。
- 文件总节点数为 `J + 2`，包含虚拟起点和虚拟终点。
- 实验脚本会校验 `task_count == J + 2`。

## 环境要求

当前代码使用 Python 标准库和 `numpy`。Excel 输出由 `excel_writer.py` 使用 `zipfile + XML` 生成，不依赖 `pandas`、`openpyxl` 或 `xlsxwriter`。

推荐使用当前机器上的 Python：

```powershell
E:\conda\python.exe
```

## 快速运行

单案例快速运行：

```powershell
E:\conda\python.exe main.py
```

`main.py` 使用 `config.py` 中的 `DEFAULT_TARGET_FILE` 作为输入文件，适合快速检查解析、训练和最终构解流程。

## 正式对比实验

正式多规模实验入口为 `experiment_suite.py`。

小规模 smoke test：

```powershell
E:\conda\python.exe experiment_suite.py --cases-dir cases --scales 10 --max-cases-per-scale 1 --seeds 1 --output-xlsx results/smoke.xlsx
```

J10/J20/J30 正式实验示例：

```powershell
E:\conda\python.exe experiment_suite.py --cases-dir cases --scales 10,20,30 --max-cases-per-scale 100 --seeds 3 --output-xlsx results/formal_j10_j20_j30_100cases_3seeds_scale_specific.xlsx
```

只运行 RQL：

```powershell
E:\conda\python.exe experiment_suite.py --cases-dir cases --scales 10,20,30 --max-cases-per-scale 50 --seeds 3 --rql-only --output-xlsx results/rql_only.xlsx
```

只运行 RH：

```powershell
E:\conda\python.exe experiment_suite.py --cases-dir cases --scales 10,20,30 --max-cases-per-scale 0 --seeds 1 --methods RH --output-xlsx results/rh_allcases_seed1.xlsx
```

说明：`--max-cases-per-scale 0` 表示不限制案例数量。

## Excel 输出

`experiment_suite.py` 会生成一个多 Sheet `.xlsx` 文件，主要包括：

| Sheet | 内容 |
|---|---|
| `metadata` | 实验时间、参数、规模、方法、状态特征说明 |
| `case_check` | 案例规模校验、任务数、人员数、里程碑数 |
| `raw_runs` | 每次运行的原始结果 |
| `instance_summary` | 每个案例和方法的实例级汇总 |
| `scale_summary` | 论文主表使用的规模级汇总 |
| `method_rank` | 各规模下按 AOTV 和 ARPD 排名 |
| `convergence` | QL/RQL 的训练历史 |
| `failures` | 异常记录，单个失败不会中断整批实验 |

## 指标口径

| 指标 | 含义 |
|---|---|
| `penalty` | 总延迟惩罚，越小越好 |
| `AOTV` | 同一规模下，各案例平均 penalty |
| `Best` | 每个案例各 seed 的最小 penalty，再对案例取平均 |
| `Std` | 实例级平均 penalty 的标准差 |
| `ARPD` | 相对同案例最优结果的平均百分比偏差 |
| `Time` | 默认使用 `total_time = train_time + solve_time` |

ARPD 定义：

```text
ARPD = mean((F_A(I) - F*(I)) / max(F*(I), 1)) * 100%
```

其中 `F*(I)` 是同一案例上所有算法得到的最小 penalty。

当前统计规则中，`raw_runs` 保留 RQL 每个 seed 的原始输出；在 `instance_summary` 和 `scale_summary` 中，RQL 的案例级 penalty 使用该案例多个 seed 中的最优值，以体现内部多次随机训练后的最好调度结果。

## RQL 参数调优

正交实验入口为 `orthogonal_experiment.py`，用于 RQL 参数调优。

示例命令：

```powershell
E:\conda\python.exe orthogonal_experiment.py --cases-dir cases --scales 10,20,30 --max-cases-per-scale 10 --seeds 1 --output-xlsx results/orthogonal_results.xlsx
```

当前正交实验设计采用分层思路：

- 第一层：训练参数 `alpha, gamma, epsilon, epsilon_decay`。
- 第二层：奖励结构和构解参数 `global_reward_scale, local_reward_scale, lookahead_reward_weight, beam_width`。
- 第三层：`max_rollouts / EP` 敏感性分析。

## RQL 收敛性实验

收敛性实验入口为 `rql_convergence_experiment.py`，用于比较 RQL 收敛改进策略，例如 scale-specific EP、低后期探索率、checkpoint、奖励归一化和 warmup。

示例命令：

```powershell
E:\conda\python.exe rql_convergence_experiment.py --cases-dir cases --scales 10,20,30 --max-cases-per-scale 5 --seeds 1 --output-xlsx results/rql_convergence_experiment.xlsx
```

## cost-to-go 耗时诊断

`cost_to_go.py` 内置了轻量诊断字段：

- `ctg_calls`
- `ctg_call_time`
- `avg_ctg_time`
- `ctg_simulation_calls`
- `ctg_simulation_time`
- `avg_ctg_simulation_time`

独立测试入口为 `ctg_profile.py`，不会运行正式对比实验。

小规模诊断：

```powershell
E:\conda\python.exe ctg_profile.py --scales 10 --max-cases-per-scale 1 --seeds 1 --max-rollouts 2
```

全规模轻量诊断：

```powershell
E:\conda\python.exe ctg_profile.py --scales 10,20,30,60,90,120 --max-cases-per-scale 3 --seeds 1 --max-rollouts 3 --output-csv results/ctg_profile_all_scales_3cases_3rollouts.csv
```

该诊断用于确认 RQL 的主要耗时来自 cost-to-go rollout 前瞻仿真。测试结果显示，J60 及以上规模中 CTG 耗时占总训练耗时的 90% 以上。

## 主要文件

| 文件 | 作用 |
|---|---|
| `main.py` | 单案例快速入口 |
| `experiment_suite.py` | 正式多规模对比实验入口 |
| `orthogonal_experiment.py` | RQL 参数正交实验 |
| `rql_convergence_experiment.py` | RQL 收敛性实验 |
| `ctg_profile.py` | cost-to-go 耗时诊断 |
| `baselines.py` | 启发式、RH baseline 构解逻辑 |
| `training.py` | QL/RQL 训练和最终构解 |
| `cost_to_go.py` | rollout cost-to-go 仿真、缓存和诊断 |
| `metrics.py` | raw、instance、scale、rank 指标汇总 |
| `excel_writer.py` | 标准库 Excel 输出 |
| `parser.py` | `.mm` 案例解析 |
| `environment.py` | 环境初始化和状态特征计算 |
| `actions.py` | 动作生成、冲突判断和自动中断 |
| `reward.py` | 动作执行和惩罚/奖励计算 |
| `qlearning.py` | Q 表初始化和 Q-learning 更新 |
| `parameters.py` | 默认训练参数和构解参数 |
| `state_space.py` | 6 维状态空间与动作类型定义 |

## 当前正式参数

默认训练参数位于 `parameters.py`：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `max_rollouts` | 500 | 默认最大训练 episode 数 |
| `alpha` | 0.2 | Q-learning 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `epsilon` | 0.2 | 初始探索率 |
| `epsilon_decay` | 0.99 | 探索率衰减系数 |
| `epsilon_min` | 0.02 | 最小探索率 |
| `global_reward_scale` | 0.1 | 全局奖励权重 |
| `local_reward_scale` | 0.4 | 局部奖励权重 |
| `lookahead_reward_weight` | 1.5 | 前瞻奖励权重 |

默认最终构解参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `q_distinction_threshold` | `1e-9` | Q 值区分阈值 |
| `beam_width` | 5 | beam search 保留动作集数量 |
| `beam_branch_limit` | 8 | 每个 beam 节点最大扩展动作数 |
| `beam_improvement_margin` | 0.1 | beam 结果相对贪心结果的最小改进幅度 |

## 结果文件和版本管理

实验输出默认写入 `results/`。`.gitignore` 已排除：

- `results/`
- `*.xlsx`
- `*.csv`
- `__pycache__/`

因此正式实验结果不会被默认提交到 GitHub。若需要保存关键结果，应单独整理为论文表格或报告文档。

## 论文写作建议

论文主实验建议使用 J10/J20/J30/J60，J90/J120 可作为补充说明。时间指标建议报告 `total_time`，同时在正文中说明：

```text
RQL 的主要额外成本来自训练阶段的 cost-to-go 前瞻仿真；
online solve_time 较小，但不能单独作为算法总时间。
```

当前方法适合定位为解质量优先的动态调度算法，而不是低耗时快速启发式算法。
