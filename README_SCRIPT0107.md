# script0107 模块说明

原单文件 `script0107.py` 已按运行逻辑拆分为多个模块，便于阅读和维护。运行方式不变：**`python script0107.py`** 或 **`python main.py`**。

## 模块与职责

| 文件 | 职责 |
|------|------|
| **config.py** | 数据文件路径 `DEFAULT_TARGET_FILE`、解析标记 `MARKERS` |
| **data_models.py** | 数据类：`ParsedData`、`TaskState`、`EnvironmentState` |
| **parser.py** | 数据文件解析：`parse_custom_file` 及内部辅助函数 |
| **state_space.py** | 状态/动作空间：`network_levels`、`get_skill_bottleneck_levels`、`generate_state_space`、`ACTIONS` |
| **environment.py** | 环境与状态特征：`init_environment`、`calculate_task_levels`、`get_state_features` 等 |
| **actions.py** | 动作生成与冲突：`generate_available_actions`、`is_action_conflict`、`add_required_interrupts` |
| **reward.py** | 动作执行与奖励：`execute_action`、`calculate_global_reward`、`calculate_global_reward_for_cost_to_go` |
| **cost_to_go.py** | Cost-to-go 模拟：`calculate_cost_to_go` |
| **qlearning.py** | Q 学习：`init_q_table`、`epsilon_greedy_action`、`update_q` |
| **training.py** | 训练与方案：`sample_training_loop`、`build_final_solution` |
| **output.py** | 输出：`pretty_print`、`print_solution` |
| **main.py** | 主流程：解析 → 打印 → 初始化 Q 表 → 训练 → 构建方案 → 输出 |
| **script0107.py** | 入口脚本，仅调用 `main.main()` |

## 依赖关系（简要）

- `main` 依赖：config, parser, output, state_space, qlearning, training
- `training` 依赖：environment, actions, reward, qlearning, cost_to_go, state_space, data_models
- `cost_to_go` 依赖：data_models, reward
- 其余模块依赖见各文件顶部 `import`。

## 运行

在项目根目录（与 `config.py` 同级）执行：

```bash
python script0107.py
```

或：

```bash
python main.py
```

数据文件路径在 **config.py** 的 `DEFAULT_TARGET_FILE` 中修改。
