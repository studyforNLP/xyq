"""
调度脚本入口：按运行逻辑已拆分为多模块，本文件仅作为启动器保留。

模块说明：
- config.py         : 数据路径与解析标记
- data_models.py    : ParsedData, TaskState, EnvironmentState
- parser.py         : 数据文件解析 parse_custom_file
- state_space.py    : 状态/动作空间与 generate_state_space, ACTIONS
- environment.py    : 环境初始化与状态特征 (init_environment, get_state_features 等)
- actions.py        : 可行动作生成、冲突判断、必要中断 (generate_available_actions 等)
- reward.py         : 动作执行与奖励 (execute_action, calculate_global_reward 等)
- cost_to_go.py     : Cost-to-go 模拟 (calculate_cost_to_go)
- qlearning.py      : Q 表与更新 (init_q_table, epsilon_greedy_action, update_q)
- training.py       : Rollout 训练与最终方案 (sample_training_loop, build_final_solution)
- output.py         : 解析结果与方案输出 (pretty_print, print_solution)
- main.py           : 主流程 (main)
"""

from main import main

if __name__ == "__main__":
    main()
