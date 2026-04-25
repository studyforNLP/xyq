"""Default algorithm parameters and their meanings."""


TRAINING_PARAMS = {
    # Maximum number of rollout episodes used to train the Q table.
    "max_rollouts": 1000,
    # Q-learning step size. Larger values learn faster but can oscillate.
    "alpha": 0.1,
    # Discount factor for future value in Q-learning updates.
    "gamma": 0.95,
    # Exploration probability used by epsilon-greedy action selection.
    "epsilon": 0.2,
    # Stop training early when the max absolute Q-table change is below this value.
    "convergence_threshold": 1e-3,
    # Weight of the episode-level global reward, where global reward = -total penalty.
    "global_reward_scale": 0.3,
    # Weight of the per-decision local reward before it is combined with global reward.
    "local_reward_scale": 0.2,
    # Weight of normalized immediate action-set feedback inside local reward.
    "immediate_reward_weight": 1.0,
    # Weight of cost-to-go lookahead gain inside local reward.
    "lookahead_reward_weight": 1.0,
    # Normalize immediate feedback by the number of executed actions in the action set.
    "normalize_immediate_reward": True,
    # Clip lookahead gain into [-lookahead_clip, lookahead_clip]. Set None to disable.
    "lookahead_clip": 1.0,
}


FINAL_SOLUTION_PARAMS = {
    # If top Q values are closer than this threshold, use cost-to-go as fallback.
    "q_distinction_threshold": 1e-9,
    # Number of partial action sets kept during final beam-search decoding.
    "beam_width": 3,
    # Maximum actions expanded from each beam node. Set 0 or None to expand all.
    "beam_branch_limit": 8,
    # Use beam result only when its cost-to-go improves over greedy by this margin.
    "beam_improvement_margin": 0.1,
}


PARAMETER_DESCRIPTIONS = {
    "max_rollouts": "训练最大 rollout / episode 次数。",
    "alpha": "Q-learning 学习率，控制新样本覆盖旧 Q 值的速度。",
    "gamma": "折扣因子，控制未来收益在当前 Q 更新中的占比。",
    "epsilon": "epsilon-greedy 探索率，训练时以该概率随机选动作。",
    "convergence_threshold": "Q 表收敛阈值，相邻两轮最大 Q 差异低于该值则提前停止。",
    "global_reward_scale": "全局奖励权重，全局奖励等于总延迟惩罚的相反数。",
    "local_reward_scale": "局部奖励权重，用于控制动作集级局部反馈对 Q 更新的影响。",
    "immediate_reward_weight": "局部奖励内部的即时执行反馈权重。",
    "lookahead_reward_weight": "局部奖励内部的 cost-to-go 前瞻收益权重。",
    "normalize_immediate_reward": "是否按动作数归一化即时反馈，避免大动作集天然奖励更大。",
    "lookahead_clip": "前瞻收益裁剪范围，避免异常 cost-to-go 差值主导训练。",
    "q_distinction_threshold": "最终构解时判断 Q 值是否足够区分的阈值。",
    "beam_width": "最终构解 beam search 保留的候选动作集数量，越大越稳但越慢。",
    "beam_branch_limit": "每个 beam 节点最多扩展的候选动作数，用于控制 cost-to-go 评估次数。",
    "beam_improvement_margin": "beam 动作集相对贪心动作集至少需要降低的 cost-to-go 幅度。",
}
