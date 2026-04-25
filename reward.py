# 动作执行与奖励计算

from typing import Tuple

from data_models import EnvironmentState, ParsedData


def _is_zero_duration_task(task_id: int, parsed: ParsedData, person_id: int | None = None) -> bool:
    """
    判断任务是否为“执行时间为 0 的虚拟任务”。
    - 优先使用解析得到的 average_duration
    - 兜底：检查 activity_people_time 中该人员（或全部人员）的时间是否为 0
    """
    if 0 <= task_id < len(parsed.average_duration) and parsed.average_duration[task_id] == 0:
        return True
    if 0 <= task_id < len(parsed.activity_people_time):
        pts = parsed.activity_people_time[task_id]
        if not pts:
            return False
        if person_id is None:
            return all(pt[1] == 0 for pt in pts)
        for pt in pts:
            if pt[0] == person_id:
                return pt[1] == 0
    return False


def execute_action(
    action: Tuple[str, int, int],
    env: EnvironmentState,
    parsed: ParsedData,
) -> float:
    """
    执行动作并返回即时奖励
    约束检查：每个任务只可由一名人员执行；人员在同一时刻只能执行一个任务；
    分配动作若人员忙碌需先执行中断动作（由 add_required_interrupts 处理）。
    """
    action_type, task_id, person_id = action
    reward = 0.0

    if action_type == "assign":
        # 分配任务
        task = env.tasks[task_id]
        if task.status != "not_started":
            return 0.0  # 任务已被分配，不执行
        if task_id in env.running_tasks or task_id in env.interrupted_tasks:
            return 0.0  # 任务已在执行或中断，不执行
        if env.people_busy.get(person_id, -1) != -1:
            return 0.0  # 人员正在执行其他任务，需要先执行中断动作
        # 执行分配
        task.status = "running"
        task.assigned_person = person_id
        task.start_time = env.current_time
        for person_time in parsed.activity_people_time[task_id]:
            if person_time[0] == person_id:
                task.remaining_time = person_time[1]
                task.original_time = person_time[1]
                break
        env.people_busy[person_id] = task_id
        # 虚拟任务（执行时间为0）在分配的同一时刻立即完成，并释放人员
        if task.remaining_time <= 0 or _is_zero_duration_task(task_id, parsed, person_id):
            task.status = "completed"
            task.finish_time = env.current_time
            env.people_busy[person_id] = -1
            env.completed_tasks.add(task_id)
            reward = 5.0  # 视为立即完成
        else:
            env.running_tasks.add(task_id)  # 标记任务执行中
            reward = 1.0  # 成功分配任务给予正奖励

    elif action_type == "interrupt":
        # 中断任务
        task = env.tasks[task_id]
        if (
            task.status != "running"
            or task.assigned_person != person_id
            or env.people_busy.get(person_id, -1) != task_id
        ):
            return 0.0  # 约束不满足，不执行
        # 执行中断
        task.status = "interrupted"
        env.interrupted_tasks[task_id] = person_id
        env.people_busy[person_id] = -1  # 释放人员
        env.running_tasks.remove(task_id)
        reward = -0.5  # 中断任务给予负奖励

    elif action_type == "resume":
        # 恢复任务
        if (
            task_id not in env.interrupted_tasks
            or env.interrupted_tasks[task_id] != person_id
        ):
            return 0.0  # 约束不满足，不执行
        task = env.tasks[task_id]
        if task.status != "interrupted":
            return 0.0  # 任务状态不正确，不执行
        if env.people_busy.get(person_id, -1) != -1:
            return 0.0  # 人员正在执行其他任务，不执行
        # 执行恢复
        task.status = "running"
        env.people_busy[person_id] = task_id  # 标记人员忙碌
        del env.interrupted_tasks[task_id]
        # 虚拟任务（执行时间为0）恢复的同一时刻立即完成，并释放人员
        if task.remaining_time <= 0 or _is_zero_duration_task(task_id, parsed, person_id):
            task.status = "completed"
            task.finish_time = env.current_time
            env.people_busy[person_id] = -1
            env.completed_tasks.add(task_id)
            reward = 5.0
        else:
            env.running_tasks.add(task_id)
            reward = 0.5  # 恢复任务给予小正奖励

    elif action_type == "continue":
        # 继续执行任务：仅表示本步该人员仍在执行该任务，不在此处扣减时间
        # 时间扣减与完成判定仅在环境“推进时间”时统一进行，避免与训练循环中的减 1 重复导致时长被算成一半
        task = env.tasks[task_id]
        if (
            task.status != "running"
            or task.assigned_person != person_id
            or env.people_busy.get(person_id, -1) != task_id
        ):
            return 0.0  # 约束不满足，不执行
        reward = 0.1  # 继续执行给予小正奖励

    return reward


def calculate_global_reward(env: EnvironmentState, parsed: ParsedData) -> float:
    """
    计算全局奖励（与里程碑延迟惩罚对应）
    延迟惩罚 = 惩罚系数 * max(0, 实际完成时间 - 应发生时间)
    最后一个里程碑惩罚系数0.5，其余0.3。
    返回：-total_penalty，即负值（有延迟时奖励为负）；无延迟时为 0。
    """
    total_penalty = 0.0
    if not parsed.milestone_event or not parsed.milestone_time:
        return 0.0
    for i, event_id in enumerate(parsed.milestone_event):
        if i < len(parsed.milestone_time):
            deadline = parsed.milestone_time[i]
            task = env.tasks[event_id]
            if task.finish_time is not None:
                delay = max(0, task.finish_time - deadline)
                penalty_coef = 0.5 if i == len(parsed.milestone_event) - 1 else 0.3
                total_penalty += penalty_coef * delay
    return -total_penalty  # 返回负的惩罚值（作为奖励，惩罚越大奖励越小）


def calculate_global_reward_for_cost_to_go(
    env: EnvironmentState, parsed: ParsedData
) -> float:
    """
    计算总延迟惩罚（作为 cost-to-go 函数值）
    延迟惩罚 = 惩罚系数 * max(0, 实际完成时间 - 应发生时间)
    Returns: 总延迟惩罚（正数）
    """
    total_penalty = 0.0
    if not parsed.milestone_event or not parsed.milestone_time:
        return 0.0
    for i, event_id in enumerate(parsed.milestone_event):
        if i < len(parsed.milestone_time):
            deadline = parsed.milestone_time[i]
            task = env.tasks[event_id]
            if task.finish_time is not None:
                delay = max(0, task.finish_time - deadline)
                penalty_coef = 0.5 if i == len(parsed.milestone_event) - 1 else 0.3
                total_penalty += penalty_coef * delay
    return total_penalty
