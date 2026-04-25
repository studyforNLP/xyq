# 动作生成、冲突判断与必要中断添加

from typing import List, Tuple

from data_models import EnvironmentState, ParsedData


def _is_zero_duration_task(task_id: int, parsed: ParsedData) -> bool:
    """执行时间为0的虚拟任务：优先看 average_duration，兜底看 activity_people_time 全为0。"""
    if 0 <= task_id < len(parsed.average_duration) and parsed.average_duration[task_id] == 0:
        return True
    if 0 <= task_id < len(parsed.activity_people_time):
        pts = parsed.activity_people_time[task_id]
        return bool(pts) and all(pt[1] == 0 for pt in pts)
    return False


def _action_is_zero_duration(action: Tuple[str, int, int], parsed: ParsedData) -> bool:
    action_type, task_id, _person_id = action
    return action_type in ("assign", "resume") and _is_zero_duration_task(task_id, parsed)


def generate_available_actions(
    env: EnvironmentState,
    parsed: ParsedData,
) -> List[Tuple[str, int, int]]:
    """
    生成当前状态下所有可行动作

    约束：
    - 每个任务只可由一名人员执行
    - 人员在同一时刻只能执行一个任务
    - 分配动作允许对忙碌人员执行（add_required_interrupts会自动添加中断动作）
    注意：interrupt 不再是独立动作，只作为 assign/resume 的附属动作自动执行

    Returns:
        动作列表，每个动作为 (动作类型, 任务ID, 人员ID)
    """
    actions = []
    # 1. 分配动作（assign）：将人员分配到可执行的新任务
    ready_tasks = []
    for i in range(parsed.task_count):
        if (
            i not in env.completed_tasks
            and i not in env.running_tasks
            and i not in env.interrupted_tasks
        ):
            preds = [p for p in parsed.immediate_predecessors[i] if p != -1]
            if not preds or all(p in env.completed_tasks for p in preds):
                ready_tasks.append(i)

    for task_id in ready_tasks:
        task = env.tasks[task_id]
        if task.status != "not_started":
            continue  # 跳过已分配的任务
        if task_id < len(parsed.activity_people_time):
            for person_time in parsed.activity_people_time[task_id]:
                person_id = person_time[0]
                # 允许对任何能执行该任务的人员分配（包括忙碌的人员）
                if person_id in env.people_busy:
                    actions.append(("assign", task_id, person_id))

    # 2. 恢复动作（resume）：由原人员恢复执行之前中断的任务
    # 允许对忙碌人员恢复（如果人员忙碌，add_required_interrupts会自动添加中断动作）
    for task_id, person_id in env.interrupted_tasks.items():
        if person_id in env.people_busy:
            task = env.tasks[task_id]
            if task.status == "interrupted":
                actions.append(("resume", task_id, person_id))

    # 3. 继续执行动作（continue）：继续执行正在执行的任务
    for task_id in env.running_tasks:
        task = env.tasks[task_id]
        if task.status == "running" and task.assigned_person is not None:
            if env.people_busy.get(task.assigned_person, -1) == task_id:
                actions.append(("continue", task_id, task.assigned_person))

    return actions


def is_action_conflict(
    action1: Tuple[str, int, int],
    action2: Tuple[str, int, int],
    parsed: ParsedData,
) -> bool:
    """
    判断两个动作是否冲突
    冲突规则：同一人员不能同时执行多个动作；同一任务不能同时被分配给多个人；
    同一任务不能同时被分配和中断/恢复。
    """
    type1, task1, person1 = action1
    type2, task2, person2 = action2
    # 约束1：同一人员不能同时执行多个动作
    if person1 == person2 and person1 != -1:
        # 虚拟任务（执行时间为0）的 assign/resume 不占用人员，允许与其它动作同一时刻发生
        if not (_action_is_zero_duration(action1, parsed) or _action_is_zero_duration(action2, parsed)):
            return True
    # 约束2：同一任务不能同时被分配给多个人
    if task1 == task2:
        if type1 == "assign" and type2 == "assign":
            return True
        # 同一任务不能同时被分配和中断/恢复
        if (type1 == "assign" and type2 in ["interrupt", "resume"]) or (
            type2 == "assign" and type1 in ["interrupt", "resume"]
        ):
            return True
    return False


def add_required_interrupts(
    selected_actions: List[Tuple[str, int, int]],
    env: EnvironmentState,
) -> List[Tuple[str, int, int]]:
    """
    为选中的分配/恢复动作添加必要的中断动作
    如果选择了 assign 或 resume，且该人员正在执行其他任务，则自动添加对应的中断动作。
    """
    result_actions = []
    person_actions = {}  # person_id -> action（assign 或 resume）
    for action in selected_actions:
        action_type, task_id, person_id = action
        if action_type == "assign" or action_type == "resume":
            person_actions[person_id] = action
        result_actions.append(action)
    # 为每个分配/恢复动作检查是否需要中断
    for person_id, person_action in person_actions.items():
        current_task_id = env.people_busy.get(person_id, -1)
        if current_task_id != -1:
            # 人员忙碌，需要中断当前任务
            current_task = env.tasks[current_task_id]
            if current_task.status == "running":
                interrupt_action = ("interrupt", current_task_id, person_id)
                if interrupt_action not in result_actions:
                    result_actions.insert(
                        result_actions.index(person_action), interrupt_action
                    )
    return result_actions
