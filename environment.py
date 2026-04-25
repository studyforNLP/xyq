from typing import Dict, List, Tuple

from data_models import EnvironmentState, ParsedData, TaskState


def init_environment(parsed: ParsedData) -> EnvironmentState:
    """Initialize the scheduling environment."""
    tasks = []
    for task_id in range(parsed.task_count):
        avg_time = parsed.average_duration[task_id] if task_id < len(parsed.average_duration) else 0
        tasks.append(
            TaskState(
                task_id=task_id,
                status="not_started",
                remaining_time=avg_time,
                original_time=avg_time,
            )
        )

    people_busy = {}
    if parsed.people_num:
        for person_id in range(parsed.people_num):
            people_busy[person_id] = -1

    return EnvironmentState(
        current_time=0,
        tasks=tasks,
        people_busy=people_busy,
        completed_tasks=set(),
        running_tasks=set(),
        interrupted_tasks={},
    )


def calculate_task_levels(
    activity_num: int,
    immediate_predecessors: List[List[int]],
    completed_tasks: set,
) -> List[int]:
    """Compute each task's level in the precedence graph."""
    del completed_tasks

    levels = [-1] * activity_num
    if activity_num > 0:
        levels[0] = 0

    changed = True
    while changed:
        changed = False
        for task_id in range(activity_num):
            if levels[task_id] != -1:
                continue
            preds = [pred for pred in immediate_predecessors[task_id] if pred != -1]
            if not preds or all(levels[pred] != -1 for pred in preds):
                levels[task_id] = max((levels[pred] for pred in preds), default=-1) + 1
                changed = True
    return levels


def calculate_network_level_feature(
    activity_num: int,
    immediate_predecessors: List[List[int]],
    completed_tasks: set,
    running_tasks: set,
) -> int:
    """Discretize the average level of ready tasks."""
    levels = calculate_task_levels(activity_num, immediate_predecessors, completed_tasks)
    if not levels:
        return 0

    max_level = max(levels) if levels else 1
    ready_tasks = []
    for task_id in range(activity_num):
        if task_id in completed_tasks or task_id in running_tasks:
            continue
        preds = [pred for pred in immediate_predecessors[task_id] if pred != -1]
        if not preds or all(pred in completed_tasks for pred in preds):
            ready_tasks.append(task_id)

    if not ready_tasks:
        return 0

    avg_level = sum(levels[task_id] for task_id in ready_tasks) / len(ready_tasks)
    nlf = avg_level / max_level if max_level > 0 else 0.0
    if nlf < 0.2:
        return 0
    if nlf < 0.4:
        return 1
    if nlf < 0.6:
        return 2
    if nlf < 0.8:
        return 3
    return 4


def calculate_skill_bottleneck_index(
    activity_num: int,
    immediate_predecessors: List[List[int]],
    activity_people_time: List[List[List[int]]],
    completed_tasks: set,
    running_tasks: set,
    people_busy: Dict[int, int],
    max_level: int = 4,
) -> int:
    """Discretize the minimum number of capable people among ready tasks."""
    del people_busy

    ready_tasks = []
    for task_id in range(activity_num):
        if task_id in completed_tasks or task_id in running_tasks:
            continue
        preds = [pred for pred in immediate_predecessors[task_id] if pred != -1]
        if not preds or all(pred in completed_tasks for pred in preds):
            ready_tasks.append(task_id)

    if not ready_tasks:
        return max_level

    min_capable_people = float("inf")
    for task_id in ready_tasks:
        if task_id < len(activity_people_time):
            min_capable_people = min(min_capable_people, len(activity_people_time[task_id]))

    if min_capable_people == float("inf"):
        return 0

    max_levels = max_level + 1
    if max_levels <= 5:
        return min(int(min_capable_people) - 1, max_level)

    people_num = max_level + 1
    normalized = (min_capable_people - 1) / (people_num - 1) if people_num > 1 else 0.0
    if normalized <= 0.2:
        return 0
    if normalized <= 0.4:
        return 1
    if normalized <= 0.6:
        return 2
    if normalized <= 0.8:
        return 3
    return 4


def calculate_milestone_urgency(
    milestone_event: List[int],
    milestone_time: List[int],
    current_time: int,
    completed_tasks: set,
) -> int:
    """Discretize the urgency of unfinished milestones."""
    if not milestone_event or not milestone_time:
        return 0

    max_urgency = 0.0
    for index, event_id in enumerate(milestone_event):
        if event_id in completed_tasks or index >= len(milestone_time):
            continue

        deadline = milestone_time[index]
        slack = deadline - current_time
        if index == 0:
            reference_window = max(1, deadline)
        else:
            reference_window = max(1, deadline - milestone_time[index - 1])

        urgency = 1 - (slack / reference_window) if slack >= 0 else 1.0
        urgency = max(0.0, min(1.0, urgency))
        max_urgency = max(max_urgency, urgency)

    if max_urgency < 0.2:
        return 0
    if max_urgency < 0.4:
        return 1
    if max_urgency < 0.6:
        return 2
    if max_urgency < 0.8:
        return 3
    return 4


def _discretize_ratio(value: float) -> int:
    """Map a ratio in [0, 1] to a 0-4 level."""
    value = max(0.0, min(1.0, value))
    if value < 0.2:
        return 0
    if value < 0.4:
        return 1
    if value < 0.6:
        return 2
    if value < 0.8:
        return 3
    return 4


def calculate_resource_utilization_rate(
    people_busy: Dict[int, int],
    people_num: int | None,
) -> int:
    """Discretize the share of busy people as RUR."""
    if not people_num:
        return 0
    busy_people = sum(1 for task_id in people_busy.values() if task_id != -1)
    return _discretize_ratio(busy_people / people_num)


def calculate_critical_running_task_pressure(
    running_tasks: set,
    tasks: List[TaskState],
    time_difference: List[int],
) -> int:
    """Discretize pressure from running tasks near the critical path as CRT."""
    if not running_tasks:
        return 0

    max_pressure = 0.0
    for task_id in running_tasks:
        task = tasks[task_id]
        remaining_time = max(0, task.remaining_time)
        slack = time_difference[task_id] if task_id < len(time_difference) else 0
        slack = max(0, slack)
        if remaining_time <= 0:
            pressure = 0.0
        else:
            pressure = remaining_time / max(1, remaining_time + slack)
        max_pressure = max(max_pressure, pressure)

    return _discretize_ratio(max_pressure)


def calculate_interrupted_task_number(interrupted_tasks: Dict[int, int]) -> int:
    """Discretize the number of interrupted tasks as ITN."""
    interrupted_count = len(interrupted_tasks)
    if interrupted_count <= 0:
        return 0
    if interrupted_count == 1:
        return 1
    if interrupted_count == 2:
        return 2
    if interrupted_count == 3:
        return 3
    return 4


def get_state_features(
    env: EnvironmentState,
    parsed: ParsedData,
    max_skill_level: int = 4,
) -> Tuple[int, ...]:
    """Return the discretized state features (NLF, SBI, MUR, RUR, CRT, ITN)."""
    nlf = calculate_network_level_feature(
        parsed.task_count,
        parsed.immediate_predecessors,
        env.completed_tasks,
        env.running_tasks,
    )
    sbi = calculate_skill_bottleneck_index(
        parsed.task_count,
        parsed.immediate_predecessors,
        parsed.activity_people_time,
        env.completed_tasks,
        env.running_tasks,
        env.people_busy,
        max_skill_level,
    )
    mur = calculate_milestone_urgency(
        parsed.milestone_event,
        parsed.milestone_time,
        env.current_time,
        env.completed_tasks,
    )
    rur = calculate_resource_utilization_rate(env.people_busy, parsed.people_num)
    crt = calculate_critical_running_task_pressure(
        env.running_tasks,
        env.tasks,
        parsed.time_difference,
    )
    itn = calculate_interrupted_task_number(env.interrupted_tasks)
    return (nlf, sbi, mur, rur, crt, itn)
