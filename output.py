from typing import Dict, List

from data_models import ParsedData
from environment import calculate_task_levels, init_environment
from reward import execute_action


def _is_zero_duration_task(task_id: int, parsed: ParsedData) -> bool:
    """执行时间为0的虚拟任务：用于在按时刻汇总中显示其执行时刻。"""
    if 0 <= task_id < len(parsed.average_duration) and parsed.average_duration[task_id] == 0:
        return True
    if 0 <= task_id < len(parsed.activity_people_time):
        pts = parsed.activity_people_time[task_id]
        return bool(pts) and all(pt[1] == 0 for pt in pts)
    return False


def pretty_print(parsed: ParsedData) -> None:
    """输出字段类型信息和任务网络"""
    print("\n[字段类型信息]")
    for field_name, value in parsed.__dict__.items():
        print(f"{field_name}: {type(value).__name__}")

    # 输出任务网络
    print("\n" + "=" * 80)
    print("任务网络结构")
    print("=" * 80)
    # 计算任务层级
    task_levels = calculate_task_levels(
        parsed.task_count, parsed.immediate_predecessors, set()
    )
    # 1. 输出邻接矩阵
    print("\n[邻接矩阵]")
    print("说明: adjacency_matrix[i][j] = 1 表示任务 i 是任务 j 的前序任务")
    n, m = len(parsed.adjacency_matrix), len(parsed.adjacency_matrix[0]) if parsed.adjacency_matrix else 0
    print(f"矩阵大小: {n} x {m}")
    for i in range(min(10, n)):
        row = parsed.adjacency_matrix[i][:10] if parsed.adjacency_matrix else []
        print(" ".join(f"{x:3d}" for x in row) + (" ..." if m > 10 else ""))
    if n > 10:
        print(" ...")

    # 2. 输出每个任务的详细信息
    print("\n" + "=" * 80)
    print("任务详细信息")
    print("=" * 80)
    print(f"{'任务ID':<8} {'层级':<6} {'紧前任务':<20} {'紧后任务':<20} {'优先值':<8} {'平均时长':<10} {'最早开始':<10} {'最晚开始':<10} {'总时差':<8}")
    print("-" * 120)
    for i in range(parsed.task_count):
        preds = [str(p) for p in parsed.immediate_predecessors[i] if p != -1] if i < len(parsed.immediate_predecessors) else []
        succs = [str(s) for s in parsed.immediate_successors[i] if s != -1] if i < len(parsed.immediate_successors) else []
        preds_str = ", ".join(preds)[:18] + ("..." if len(preds) > 3 else "") or "无"
        succs_str = ", ".join(succs)[:18] + ("..." if len(succs) > 3 else "") or "无"
        level = task_levels[i] if i < len(task_levels) else -1
        priority = parsed.priority_value[i] if i < len(parsed.priority_value) else 0
        avg_dur = parsed.average_duration[i] if i < len(parsed.average_duration) else 0
        earliest = parsed.earliest_start[i] if i < len(parsed.earliest_start) else 0
        latest = parsed.latest_start[i] if i < len(parsed.latest_start) else 0
        time_diff = parsed.time_difference[i] if i < len(parsed.time_difference) else 0
        print(f"{i:<8} {level:<6} {preds_str:<20} {succs_str:<20} {priority:<8} {avg_dur:<10} {earliest:<10} {latest:<10} {time_diff:<8}")

    # 3. 输出任务层级分布
    print("\n" + "=" * 80)
    print("任务层级分布")
    print("=" * 80)
    level_distribution = {}
    for i, level in enumerate(task_levels):
        level_distribution.setdefault(level, []).append(i)
    for level in sorted(level_distribution.keys()):
        tasks = level_distribution[level]
        print(f"层级 {level}: 任务 {', '.join(map(str, tasks))} (共 {len(tasks)} 个任务)")

    # 4. 输出里程碑信息
    if parsed.milestone_event and parsed.milestone_time:
        print("\n" + "=" * 80)
        print("里程碑事件")
        print("=" * 80)
        print(f"{'序号':<6} {'任务ID':<8} {'预期时间':<10} {'惩罚系数':<10}")
        print("-" * 40)
        for i, (event_id, mt) in enumerate(zip(parsed.milestone_event, parsed.milestone_time)):
            penalty_coef = 0.5 if i == len(parsed.milestone_event) - 1 else 0.3
            print(f"{i+1:<6} {event_id:<8} {mt:<10} {penalty_coef:<10}")

    # 5. 输出人员信息
    if parsed.people_num:
        print("\n" + "=" * 80)
        print("人员信息")
        print("=" * 80)
        print(f"总人员数: {parsed.people_num}")
        if parsed.activity_people_time:
            for i, person_time_list in enumerate(parsed.activity_people_time):
                if person_time_list:
                    person_ids = [str(pt[0]) for pt in person_time_list]
                    print(f"  任务 {i}: {len(person_time_list)} 人 (人员ID: {', '.join(person_ids)})")
    print("\n" + "=" * 80)


def print_solution(
    solution: List[Dict],
    total_penalty: float,
    parsed: ParsedData,
) -> None:
    """
    输出最终调度方案
    Args:
        solution: 调度方案列表
        total_penalty: 总延迟惩罚
        parsed: 解析后的数据
    """
    print("\n" + "=" * 80)
    print("最终调度方案")
    print("=" * 80)
    print(f"总延迟惩罚: {total_penalty:.2f}")
    print("\n时刻 | 任务ID | 人员ID | 动作类型 | 说明")
    print("-" * 80)
    # 按时间排序
    sorted_solution = sorted(solution, key=lambda x: x["time"])
    action_desc = {"assign": "分配任务", "interrupt": "中断任务", "resume": "恢复任务", "continue": "继续执行"}
    for item in sorted_solution:
        action_type = item["action"]
        desc = action_desc.get(action_type, action_type)
        print(f"{item['time']:5d} | {item['task_id']:6d} | {item['person_id']:6d} | {action_type:8s} | {desc}")

    # 输出每个时刻的任务执行情况（仅保留当时真正处于“运行中”的任务）
    print("\n" + "=" * 80)
    print("按时刻汇总：每个时刻正在执行的任务")
    print("=" * 80)

    # 使用与训练/构建方案相同的环境逻辑，基于动作序列重放一次，
    # 这样可以自动剔除已中断和已完成的任务
    env = init_environment(parsed)
    max_time = max(item["time"] for item in sorted_solution) if sorted_solution else 0

    # 预先按时刻分组动作，方便重放
    actions_by_time: Dict[int, List[Dict]] = {}
    for item in sorted_solution:
        t = item["time"]
        actions_by_time.setdefault(t, []).append(item)

    for t in range(max_time + 1):
        # 对齐环境时间
        env.current_time = t

        # 先执行该时刻的所有动作（assign / interrupt / resume / continue），更新运行状态
        for item in actions_by_time.get(t, []):
            action_type = item["action"]
            task_id = item["task_id"]
            person_id = item["person_id"]
            execute_action((action_type, task_id, person_id), env, parsed)

        # 显示集合 = 当前正在执行的任务 + 该时刻被分配/恢复的虚拟任务（执行时刻也需输出）
        display_pairs = set()
        for task_id in env.running_tasks:
            p = env.tasks[task_id].assigned_person
            if p is not None:
                display_pairs.add((task_id, p))
        for item in actions_by_time.get(t, []):
            if item["action"] in ("assign", "resume"):
                tid, pid = item["task_id"], item["person_id"]
                if _is_zero_duration_task(tid, parsed):
                    display_pairs.add((tid, pid))
        if display_pairs:
            running_info = ", ".join(
                f"任务{task_id}(人员{person_id})"
                for task_id, person_id in sorted(display_pairs)
            )
            print(f"时刻 {t:5d}: {running_info}")

        # 时间推进一步，按训练/构建方案中的逻辑更新剩余时间和完成情况
        for task_id in list(env.running_tasks):
            task = env.tasks[task_id]
            if task.remaining_time > 0:
                task.remaining_time -= 1
                if task.remaining_time <= 0:
                    task.status = "completed"
                    task.finish_time = t + 1
                    env.people_busy[task.assigned_person] = -1
                    env.running_tasks.remove(task_id)
                    env.completed_tasks.add(task_id)
