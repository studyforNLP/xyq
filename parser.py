# 数据文件解析：读取并解析调度数据文件
#
# 数据格式说明：
#   - 首行：任务总数
#   - *    ：任务邻接矩阵（持续到下一个标记或文件结束）
#   - **   ：服务人员总数
#   - ***  ：每个任务可执行它的人员数量
#   - **** ：活动+人员+时长（每个任务的人员执行时间明细）
#   - *****：里程碑数量
#   - ******：里程碑事件（第一行）和发生时间（第二行）

import re
from pathlib import Path
from typing import List, Optional, Tuple

from config import MARKERS
from data_models import ParsedData


def _collect_until_marker(lines: List[str], start_idx: int) -> Tuple[List[str], int]:
    """收集从 start_idx 开始直到遇到下一标记前的所有行"""
    collected: List[str] = []
    idx = start_idx
    while idx < len(lines) and lines[idx] not in MARKERS:
        collected.append(lines[idx])
        idx += 1
    return collected, idx


def _rows_to_int_matrix(rows: List[str]) -> List[List[int]]:
    """将每行拆分为整数列表"""
    matrix: List[List[int]] = []
    for row in rows:
        if not row:
            continue
        parts = row.replace("\t", " ").split()
        matrix.append([int(p) for p in parts])
    return matrix


def _calculate_priority_value(
    activity_num: int,
    adjacency_matrix: List[List[int]],
    milestone_event: List[int],
) -> List[int]:
    """
    计算任务优先值（严格按照Java代码逻辑实现）

    优先值逻辑：里程碑事件关联的所有前序相关任务都应该优先安排。
    通过对里程碑事件的前序相关任务进行计数来设置优先值。
    起始事件（0）和结束事件（activity_num-1）虽然都是里程碑事件，但不包含在优先值的计数中。

    Args:
        activity_num: 任务总数
        adjacency_matrix: 邻接矩阵，adjacency_matrix[i][j] == 1 表示任务i是任务j的前序
        milestone_event: 里程碑事件（任务号）列表

    Returns:
        任务优先值列表，值越大优先级越高
    """
    # 初始化所有任务的优先值为0
    priority_value = [0] * activity_num
    # 获取所有里程碑事件（排除结束事件，即activity_num-1）
    mile = []
    for i in range(len(milestone_event)):
        if milestone_event[i] != activity_num - 1:
            mile.append(milestone_event[i])

    for integer in mile:
        # 里程碑事件本身也要计算优先值，优先值+1
        priority_value[integer] = priority_value[integer] + 1
        # preActivity存储每一层的前序任务
        pre_activity: List[List[int]] = []
        temp_pre_activity = [integer]
        pre_activity.append(temp_pre_activity)
        # 继续向上搜索，直到到达起始事件（0）
        while pre_activity[-1][0] != 0:
            current_layer = pre_activity[-1]
            for k in range(len(current_layer)):
                temp = []
                n = 0
                # 查找任务current_layer[k]的前序任务
                for j in range(len(adjacency_matrix[0])):
                    if adjacency_matrix[j][current_layer[k]] == 1:
                        priority_value[j] = priority_value[j] + 1
                        temp.append(j)
                        n += 1
                if n == 0:
                    temp.append(0)
                pre_activity.append(temp)
    # 起始事件的优先值设为0
    priority_value[0] = 0
    return priority_value


def _calculate_time_difference(
    activity_num: int,
    activity_people_time: List[List[List[int]]],
    immediate_predecessors: List[List[int]],
    immediate_successors: List[List[int]],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    计算各任务的总时差

    步骤：
    1. 先对每个任务各人员的处理时间进行平均值四舍五入取整得到任务的平均处理时间
    2. 正向计算：对项目网络图进行正向计算，得到任务最早开始时间
    3. 逆向计算：对项目网络图进行逆向计算，得到任务最晚开始时间
    4. 计算总时差：最晚开始时间 - 最早开始时间

    Returns:
        (average_duration, earliest_start, latest_start, time_difference)
    """
    # 步骤1：计算每个任务的平均处理时间
    average_duration = [0] * activity_num
    for i in range(activity_num):
        if len(activity_people_time[i]) > 0:
            total_time = sum(activity_people_time[i][j][1] for j in range(len(activity_people_time[i])))
            average = total_time / len(activity_people_time[i])
            average_duration[i] = int(average + 0.5)
        else:
            average_duration[i] = 0

    # 步骤2：正向计算 - 计算最早开始时间
    earliest_start = [0] * activity_num
    earliest_finish = [0] * activity_num
    for i in range(activity_num):
        if immediate_predecessors[i][0] == -1:
            earliest_start[i] = 0
            earliest_finish[i] = earliest_start[i] + average_duration[i]
        else:
            max_finish_time = float("-inf")
            for j in range(len(immediate_predecessors[i])):
                predecessor = immediate_predecessors[i][j]
                finish_time = earliest_finish[predecessor]
                if finish_time > max_finish_time:
                    max_finish_time = finish_time
            earliest_start[i] = max_finish_time
            earliest_finish[i] = earliest_start[i] + average_duration[i]

    # 步骤3：逆向计算 - 计算最晚开始时间
    latest_start = [0] * activity_num
    latest_finish = [0] * activity_num
    latest_finish[activity_num - 1] = earliest_finish[activity_num - 1]
    latest_start[activity_num - 1] = latest_finish[activity_num - 1] - average_duration[activity_num - 1]
    for i in range(activity_num - 2, -1, -1):
        if immediate_successors[i][0] == -1:
            latest_finish[i] = earliest_finish[activity_num - 1]
            latest_start[i] = latest_finish[i] - average_duration[i]
        else:
            min_start_time = float("inf")
            for j in range(len(immediate_successors[i])):
                successor = immediate_successors[i][j]
                start_time = latest_start[successor]
                if start_time < min_start_time:
                    min_start_time = start_time
            latest_finish[i] = min_start_time
            latest_start[i] = latest_finish[i] - average_duration[i]

    # 步骤4：计算总时差 = 最晚开始时间 - 最早开始时间
    time_difference = [latest_start[i] - earliest_start[i] for i in range(activity_num)]
    return average_duration, earliest_start, latest_start, time_difference


def parse_custom_file(path: Path) -> ParsedData:
    """按约定格式解析文件"""
    with path.open("r", encoding="utf-8-sig") as f:
        raw_lines = [line.strip() for line in f]

    lines: List[str] = [ln for ln in raw_lines if ln or ln in MARKERS]
    if not lines:
        raise ValueError("文件为空或仅包含空白行。")

    idx = 0
    task_count = int(lines[idx])
    idx += 1

    def _expect(marker: str, current_idx: int) -> int:
        if current_idx >= len(lines) or lines[current_idx] != marker:
            raise ValueError(f"期望出现标记 {marker}，但在位置 {current_idx} 未找到。")
        return current_idx + 1

    idx = _expect("*", idx)
    matrix_rows, idx = _collect_until_marker(lines, idx)
    adjacency_matrix = _rows_to_int_matrix(matrix_rows)

    # 根据邻接矩阵计算紧前与紧后集合
    activity_num = len(adjacency_matrix)
    predecessors = [[] for _ in range(activity_num)]
    successors = [[] for _ in range(activity_num)]
    for i in range(activity_num):
        for j in range(activity_num):
            if adjacency_matrix[i][j] == 1:
                predecessors[j].append(i)
                successors[i].append(j)
    if activity_num > 0:
        predecessors[0].append(-1)
        successors[activity_num - 1].append(-1)
    immediate_predecessors = [list(pred) for pred in predecessors]
    immediate_successors = [list(succ) for succ in successors]

    # 读取服务人员总数（**段的第一行）
    people_num: Optional[int] = None
    idx = _expect("**", idx)
    if idx < len(lines):
        first_line = lines[idx].replace("\t", " ").split()
        if first_line:
            people_num = int(first_line[0])
        idx += 1

    # 读取每个活动的人员数量（***段）
    peopel_num_list: List[int] = []
    idx = _expect("***", idx)
    worker_count_rows, idx = _collect_until_marker(lines, idx)
    for row in worker_count_rows:
        if not row:
            continue
        str1 = row.replace("\t", " ")
        parts = [p for p in str1.split() if p]
        if len(parts) >= 2:
            peopel_num_list.append(int(parts[1]))  # 提取第二个数字作为人员数量

    # 读取活动+人+时长（****段）
    activity_people_time: List[List[List[int]]] = []
    if idx < len(lines) and lines[idx] == "****":
        idx += 1
        # 读取活动+人员+时长
        for i in range(activity_num):
            n = peopel_num_list[i] if i < len(peopel_num_list) else 0
            ptime: List[List[int]] = []
            for j in range(n):
                if idx >= len(lines):
                    break
                str1 = lines[idx].replace("\t", " ")
                parts = [p for p in str1.split() if p]
                if len(parts) >= 2:
                    ptime.append([int(parts[0]), int(parts[1])])
                idx += 1
            activity_people_time.append(ptime)

    # 读取里程碑数量（*****段）
    milestone_count: Optional[int] = None
    if idx < len(lines) and lines[idx] == "*****":
        idx += 1
        if idx < len(lines):
            first_line = lines[idx].replace("\t", " ").split()
            if first_line and first_line[0].isdigit():
                milestone_count = int(first_line[0])
            idx += 1

    # 读取里程碑事件和时间（******段）
    milestone_event: List[int] = []
    milestone_time: List[int] = []
    if idx < len(lines) and lines[idx] == "******":
        idx += 1
        # 第一行：里程碑事件（任务号）
        if idx < len(lines) and milestone_count is not None:
            str1 = re.sub(r"\D+", " ", lines[idx])
            parts = [p for p in str1.split() if p]
            for i in range(milestone_count):
                if i < len(parts):
                    milestone_event.append(int(parts[i]))
            idx += 1
        # 第二行：里程碑时间
        if idx < len(lines) and milestone_count is not None:
            str1 = re.sub(r"\D+", " ", lines[idx])
            parts = [p for p in str1.split() if p]
            for i in range(milestone_count):
                if i < len(parts):
                    milestone_time.append(int(parts[i]))
            idx += 1

    # 在读取里程碑之后，立即计算任务优先值
    priority_value = _calculate_priority_value(activity_num, adjacency_matrix, milestone_event)

    # 计算各任务总时差（在计算优先值之后）
    (
        average_duration,
        earliest_start,
        latest_start,
        time_difference,
    ) = _calculate_time_difference(
        activity_num,
        activity_people_time,
        immediate_predecessors,
        immediate_successors,
    )

    tail_section: List[List[int]] = []
    return ParsedData(
        task_count=task_count,
        adjacency_matrix=adjacency_matrix,
        people_num=people_num,
        activity_people_time=activity_people_time,
        milestone_count=milestone_count,
        milestone_event=milestone_event,
        milestone_time=milestone_time,
        tail_section=tail_section,
        immediate_predecessors=immediate_predecessors,
        immediate_successors=immediate_successors,
        priority_value=priority_value,
        average_duration=average_duration,
        earliest_start=earliest_start,
        latest_start=latest_start,
        time_difference=time_difference,
    )
