# 数据模型：解析结果、任务状态、环境状态

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ParsedData:
    """存储完整的解析结果"""

    task_count: int
    adjacency_matrix: List[List[int]] = field(default_factory=list)
    people_num: Optional[int] = None  # 服务人员总数（**段的第一行）
    activity_people_time: List[List[List[int]]] = field(default_factory=list)  # 每个任务的人员和时间列表，格式：[[[人员ID, 时间], ...], ...]
    milestone_count: Optional[int] = None
    milestone_event: List[int] = field(default_factory=list)  # 里程碑事件（任务号）列表
    milestone_time: List[int] = field(default_factory=list)  # 里程碑时间列表，与milestone_event对应
    tail_section: List[List[int]] = field(default_factory=list)  # 尾段
    immediate_predecessors: List[List[int]] = field(default_factory=list)  # 紧前活动集合
    immediate_successors: List[List[int]] = field(default_factory=list)  # 紧后活动集合
    priority_value: List[int] = field(default_factory=list)  # 任务优先值列表，值越大优先级越高
    average_duration: List[int] = field(default_factory=list)  # 任务平均处理时间列表
    earliest_start: List[int] = field(default_factory=list)  # 最早开始时间列表
    latest_start: List[int] = field(default_factory=list)  # 最晚开始时间列表
    time_difference: List[int] = field(default_factory=list)  # 时间差列表（总时差）


@dataclass
class TaskState:
    """任务状态"""

    task_id: int
    status: str  # "not_started", "running", "completed", "interrupted"
    assigned_person: Optional[int] = None  # 分配的人员ID
    remaining_time: int = 0  # 剩余执行时间
    original_time: int = 0  # 原始执行时间
    start_time: Optional[int] = None  # 开始执行时间
    finish_time: Optional[int] = None  # 完成时间


@dataclass
class EnvironmentState:
    """环境状态"""

    current_time: int
    tasks: List[TaskState]  # 所有任务的状态
    people_busy: Dict[int, int]  # {人员ID: 任务ID}，-1表示空闲
    completed_tasks: set  # 已完成的任务集合
    running_tasks: set  # 正在执行的任务集合
    interrupted_tasks: Dict[int, int]  # {任务ID: 中断时的人员ID}，用于恢复
