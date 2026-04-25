import itertools
from typing import List, Optional, Tuple


# NLF: network level feature, discretized into five levels.
network_levels = [0, 1, 2, 3, 4]

# MUR: milestone urgency rate, discretized into five levels.
milestone_urgency_levels = [0, 1, 2, 3, 4]

# RUR: resource utilization rate, discretized into five levels.
resource_utilization_levels = [0, 1, 2, 3, 4]

# CRT: critical running task pressure, discretized into five levels.
critical_running_task_levels = [0, 1, 2, 3, 4]

# ITN: interrupted task number, discretized into five levels.
interrupted_task_levels = [0, 1, 2, 3, 4]


def get_skill_bottleneck_levels(people_num: Optional[int]) -> List[int]:
    """Return SBI levels based on the number of available people."""
    if people_num is None:
        return [0, 1, 2, 3, 4]
    if people_num <= 5:
        return list(range(people_num))
    return [0, 1, 2, 3, 4]


skill_bottleneck_levels = [0, 1, 2, 3, 4]


def generate_state_space(people_num: Optional[int]) -> List[Tuple[int, ...]]:
    """Generate the discrete state space: (NLF, SBI, MUR, RUR, CRT, ITN)."""
    skill_levels = get_skill_bottleneck_levels(people_num)
    return list(
        itertools.product(
            network_levels,
            skill_levels,
            milestone_urgency_levels,
            resource_utilization_levels,
            critical_running_task_levels,
            interrupted_task_levels,
        )
    )


state_space = list(
    itertools.product(
        network_levels,
        skill_bottleneck_levels,
        milestone_urgency_levels,
        resource_utilization_levels,
        critical_running_task_levels,
        interrupted_task_levels,
    )
)

# Interrupt is an automatic resource adjustment, not a learned decision action.
ACTIONS = ["assign", "resume", "continue"]
