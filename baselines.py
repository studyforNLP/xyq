from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Sequence, Tuple

from actions import add_required_interrupts, generate_available_actions, is_action_conflict
from cost_to_go import calculate_action_set_cost_to_go
from data_models import EnvironmentState, ParsedData
from environment import init_environment
from reward import calculate_global_reward
from training import (
    _advance_running_tasks,
    _append_solution_actions,
    _execute_action_set,
    _process_zero_duration_actions,
)


Action = Tuple[str, int, int]
RuleKey = Callable[[Action, EnvironmentState, ParsedData, Dict], Tuple]

HEURISTIC_METHODS = [
    "MPV-SLK-EFT",
    "MXS+MF",
    "SLK-EFT",
    "SPT",
    "ECT",
    "FIFO",
]
ROLLOUT_METHODS = ["RH"]
BASELINE_METHODS = HEURISTIC_METHODS + ROLLOUT_METHODS
DETERMINISTIC_METHODS = set(HEURISTIC_METHODS)
RANDOM_METHODS = {"QL", "RH", "RQL"}


def normalize_method(method: str) -> str:
    normalized = method.strip().upper().replace("_", "-")
    aliases = {
        "MPV-SLK-EFT": "MPV-SLK-EFT",
        "MPV+SLK+EFT": "MPV-SLK-EFT",
        "MXS-MF": "MXS+MF",
        "MXS+MF": "MXS+MF",
        "SLK-EFT": "SLK-EFT",
        "SPT": "SPT",
        "ECT": "ECT",
        "FIFO": "FIFO",
        "RH": "RH",
        "QL": "QL",
        "RQL": "RQL",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown method: {method}")
    return aliases[normalized]


def _duration_for_person(task_id: int, person_id: int, parsed: ParsedData) -> float:
    if task_id >= len(parsed.activity_people_time):
        return float("inf")
    for candidate_person, duration in parsed.activity_people_time[task_id]:
        if candidate_person == person_id:
            return duration
    return float("inf")


def _action_duration(action: Action, env: EnvironmentState, parsed: ParsedData) -> float:
    action_type, task_id, person_id = action
    if action_type == "assign":
        return _duration_for_person(task_id, person_id, parsed)
    if action_type in {"resume", "continue"}:
        return max(0, env.tasks[task_id].remaining_time)
    return 0.0


def _action_finish_time(action: Action, env: EnvironmentState, parsed: ParsedData) -> float:
    return env.current_time + _action_duration(action, env, parsed)


def _priority_value(task_id: int, parsed: ParsedData) -> int:
    if task_id < len(parsed.priority_value):
        return parsed.priority_value[task_id]
    return 0


def _slack(task_id: int, parsed: ParsedData) -> int:
    if task_id < len(parsed.time_difference):
        return parsed.time_difference[task_id]
    return 0


def _action_order(action_type: str) -> int:
    return {"resume": 0, "continue": 1, "assign": 2, "interrupt": 3}.get(action_type, 99)


def _successor_counts(parsed: ParsedData) -> Dict[int, int]:
    @lru_cache(maxsize=None)
    def descendants(task_id: int) -> frozenset[int]:
        successors = [
            succ
            for succ in parsed.immediate_successors[task_id]
            if succ != -1 and 0 <= succ < parsed.task_count
        ]
        result = set(successors)
        for successor in successors:
            result.update(descendants(successor))
        return frozenset(result)

    return {task_id: len(descendants(task_id)) for task_id in range(parsed.task_count)}


def _rule_key(
    method: str,
    action: Action,
    env: EnvironmentState,
    parsed: ParsedData,
    context: Dict,
) -> Tuple:
    action_type, task_id, person_id = action
    duration = _action_duration(action, env, parsed)
    finish_time = _action_finish_time(action, env, parsed)
    priority = _priority_value(task_id, parsed)
    slack = _slack(task_id, parsed)

    if method == "MPV-SLK-EFT":
        return (-priority, slack, finish_time, task_id, person_id, _action_order(action_type))
    if method == "MXS+MF":
        successor_count = context["successor_counts"].get(task_id, 0)
        return (
            -successor_count,
            -priority,
            slack,
            finish_time,
            duration,
            task_id,
            person_id,
            _action_order(action_type),
        )
    if method == "SLK-EFT":
        return (slack, finish_time, -priority, task_id, person_id, _action_order(action_type))
    if method == "SPT":
        return (duration, finish_time, slack, -priority, task_id, person_id, _action_order(action_type))
    if method == "ECT":
        return (finish_time, duration, slack, -priority, task_id, person_id, _action_order(action_type))
    if method == "FIFO":
        fifo_order = context.setdefault("fifo_order", {})
        return (
            fifo_order.get(task_id, len(fifo_order)),
            finish_time,
            duration,
            task_id,
            person_id,
            _action_order(action_type),
        )
    raise ValueError(f"Unsupported heuristic method: {method}")


def _select_rule_action_set(
    available_actions: List[Action],
    env: EnvironmentState,
    parsed: ParsedData,
    method: str,
    context: Dict,
) -> List[Action]:
    if method == "FIFO":
        fifo_order = context.setdefault("fifo_order", {})
        for action in available_actions:
            task_id = action[1]
            if task_id not in fifo_order:
                fifo_order[task_id] = len(fifo_order)

    selected_actions: List[Action] = []
    candidate_actions = available_actions.copy()

    while candidate_actions:
        best_action = min(
            candidate_actions,
            key=lambda action: _rule_key(method, action, env, parsed, context),
        )
        selected_actions.append(best_action)
        candidate_actions = [
            action
            for action in candidate_actions
            if not is_action_conflict(action, best_action, parsed)
        ]

    return selected_actions


def _rank_rollout_action_sets(
    action_sets: List[Tuple[Action, ...]],
    env: EnvironmentState,
    parsed: ParsedData,
) -> List[Tuple[Action, ...]]:
    metrics = {}
    for actions in action_sets:
        executable_actions = add_required_interrupts(list(actions), env)
        metrics[actions] = (
            calculate_action_set_cost_to_go(executable_actions, parsed, env),
            -len(actions),
            actions,
        )
    return sorted(action_sets, key=lambda actions: metrics[actions])


def _rollout_candidates(
    available_actions: Sequence[Action],
    selected_actions: Tuple[Action, ...],
    parsed: ParsedData,
) -> List[Action]:
    return [
        action
        for action in available_actions
        if action not in selected_actions
        and all(
            not is_action_conflict(action, selected_action, parsed)
            for selected_action in selected_actions
        )
    ]


def _select_rollout_action_set(
    available_actions: List[Action],
    env: EnvironmentState,
    parsed: ParsedData,
    beam_width: int = 1,
    beam_branch_limit: int | None = 8,
) -> List[Action]:
    if beam_width <= 1:
        selected_actions: List[Action] = []
        candidate_actions = available_actions.copy()
        while candidate_actions:
            best_action = min(
                candidate_actions,
                key=lambda action: calculate_action_set_cost_to_go(
                    add_required_interrupts(selected_actions + [action], env),
                    parsed,
                    env,
                ),
            )
            selected_actions.append(best_action)
            candidate_actions = [
                action
                for action in candidate_actions
                if not is_action_conflict(action, best_action, parsed)
            ]
        return selected_actions

    beams: List[Tuple[Action, ...]] = [tuple()]
    completed: List[Tuple[Action, ...]] = []
    beam_width = max(1, beam_width)
    while beams:
        expanded: List[Tuple[Action, ...]] = []
        for selected_actions in beams:
            candidates = _rollout_candidates(available_actions, selected_actions, parsed)
            if not candidates:
                completed.append(selected_actions)
                continue

            ranked_candidates = sorted(
                candidates,
                key=lambda action: calculate_action_set_cost_to_go(
                    add_required_interrupts(list(selected_actions + (action,)), env),
                    parsed,
                    env,
                ),
            )
            if beam_branch_limit and beam_branch_limit > 0:
                ranked_candidates = ranked_candidates[:beam_branch_limit]
            expanded.extend(selected_actions + (action,) for action in ranked_candidates)

        if not expanded:
            break
        beams = _rank_rollout_action_sets(list(dict.fromkeys(expanded)), env, parsed)[:beam_width]

    action_sets = completed or beams
    if not action_sets:
        return []
    return list(_rank_rollout_action_sets(action_sets, env, parsed)[0])


def build_baseline_solution(
    parsed: ParsedData,
    method: str,
    rollout_beam_width: int = 1,
    rollout_branch_limit: int | None = 8,
) -> Tuple[List[Dict], float]:
    method = normalize_method(method)
    if method not in BASELINE_METHODS:
        raise ValueError(f"{method} is not a baseline method.")

    env = init_environment(parsed)
    solution: List[Dict] = []
    context: Dict = {"successor_counts": _successor_counts(parsed)}
    max_steps = 10000

    for _ in range(max_steps):
        instant_actions = _process_zero_duration_actions(env, parsed)
        _append_solution_actions(solution, env, instant_actions)

        if len(env.completed_tasks) == parsed.task_count:
            break

        available_actions = generate_available_actions(env, parsed)
        if not available_actions:
            if not env.running_tasks:
                break
            _advance_running_tasks(env)
            instant_actions = _process_zero_duration_actions(env, parsed)
            _append_solution_actions(solution, env, instant_actions)
            env.current_time += 1
            continue

        if method == "RH":
            selected_actions = _select_rollout_action_set(
                available_actions,
                env,
                parsed,
                beam_width=rollout_beam_width,
                beam_branch_limit=rollout_branch_limit,
            )
        else:
            selected_actions = _select_rule_action_set(
                available_actions,
                env,
                parsed,
                method,
                context,
            )

        executable_actions, _immediate_feedback = _execute_action_set(
            selected_actions,
            env,
            parsed,
        )
        _append_solution_actions(solution, env, executable_actions)

        _advance_running_tasks(env)
        instant_actions = _process_zero_duration_actions(env, parsed)
        _append_solution_actions(solution, env, instant_actions)
        env.current_time += 1

    total_penalty = -calculate_global_reward(env, parsed)
    return solution, total_penalty
