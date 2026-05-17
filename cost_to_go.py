import copy
import time
from typing import List, Sequence, Tuple

from actions import add_required_interrupts, generate_available_actions, is_action_conflict
from data_models import EnvironmentState, ParsedData
from reward import calculate_global_reward_for_cost_to_go, execute_action


Action = Tuple[str, int, int]
MAX_SIMULATION_STEPS = 10000
_COST_TO_GO_CACHE: dict[Tuple, float] = {}
_CACHE_HITS = 0
_CACHE_MISSES = 0
_CTG_CALLS = 0
_CTG_CALL_TIME = 0.0
_CTG_SIMULATION_CALLS = 0
_CTG_SIMULATION_TIME = 0.0


def clear_cost_to_go_cache() -> None:
    """Clear cached cost-to-go simulations and reset diagnostics."""
    global _CACHE_HITS, _CACHE_MISSES
    global _CTG_CALLS, _CTG_CALL_TIME, _CTG_SIMULATION_CALLS, _CTG_SIMULATION_TIME
    _COST_TO_GO_CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0
    _CTG_CALLS = 0
    _CTG_CALL_TIME = 0.0
    _CTG_SIMULATION_CALLS = 0
    _CTG_SIMULATION_TIME = 0.0


def get_cost_to_go_cache_info() -> dict[str, int | float]:
    """Return lightweight cache and runtime diagnostics."""
    avg_ctg_time = _CTG_CALL_TIME / _CTG_CALLS if _CTG_CALLS else 0.0
    avg_simulation_time = (
        _CTG_SIMULATION_TIME / _CTG_SIMULATION_CALLS
        if _CTG_SIMULATION_CALLS
        else 0.0
    )
    return {
        "size": len(_COST_TO_GO_CACHE),
        "hits": _CACHE_HITS,
        "misses": _CACHE_MISSES,
        "ctg_calls": _CTG_CALLS,
        "ctg_call_time": _CTG_CALL_TIME,
        "avg_ctg_time": avg_ctg_time,
        "ctg_simulation_calls": _CTG_SIMULATION_CALLS,
        "ctg_simulation_time": _CTG_SIMULATION_TIME,
        "avg_ctg_simulation_time": avg_simulation_time,
    }


def _task_state_signature(env: EnvironmentState) -> Tuple:
    return tuple(
        (
            task.task_id,
            task.status,
            task.remaining_time,
            task.assigned_person,
            task.start_time,
            task.finish_time,
        )
        for task in env.tasks
    )


def _cost_to_go_cache_key(
    actions: Sequence[Action],
    parsed: ParsedData,
    env: EnvironmentState,
) -> Tuple:
    return (
        id(parsed),
        env.current_time,
        tuple(actions),
        tuple(sorted(env.people_busy.items())),
        tuple(sorted(env.completed_tasks)),
        tuple(sorted(env.running_tasks)),
        tuple(sorted(env.interrupted_tasks.items())),
        _task_state_signature(env),
    )


def _is_zero_duration_task(
    task_id: int,
    parsed: ParsedData,
    person_id: int | None = None,
) -> bool:
    """Check whether a task should complete immediately."""
    if 0 <= task_id < len(parsed.average_duration) and parsed.average_duration[task_id] == 0:
        return True
    if 0 <= task_id < len(parsed.activity_people_time):
        people_times = parsed.activity_people_time[task_id]
        if not people_times:
            return False
        if person_id is None:
            return all(person_time[1] == 0 for person_time in people_times)
        for candidate_person, candidate_time in people_times:
            if candidate_person == person_id:
                return candidate_time == 0
    return False


def _duration_for_person(task_id: int, person_id: int, parsed: ParsedData) -> int | None:
    if task_id >= len(parsed.activity_people_time):
        return None
    for candidate_person, duration in parsed.activity_people_time[task_id]:
        if candidate_person == person_id:
            return duration
    return None


def _task_priority_key(task_id: int, parsed: ParsedData) -> Tuple[int, int, int]:
    priority = parsed.priority_value[task_id] if task_id < len(parsed.priority_value) else 0
    slack = parsed.time_difference[task_id] if task_id < len(parsed.time_difference) else 0
    return (-priority, slack, task_id)


def _action_finish_time(action: Action, env: EnvironmentState, parsed: ParsedData) -> float:
    action_type, task_id, person_id = action
    if action_type == "assign":
        duration = _duration_for_person(task_id, person_id, parsed)
        if duration is None:
            return float("inf")
    elif action_type in ("resume", "continue"):
        duration = max(0, env.tasks[task_id].remaining_time)
    else:
        duration = 0
    return env.current_time + duration


def _action_priority_key(action: Action, env: EnvironmentState, parsed: ParsedData) -> Tuple:
    action_type, task_id, person_id = action
    action_order = {"resume": 0, "continue": 1, "assign": 2, "interrupt": 3}
    return (
        *_task_priority_key(task_id, parsed),
        _action_finish_time(action, env, parsed),
        person_id,
        action_order.get(action_type, 99),
    )


def _advance_running_tasks(env: EnvironmentState) -> None:
    for task_id in list(env.running_tasks):
        task = env.tasks[task_id]
        if task.remaining_time > 0:
            task.remaining_time -= 1
            if task.remaining_time <= 0:
                task.status = "completed"
                task.finish_time = env.current_time
                env.people_busy[task.assigned_person] = -1
                env.running_tasks.remove(task_id)
                env.completed_tasks.add(task_id)


def _zero_duration_actions(env: EnvironmentState, parsed: ParsedData) -> List[Action]:
    actions = [
        action
        for action in generate_available_actions(env, parsed)
        if action[0] in ("assign", "resume") and _is_zero_duration_task(action[1], parsed, action[2])
    ]
    return sorted(actions, key=lambda action: _action_priority_key(action, env, parsed))


def _process_zero_duration_actions(env: EnvironmentState, parsed: ParsedData) -> float:
    """Complete all currently available zero-duration tasks and return their reward."""
    total_reward = 0.0
    for _ in range(MAX_SIMULATION_STEPS):
        instant_actions = _zero_duration_actions(env, parsed)
        if not instant_actions:
            return total_reward

        progressed = False
        handled_tasks = set()
        for action in instant_actions:
            task_id = action[1]
            if task_id in handled_tasks or task_id in env.completed_tasks:
                continue

            before_completed = len(env.completed_tasks)
            for executable_action in add_required_interrupts([action], env):
                total_reward += execute_action(executable_action, env, parsed)
            progressed = progressed or len(env.completed_tasks) > before_completed
            handled_tasks.add(task_id)

        if not progressed:
            return total_reward
    return total_reward


def _execute_action_set(
    actions: Sequence[Action],
    env: EnvironmentState,
    parsed: ParsedData,
) -> float:
    total_reward = 0.0
    for action in add_required_interrupts(list(actions), env):
        total_reward += execute_action(action, env, parsed)
    return total_reward


def _advance_one_decision_time(env: EnvironmentState, parsed: ParsedData) -> None:
    _advance_running_tasks(env)
    _process_zero_duration_actions(env, parsed)
    env.current_time += 1


def build_heuristic_action_set(env: EnvironmentState, parsed: ParsedData) -> List[Action]:
    """Build the baseline action set using the document's heuristic rules."""
    selected_actions: List[Action] = []
    candidate_actions = generate_available_actions(env, parsed)

    while candidate_actions:
        best_action = min(
            candidate_actions,
            key=lambda action: _action_priority_key(action, env, parsed),
        )
        selected_actions.append(best_action)
        candidate_actions = [
            action
            for action in candidate_actions
            if not is_action_conflict(action, best_action, parsed)
        ]

    return add_required_interrupts(selected_actions, env)


def calculate_action_set_cost_to_go(
    actions: Sequence[Action],
    parsed: ParsedData,
    env: EnvironmentState,
    use_cache: bool = True,
) -> float:
    """Simulate a fixed current action set, then finish by heuristic rollout."""
    global _CACHE_HITS, _CACHE_MISSES, _CTG_CALLS, _CTG_CALL_TIME
    global _CTG_SIMULATION_CALLS, _CTG_SIMULATION_TIME
    call_started = time.perf_counter()
    _CTG_CALLS += 1
    try:
        cache_key = _cost_to_go_cache_key(actions, parsed, env)
        if use_cache and cache_key in _COST_TO_GO_CACHE:
            _CACHE_HITS += 1
            return _COST_TO_GO_CACHE[cache_key]
        if use_cache:
            _CACHE_MISSES += 1

        simulation_started = time.perf_counter()
        _CTG_SIMULATION_CALLS += 1
        try:
            sim_env = copy.deepcopy(env)
            _process_zero_duration_actions(sim_env, parsed)
            _execute_action_set(actions, sim_env, parsed)
            _advance_one_decision_time(sim_env, parsed)

            result = 10000.0
            for _ in range(MAX_SIMULATION_STEPS):
                _process_zero_duration_actions(sim_env, parsed)
                if len(sim_env.completed_tasks) == parsed.task_count:
                    result = calculate_global_reward_for_cost_to_go(sim_env, parsed)
                    break

                heuristic_actions = build_heuristic_action_set(sim_env, parsed)
                if heuristic_actions:
                    _execute_action_set(heuristic_actions, sim_env, parsed)
                elif not sim_env.running_tasks:
                    result = 10000.0
                    break

                _advance_one_decision_time(sim_env, parsed)
        finally:
            _CTG_SIMULATION_TIME += time.perf_counter() - simulation_started

        if use_cache:
            _COST_TO_GO_CACHE[cache_key] = result
        return result
    finally:
        _CTG_CALL_TIME += time.perf_counter() - call_started


def calculate_baseline_cost_to_go(parsed: ParsedData, env: EnvironmentState) -> float:
    baseline_actions = build_heuristic_action_set(env, parsed)
    return calculate_action_set_cost_to_go(baseline_actions, parsed, env)


def calculate_lookahead_gain(
    actions: Sequence[Action],
    parsed: ParsedData,
    env: EnvironmentState,
    epsilon: float = 1e-6,
) -> float:
    set_cost = calculate_action_set_cost_to_go(actions, parsed, env)
    baseline_cost = calculate_baseline_cost_to_go(parsed, env)
    if baseline_cost <= epsilon:
        return 0.0 if set_cost <= epsilon else -set_cost
    return (baseline_cost - set_cost) / max(baseline_cost, epsilon)


def calculate_cost_to_go(
    action: Action,
    parsed: ParsedData,
    env: EnvironmentState,
) -> float:
    """Compatibility wrapper for single-action fallback callers."""
    return calculate_action_set_cost_to_go([action], parsed, env)
