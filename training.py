import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

from actions import add_required_interrupts, generate_available_actions, is_action_conflict
from cost_to_go import calculate_action_set_cost_to_go, calculate_lookahead_gain
from data_models import EnvironmentState, ParsedData
from environment import get_state_features, init_environment
from qlearning import update_q
from reward import calculate_global_reward, execute_action
from state_space import ACTIONS


Action = Tuple[str, int, int]
State = Tuple[int, ...]
TrajectoryItem = Tuple[State, Action, float, State]


def _is_zero_duration_task(task_id: int, parsed: ParsedData, person_id: int | None = None) -> bool:
    """Check whether a task completes immediately once assigned or resumed."""
    if 0 <= task_id < len(parsed.average_duration) and parsed.average_duration[task_id] == 0:
        return True
    if 0 <= task_id < len(parsed.activity_people_time):
        people_times = parsed.activity_people_time[task_id]
        if not people_times:
            return False
        if person_id is None:
            return all(person_time[1] == 0 for person_time in people_times)
        return any(person == person_id and duration == 0 for person, duration in people_times)
    return False


def _copy_q_table(q_table: Dict) -> Dict:
    """Create a shallow copy of the sparse Q table per state."""
    return {state: dict(action_values) for state, action_values in q_table.items()}


def _max_q_diff(current_q: Dict, previous_q: Dict) -> float:
    """Return the maximum absolute difference between two sparse Q tables."""
    max_diff = 0.0
    all_states = set(current_q) | set(previous_q)
    for state in all_states:
        current_actions = current_q.get(state, {})
        previous_actions = previous_q.get(state, {})
        all_actions = set(current_actions) | set(previous_actions)
        for action in all_actions:
            max_diff = max(
                max_diff,
                abs(current_actions.get(action, 0.0) - previous_actions.get(action, 0.0)),
            )
    return max_diff


def _zero_duration_actions(env: EnvironmentState, parsed: ParsedData) -> List[Action]:
    actions = [
        action
        for action in generate_available_actions(env, parsed)
        if action[0] in ("assign", "resume")
        and _is_zero_duration_task(action[1], parsed, action[2])
    ]
    return sorted(actions, key=lambda action: (action[1], action[2], action[0]))


def _process_zero_duration_actions(env: EnvironmentState, parsed: ParsedData) -> List[Action]:
    """Apply all currently available zero-duration tasks as state updates."""
    executed_actions: List[Action] = []
    max_zero_steps = max(1, parsed.task_count * max(1, parsed.people_num or 1) * 2)

    for _ in range(max_zero_steps):
        instant_actions = _zero_duration_actions(env, parsed)
        if not instant_actions:
            return executed_actions

        progressed = False
        handled_tasks = set()
        for action in instant_actions:
            task_id = action[1]
            if task_id in handled_tasks or task_id in env.completed_tasks:
                continue

            before_completed = len(env.completed_tasks)
            for executable_action in add_required_interrupts([action], env):
                execute_action(executable_action, env, parsed)
                executed_actions.append(executable_action)
            progressed = progressed or len(env.completed_tasks) > before_completed
            handled_tasks.add(task_id)

        if not progressed:
            return executed_actions

    return executed_actions


def _advance_running_tasks(env: EnvironmentState) -> None:
    """Advance one unit of time for running tasks."""
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


def _execute_action_set(
    selected_actions: Sequence[Action],
    env: EnvironmentState,
    parsed: ParsedData,
) -> Tuple[List[Action], float]:
    executable_actions = add_required_interrupts(list(selected_actions), env)
    immediate_feedback = 0.0
    for action in executable_actions:
        immediate_feedback += execute_action(action, env, parsed)
    return executable_actions, immediate_feedback


def _clip(value: float, limit: float | None) -> float:
    if limit is None or limit <= 0:
        return value
    return max(-limit, min(limit, value))


def _calculate_local_reward(
    immediate_feedback: float,
    action_count: int,
    lookahead_gain: float,
    immediate_reward_weight: float,
    lookahead_reward_weight: float,
    normalize_immediate_reward: bool,
    lookahead_clip: float | None,
) -> float:
    if normalize_immediate_reward:
        immediate_feedback = immediate_feedback / max(1, action_count)
    lookahead_gain = _clip(lookahead_gain, lookahead_clip)
    return (
        immediate_reward_weight * immediate_feedback
        + lookahead_reward_weight * lookahead_gain
    )


def _append_solution_actions(solution: List[Dict], env: EnvironmentState, actions: Sequence[Action]) -> None:
    for action_type, task_id, person_id in actions:
        solution.append(
            {
                "time": env.current_time,
                "task_id": task_id,
                "person_id": person_id,
                "action": action_type,
            }
        )


def _select_training_action_set(
    Q_table: Dict,
    state: State,
    available_actions: List[Action],
    parsed: ParsedData,
    epsilon: float,
) -> List[Action]:
    selected_actions: List[Action] = []
    candidate_actions = available_actions.copy()

    while candidate_actions:
        if np.random.rand() < epsilon:
            selected_action = random.choice(candidate_actions)
        else:
            selected_action = max(
                candidate_actions,
                key=lambda action: Q_table.get(state, {}).get(action, 0.0),
            )

        selected_actions.append(selected_action)
        candidate_actions = [
            action
            for action in candidate_actions
            if not is_action_conflict(action, selected_action, parsed)
        ]

    return selected_actions


def _q_values_are_distinct(q_values: List[float], threshold: float) -> bool:
    if not q_values or all(abs(value) <= threshold for value in q_values):
        return False
    if len(q_values) == 1:
        return True
    ordered = sorted(q_values, reverse=True)
    return ordered[0] - ordered[1] > threshold


def _beam_candidates(
    available_actions: List[Action],
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


def _beam_metric(
    selected_actions: Tuple[Action, ...],
    Q_table: Dict,
    state: State,
    parsed: ParsedData,
    env: EnvironmentState,
) -> Tuple[float, float, int, Tuple[Action, ...]]:
    executable_actions = add_required_interrupts(list(selected_actions), env)
    cost_to_go = calculate_action_set_cost_to_go(executable_actions, parsed, env)
    q_score = sum(Q_table.get(state, {}).get(action, 0.0) for action in selected_actions)
    return cost_to_go, q_score, len(selected_actions), selected_actions


def _rank_beam_sets(
    action_sets: List[Tuple[Action, ...]],
    Q_table: Dict,
    state: State,
    parsed: ParsedData,
    env: EnvironmentState,
    q_distinction_threshold: float,
) -> List[Tuple[Action, ...]]:
    del q_distinction_threshold
    metrics = {
        actions: _beam_metric(actions, Q_table, state, parsed, env)
        for actions in action_sets
    }

    def sort_key(actions: Tuple[Action, ...]) -> Tuple:
        cost_to_go, q_score, action_count, normalized_actions = metrics[actions]
        return (cost_to_go, -q_score, -action_count, normalized_actions)

    return sorted(action_sets, key=sort_key)


def _rank_candidate_actions(
    candidates: List[Action],
    selected_actions: Tuple[Action, ...],
    Q_table: Dict,
    state: State,
    parsed: ParsedData,
    env: EnvironmentState,
    q_distinction_threshold: float,
) -> List[Action]:
    q_values = [Q_table.get(state, {}).get(action, 0.0) for action in candidates]
    q_is_distinct = _q_values_are_distinct(q_values, q_distinction_threshold)

    def sort_key(action: Action) -> Tuple:
        q_value = Q_table.get(state, {}).get(action, 0.0)
        if q_is_distinct:
            return (-q_value, action)
        action_set = add_required_interrupts(list(selected_actions + (action,)), env)
        return (calculate_action_set_cost_to_go(action_set, parsed, env), -q_value, action)

    return sorted(candidates, key=sort_key)


def _select_greedy_final_action_set(
    Q_table: Dict,
    state: State,
    available_actions: List[Action],
    parsed: ParsedData,
    env: EnvironmentState,
    q_distinction_threshold: float,
) -> List[Action]:
    selected_actions: List[Action] = []
    candidate_actions = available_actions.copy()

    while candidate_actions:
        q_values = [Q_table.get(state, {}).get(action, 0.0) for action in candidate_actions]
        if _q_values_are_distinct(q_values, q_distinction_threshold):
            selected_action = max(
                candidate_actions,
                key=lambda action: Q_table.get(state, {}).get(action, 0.0),
            )
        else:
            selected_action = min(
                candidate_actions,
                key=lambda action: calculate_action_set_cost_to_go(
                    add_required_interrupts(selected_actions + [action], env),
                    parsed,
                    env,
                ),
            )

        selected_actions.append(selected_action)
        candidate_actions = [
            action
            for action in candidate_actions
            if not is_action_conflict(action, selected_action, parsed)
        ]

    return selected_actions


def _select_beam_final_action_set(
    Q_table: Dict,
    state: State,
    available_actions: List[Action],
    parsed: ParsedData,
    env: EnvironmentState,
    q_distinction_threshold: float,
    beam_width: int,
    beam_branch_limit: int | None,
) -> List[Action]:
    beam_width = max(1, beam_width)
    beams: List[Tuple[Action, ...]] = [tuple()]
    completed: List[Tuple[Action, ...]] = []

    while beams:
        expanded: List[Tuple[Action, ...]] = []
        for selected_actions in beams:
            candidates = _beam_candidates(available_actions, selected_actions, parsed)
            if not candidates:
                completed.append(selected_actions)
                continue

            ranked_candidates = _rank_candidate_actions(
                candidates,
                selected_actions,
                Q_table,
                state,
                parsed,
                env,
                q_distinction_threshold,
            )
            if beam_branch_limit and beam_branch_limit > 0:
                ranked_candidates = ranked_candidates[:beam_branch_limit]

            for action in ranked_candidates:
                expanded.append(selected_actions + (action,))

        if not expanded:
            break

        deduped = list(dict.fromkeys(expanded))
        beams = _rank_beam_sets(
            deduped,
            Q_table,
            state,
            parsed,
            env,
            q_distinction_threshold,
        )[:beam_width]

    action_sets = completed or beams
    if not action_sets:
        return []
    best_actions = _rank_beam_sets(
        action_sets,
        Q_table,
        state,
        parsed,
        env,
        q_distinction_threshold,
    )[0]
    return list(best_actions)


def _select_final_action_set(
    Q_table: Dict,
    state: State,
    available_actions: List[Action],
    parsed: ParsedData,
    env: EnvironmentState,
    q_distinction_threshold: float,
    beam_width: int = 3,
    beam_branch_limit: int | None = 8,
    beam_improvement_margin: float = 0.1,
) -> List[Action]:
    greedy_actions = _select_greedy_final_action_set(
        Q_table,
        state,
        available_actions,
        parsed,
        env,
        q_distinction_threshold,
    )

    q_values = [Q_table.get(state, {}).get(action, 0.0) for action in available_actions]
    if beam_width <= 1 or _q_values_are_distinct(q_values, q_distinction_threshold):
        return greedy_actions

    beam_actions = _select_beam_final_action_set(
        Q_table,
        state,
        available_actions,
        parsed,
        env,
        q_distinction_threshold,
        beam_width,
        beam_branch_limit,
    )
    greedy_cost = calculate_action_set_cost_to_go(
        add_required_interrupts(greedy_actions, env),
        parsed,
        env,
    )
    beam_cost = calculate_action_set_cost_to_go(
        add_required_interrupts(beam_actions, env),
        parsed,
        env,
    )
    if beam_cost + beam_improvement_margin < greedy_cost:
        return beam_actions
    return greedy_actions


def _update_q_from_trajectory(
    Q_table: Dict,
    trajectory: List[TrajectoryItem],
    global_reward: float,
    alpha: float,
    gamma: float,
    local_reward_scale: float,
    global_reward_scale: float,
) -> None:
    for traj_state, traj_action, traj_reward, traj_next in trajectory:
        reward_for_update = (
            local_reward_scale * traj_reward
            + global_reward_scale * global_reward
        )
        update_q(
            Q_table,
            traj_state,
            traj_action,
            reward_for_update,
            traj_next,
            alpha,
            gamma,
        )


def sample_training_loop(
    Q_table: Dict,
    parsed: ParsedData,
    max_rollouts: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    convergence_threshold: float = 1e-4,
    max_skill_level: int = 4,
    global_reward_scale: float = 0.3,
    local_reward_scale: float = 0.2,
    immediate_reward_weight: float = 1.0,
    lookahead_reward_weight: float = 1.0,
    normalize_immediate_reward: bool = True,
    lookahead_clip: float | None = 1.0,
    verbose: bool = True,
    log_interval: int = 10,
    return_metrics: bool = False,
) -> Dict:
    """Run rollout-based Q-learning and return the trained Q table."""
    if verbose:
        print(f"Planned rollouts: {max_rollouts}")
    prev_q = _copy_q_table(Q_table)
    converged = False
    completed_rollouts = 0
    last_max_diff = float("inf")

    for rollout_i in range(1, max_rollouts + 1):
        alpha_eff = alpha * (0.995 ** (rollout_i - 1))

        if rollout_i > 1:
            last_max_diff = _max_q_diff(Q_table, prev_q)
            if last_max_diff < convergence_threshold:
                completed_rollouts = rollout_i - 1
                if verbose:
                    print(
                        f"Q table converged after rollout {completed_rollouts}, "
                        f"max diff={last_max_diff:.6f}"
                    )
                converged = True
                break
        prev_q = _copy_q_table(Q_table)

        env = init_environment(parsed)
        trajectory: List[TrajectoryItem] = []
        max_steps = 10000
        step = 0

        while step < max_steps:
            step += 1

            _process_zero_duration_actions(env, parsed)
            state_features = get_state_features(env, parsed, max_skill_level)

            if len(env.completed_tasks) == parsed.task_count:
                global_reward = calculate_global_reward(env, parsed)
                _update_q_from_trajectory(
                    Q_table,
                    trajectory,
                    global_reward,
                    alpha_eff,
                    gamma,
                    local_reward_scale,
                    global_reward_scale,
                )
                break

            available_actions = generate_available_actions(env, parsed)
            if not available_actions:
                if not env.running_tasks:
                    break
                _advance_running_tasks(env)
                _process_zero_duration_actions(env, parsed)
                env.current_time += 1
                continue

            selected_actions = _select_training_action_set(
                Q_table,
                state_features,
                available_actions,
                parsed,
                epsilon,
            )
            executable_actions = add_required_interrupts(selected_actions, env)
            lookahead_gain = calculate_lookahead_gain(executable_actions, parsed, env)

            _executed_actions, immediate_feedback = _execute_action_set(
                selected_actions,
                env,
                parsed,
            )
            local_reward = _calculate_local_reward(
                immediate_feedback=immediate_feedback,
                action_count=len(executable_actions),
                lookahead_gain=lookahead_gain,
                immediate_reward_weight=immediate_reward_weight,
                lookahead_reward_weight=lookahead_reward_weight,
                normalize_immediate_reward=normalize_immediate_reward,
                lookahead_clip=lookahead_clip,
            )

            _advance_running_tasks(env)
            _process_zero_duration_actions(env, parsed)
            env.current_time += 1
            next_state_features = get_state_features(env, parsed, max_skill_level)

            for action in selected_actions:
                if action[0] in ACTIONS:
                    trajectory.append((state_features, action, local_reward, next_state_features))

        completed_rollouts = rollout_i
        if verbose and log_interval > 0 and rollout_i % log_interval == 0:
            print(f"Completed rollout {rollout_i}/{max_rollouts}")
            _, penalty = build_final_solution(Q_table, parsed, max_skill_level)
            print(f"  Current decoded penalty: {penalty:.2f}")

    if verbose:
        if converged:
            print("Training stopped because the Q table converged.")
        else:
            print(f"Training stopped after reaching max_rollouts={max_rollouts}.")
    if return_metrics:
        return Q_table, {
            "converged": converged,
            "rollouts": completed_rollouts,
            "max_q_diff": last_max_diff,
        }
    return Q_table


def build_final_solution(
    Q_table: Dict,
    parsed: ParsedData,
    max_skill_level: int = 4,
    q_distinction_threshold: float = 1e-9,
    beam_width: int = 3,
    beam_branch_limit: int | None = 8,
    beam_improvement_margin: float = 0.1,
) -> Tuple[List[Dict], float]:
    """Build the final schedule greedily from the trained Q table."""
    env = init_environment(parsed)
    solution = []
    max_steps = 10000
    step = 0

    while step < max_steps:
        step += 1

        instant_actions = _process_zero_duration_actions(env, parsed)
        _append_solution_actions(solution, env, instant_actions)

        state_features = get_state_features(env, parsed, max_skill_level)
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

        selected_actions = _select_final_action_set(
            Q_table,
            state_features,
            available_actions,
            parsed,
            env,
            q_distinction_threshold,
            beam_width,
            beam_branch_limit,
            beam_improvement_margin,
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
