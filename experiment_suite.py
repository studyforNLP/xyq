from __future__ import annotations

import argparse
import random
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from actions import add_required_interrupts, generate_available_actions
from baselines import BASELINE_METHODS, RANDOM_METHODS, build_baseline_solution, normalize_method
from cost_to_go import calculate_lookahead_gain, clear_cost_to_go_cache, get_cost_to_go_cache_info
from environment import get_state_features, init_environment
from excel_writer import write_xlsx
from metrics import build_all_summaries
from parameters import FINAL_SOLUTION_PARAMS, TRAINING_PARAMS
from parser import parse_custom_file
from qlearning import init_q_table
from reward import calculate_global_reward
from state_space import ACTIONS, generate_state_space
from training import (
    _advance_running_tasks,
    _calculate_local_reward,
    _copy_q_table,
    _execute_action_set,
    _max_q_diff,
    _process_zero_duration_actions,
    _select_training_action_set,
    _update_q_from_trajectory,
    build_final_solution,
    sample_training_loop,
)


DEFAULT_SCALES = [10, 20, 30]
DEFAULT_MAX_CASES_PER_SCALE = 50
DEFAULT_SEEDS = 3
DEFAULT_OUTPUT_XLSX = "results/formal_j10_j20_j30_50cases_3seeds3.xlsx"
DEFAULT_RQL_CONVERGENCE_PROFILE = "scale_specific"
DEFAULT_RQL_BASE_MILESTONE_PENALTY_WEIGHT_SUM = 0.8

Action = Tuple[str, int, int]
State = Tuple[int, ...]
TrajectoryItem = Tuple[State, Action, float, State]


@dataclass(frozen=True)
class RqlConvergenceProfile:
    description: str
    ep_by_scale: Dict[int, int]
    late_epsilon_min: float | None = None
    late_epsilon_start_ratio: float = 0.70
    checkpoint: bool = False
    normalize_global_reward_by_milestone_weight_sum: bool = False
    lookahead_warmup: bool = False
    disable_lookahead_clip: bool = False
    lookahead_clip: float | None = None
    alpha_decay: float = 0.995
    alpha_min: float = 0.0


RQL_CONVERGENCE_PROFILES: Dict[str, RqlConvergenceProfile] = {
    "checkpoint": RqlConvergenceProfile(
        description=(
            "Formal RQL convergence setting: scale-specific EP, lower late epsilon, "
            "alpha floor, and best-Q checkpoint selection."
        ),
        ep_by_scale={10: 500, 20: 1000, 30: 1500},
        late_epsilon_min=0.005,
        checkpoint=True,
        alpha_decay=0.99,
        alpha_min=0.01,
    ),
    "combined": RqlConvergenceProfile(
        description=(
            "Combined convergence setting: scale-specific EP, lower late epsilon, "
            "best-Q checkpoint, milestone-penalty-weight reward normalization, "
            "lookahead warm-up, and no lookahead clipping."
        ),
        ep_by_scale={10: 500, 20: 1000, 30: 1500},
        late_epsilon_min=0.005,
        checkpoint=True,
        normalize_global_reward_by_milestone_weight_sum=True,
        lookahead_warmup=True,
        disable_lookahead_clip=True,
        alpha_decay=0.99,
        alpha_min=0.01,
    ),
}

DEFAULT_METHODS = [
    "MPV-SLK-EFT",
    "MXS+MF",
    "SLK-EFT",
    "SPT",
    "ECT",
    "FIFO",
    "QL",
    "RH",
    "RQL",
]


def _natural_key(value: str) -> List:
    parts = re.split(r"(\d+)", value.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def _parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _is_learning_method(method: str) -> bool:
    return method in {"QL", "RQL"}


def _is_random_method(method: str) -> bool:
    return method in RANDOM_METHODS


def _parse_method_csv(value: str) -> List[str]:
    return [normalize_method(item) for item in value.split(",") if item.strip()]


def _resolve_rql_profile_name(scale: int, args: argparse.Namespace) -> str:
    if args.rql_convergence_profile == "scale_specific":
        return "checkpoint" if scale == 10 else "combined"
    return args.rql_convergence_profile


def _rql_profile_description(args: argparse.Namespace) -> str:
    if args.rql_convergence_profile == "off":
        return "Use the original sample_training_loop RQL path."
    if args.rql_convergence_profile == "scale_specific":
        return (
            "Scale-specific formal RQL setting: J10 uses checkpoint; "
            "J20 and J30 use combined."
        )
    return RQL_CONVERGENCE_PROFILES[args.rql_convergence_profile].description


def _max_skill_level(people_num: int | None) -> int:
    if people_num is None:
        return 4
    if people_num <= 5:
        return people_num - 1
    return 4


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _solution_stats(solution: List[Dict]) -> Dict[str, int]:
    return {
        "action_count": len(solution),
        "interrupt_count": sum(1 for item in solution if item["action"] == "interrupt"),
    }


def _discover_cases(args: argparse.Namespace) -> tuple[List[Dict], List[Dict]]:
    cases_dir = Path(args.cases_dir)
    case_rows: List[Dict] = []
    valid_cases: List[Dict] = []

    for scale in args.scales:
        scale_dir = cases_dir / f"j{scale}.mm"
        files = sorted(scale_dir.glob("*.mm"), key=lambda path: _natural_key(path.name))
        if args.max_cases_per_scale and args.max_cases_per_scale > 0:
            files = files[: args.max_cases_per_scale]

        for index, file_path in enumerate(files, start=1):
            instance_id = file_path.stem
            expected_task_count = scale + 2
            row = {
                "scale": f"J{scale}",
                "instance_id": instance_id,
                "file": file_path.name,
                "path": str(file_path),
                "task_count": "",
                "expected_task_count": expected_task_count,
                "people_num": "",
                "milestone_count": "",
                "valid": False,
                "message": "",
            }
            try:
                parsed = parse_custom_file(file_path)
                valid = parsed.task_count == expected_task_count
                row.update(
                    {
                        "task_count": parsed.task_count,
                        "people_num": parsed.people_num,
                        "milestone_count": parsed.milestone_count,
                        "valid": valid,
                        "message": "" if valid else "task_count does not equal J+2",
                    }
                )
                if valid:
                    valid_cases.append(
                        {
                            "scale": f"J{scale}",
                            "scale_value": scale,
                            "scale_index": index,
                            "instance_id": instance_id,
                            "file": file_path.name,
                            "path": file_path,
                            "parsed": parsed,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - recorded in failures/case_check.
                row["message"] = repr(exc)
            case_rows.append(row)

    return case_rows, valid_cases


def _should_record_history(case: Dict, method: str, seed: int | None, args: argparse.Namespace) -> bool:
    if not _is_learning_method(method):
        return False
    if args.convergence_cases_per_scale <= 0 or args.convergence_seeds <= 0:
        return False
    if case["scale_index"] > args.convergence_cases_per_scale:
        return False
    if seed is None or seed >= args.convergence_seeds:
        return False
    return True


def _rql_profile_max_rollouts(profile: RqlConvergenceProfile, scale: int, args: argparse.Namespace) -> int:
    max_rollouts = profile.ep_by_scale.get(scale, args.max_rollouts)
    if args.rql_max_rollouts_cap and args.rql_max_rollouts_cap > 0:
        max_rollouts = min(max_rollouts, args.rql_max_rollouts_cap)
    return max_rollouts


def _rql_epsilon_for_episode(
    rollout_i: int,
    max_rollouts: int,
    profile: RqlConvergenceProfile,
    args: argparse.Namespace,
) -> float:
    epsilon_floor = args.epsilon_min
    if (
        profile.late_epsilon_min is not None
        and rollout_i / max(1, max_rollouts) >= profile.late_epsilon_start_ratio
    ):
        epsilon_floor = profile.late_epsilon_min
    return max(epsilon_floor, args.epsilon * (args.epsilon_decay ** (rollout_i - 1)))


def _rql_alpha_for_episode(rollout_i: int, profile: RqlConvergenceProfile, args: argparse.Namespace) -> float:
    alpha = args.alpha * (profile.alpha_decay ** (rollout_i - 1))
    if profile.alpha_min and profile.alpha_min > 0:
        alpha = max(profile.alpha_min, alpha)
    return alpha


def _rql_lookahead_weight_for_episode(
    rollout_i: int,
    max_rollouts: int,
    profile: RqlConvergenceProfile,
    args: argparse.Namespace,
) -> float:
    final_weight = args.lookahead_reward_weight
    if not profile.lookahead_warmup:
        return final_weight

    progress = rollout_i / max(1, max_rollouts)
    if progress <= 0.20:
        return 0.5
    if progress <= 0.60:
        return 0.5 + (1.0 - 0.5) * ((progress - 0.20) / 0.40)
    return 1.0 + (final_weight - 1.0) * ((progress - 0.60) / 0.40)


def _rql_lookahead_clip(profile: RqlConvergenceProfile, args: argparse.Namespace) -> float | None:
    if profile.disable_lookahead_clip:
        return None
    if profile.lookahead_clip is not None:
        return profile.lookahead_clip
    return args.lookahead_clip


def _milestone_penalty_weight_sum(parsed) -> float:
    """Return the penalty-coefficient sum used by the milestone-delay objective."""
    milestone_count = len(parsed.milestone_event) or parsed.milestone_count or 0
    if milestone_count <= 0:
        return DEFAULT_RQL_BASE_MILESTONE_PENALTY_WEIGHT_SUM
    return 0.3 * max(0, milestone_count - 1) + 0.5


def _rql_global_reward_normalization_factor(profile: RqlConvergenceProfile, parsed) -> float:
    if not profile.normalize_global_reward_by_milestone_weight_sum:
        return 1.0
    current_weight_sum = _milestone_penalty_weight_sum(parsed)
    return DEFAULT_RQL_BASE_MILESTONE_PENALTY_WEIGHT_SUM / max(current_weight_sum, 1e-9)


def _decode_rql_penalty(q_table: Dict, parsed, max_skill_level: int, args: argparse.Namespace) -> float:
    _, penalty = build_final_solution(
        q_table,
        parsed,
        max_skill_level,
        q_distinction_threshold=args.q_distinction_threshold,
        beam_width=args.beam_width,
        beam_branch_limit=args.beam_branch_limit,
        beam_improvement_margin=args.beam_improvement_margin,
        use_cost_to_go_fallback=True,
    )
    return penalty


def _train_rql_convergence_profile(
    q_table: Dict,
    parsed,
    scale: int,
    profile_name: str,
    profile: RqlConvergenceProfile,
    args: argparse.Namespace,
    record_history: bool,
) -> tuple[Dict, Dict]:
    max_skill_level = _max_skill_level(parsed.people_num)
    max_rollouts = _rql_profile_max_rollouts(profile, scale, args)
    milestone_penalty_weight_sum = _milestone_penalty_weight_sum(parsed)
    global_reward_normalization_factor = _rql_global_reward_normalization_factor(profile, parsed)
    effective_lookahead_clip = _rql_lookahead_clip(profile, args)
    prev_q = _copy_q_table(q_table)
    converged = False
    completed_rollouts = 0
    last_max_diff = float("inf")
    history: List[Dict] = []

    best_q_table = _copy_q_table(q_table)
    best_checkpoint_penalty = float("inf")
    history_interval = max(1, args.history_decode_interval)
    checkpoint_interval = max(1, args.rql_checkpoint_interval)

    for rollout_i in range(1, max_rollouts + 1):
        alpha_eff = _rql_alpha_for_episode(rollout_i, profile, args)
        epsilon_eff = _rql_epsilon_for_episode(rollout_i, max_rollouts, profile, args)
        lookahead_weight = _rql_lookahead_weight_for_episode(rollout_i, max_rollouts, profile, args)

        env = init_environment(parsed)
        trajectory: List[TrajectoryItem] = []
        episode_reward = 0.0
        max_steps = 10000

        for _step in range(max_steps):
            _process_zero_duration_actions(env, parsed)
            state_features = get_state_features(env, parsed, max_skill_level)

            if len(env.completed_tasks) == parsed.task_count:
                raw_global_reward = calculate_global_reward(env, parsed)
                episode_reward = raw_global_reward
                global_reward_for_update = raw_global_reward * global_reward_normalization_factor
                _update_q_from_trajectory(
                    q_table,
                    trajectory,
                    global_reward_for_update,
                    alpha_eff,
                    args.gamma,
                    args.local_reward_scale,
                    args.global_reward_scale,
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
                q_table,
                state_features,
                available_actions,
                parsed,
                epsilon_eff,
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
                immediate_reward_weight=args.immediate_reward_weight,
                lookahead_reward_weight=lookahead_weight,
                normalize_immediate_reward=not args.disable_reward_normalization,
                lookahead_clip=effective_lookahead_clip,
            )

            _advance_running_tasks(env)
            _process_zero_duration_actions(env, parsed)
            env.current_time += 1
            next_state_features = get_state_features(env, parsed, max_skill_level)

            for action in selected_actions:
                if action[0] in ACTIONS:
                    trajectory.append((state_features, action, local_reward, next_state_features))

        completed_rollouts = rollout_i
        last_max_diff = _max_q_diff(q_table, prev_q)
        prev_q = _copy_q_table(q_table)

        should_decode_history = (
            record_history
            and args.history_decode_interval > 0
            and rollout_i % history_interval == 0
        )
        should_decode_checkpoint = (
            profile.checkpoint
            and args.rql_checkpoint_interval > 0
            and rollout_i % checkpoint_interval == 0
        )
        decoded_penalty = ""
        if should_decode_history or should_decode_checkpoint:
            decoded_penalty = _decode_rql_penalty(q_table, parsed, max_skill_level, args)
            if should_decode_checkpoint and decoded_penalty < best_checkpoint_penalty:
                best_checkpoint_penalty = decoded_penalty
                best_q_table = _copy_q_table(q_table)

        if should_decode_history:
            history.append(
                {
                    "episode": rollout_i,
                    "episode_reward": episode_reward,
                    "decoded_penalty": decoded_penalty,
                    "max_q_diff": last_max_diff,
                    "epsilon": epsilon_eff,
                    "alpha": alpha_eff,
                    "lookahead_reward_weight": lookahead_weight,
                }
            )

        if rollout_i > 1 and last_max_diff < args.convergence_threshold:
            converged = True
            break

    if profile.checkpoint and best_checkpoint_penalty < float("inf"):
        returned_q_table = best_q_table
        used_checkpoint = True
    else:
        returned_q_table = q_table
        used_checkpoint = False
        best_checkpoint_penalty = ""

    metrics = {
        "converged": converged,
        "rollouts": completed_rollouts,
        "max_rollouts": max_rollouts,
        "max_q_diff": last_max_diff,
        "history": history,
        "used_checkpoint": used_checkpoint,
        "best_checkpoint_penalty": best_checkpoint_penalty,
        "rql_convergence_profile": profile_name,
        "late_epsilon_min": profile.late_epsilon_min if profile.late_epsilon_min is not None else "",
        "alpha_decay": profile.alpha_decay,
        "alpha_min": profile.alpha_min,
        "normalize_global_reward_by_milestone_weight_sum": (
            profile.normalize_global_reward_by_milestone_weight_sum
        ),
        "milestone_penalty_weight_sum": milestone_penalty_weight_sum,
        "global_reward_normalization_factor": global_reward_normalization_factor,
        "lookahead_warmup": profile.lookahead_warmup,
        "lookahead_clip_disabled": profile.disable_lookahead_clip,
        "effective_lookahead_clip": "" if effective_lookahead_clip is None else effective_lookahead_clip,
    }
    return returned_q_table, metrics


def _run_learning_method(
    case: Dict,
    method: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[Dict, List[Dict]]:
    parsed = case["parsed"]
    max_skill_level = _max_skill_level(parsed.people_num)
    state_space = generate_state_space(parsed.people_num)
    q_table = init_q_table(state_space, ACTIONS)
    should_record_history = _should_record_history(case, method, seed, args)

    _set_seed(seed)
    started_train = time.perf_counter()
    rql_profile_name = _resolve_rql_profile_name(int(case["scale_value"]), args)
    if method == "RQL" and rql_profile_name != "off":
        profile = RQL_CONVERGENCE_PROFILES[rql_profile_name]
        q_table, train_metrics = _train_rql_convergence_profile(
            q_table,
            parsed,
            int(case["scale_value"]),
            rql_profile_name,
            profile,
            args,
            should_record_history,
        )
    else:
        q_table, train_metrics = sample_training_loop(
            Q_table=q_table,
            parsed=parsed,
            max_rollouts=args.max_rollouts,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            convergence_threshold=args.convergence_threshold,
            max_skill_level=max_skill_level,
            global_reward_scale=args.global_reward_scale,
            local_reward_scale=args.local_reward_scale,
            immediate_reward_weight=args.immediate_reward_weight,
            lookahead_reward_weight=args.lookahead_reward_weight,
            normalize_immediate_reward=not args.disable_reward_normalization,
            lookahead_clip=args.lookahead_clip,
            verbose=args.verbose_training,
            log_interval=args.log_interval,
            return_metrics=True,
            training_mode=method.lower(),
            return_history=should_record_history,
            history_decode_interval=args.history_decode_interval,
        )
    train_time = time.perf_counter() - started_train

    started_solve = time.perf_counter()
    if method == "QL":
        solution, penalty = build_final_solution(
            q_table,
            parsed,
            max_skill_level,
            beam_width=1,
            use_cost_to_go_fallback=False,
        )
    else:
        solution, penalty = build_final_solution(
            q_table,
            parsed,
            max_skill_level,
            q_distinction_threshold=args.q_distinction_threshold,
            beam_width=args.beam_width,
            beam_branch_limit=args.beam_branch_limit,
            beam_improvement_margin=args.beam_improvement_margin,
            use_cost_to_go_fallback=True,
        )
    solve_time = time.perf_counter() - started_solve

    history_rows: List[Dict] = []
    for item in train_metrics.get("history", []):
        history_rows.append(
            {
                "scale": case["scale"],
                "instance_id": case["instance_id"],
                "file": case["file"],
                "method": method,
                "seed": seed,
                "episode": item["episode"],
                "episode_reward": item["episode_reward"],
                "decoded_penalty": item["decoded_penalty"],
                "max_q_diff": item["max_q_diff"],
                "epsilon": item.get("epsilon", ""),
                "alpha": item.get("alpha", ""),
                "lookahead_reward_weight": item.get("lookahead_reward_weight", ""),
            }
        )

    result = {
        "penalty": penalty,
        "train_time": train_time,
        "solve_time": solve_time,
        "total_time": train_time + solve_time,
        "converged": train_metrics["converged"],
        "rollouts": train_metrics["rollouts"],
        "max_q_diff": train_metrics["max_q_diff"],
        "max_rollouts": train_metrics.get("max_rollouts", args.max_rollouts),
        "rql_convergence_profile": train_metrics.get("rql_convergence_profile", ""),
        "used_checkpoint": train_metrics.get("used_checkpoint", ""),
        "best_checkpoint_penalty": train_metrics.get("best_checkpoint_penalty", ""),
        "late_epsilon_min": train_metrics.get("late_epsilon_min", ""),
        "alpha_decay": train_metrics.get("alpha_decay", ""),
        "alpha_min": train_metrics.get("alpha_min", ""),
        "normalize_global_reward_by_milestone_weight_sum": train_metrics.get(
            "normalize_global_reward_by_milestone_weight_sum", ""
        ),
        "milestone_penalty_weight_sum": train_metrics.get("milestone_penalty_weight_sum", ""),
        "global_reward_normalization_factor": train_metrics.get(
            "global_reward_normalization_factor", ""
        ),
        "lookahead_warmup": train_metrics.get("lookahead_warmup", ""),
        "lookahead_clip_disabled": train_metrics.get("lookahead_clip_disabled", ""),
        "effective_lookahead_clip": train_metrics.get("effective_lookahead_clip", ""),
        **_solution_stats(solution),
    }
    return result, history_rows


def _run_baseline_method(case: Dict, method: str, args: argparse.Namespace) -> Dict:
    started = time.perf_counter()
    solution, penalty = build_baseline_solution(
        case["parsed"],
        method,
        rollout_beam_width=args.rollout_beam_width,
        rollout_branch_limit=args.rollout_branch_limit,
    )
    solve_time = time.perf_counter() - started
    return {
        "penalty": penalty,
        "train_time": 0.0,
        "solve_time": solve_time,
        "total_time": solve_time,
        "converged": "",
        "rollouts": "",
        "max_q_diff": "",
        **_solution_stats(solution),
    }


def run_suite(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    case_check_rows, valid_cases = _discover_cases(args)
    raw_rows: List[Dict] = []
    convergence_rows: List[Dict] = []
    failure_rows: List[Dict] = []

    for case_index, case in enumerate(valid_cases, start=1):
        for method in args.methods:
            seeds = list(range(args.seeds)) if _is_random_method(method) else [None]
            for seed in seeds:
                clear_cost_to_go_cache()
                seed_label = seed if seed is not None else "deterministic"
                print(
                    f"[{case_index}/{len(valid_cases)}] "
                    f"{case['scale']} {case['file']} {method} "
                    f"seed={seed_label}"
                )
                try:
                    if _is_learning_method(method):
                        run_result, history = _run_learning_method(case, method, int(seed), args)
                        convergence_rows.extend(history)
                    elif method in BASELINE_METHODS:
                        if seed is not None:
                            _set_seed(int(seed))
                        run_result = _run_baseline_method(case, method, args)
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                    cache_info = get_cost_to_go_cache_info()
                    raw_rows.append(
                        {
                            "scale": case["scale"],
                            "instance_id": case["instance_id"],
                            "file": case["file"],
                            "method": method,
                            "seed": seed_label,
                            "penalty": run_result["penalty"],
                            "train_time": run_result["train_time"],
                            "solve_time": run_result["solve_time"],
                            "total_time": run_result["total_time"],
                            "converged": run_result["converged"],
                            "rollouts": run_result["rollouts"],
                            "max_rollouts": run_result.get("max_rollouts", ""),
                            "max_q_diff": run_result["max_q_diff"],
                            "rql_convergence_profile": run_result.get("rql_convergence_profile", ""),
                            "used_checkpoint": run_result.get("used_checkpoint", ""),
                            "best_checkpoint_penalty": run_result.get("best_checkpoint_penalty", ""),
                            "late_epsilon_min": run_result.get("late_epsilon_min", ""),
                            "alpha_decay": run_result.get("alpha_decay", ""),
                            "alpha_min": run_result.get("alpha_min", ""),
                            "normalize_global_reward_by_milestone_weight_sum": run_result.get(
                                "normalize_global_reward_by_milestone_weight_sum", ""
                            ),
                            "milestone_penalty_weight_sum": run_result.get(
                                "milestone_penalty_weight_sum", ""
                            ),
                            "global_reward_normalization_factor": run_result.get(
                                "global_reward_normalization_factor", ""
                            ),
                            "lookahead_warmup": run_result.get("lookahead_warmup", ""),
                            "lookahead_clip_disabled": run_result.get("lookahead_clip_disabled", ""),
                            "effective_lookahead_clip": run_result.get("effective_lookahead_clip", ""),
                            "action_count": run_result["action_count"],
                            "interrupt_count": run_result["interrupt_count"],
                            "ctg_cache_hits": cache_info["hits"],
                            "ctg_cache_misses": cache_info["misses"],
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - continue the experiment suite.
                    failure_rows.append(
                        {
                            "scale": case["scale"],
                            "instance_id": case["instance_id"],
                            "file": case["file"],
                            "method": method,
                            "seed": seed_label,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(limit=8),
                        }
                    )

    instance_summary, scale_summary, method_rank = build_all_summaries(raw_rows)
    metadata_rows = _metadata_rows(args, case_check_rows, valid_cases, raw_rows, failure_rows)

    return {
        "metadata": metadata_rows,
        "case_check": case_check_rows,
        "raw_runs": raw_rows,
        "instance_summary": instance_summary,
        "scale_summary": scale_summary,
        "method_rank": method_rank,
        "convergence": convergence_rows,
        "failures": failure_rows
        or [
            {
                "scale": "",
                "instance_id": "",
                "file": "",
                "method": "",
                "seed": "",
                "error": "no failures",
                "traceback": "",
            }
        ],
    }


def _metadata_rows(
    args: argparse.Namespace,
    case_check_rows: List[Dict],
    valid_cases: List[Dict],
    raw_rows: List[Dict],
    failure_rows: List[Dict],
) -> List[Dict]:
    valid_by_scale: Dict[str, int] = {}
    for case in valid_cases:
        valid_by_scale[case["scale"]] = valid_by_scale.get(case["scale"], 0) + 1

    rows = [
        {"key": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
        {"key": "cases_dir", "value": str(Path(args.cases_dir).resolve())},
        {"key": "scales", "value": ",".join(f"J{scale}" for scale in args.scales)},
        {"key": "max_cases_per_scale", "value": args.max_cases_per_scale},
        {"key": "seeds", "value": args.seeds},
        {"key": "methods", "value": ",".join(args.methods)},
        {"key": "state_features", "value": "NLF,SBI,MUR,RUR,CRT,ITN"},
        {"key": "total_case_rows", "value": len(case_check_rows)},
        {"key": "valid_cases", "value": len(valid_cases)},
        {"key": "raw_runs", "value": len(raw_rows)},
        {"key": "failures", "value": len(failure_rows)},
        {"key": "max_rollouts", "value": args.max_rollouts},
        {"key": "alpha", "value": args.alpha},
        {"key": "gamma", "value": args.gamma},
        {"key": "epsilon", "value": args.epsilon},
        {"key": "epsilon_decay", "value": args.epsilon_decay},
        {"key": "epsilon_min", "value": args.epsilon_min},
        {"key": "convergence_threshold", "value": args.convergence_threshold},
        {"key": "global_reward_scale", "value": args.global_reward_scale},
        {"key": "local_reward_scale", "value": args.local_reward_scale},
        {"key": "immediate_reward_weight", "value": args.immediate_reward_weight},
        {"key": "lookahead_reward_weight", "value": args.lookahead_reward_weight},
        {"key": "lookahead_clip", "value": args.lookahead_clip},
        {"key": "q_distinction_threshold", "value": args.q_distinction_threshold},
        {"key": "beam_width", "value": args.beam_width},
        {"key": "beam_branch_limit", "value": args.beam_branch_limit},
        {"key": "beam_improvement_margin", "value": args.beam_improvement_margin},
        {"key": "rollout_beam_width", "value": args.rollout_beam_width},
        {"key": "rollout_branch_limit", "value": args.rollout_branch_limit},
        {"key": "rql_convergence_profile", "value": args.rql_convergence_profile},
        {"key": "rql_convergence_profile_description", "value": _rql_profile_description(args)},
        {
            "key": "rql_base_milestone_penalty_weight_sum",
            "value": DEFAULT_RQL_BASE_MILESTONE_PENALTY_WEIGHT_SUM,
        },
        {"key": "rql_max_rollouts_cap", "value": args.rql_max_rollouts_cap},
        {"key": "rql_checkpoint_interval", "value": args.rql_checkpoint_interval},
        {"key": "rql_summary_penalty_policy", "value": "best seed per instance"},
    ]
    for scale, count in sorted(valid_by_scale.items(), key=lambda item: int(item[0][1:])):
        rows.append({"key": f"{scale}_valid_cases", "value": count})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full scheduling experiments over cases/ and write one XLSX file.")
    parser.add_argument("--cases-dir", default="cases")
    parser.add_argument("--scales", type=_parse_int_csv, default=DEFAULT_SCALES)
    parser.add_argument("--max-cases-per-scale", type=int, default=DEFAULT_MAX_CASES_PER_SCALE)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--methods", type=_parse_method_csv, default=DEFAULT_METHODS)
    parser.add_argument("--rql-only", action="store_true")
    parser.add_argument("--output-xlsx", default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--max-rollouts", type=int, default=TRAINING_PARAMS["max_rollouts"])
    parser.add_argument("--alpha", type=float, default=TRAINING_PARAMS["alpha"])
    parser.add_argument("--gamma", type=float, default=TRAINING_PARAMS["gamma"])
    parser.add_argument("--epsilon", type=float, default=TRAINING_PARAMS["epsilon"])
    parser.add_argument("--epsilon-decay", type=float, default=TRAINING_PARAMS["epsilon_decay"])
    parser.add_argument("--epsilon-min", type=float, default=TRAINING_PARAMS["epsilon_min"])
    parser.add_argument("--convergence-threshold", type=float, default=TRAINING_PARAMS["convergence_threshold"])
    parser.add_argument("--global-reward-scale", type=float, default=TRAINING_PARAMS["global_reward_scale"])
    parser.add_argument("--local-reward-scale", type=float, default=TRAINING_PARAMS["local_reward_scale"])
    parser.add_argument("--immediate-reward-weight", type=float, default=TRAINING_PARAMS["immediate_reward_weight"])
    parser.add_argument("--lookahead-reward-weight", type=float, default=TRAINING_PARAMS["lookahead_reward_weight"])
    parser.add_argument("--lookahead-clip", type=float, default=TRAINING_PARAMS["lookahead_clip"])
    parser.add_argument("--disable-reward-normalization", action="store_true")
    parser.add_argument("--q-distinction-threshold", type=float, default=FINAL_SOLUTION_PARAMS["q_distinction_threshold"])
    parser.add_argument("--beam-width", type=int, default=FINAL_SOLUTION_PARAMS["beam_width"])
    parser.add_argument("--beam-branch-limit", type=int, default=FINAL_SOLUTION_PARAMS["beam_branch_limit"])
    parser.add_argument("--beam-improvement-margin", type=float, default=FINAL_SOLUTION_PARAMS["beam_improvement_margin"])
    parser.add_argument("--rollout-beam-width", type=int, default=1)
    parser.add_argument("--rollout-branch-limit", type=int, default=8)
    parser.add_argument("--convergence-cases-per-scale", type=int, default=1)
    parser.add_argument("--convergence-seeds", type=int, default=1)
    parser.add_argument("--history-decode-interval", type=int, default=100)
    parser.add_argument(
        "--rql-convergence-profile",
        choices=["off", "scale_specific", *RQL_CONVERGENCE_PROFILES.keys()],
        default=DEFAULT_RQL_CONVERGENCE_PROFILE,
    )
    parser.add_argument("--rql-max-rollouts-cap", type=int, default=0)
    parser.add_argument("--rql-checkpoint-interval", type=int, default=100)
    parser.add_argument("--verbose-training", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    if args.rql_only:
        args.methods = ["RQL"]
    return args


def main() -> None:
    args = parse_args()
    sheets = run_suite(args)
    write_xlsx(args.output_xlsx, sheets)
    failure_count = 0
    if sheets["failures"] and sheets["failures"][0].get("error") != "no failures":
        failure_count = len(sheets["failures"])
    print(f"XLSX written to: {Path(args.output_xlsx).resolve()}")
    print(f"raw_runs={len(sheets['raw_runs'])} failures={failure_count}")


if __name__ == "__main__":
    main()
