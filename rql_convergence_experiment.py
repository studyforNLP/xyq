from __future__ import annotations

import argparse
import random
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

import numpy as np

from actions import add_required_interrupts, generate_available_actions
from cost_to_go import (
    calculate_lookahead_gain,
    clear_cost_to_go_cache,
    get_cost_to_go_cache_info,
)
from environment import get_state_features, init_environment
from excel_writer import write_xlsx
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
)


Action = Tuple[str, int, int]
State = Tuple[int, ...]
TrajectoryItem = Tuple[State, Action, float, State]

DEFAULT_SCALES = [10, 20, 30]
DEFAULT_VARIANTS = [
    "baseline",
    "scale_ep",
    "low_epsilon",
    "checkpoint",
    "norm_warmup",
]


@dataclass(frozen=True)
class VariantConfig:
    variant: str
    description: str
    ep_by_scale: Dict[int, int]
    epsilon_min: float
    late_epsilon_min: float | None = None
    late_epsilon_start_ratio: float = 0.70
    checkpoint: bool = False
    normalize_global_reward_by_task_count: bool = False
    lookahead_warmup: bool = False
    lookahead_clip: float | None = TRAINING_PARAMS["lookahead_clip"]
    alpha_decay: float = 0.995
    alpha_min: float = 0.0


VARIANT_CONFIGS: Dict[str, VariantConfig] = {
    "baseline": VariantConfig(
        variant="baseline",
        description="Current RQL parameters: EP=500, epsilon_min=0.02, no checkpoint.",
        ep_by_scale={10: 500, 20: 500, 30: 500},
        epsilon_min=TRAINING_PARAMS["epsilon_min"],
    ),
    "scale_ep": VariantConfig(
        variant="scale_ep",
        description="Increase training budget by scale: J10=500, J20=1000, J30=2000.",
        ep_by_scale={10: 500, 20: 1000, 30: 1500},
        epsilon_min=TRAINING_PARAMS["epsilon_min"],
    ),
    "low_epsilon": VariantConfig(
        variant="low_epsilon",
        description="Scale-specific EP plus lower late-stage exploration.",
        ep_by_scale={10: 500, 20: 1000, 30: 1500},
        epsilon_min=TRAINING_PARAMS["epsilon_min"],
        late_epsilon_min=0.005,
        alpha_decay=0.99,
        alpha_min=0.01,
    ),
    "checkpoint": VariantConfig(
        variant="checkpoint",
        description="Low late exploration plus best-Q-table checkpoint selection.",
        ep_by_scale={10: 500, 20: 1000, 30: 1500},
        epsilon_min=TRAINING_PARAMS["epsilon_min"],
        late_epsilon_min=0.005,
        checkpoint=True,
        alpha_decay=0.99,
        alpha_min=0.01,
    ),
    "norm_warmup": VariantConfig(
        variant="norm_warmup",
        description=(
            "Checkpoint plus global reward normalization by task_count and "
            "lookahead reward warm-up."
        ),
        ep_by_scale={10: 500, 20: 1000, 30: 1500},
        epsilon_min=TRAINING_PARAMS["epsilon_min"],
        late_epsilon_min=0.005,
        checkpoint=True,
        normalize_global_reward_by_task_count=True,
        lookahead_warmup=True,
        lookahead_clip=0.5,
        alpha_decay=0.99,
        alpha_min=0.01,
    ),
}


def _parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _natural_key(value: str) -> List:
    parts = re.split(r"(\d+)", value.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _max_skill_level(people_num: int | None) -> int:
    if people_num is None:
        return 4
    if people_num <= 5:
        return people_num - 1
    return 4


def _safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _safe_pstdev(values: Sequence[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


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
            row = {
                "scale": f"J{scale}",
                "scale_value": scale,
                "scale_index": index,
                "instance_id": file_path.stem,
                "file": file_path.name,
                "path": str(file_path),
                "task_count": "",
                "expected_task_count": scale + 2,
                "people_num": "",
                "milestone_count": "",
                "valid": False,
                "message": "",
            }
            try:
                parsed = parse_custom_file(file_path)
                valid = parsed.task_count == scale + 2
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
                    valid_cases.append({**row, "path": file_path, "parsed": parsed})
            except Exception as exc:  # noqa: BLE001 - recorded in output.
                row["message"] = repr(exc)
            case_rows.append(row)

    return case_rows, valid_cases


def _variant_max_rollouts(variant: VariantConfig, scale: int, args: argparse.Namespace) -> int:
    max_rollouts = variant.ep_by_scale.get(scale, TRAINING_PARAMS["max_rollouts"])
    if args.max_rollouts_cap and args.max_rollouts_cap > 0:
        max_rollouts = min(max_rollouts, args.max_rollouts_cap)
    return max_rollouts


def _epsilon_for_episode(
    rollout_i: int,
    max_rollouts: int,
    variant: VariantConfig,
) -> float:
    epsilon_floor = variant.epsilon_min
    if (
        variant.late_epsilon_min is not None
        and rollout_i / max(1, max_rollouts) >= variant.late_epsilon_start_ratio
    ):
        epsilon_floor = variant.late_epsilon_min
    return max(epsilon_floor, TRAINING_PARAMS["epsilon"] * (TRAINING_PARAMS["epsilon_decay"] ** (rollout_i - 1)))


def _alpha_for_episode(rollout_i: int, variant: VariantConfig) -> float:
    alpha = TRAINING_PARAMS["alpha"] * (variant.alpha_decay ** (rollout_i - 1))
    if variant.alpha_min and variant.alpha_min > 0:
        alpha = max(variant.alpha_min, alpha)
    return alpha


def _lookahead_weight_for_episode(
    rollout_i: int,
    max_rollouts: int,
    variant: VariantConfig,
) -> float:
    final_weight = TRAINING_PARAMS["lookahead_reward_weight"]
    if not variant.lookahead_warmup:
        return final_weight

    progress = rollout_i / max(1, max_rollouts)
    if progress <= 0.20:
        return 0.5
    if progress <= 0.60:
        return 0.5 + (1.0 - 0.5) * ((progress - 0.20) / 0.40)
    return 1.0 + (final_weight - 1.0) * ((progress - 0.60) / 0.40)


def _decode_penalty(q_table: Dict, parsed, max_skill_level: int) -> float:
    _, penalty = build_final_solution(
        q_table,
        parsed,
        max_skill_level,
        q_distinction_threshold=FINAL_SOLUTION_PARAMS["q_distinction_threshold"],
        beam_width=FINAL_SOLUTION_PARAMS["beam_width"],
        beam_branch_limit=FINAL_SOLUTION_PARAMS["beam_branch_limit"],
        beam_improvement_margin=FINAL_SOLUTION_PARAMS["beam_improvement_margin"],
        use_cost_to_go_fallback=True,
    )
    return penalty


def _train_rql_variant(
    q_table: Dict,
    parsed,
    scale: int,
    variant: VariantConfig,
    args: argparse.Namespace,
) -> tuple[Dict, Dict, List[Dict]]:
    max_skill_level = _max_skill_level(parsed.people_num)
    max_rollouts = _variant_max_rollouts(variant, scale, args)
    prev_q = _copy_q_table(q_table)
    converged = False
    completed_rollouts = 0
    last_max_diff = float("inf")
    history: List[Dict] = []

    best_q_table = _copy_q_table(q_table)
    best_checkpoint_penalty = float("inf")
    checkpoint_interval = max(1, args.checkpoint_interval)
    history_interval = max(1, args.history_decode_interval)

    for rollout_i in range(1, max_rollouts + 1):
        alpha_eff = _alpha_for_episode(rollout_i, variant)
        epsilon_eff = _epsilon_for_episode(rollout_i, max_rollouts, variant)
        lookahead_weight = _lookahead_weight_for_episode(rollout_i, max_rollouts, variant)

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
                global_reward_for_update = raw_global_reward
                if variant.normalize_global_reward_by_task_count:
                    global_reward_for_update = raw_global_reward / max(1, parsed.task_count)
                _update_q_from_trajectory(
                    q_table,
                    trajectory,
                    global_reward_for_update,
                    alpha_eff,
                    TRAINING_PARAMS["gamma"],
                    TRAINING_PARAMS["local_reward_scale"],
                    TRAINING_PARAMS["global_reward_scale"],
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
                immediate_reward_weight=TRAINING_PARAMS["immediate_reward_weight"],
                lookahead_reward_weight=lookahead_weight,
                normalize_immediate_reward=TRAINING_PARAMS["normalize_immediate_reward"],
                lookahead_clip=variant.lookahead_clip,
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
            args.record_history
            and args.history_decode_interval > 0
            and rollout_i % history_interval == 0
        )
        should_decode_checkpoint = (
            variant.checkpoint
            and args.checkpoint_interval > 0
            and rollout_i % checkpoint_interval == 0
        )
        decoded_penalty = ""
        if should_decode_history or should_decode_checkpoint:
            decoded_penalty = _decode_penalty(q_table, parsed, max_skill_level)
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

    if variant.checkpoint and best_checkpoint_penalty < float("inf"):
        returned_q = best_q_table
        used_checkpoint = True
    else:
        returned_q = q_table
        used_checkpoint = False
        best_checkpoint_penalty = ""

    metrics = {
        "max_rollouts": max_rollouts,
        "converged": converged,
        "rollouts": completed_rollouts,
        "max_q_diff": last_max_diff,
        "used_checkpoint": used_checkpoint,
        "best_checkpoint_penalty": best_checkpoint_penalty,
        "alpha_decay": variant.alpha_decay,
        "alpha_min": variant.alpha_min,
        "late_epsilon_min": variant.late_epsilon_min
        if variant.late_epsilon_min is not None
        else "",
        "normalize_global_reward_by_task_count": variant.normalize_global_reward_by_task_count,
        "lookahead_warmup": variant.lookahead_warmup,
        "lookahead_clip": variant.lookahead_clip if variant.lookahead_clip is not None else "",
    }
    return returned_q, metrics, history


def _run_once(case: Dict, seed: int, variant: VariantConfig, args: argparse.Namespace) -> tuple[Dict, List[Dict]]:
    parsed = case["parsed"]
    max_skill_level = _max_skill_level(parsed.people_num)
    state_space = generate_state_space(parsed.people_num)
    q_table = init_q_table(state_space, ACTIONS)

    _set_seed(seed)
    started_train = time.perf_counter()
    trained_q, train_metrics, history_rows = _train_rql_variant(
        q_table,
        parsed,
        int(case["scale_value"]),
        variant,
        args,
    )
    train_time = time.perf_counter() - started_train

    started_solve = time.perf_counter()
    solution, penalty = build_final_solution(
        trained_q,
        parsed,
        max_skill_level,
        q_distinction_threshold=FINAL_SOLUTION_PARAMS["q_distinction_threshold"],
        beam_width=FINAL_SOLUTION_PARAMS["beam_width"],
        beam_branch_limit=FINAL_SOLUTION_PARAMS["beam_branch_limit"],
        beam_improvement_margin=FINAL_SOLUTION_PARAMS["beam_improvement_margin"],
        use_cost_to_go_fallback=True,
    )
    solve_time = time.perf_counter() - started_solve

    result = {
        "penalty": penalty,
        "train_time": train_time,
        "solve_time": solve_time,
        "total_time": train_time + solve_time,
        "action_count": len(solution),
        "interrupt_count": sum(1 for item in solution if item["action"] == "interrupt"),
        **train_metrics,
    }
    return result, history_rows


def _build_variant_rows(variants: Sequence[VariantConfig], args: argparse.Namespace) -> List[Dict]:
    rows: List[Dict] = []
    for variant in variants:
        for scale in args.scales:
            rows.append(
                {
                    "variant": variant.variant,
                    "scale": f"J{scale}",
                    "description": variant.description,
                    "max_rollouts": _variant_max_rollouts(variant, scale, args),
                    "epsilon_min": variant.epsilon_min,
                    "late_epsilon_min": variant.late_epsilon_min
                    if variant.late_epsilon_min is not None
                    else "",
                    "late_epsilon_start_ratio": variant.late_epsilon_start_ratio,
                    "checkpoint": variant.checkpoint,
                    "normalize_global_reward_by_task_count": variant.normalize_global_reward_by_task_count,
                    "lookahead_warmup": variant.lookahead_warmup,
                    "lookahead_clip": variant.lookahead_clip if variant.lookahead_clip is not None else "",
                    "alpha_decay": variant.alpha_decay,
                    "alpha_min": variant.alpha_min,
                }
            )
    return rows


def _summarize(raw_rows: List[Dict], group_keys: Sequence[str]) -> List[Dict]:
    grouped: Dict[Tuple, List[Dict]] = {}
    for row in raw_rows:
        key = tuple(row[item] for item in group_keys)
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []
    for key, rows in sorted(grouped.items()):
        penalties = [float(row["penalty"]) for row in rows]
        times = [float(row["total_time"]) for row in rows]
        rollouts = [float(row["rollouts"]) for row in rows]
        q_diffs = [float(row["max_q_diff"]) for row in rows]
        converged_count = sum(1 for row in rows if row["converged"] is True)
        item = {name: value for name, value in zip(group_keys, key)}
        item.update(
            {
                "runs": len(rows),
                "mean_penalty": _safe_mean(penalties),
                "best_penalty": min(penalties) if penalties else "",
                "std_penalty": _safe_pstdev(penalties),
                "mean_total_time": _safe_mean(times),
                "converged_runs": converged_count,
                "convergence_rate": converged_count / max(1, len(rows)),
                "mean_rollouts": _safe_mean(rollouts),
                "mean_max_q_diff": _safe_mean(q_diffs),
                "checkpoint_used_runs": sum(1 for row in rows if row["used_checkpoint"] is True),
            }
        )
        summary_rows.append(item)
    return summary_rows


def _build_analysis_rows(overall_summary: List[Dict], scale_summary: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    if not overall_summary:
        return rows

    best_quality = min(overall_summary, key=lambda row: (row["mean_penalty"], row["mean_max_q_diff"]))
    best_convergence = max(
        overall_summary,
        key=lambda row: (row["convergence_rate"], -row["mean_penalty"]),
    )
    rows.append(
        {
            "section": "overall",
            "finding": "Best mean penalty variant",
            "evidence": (
                f"{best_quality['variant']} mean_penalty="
                f"{best_quality['mean_penalty']:.4f}, convergence_rate="
                f"{best_quality['convergence_rate']:.4f}"
            ),
        }
    )
    rows.append(
        {
            "section": "overall",
            "finding": "Best convergence variant",
            "evidence": (
                f"{best_convergence['variant']} convergence_rate="
                f"{best_convergence['convergence_rate']:.4f}, mean_penalty="
                f"{best_convergence['mean_penalty']:.4f}"
            ),
        }
    )

    baseline = next((row for row in overall_summary if row["variant"] == "baseline"), None)
    if baseline:
        for row in overall_summary:
            if row["variant"] == "baseline":
                continue
            penalty_change = (
                (row["mean_penalty"] - baseline["mean_penalty"])
                / max(baseline["mean_penalty"], 1.0)
                * 100.0
            )
            convergence_change = row["convergence_rate"] - baseline["convergence_rate"]
            rows.append(
                {
                    "section": "baseline_comparison",
                    "finding": row["variant"],
                    "evidence": (
                        f"penalty_change_vs_baseline={penalty_change:.2f}%, "
                        f"convergence_rate_delta={convergence_change:.4f}, "
                        f"mean_max_q_diff={row['mean_max_q_diff']:.6f}"
                    ),
                }
            )

    scales = sorted({row["scale"] for row in scale_summary}, key=lambda value: int(value[1:]))
    for scale in scales:
        rows_for_scale = [row for row in scale_summary if row["scale"] == scale]
        best_scale = min(rows_for_scale, key=lambda row: (row["mean_penalty"], row["mean_max_q_diff"]))
        conv_scale = max(rows_for_scale, key=lambda row: (row["convergence_rate"], -row["mean_penalty"]))
        rows.append(
            {
                "section": "scale_best",
                "finding": scale,
                "evidence": (
                    f"best_quality={best_scale['variant']} "
                    f"mean_penalty={best_scale['mean_penalty']:.4f}; "
                    f"best_convergence={conv_scale['variant']} "
                    f"convergence_rate={conv_scale['convergence_rate']:.4f}"
                ),
            }
        )
    return rows


def _write_analysis_md(
    output_path: str,
    args: argparse.Namespace,
    overall_summary: List[Dict],
    scale_summary: List[Dict],
    analysis_rows: List[Dict],
) -> None:
    lines = [
        "# RQL 收敛性实验分析",
        "",
        f"- 规模: {','.join(f'J{scale}' for scale in args.scales)}",
        f"- 每规模案例数: {args.max_cases_per_scale}",
        f"- seed 数: {args.seeds}",
        f"- 方案: {','.join(args.variants)}",
        "",
        "## 总体汇总",
        "",
        "| variant | mean_penalty | convergence_rate | mean_rollouts | mean_max_q_diff | mean_time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall_summary:
        lines.append(
            "| {variant} | {penalty:.4f} | {conv:.4f} | {rollouts:.1f} | {qdiff:.6f} | {time:.3f} |".format(
                variant=row["variant"],
                penalty=row["mean_penalty"],
                conv=row["convergence_rate"],
                rollouts=row["mean_rollouts"],
                qdiff=row["mean_max_q_diff"],
                time=row["mean_total_time"],
            )
        )

    lines.extend(["", "## 关键发现", ""])
    for row in analysis_rows:
        lines.append(f"- {row['section']} / {row['finding']}: {row['evidence']}")

    lines.extend(["", "## 分规模汇总", ""])
    lines.append("| scale | variant | mean_penalty | convergence_rate | mean_max_q_diff | mean_time |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in scale_summary:
        lines.append(
            "| {scale} | {variant} | {penalty:.4f} | {conv:.4f} | {qdiff:.6f} | {time:.3f} |".format(
                scale=row["scale"],
                variant=row["variant"],
                penalty=row["mean_penalty"],
                conv=row["convergence_rate"],
                qdiff=row["mean_max_q_diff"],
                time=row["mean_total_time"],
            )
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    selected_variants = [VARIANT_CONFIGS[name] for name in args.variants]
    if args.smoke:
        args = replace_namespace(
            args,
            max_cases_per_scale=1,
            seeds=1,
            variants=["baseline"],
            max_rollouts_cap=2,
            record_history=True,
            history_decode_interval=1,
            checkpoint_interval=1,
        )
        selected_variants = [VARIANT_CONFIGS["baseline"]]

    case_rows, valid_cases = _discover_cases(args)
    variant_rows = _build_variant_rows(selected_variants, args)
    raw_rows: List[Dict] = []
    history_rows: List[Dict] = []
    failure_rows: List[Dict] = []

    if args.dry_run:
        return {
            "metadata": _metadata_rows(args, valid_cases, 0, 0),
            "case_check": case_rows,
            "variant_settings": variant_rows,
            "raw_runs": [],
            "scale_summary": [],
            "overall_summary": [],
            "history": [],
            "analysis": [{"section": "dry_run", "finding": "No experiment executed", "evidence": ""}],
            "failures": [{"error": "dry run"}],
        }

    total_runs = len(valid_cases) * len(selected_variants) * args.seeds
    run_index = 0
    started_all = time.perf_counter()

    for case in valid_cases:
        for variant in selected_variants:
            for seed in range(args.seeds):
                run_index += 1
                clear_cost_to_go_cache()
                if args.verbose:
                    print(
                        f"[{run_index}/{total_runs}] {case['scale']} {case['file']} "
                        f"{variant.variant} seed={seed}"
                    )
                try:
                    result, run_history = _run_once(case, seed, variant, args)
                    cache_info = get_cost_to_go_cache_info()
                    raw_rows.append(
                        {
                            "scale": case["scale"],
                            "scale_value": case["scale_value"],
                            "instance_id": case["instance_id"],
                            "file": case["file"],
                            "variant": variant.variant,
                            "seed": seed,
                            "penalty": result["penalty"],
                            "train_time": result["train_time"],
                            "solve_time": result["solve_time"],
                            "total_time": result["total_time"],
                            "converged": result["converged"],
                            "rollouts": result["rollouts"],
                            "max_rollouts": result["max_rollouts"],
                            "max_q_diff": result["max_q_diff"],
                            "used_checkpoint": result["used_checkpoint"],
                            "best_checkpoint_penalty": result["best_checkpoint_penalty"],
                            "action_count": result["action_count"],
                            "interrupt_count": result["interrupt_count"],
                            "ctg_cache_hits": cache_info["hits"],
                            "ctg_cache_misses": cache_info["misses"],
                            "alpha_decay": result["alpha_decay"],
                            "alpha_min": result["alpha_min"],
                            "late_epsilon_min": result["late_epsilon_min"],
                            "normalize_global_reward_by_task_count": result[
                                "normalize_global_reward_by_task_count"
                            ],
                            "lookahead_warmup": result["lookahead_warmup"],
                            "lookahead_clip": result["lookahead_clip"],
                        }
                    )
                    for item in run_history:
                        history_rows.append(
                            {
                                "scale": case["scale"],
                                "instance_id": case["instance_id"],
                                "file": case["file"],
                                "variant": variant.variant,
                                "seed": seed,
                                **item,
                            }
                        )
                except Exception as exc:  # noqa: BLE001 - keep the experiment running.
                    failure_rows.append(
                        {
                            "scale": case["scale"],
                            "instance_id": case["instance_id"],
                            "file": case["file"],
                            "variant": variant.variant,
                            "seed": seed,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(limit=8),
                        }
                    )

    elapsed = time.perf_counter() - started_all
    scale_summary = _summarize(raw_rows, ["scale", "variant"])
    overall_summary = _summarize(raw_rows, ["variant"])
    analysis_rows = _build_analysis_rows(overall_summary, scale_summary)
    _write_analysis_md(args.output_md, args, overall_summary, scale_summary, analysis_rows)

    return {
        "metadata": _metadata_rows(args, valid_cases, len(raw_rows), elapsed),
        "case_check": case_rows,
        "variant_settings": variant_rows,
        "raw_runs": raw_rows,
        "scale_summary": scale_summary,
        "overall_summary": overall_summary,
        "history": history_rows or [{"message": "history disabled or no history rows"}],
        "analysis": analysis_rows,
        "failures": failure_rows
        or [
            {
                "scale": "",
                "instance_id": "",
                "file": "",
                "variant": "",
                "seed": "",
                "error": "no failures",
                "traceback": "",
            }
        ],
    }


def _metadata_rows(
    args: argparse.Namespace,
    valid_cases: List[Dict],
    raw_count: int,
    elapsed_seconds: float,
) -> List[Dict]:
    rows = [
        {"key": "generated_at", "value": time.strftime("%Y-%m-%d %H:%M:%S")},
        {"key": "cases_dir", "value": str(Path(args.cases_dir).resolve())},
        {"key": "scales", "value": ",".join(f"J{scale}" for scale in args.scales)},
        {"key": "max_cases_per_scale", "value": args.max_cases_per_scale},
        {"key": "seeds", "value": args.seeds},
        {"key": "variants", "value": ",".join(args.variants)},
        {"key": "valid_cases", "value": len(valid_cases)},
        {"key": "raw_runs", "value": raw_count},
        {"key": "elapsed_seconds", "value": elapsed_seconds},
        {"key": "convergence_threshold", "value": args.convergence_threshold},
        {"key": "history_decode_interval", "value": args.history_decode_interval},
        {"key": "checkpoint_interval", "value": args.checkpoint_interval},
        {"key": "base_alpha", "value": TRAINING_PARAMS["alpha"]},
        {"key": "base_gamma", "value": TRAINING_PARAMS["gamma"]},
        {"key": "base_epsilon", "value": TRAINING_PARAMS["epsilon"]},
        {"key": "base_epsilon_decay", "value": TRAINING_PARAMS["epsilon_decay"]},
        {"key": "base_epsilon_min", "value": TRAINING_PARAMS["epsilon_min"]},
        {"key": "base_global_reward_scale", "value": TRAINING_PARAMS["global_reward_scale"]},
        {"key": "base_local_reward_scale", "value": TRAINING_PARAMS["local_reward_scale"]},
        {"key": "base_lookahead_reward_weight", "value": TRAINING_PARAMS["lookahead_reward_weight"]},
        {"key": "final_beam_width", "value": FINAL_SOLUTION_PARAMS["beam_width"]},
        {"key": "output_md", "value": str(Path(args.output_md).resolve())},
    ]
    valid_by_scale: Dict[str, int] = {}
    for case in valid_cases:
        valid_by_scale[case["scale"]] = valid_by_scale.get(case["scale"], 0) + 1
    for scale, count in sorted(valid_by_scale.items(), key=lambda item: int(item[0][1:])):
        rows.append({"key": f"{scale}_valid_cases", "value": count})
    return rows


def replace_namespace(args: argparse.Namespace, **updates) -> argparse.Namespace:
    data = vars(args).copy()
    data.update(updates)
    return argparse.Namespace(**data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RQL-only convergence experiments and summarize the results."
    )
    parser.add_argument("--cases-dir", default="cases")
    parser.add_argument("--scales", type=_parse_int_csv, default=DEFAULT_SCALES)
    parser.add_argument("--max-cases-per-scale", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--variants", type=_parse_str_csv, default=DEFAULT_VARIANTS)
    parser.add_argument("--output-xlsx", default="results/rql_convergence_experiment.xlsx")
    parser.add_argument("--output-md", default="results/rql_convergence_experiment_analysis.md")
    parser.add_argument("--convergence-threshold", type=float, default=TRAINING_PARAMS["convergence_threshold"])
    parser.add_argument("--history-decode-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--max-rollouts-cap", type=int, default=0)
    parser.add_argument("--disable-history", dest="record_history", action="store_false")
    parser.add_argument("--record-history", dest="record_history", action="store_true")
    parser.set_defaults(record_history=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    unknown = [variant for variant in args.variants if variant not in VARIANT_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {sorted(VARIANT_CONFIGS)}")
    return args


def main() -> None:
    args = parse_args()
    sheets = run_experiment(args)
    write_xlsx(args.output_xlsx, sheets)
    failure_count = 0
    if sheets["failures"] and sheets["failures"][0].get("error") != "no failures":
        failure_count = len(sheets["failures"])

    print(f"XLSX written to: {Path(args.output_xlsx).resolve()}")
    print(f"Analysis markdown written to: {Path(args.output_md).resolve()}")
    print(f"raw_runs={len(sheets['raw_runs'])} failures={failure_count}")
    if sheets["analysis"]:
        print("Key findings:")
        for row in sheets["analysis"][:8]:
            print(f"- {row['section']} / {row['finding']}: {row['evidence']}")


if __name__ == "__main__":
    main()
