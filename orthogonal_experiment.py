from __future__ import annotations

import argparse
import random
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np

from cost_to_go import clear_cost_to_go_cache, get_cost_to_go_cache_info
from excel_writer import write_xlsx
from parameters import FINAL_SOLUTION_PARAMS, TRAINING_PARAMS
from parser import parse_custom_file
from qlearning import init_q_table
from state_space import ACTIONS, generate_state_space
from training import build_final_solution, sample_training_loop


DEFAULT_SCALES = [10, 20, 30]
L9_TABLE = [
    [1, 1, 1, 1],
    [1, 2, 2, 2],
    [1, 3, 3, 3],
    [2, 1, 2, 3],
    [2, 2, 3, 1],
    [2, 3, 1, 2],
    [3, 1, 3, 2],
    [3, 2, 1, 3],
    [3, 3, 2, 1],
]

TRAINING_FACTORS = {
    "A": ("alpha", [0.05, 0.10, 0.20]),
    "B": ("gamma", [0.90, 0.95, 0.99]),
    "C": ("epsilon", [0.10, 0.20, 0.30]),
    "D": ("epsilon_decay", [0.990, 0.995, 0.999]),
}

REWARD_FACTORS = {
    "A": ("global_reward_scale", [0.10, 0.30, 0.50]),
    "B": ("local_reward_scale", [0.10, 0.20, 0.40]),
    "C": ("lookahead_reward_weight", [0.50, 1.00, 1.50]),
    "D": ("beam_width", [1, 3, 5]),
}

DEFAULT_EP_VALUES = [200, 500, 1000, 1500]


def _parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def _safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def _safe_pstdev(values: List[float]) -> float:
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
                    valid_cases.append(
                        {
                            "scale": f"J{scale}",
                            "scale_value": scale,
                            "scale_index": index,
                            "instance_id": file_path.stem,
                            "file": file_path.name,
                            "path": file_path,
                            "parsed": parsed,
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - recorded in the Excel output.
                row["message"] = repr(exc)
            case_rows.append(row)

    return case_rows, valid_cases


def _base_params(args: argparse.Namespace) -> Dict:
    return {
        "max_rollouts": args.max_rollouts,
        "alpha": TRAINING_PARAMS["alpha"],
        "gamma": TRAINING_PARAMS["gamma"],
        "epsilon": TRAINING_PARAMS["epsilon"],
        "epsilon_decay": TRAINING_PARAMS["epsilon_decay"],
        "epsilon_min": args.epsilon_min,
        "convergence_threshold": TRAINING_PARAMS["convergence_threshold"],
        "global_reward_scale": TRAINING_PARAMS["global_reward_scale"],
        "local_reward_scale": TRAINING_PARAMS["local_reward_scale"],
        "immediate_reward_weight": TRAINING_PARAMS["immediate_reward_weight"],
        "lookahead_reward_weight": TRAINING_PARAMS["lookahead_reward_weight"],
        "lookahead_clip": TRAINING_PARAMS["lookahead_clip"],
        "q_distinction_threshold": FINAL_SOLUTION_PARAMS["q_distinction_threshold"],
        "beam_width": FINAL_SOLUTION_PARAMS["beam_width"],
        "beam_branch_limit": FINAL_SOLUTION_PARAMS["beam_branch_limit"],
        "beam_improvement_margin": FINAL_SOLUTION_PARAMS["beam_improvement_margin"],
    }


def _build_l9_combos(phase: str, factors: Dict[str, tuple[str, List]], fixed_params: Dict, max_combos: int) -> List[Dict]:
    combos: List[Dict] = []
    factor_items = list(factors.items())
    rows = L9_TABLE[: max_combos if max_combos and max_combos > 0 else len(L9_TABLE)]
    for index, levels in enumerate(rows, start=1):
        params = dict(fixed_params)
        combo: Dict = {
            "phase": phase,
            "combo_id": f"{phase.upper()}-{index:02d}",
            "row": index,
        }
        for (factor_code, (param_name, values)), level in zip(factor_items, levels):
            value = values[level - 1]
            params[param_name] = value
            combo[f"{factor_code}_factor"] = param_name
            combo[f"{factor_code}_level"] = level
            combo[f"{factor_code}_value"] = value
        combo.update(params)
        combos.append(combo)
    return combos


def _run_rql_once(case: Dict, seed: int, params: Dict, args: argparse.Namespace) -> tuple[Dict, List[Dict]]:
    parsed = case["parsed"]
    max_skill_level = _max_skill_level(parsed.people_num)
    state_space = generate_state_space(parsed.people_num)
    q_table = init_q_table(state_space, ACTIONS)

    _set_seed(seed)
    started_train = time.perf_counter()
    q_table, train_metrics = sample_training_loop(
        Q_table=q_table,
        parsed=parsed,
        max_rollouts=int(params["max_rollouts"]),
        alpha=float(params["alpha"]),
        gamma=float(params["gamma"]),
        epsilon=float(params["epsilon"]),
        epsilon_decay=float(params["epsilon_decay"]),
        epsilon_min=float(params["epsilon_min"]),
        convergence_threshold=float(params["convergence_threshold"]),
        max_skill_level=max_skill_level,
        global_reward_scale=float(params["global_reward_scale"]),
        local_reward_scale=float(params["local_reward_scale"]),
        immediate_reward_weight=float(params["immediate_reward_weight"]),
        lookahead_reward_weight=float(params["lookahead_reward_weight"]),
        normalize_immediate_reward=True,
        lookahead_clip=float(params["lookahead_clip"]),
        verbose=args.verbose_training,
        log_interval=args.log_interval,
        return_metrics=True,
        training_mode="rql",
        return_history=args.record_history,
        history_decode_interval=args.history_decode_interval,
    )
    train_time = time.perf_counter() - started_train

    started_solve = time.perf_counter()
    solution, penalty = build_final_solution(
        q_table,
        parsed,
        max_skill_level,
        q_distinction_threshold=float(params["q_distinction_threshold"]),
        beam_width=int(params["beam_width"]),
        beam_branch_limit=int(params["beam_branch_limit"]),
        beam_improvement_margin=float(params["beam_improvement_margin"]),
        use_cost_to_go_fallback=True,
    )
    solve_time = time.perf_counter() - started_solve

    history_rows = []
    for item in train_metrics.get("history", []):
        history_rows.append(
            {
                "scale": case["scale"],
                "instance_id": case["instance_id"],
                "file": case["file"],
                "seed": seed,
                "episode": item["episode"],
                "episode_reward": item["episode_reward"],
                "decoded_penalty": item["decoded_penalty"],
                "max_q_diff": item["max_q_diff"],
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
        "action_count": len(solution),
        "interrupt_count": sum(1 for item in solution if item["action"] == "interrupt"),
    }
    return result, history_rows


def _run_combos(
    phase: str,
    combos: List[Dict],
    cases: List[Dict],
    args: argparse.Namespace,
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    raw_rows: List[Dict] = []
    history_rows: List[Dict] = []
    failures: List[Dict] = []

    for combo_index, combo in enumerate(combos, start=1):
        params = _params_from_combo(combo)
        for case_index, case in enumerate(cases, start=1):
            for seed in range(args.seeds):
                clear_cost_to_go_cache()
                print(
                    f"{phase} combo={combo['combo_id']} "
                    f"({combo_index}/{len(combos)}) case={case['file']} "
                    f"({case_index}/{len(cases)}) seed={seed}"
                )
                try:
                    run_result, run_history = _run_rql_once(case, seed, params, args)
                    cache_info = get_cost_to_go_cache_info()
                    row = {
                        "phase": phase,
                        "combo_id": combo["combo_id"],
                        "scale": case["scale"],
                        "instance_id": case["instance_id"],
                        "file": case["file"],
                        "seed": seed,
                        "penalty": run_result["penalty"],
                        "train_time": run_result["train_time"],
                        "solve_time": run_result["solve_time"],
                        "total_time": run_result["total_time"],
                        "converged": run_result["converged"],
                        "rollouts": run_result["rollouts"],
                        "max_q_diff": run_result["max_q_diff"],
                        "action_count": run_result["action_count"],
                        "interrupt_count": run_result["interrupt_count"],
                        "ctg_cache_hits": cache_info["hits"],
                        "ctg_cache_misses": cache_info["misses"],
                    }
                    row.update(_compact_param_columns(params))
                    raw_rows.append(row)
                    for history in run_history:
                        history.update({"phase": phase, "combo_id": combo["combo_id"]})
                        history_rows.append(history)
                except Exception as exc:  # noqa: BLE001 - continue parameter screening.
                    failures.append(
                        {
                            "phase": phase,
                            "combo_id": combo["combo_id"],
                            "scale": case["scale"],
                            "instance_id": case["instance_id"],
                            "file": case["file"],
                            "seed": seed,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(limit=8),
                        }
                    )

    return raw_rows, history_rows, failures


def _run_ep_sensitivity(
    best_params: Dict,
    cases: List[Dict],
    args: argparse.Namespace,
) -> tuple[List[Dict], List[Dict]]:
    raw_rows: List[Dict] = []
    failures: List[Dict] = []
    for ep in args.ep_values:
        params = dict(best_params)
        params["max_rollouts"] = ep
        for case_index, case in enumerate(cases, start=1):
            for seed in range(args.seeds):
                clear_cost_to_go_cache()
                print(f"EP={ep} case={case['file']} ({case_index}/{len(cases)}) seed={seed}")
                try:
                    run_result, _history = _run_rql_once(case, seed, params, args)
                    cache_info = get_cost_to_go_cache_info()
                    row = {
                        "phase": "ep_sensitivity",
                        "EP": ep,
                        "scale": case["scale"],
                        "instance_id": case["instance_id"],
                        "file": case["file"],
                        "seed": seed,
                        "penalty": run_result["penalty"],
                        "train_time": run_result["train_time"],
                        "solve_time": run_result["solve_time"],
                        "total_time": run_result["total_time"],
                        "converged": run_result["converged"],
                        "rollouts": run_result["rollouts"],
                        "max_q_diff": run_result["max_q_diff"],
                        "ctg_cache_hits": cache_info["hits"],
                        "ctg_cache_misses": cache_info["misses"],
                    }
                    raw_rows.append(row)
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        {
                            "phase": "ep_sensitivity",
                            "EP": ep,
                            "scale": case["scale"],
                            "instance_id": case["instance_id"],
                            "file": case["file"],
                            "seed": seed,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(limit=8),
                        }
                    )
    return raw_rows, failures


def _params_from_combo(combo: Dict) -> Dict:
    keys = [
        "max_rollouts",
        "alpha",
        "gamma",
        "epsilon",
        "epsilon_decay",
        "epsilon_min",
        "convergence_threshold",
        "global_reward_scale",
        "local_reward_scale",
        "immediate_reward_weight",
        "lookahead_reward_weight",
        "lookahead_clip",
        "q_distinction_threshold",
        "beam_width",
        "beam_branch_limit",
        "beam_improvement_margin",
    ]
    return {key: combo[key] for key in keys}


def _compact_param_columns(params: Dict) -> Dict:
    return {
        "alpha": params["alpha"],
        "gamma": params["gamma"],
        "epsilon": params["epsilon"],
        "epsilon_decay": params["epsilon_decay"],
        "epsilon_min": params["epsilon_min"],
        "global_reward_scale": params["global_reward_scale"],
        "local_reward_scale": params["local_reward_scale"],
        "lookahead_reward_weight": params["lookahead_reward_weight"],
        "beam_width": params["beam_width"],
        "max_rollouts": params["max_rollouts"],
    }


def _summarize_combos(raw_rows: List[Dict], combos: List[Dict], phase: str) -> List[Dict]:
    best_by_case: Dict[tuple[str, str], float] = {}
    for row in raw_rows:
        key = (row["scale"], row["instance_id"])
        penalty = float(row["penalty"])
        best_by_case[key] = min(best_by_case.get(key, penalty), penalty)

    rows_by_combo_case: Dict[tuple[str, str, str], List[Dict]] = {}
    for row in raw_rows:
        key = (row["combo_id"], row["scale"], row["instance_id"])
        rows_by_combo_case.setdefault(key, []).append(row)

    combo_case_metrics: Dict[str, List[Dict]] = {}
    for (combo_id, scale, instance_id), rows in rows_by_combo_case.items():
        penalties = [float(row["penalty"]) for row in rows]
        best = best_by_case[(scale, instance_id)]
        mean_penalty = _safe_mean(penalties)
        arpd = ((mean_penalty - best) / max(best, 1.0)) * 100.0
        combo_case_metrics.setdefault(combo_id, []).append(
            {
                "mean_penalty": mean_penalty,
                "best_penalty": min(penalties),
                "arpd": arpd,
                "mean_time": _safe_mean([float(row["total_time"]) for row in rows]),
                "convergence_rate": _safe_mean([1.0 if row["converged"] is True else 0.0 for row in rows]),
                "mean_rollouts": _safe_mean([float(row["rollouts"]) for row in rows]),
            }
        )

    combo_lookup = {combo["combo_id"]: combo for combo in combos}
    summary_rows: List[Dict] = []
    for combo_id, metrics in sorted(combo_case_metrics.items()):
        combo = combo_lookup[combo_id]
        summary = {
            "phase": phase,
            "combo_id": combo_id,
            "cases": len(metrics),
            "mean_penalty": _safe_mean([item["mean_penalty"] for item in metrics]),
            "best_penalty": _safe_mean([item["best_penalty"] for item in metrics]),
            "std_penalty": _safe_pstdev([item["mean_penalty"] for item in metrics]),
            "mean_arpd": _safe_mean([item["arpd"] for item in metrics]),
            "mean_time": _safe_mean([item["mean_time"] for item in metrics]),
            "convergence_rate": _safe_mean([item["convergence_rate"] for item in metrics]),
            "mean_rollouts": _safe_mean([item["mean_rollouts"] for item in metrics]),
        }
        summary.update(_factor_columns(combo))
        summary.update(_compact_param_columns(_params_from_combo(combo)))
        summary_rows.append(summary)
    return summary_rows


def _factor_columns(combo: Dict) -> Dict:
    result: Dict = {}
    for factor in ["A", "B", "C", "D"]:
        result[f"{factor}_factor"] = combo.get(f"{factor}_factor", "")
        result[f"{factor}_level"] = combo.get(f"{factor}_level", "")
        result[f"{factor}_value"] = combo.get(f"{factor}_value", "")
    return result


def _best_combo(summary_rows: List[Dict]) -> Dict:
    if not summary_rows:
        return {}
    ordered = sorted(
        summary_rows,
        key=lambda row: (
            float(row["mean_arpd"]),
            float(row["mean_time"]),
            -float(row["convergence_rate"]),
            float(row["mean_rollouts"]),
            row["combo_id"],
        ),
    )
    best = ordered[0]
    one_percent_cutoff = float(best["mean_arpd"]) + 1.0
    close_rows = [row for row in ordered if float(row["mean_arpd"]) <= one_percent_cutoff]
    return sorted(
        close_rows,
        key=lambda row: (
            float(row["mean_time"]),
            -float(row["convergence_rate"]),
            float(row["mean_rollouts"]),
            float(row["mean_arpd"]),
            row["combo_id"],
        ),
    )[0]


def _range_analysis(summary_rows: List[Dict], factors: Dict[str, tuple[str, List]]) -> List[Dict]:
    rows: List[Dict] = []
    for factor_code, (param_name, values) in factors.items():
        level_rows = []
        for level, value in enumerate(values, start=1):
            matching = [
                row
                for row in summary_rows
                if int(row.get(f"{factor_code}_level", 0) or 0) == level
            ]
            level_rows.append(
                {
                    "phase": summary_rows[0]["phase"] if summary_rows else "",
                    "factor": factor_code,
                    "parameter": param_name,
                    "level": level,
                    "parameter_value": value,
                    "mean_arpd": _safe_mean([float(row["mean_arpd"]) for row in matching]),
                    "mean_penalty": _safe_mean([float(row["mean_penalty"]) for row in matching]),
                    "mean_time": _safe_mean([float(row["mean_time"]) for row in matching]),
                }
            )
        arpd_values = [row["mean_arpd"] for row in level_rows]
        range_r = max(arpd_values) - min(arpd_values) if arpd_values else 0.0
        for row in level_rows:
            row["range_R"] = range_r
            rows.append(row)
    return rows


def _summarize_ep(ep_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[int, List[Dict]] = {}
    for row in ep_rows:
        grouped.setdefault(int(row["EP"]), []).append(row)

    best_by_case: Dict[tuple[str, str], float] = {}
    for row in ep_rows:
        key = (row["scale"], row["instance_id"])
        penalty = float(row["penalty"])
        best_by_case[key] = min(best_by_case.get(key, penalty), penalty)

    summary_rows: List[Dict] = []
    for ep, rows in sorted(grouped.items()):
        case_metrics: Dict[tuple[str, str], List[Dict]] = {}
        for row in rows:
            case_metrics.setdefault((row["scale"], row["instance_id"]), []).append(row)
        case_penalties = []
        case_arpds = []
        for key, case_rows in case_metrics.items():
            mean_penalty = _safe_mean([float(row["penalty"]) for row in case_rows])
            case_penalties.append(mean_penalty)
            best = best_by_case[key]
            case_arpds.append(((mean_penalty - best) / max(best, 1.0)) * 100.0)
        summary_rows.append(
            {
                "EP": ep,
                "cases": len(case_metrics),
                "mean_penalty": _safe_mean(case_penalties),
                "std_penalty": _safe_pstdev(case_penalties),
                "mean_arpd": _safe_mean(case_arpds),
                "mean_time": _safe_mean([float(row["total_time"]) for row in rows]),
                "convergence_rate": _safe_mean([1.0 if row["converged"] is True else 0.0 for row in rows]),
                "mean_rollouts": _safe_mean([float(row["rollouts"]) for row in rows]),
            }
        )
    return summary_rows


def _tag_rows(rows: List[Dict], tuning_scope: str) -> List[Dict]:
    return [{"tuning_scope": tuning_scope, **row} for row in rows]


def _best_params_rows(
    training_best: Dict,
    reward_best: Dict,
    ep_best: Dict,
    final_params: Dict,
    tuning_scope: str,
) -> List[Dict]:
    rows = [
        {"tuning_scope": tuning_scope, "stage": "training", "selected_combo": training_best.get("combo_id", "")},
        {"tuning_scope": tuning_scope, "stage": "reward", "selected_combo": reward_best.get("combo_id", "")},
        {"tuning_scope": tuning_scope, "stage": "EP", "selected_EP": ep_best.get("EP", "")},
    ]
    for key, value in final_params.items():
        rows.append({"tuning_scope": tuning_scope, "stage": "final_param", "parameter": key, "value": value})
    return rows


def _scale_best_param_row(
    training_best: Dict,
    reward_best: Dict,
    ep_best: Dict,
    final_params: Dict,
    tuning_scope: str,
) -> Dict:
    row = {
        "tuning_scope": tuning_scope,
        "training_combo": training_best.get("combo_id", ""),
        "reward_combo": reward_best.get("combo_id", ""),
        "selected_EP": ep_best.get("EP", ""),
    }
    row.update(final_params)
    return row


def _metadata_rows(
    args: argparse.Namespace,
    case_rows: List[Dict],
    valid_cases: List[Dict],
    failures: List[Dict],
    tuning_scope: str,
) -> List[Dict]:
    return [
        {"key": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
        {"key": "experiment_type", "value": "layered_orthogonal_rql"},
        {"key": "tuning_mode", "value": "pooled" if args.pooled else "per_scale"},
        {"key": "tuning_scope", "value": tuning_scope},
        {"key": "cases_dir", "value": str(Path(args.cases_dir).resolve())},
        {"key": "scales", "value": ",".join(f"J{scale}" for scale in args.scales)},
        {"key": "max_cases_per_scale", "value": args.max_cases_per_scale},
        {"key": "seeds", "value": args.seeds},
        {"key": "max_combos", "value": args.max_combos or "all"},
        {"key": "case_rows", "value": len(case_rows)},
        {"key": "valid_cases", "value": len(valid_cases)},
        {"key": "failures", "value": len(failures)},
        {"key": "epsilon_min", "value": args.epsilon_min},
        {"key": "default_max_rollouts", "value": args.max_rollouts},
        {"key": "EP_values", "value": ",".join(str(value) for value in args.ep_values)},
    ]


def _failure_placeholder() -> List[Dict]:
    return [
        {
            "tuning_scope": "",
            "phase": "",
            "combo_id": "",
            "scale": "",
            "instance_id": "",
            "file": "",
            "seed": "",
            "error": "no failures",
            "traceback": "",
        }
    ]


def _history_placeholder() -> List[Dict]:
    return [{"message": "history recording disabled"}]


def _run_layered_orthogonal_for_cases(
    args: argparse.Namespace,
    case_rows: List[Dict],
    valid_cases: List[Dict],
    tuning_scope: str,
) -> Dict[str, List[Dict]]:
    base_params = _base_params(args)

    training_combos = _build_l9_combos("training", TRAINING_FACTORS, base_params, args.max_combos)
    training_raw, training_history, training_failures = _run_combos("training", training_combos, valid_cases, args)
    training_summary = _summarize_combos(training_raw, training_combos, "training")
    training_best = _best_combo(training_summary)
    training_best_params = dict(base_params)
    if training_best:
        for key in ["alpha", "gamma", "epsilon", "epsilon_decay"]:
            training_best_params[key] = training_best[key]

    reward_combos = _build_l9_combos("reward", REWARD_FACTORS, training_best_params, args.max_combos)
    reward_raw, reward_history, reward_failures = _run_combos("reward", reward_combos, valid_cases, args)
    reward_summary = _summarize_combos(reward_raw, reward_combos, "reward")
    reward_best = _best_combo(reward_summary)
    final_params = dict(training_best_params)
    if reward_best:
        for key in ["global_reward_scale", "local_reward_scale", "lookahead_reward_weight", "beam_width"]:
            final_params[key] = reward_best[key]

    ep_raw, ep_failures = _run_ep_sensitivity(final_params, valid_cases, args)
    ep_summary = _summarize_ep(ep_raw)
    ep_best = _best_combo([
        {
            "combo_id": f"EP-{row['EP']}",
            "mean_arpd": row["mean_arpd"],
            "mean_time": row["mean_time"],
            "convergence_rate": row["convergence_rate"],
            "mean_rollouts": row["mean_rollouts"],
            **row,
        }
        for row in ep_summary
    ])
    if ep_best and "EP" in ep_best:
        final_params["max_rollouts"] = ep_best["EP"]

    failures = training_failures + reward_failures + ep_failures
    metadata = _metadata_rows(args, case_rows, valid_cases, failures, tuning_scope)

    orthogonal_table = _tag_rows(training_combos + reward_combos, tuning_scope)
    raw_runs = _tag_rows(training_raw + reward_raw + ep_raw, tuning_scope)
    combo_summary = _tag_rows(training_summary + reward_summary, tuning_scope)
    range_analysis = _tag_rows(
        _range_analysis(training_summary, TRAINING_FACTORS)
        + _range_analysis(reward_summary, REWARD_FACTORS),
        tuning_scope,
    )
    ep_summary = _tag_rows(ep_summary, tuning_scope)
    convergence = _tag_rows(training_history + reward_history, tuning_scope)

    return {
        "metadata": metadata,
        "orthogonal_table": orthogonal_table,
        "case_check": case_rows,
        "raw_runs": raw_runs,
        "combo_summary": combo_summary,
        "range_analysis": range_analysis,
        "ep_sensitivity": ep_summary,
        "scale_best_params": [
            _scale_best_param_row(training_best, reward_best, ep_best, final_params, tuning_scope)
        ],
        "best_params": _best_params_rows(training_best, reward_best, ep_best, final_params, tuning_scope),
        "convergence": convergence,
        "failures": _tag_rows(failures, tuning_scope),
    }


def _merge_scope_sheets(scope_sheets: List[Dict[str, List[Dict]]], case_rows: List[Dict]) -> Dict[str, List[Dict]]:
    sheet_names = [
        "metadata",
        "orthogonal_table",
        "raw_runs",
        "combo_summary",
        "range_analysis",
        "ep_sensitivity",
        "scale_best_params",
        "best_params",
        "convergence",
        "failures",
    ]
    merged = {name: [] for name in sheet_names}
    for sheets in scope_sheets:
        for name in sheet_names:
            merged[name].extend(sheets.get(name, []))

    merged["case_check"] = case_rows
    if not merged["convergence"]:
        merged["convergence"] = _history_placeholder()
    if not merged["failures"]:
        merged["failures"] = _failure_placeholder()
    return {
        "metadata": merged["metadata"],
        "orthogonal_table": merged["orthogonal_table"],
        "case_check": merged["case_check"],
        "raw_runs": merged["raw_runs"],
        "combo_summary": merged["combo_summary"],
        "range_analysis": merged["range_analysis"],
        "ep_sensitivity": merged["ep_sensitivity"],
        "scale_best_params": merged["scale_best_params"],
        "best_params": merged["best_params"],
        "convergence": merged["convergence"],
        "failures": merged["failures"],
    }


def run_layered_orthogonal(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    case_rows, valid_cases = _discover_cases(args)
    if args.pooled:
        sheets = _run_layered_orthogonal_for_cases(args, case_rows, valid_cases, "ALL")
        if not sheets["convergence"]:
            sheets["convergence"] = _history_placeholder()
        if not sheets["failures"]:
            sheets["failures"] = _failure_placeholder()
        return sheets

    scope_sheets: List[Dict[str, List[Dict]]] = []
    for scale in args.scales:
        scope = f"J{scale}"
        scoped_case_rows = [row for row in case_rows if row["scale"] == scope]
        scoped_valid_cases = [case for case in valid_cases if case["scale"] == scope]
        if not scoped_valid_cases:
            scope_sheets.append(
                {
                    "metadata": _metadata_rows(args, scoped_case_rows, scoped_valid_cases, [], scope),
                    "orthogonal_table": [],
                    "case_check": scoped_case_rows,
                    "raw_runs": [],
                    "combo_summary": [],
                    "range_analysis": [],
                    "ep_sensitivity": [],
                    "scale_best_params": [],
                    "best_params": [{"tuning_scope": scope, "stage": "skipped", "message": "no valid cases"}],
                    "convergence": [],
                    "failures": [
                        {
                            "tuning_scope": scope,
                            "phase": "",
                            "combo_id": "",
                            "scale": scope,
                            "instance_id": "",
                            "file": "",
                            "seed": "",
                            "error": "no valid cases",
                            "traceback": "",
                        }
                    ],
                }
            )
            continue

        print(f"=== Tuning scope {scope}: {len(scoped_valid_cases)} valid cases ===")
        scope_sheets.append(
            _run_layered_orthogonal_for_cases(
                args,
                scoped_case_rows,
                scoped_valid_cases,
                scope,
            )
        )

    return _merge_scope_sheets(scope_sheets, case_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run layered orthogonal experiments for RQL parameter tuning.")
    parser.add_argument("--cases-dir", default="cases")
    parser.add_argument("--scales", type=_parse_int_csv, default=DEFAULT_SCALES)
    parser.add_argument("--max-cases-per-scale", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-combos", type=int, default=0, help="0 means all L9 combinations.")
    parser.add_argument("--output-xlsx", default="results/orthogonal_by_scale_results.xlsx")
    parser.add_argument("--max-rollouts", type=int, default=TRAINING_PARAMS["max_rollouts"])
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--ep-values", type=_parse_int_csv, default=DEFAULT_EP_VALUES)
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Use the old pooled mode and tune one shared parameter set across all selected scales.",
    )
    parser.add_argument("--record-history", action="store_true")
    parser.add_argument("--history-decode-interval", type=int, default=10)
    parser.add_argument("--verbose-training", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sheets = run_layered_orthogonal(args)
    write_xlsx(args.output_xlsx, sheets)
    failure_count = 0
    if sheets["failures"] and sheets["failures"][0].get("error") != "no failures":
        failure_count = len(sheets["failures"])
    print(f"XLSX written to: {Path(args.output_xlsx).resolve()}")
    print(f"raw_runs={len(sheets['raw_runs'])} failures={failure_count}")


if __name__ == "__main__":
    main()
