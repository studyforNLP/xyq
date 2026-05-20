from __future__ import annotations

import argparse
import math
import random
import re
import time
import traceback
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List
from xml.sax.saxutils import escape

import numpy as np

from cost_to_go import clear_cost_to_go_cache, get_cost_to_go_cache_info
from excel_writer import write_xlsx
from experiment_suite import (
    RQL_CONVERGENCE_PROFILES,
    _max_skill_level,
    _train_rql_convergence_profile,
)
from parameters import FINAL_SOLUTION_PARAMS, TRAINING_PARAMS
from parser import parse_custom_file
from qlearning import init_q_table
from state_space import ACTIONS, generate_state_space
from training import build_final_solution


DEFAULT_SCALES = [10, 20, 30, 60]
DEFAULT_MAX_ROLLOUTS_BY_SCALE = {
    10: 1000,
    20: 2000,
    30: 3000,
    60: 4000,
}


def _parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_scale_rollouts(value: str) -> Dict[int, int]:
    if not value.strip():
        return dict(DEFAULT_MAX_ROLLOUTS_BY_SCALE)
    result: Dict[int, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        scale_text, rollout_text = item.split(":", maxsplit=1)
        result[int(scale_text.strip())] = int(rollout_text.strip())
    return result


def _natural_key(value: str) -> List[object]:
    parts = re.split(r"(\d+)", value.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _safe_pstdev(values: Iterable[float]) -> float:
    values = list(values)
    return pstdev(values) if len(values) > 1 else 0.0


def _discover_cases(args: argparse.Namespace) -> tuple[List[Dict], List[Dict]]:
    case_check_rows: List[Dict] = []
    valid_cases: List[Dict] = []
    cases_dir = Path(args.cases_dir)

    for scale in args.scales:
        scale_dir = cases_dir / f"j{scale}.mm"
        files = sorted(scale_dir.glob("*.mm"), key=lambda path: _natural_key(path.name))
        if args.max_cases_per_scale > 0:
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
            case_check_rows.append(row)

    return case_check_rows, valid_cases


def _profile_name_for_scale(scale: int, args: argparse.Namespace) -> str:
    if args.rql_convergence_profile == "scale_specific":
        return "checkpoint" if scale == 10 else "combined"
    return args.rql_convergence_profile


def _profile_for_scale(scale: int, args: argparse.Namespace):
    profile_name = _profile_name_for_scale(scale, args)
    base_profile = RQL_CONVERGENCE_PROFILES[profile_name]
    max_rollouts = args.max_rollouts_by_scale.get(
        scale,
        DEFAULT_MAX_ROLLOUTS_BY_SCALE.get(scale, TRAINING_PARAMS["max_rollouts"]),
    )
    return profile_name, replace(base_profile, ep_by_scale={scale: max_rollouts})


def _decode_penalty(q_table: Dict, parsed, max_skill_level: int, args: argparse.Namespace) -> float:
    _solution, penalty = build_final_solution(
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


def _run_case_seed(case: Dict, seed: int, args: argparse.Namespace) -> tuple[List[Dict], Dict]:
    parsed = case["parsed"]
    scale = int(case["scale_value"])
    profile_name, profile = _profile_for_scale(scale, args)
    max_rollouts = profile.ep_by_scale[scale]
    max_skill_level = _max_skill_level(parsed.people_num)
    state_space = generate_state_space(parsed.people_num)
    q_table = init_q_table(state_space, ACTIONS)

    _set_seed(seed)
    clear_cost_to_go_cache()
    started = time.perf_counter()
    trained_q_table, train_metrics = _train_rql_convergence_profile(
        q_table,
        parsed,
        scale,
        profile_name,
        profile,
        args,
        record_history=True,
    )
    train_time = time.perf_counter() - started
    final_penalty = _decode_penalty(trained_q_table, parsed, max_skill_level, args)
    cache_info = get_cost_to_go_cache_info()

    history_rows: List[Dict] = []
    best_so_far = float("inf")
    best_reward_so_far = -float("inf")
    reward_window: List[float] = []
    for item in train_metrics.get("history", []):
        episode_reward = float(item["episode_reward"])
        decoded_penalty = float(item["decoded_penalty"])
        best_so_far = min(best_so_far, decoded_penalty)
        best_reward_so_far = max(best_reward_so_far, episode_reward)
        reward_window.append(episode_reward)
        if len(reward_window) > args.reward_moving_window:
            reward_window.pop(0)
        max_q_diff = float(item["max_q_diff"])
        episode = int(item["episode"])
        history_rows.append(
            {
                "scale": case["scale"],
                "scale_value": scale,
                "instance_id": case["instance_id"],
                "file": case["file"],
                "seed": seed,
                "profile": profile_name,
                "episode": episode,
                "training_step": episode,
                "max_rollouts": max_rollouts,
                "progress": episode / max(1, max_rollouts),
                "episode_reward": episode_reward,
                "smoothed_episode_reward": _safe_mean(reward_window),
                "best_so_far_reward": best_reward_so_far,
                "decoded_penalty": decoded_penalty,
                "best_so_far_penalty": best_so_far,
                "max_q_diff": max_q_diff,
                "log_max_q_diff": math.log10(max(max_q_diff, 1e-12)),
                "epsilon": float(item["epsilon"]),
                "alpha": float(item["alpha"]),
                "lookahead_reward_weight": float(item["lookahead_reward_weight"]),
            }
        )

    run_summary = {
        "scale": case["scale"],
        "instance_id": case["instance_id"],
        "file": case["file"],
        "seed": seed,
        "profile": profile_name,
        "max_rollouts": max_rollouts,
        "completed_rollouts": train_metrics["rollouts"],
        "history_points": len(history_rows),
        "final_penalty": final_penalty,
        "best_history_penalty": min(
            (row["best_so_far_penalty"] for row in history_rows),
            default="",
        ),
        "train_time": train_time,
        "converged": train_metrics["converged"],
        "max_q_diff": train_metrics["max_q_diff"],
        "used_checkpoint": train_metrics.get("used_checkpoint", ""),
        "best_checkpoint_penalty": train_metrics.get("best_checkpoint_penalty", ""),
        "ctg_calls": cache_info["ctg_calls"],
        "ctg_call_time": cache_info["ctg_call_time"],
        "avg_ctg_time": cache_info["avg_ctg_time"],
        "ctg_cache_hits": cache_info["hits"],
        "ctg_cache_misses": cache_info["misses"],
    }
    return history_rows, run_summary


def _build_curve_summary(history_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple[str, int], List[Dict]] = {}
    for row in history_rows:
        key = (str(row["scale"]), int(row["episode"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []
    for (scale, episode), rows in sorted(grouped.items(), key=lambda item: (_scale_key(item[0][0]), item[0][1])):
        decoded = [float(row["decoded_penalty"]) for row in rows]
        best_so_far = [float(row["best_so_far_penalty"]) for row in rows]
        rewards = [float(row["episode_reward"]) for row in rows]
        smoothed_rewards = [float(row["smoothed_episode_reward"]) for row in rows]
        best_rewards = [float(row["best_so_far_reward"]) for row in rows]
        max_q_diff = [float(row["max_q_diff"]) for row in rows]
        log_q_diff = [float(row["log_max_q_diff"]) for row in rows]
        progress = _safe_mean(float(row["progress"]) for row in rows)
        summary_rows.append(
            {
                "scale": scale,
                "episode": episode,
                "training_step": episode,
                "progress": progress,
                "runs": len(rows),
                "avg_episode_reward": _safe_mean(rewards),
                "std_episode_reward": _safe_pstdev(rewards),
                "avg_smoothed_reward": _safe_mean(smoothed_rewards),
                "std_smoothed_reward": _safe_pstdev(smoothed_rewards),
                "avg_best_so_far_reward": _safe_mean(best_rewards),
                "std_best_so_far_reward": _safe_pstdev(best_rewards),
                "avg_decoded_penalty": _safe_mean(decoded),
                "std_decoded_penalty": _safe_pstdev(decoded),
                "avg_best_so_far_penalty": _safe_mean(best_so_far),
                "std_best_so_far_penalty": _safe_pstdev(best_so_far),
                "avg_max_q_diff": _safe_mean(max_q_diff),
                "avg_log_max_q_diff": _safe_mean(log_q_diff),
            }
        )
    return summary_rows


def _build_reward_curve_rows(curve_summary_rows: List[Dict]) -> List[Dict]:
    """Compact plotting table: x-axis is training step, y-axis is reward."""
    return [
        {
            "scale": row["scale"],
            "training_step": row["training_step"],
            "progress": row["progress"],
            "runs": row["runs"],
            "avg_episode_reward": row["avg_episode_reward"],
            "std_episode_reward": row["std_episode_reward"],
            "avg_smoothed_reward": row["avg_smoothed_reward"],
            "std_smoothed_reward": row["std_smoothed_reward"],
            "avg_best_so_far_reward": row["avg_best_so_far_reward"],
            "std_best_so_far_reward": row["std_best_so_far_reward"],
        }
        for row in curve_summary_rows
    ]


def _scale_key(scale: str) -> int:
    digits = "".join(ch for ch in scale if ch.isdigit())
    return int(digits) if digits else 0


def _metadata_rows(args: argparse.Namespace, case_count: int) -> List[Dict]:
    return [
        {"key": "description", "value": "Paper-oriented RQL convergence curve experiment"},
        {"key": "cases_dir", "value": str(Path(args.cases_dir).resolve())},
        {"key": "scales", "value": ",".join(f"J{scale}" for scale in args.scales)},
        {"key": "max_cases_per_scale", "value": args.max_cases_per_scale},
        {"key": "seeds", "value": args.seeds},
        {
            "key": "max_rollouts_by_scale",
            "value": ",".join(
                f"J{scale}:{args.max_rollouts_by_scale[scale]}"
                for scale in sorted(args.max_rollouts_by_scale)
            ),
        },
        {"key": "history_decode_interval", "value": args.history_decode_interval},
        {"key": "reward_moving_window", "value": args.reward_moving_window},
        {"key": "rql_convergence_profile", "value": args.rql_convergence_profile},
        {"key": "rql_checkpoint_interval", "value": args.rql_checkpoint_interval},
        {"key": "valid_case_count", "value": case_count},
        {
            "key": "primary_convergence_curve",
            "value": "training_step vs avg_smoothed_reward",
        },
        {
            "key": "reward_definition",
            "value": "episode_reward = global reward = -total weighted milestone delay penalty",
        },
        {"key": "auxiliary_curve_penalty", "value": "training_step vs avg_best_so_far_penalty"},
        {"key": "auxiliary_curve_qdiff", "value": "training_step vs avg_log_max_q_diff"},
    ]


def _write_svg_curve(
    rows: List[Dict],
    output_path: Path,
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    if not rows:
        return
    width = 980
    height = 600
    left = 90
    right = 210
    top = 55
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]

    by_scale: Dict[str, List[Dict]] = {}
    for row in rows:
        by_scale.setdefault(str(row["scale"]), []).append(row)
    scales = sorted(by_scale, key=_scale_key)
    x_values = [float(row[x_key]) for row in rows]
    x_min = min(x_values)
    x_max = max(x_values)
    if abs(x_max - x_min) < 1e-12:
        x_min = 0.0
        x_max += 1.0
    y_values = [float(row[y_key]) for row in rows]
    y_min = min(y_values)
    y_max = max(y_values)
    if abs(y_max - y_min) < 1e-12:
        y_min -= 1.0
        y_max += 1.0
    y_padding = (y_max - y_min) * 0.08
    y_min -= y_padding
    y_max += y_padding

    def x_pos(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in range(0, 6):
        value = x_min + (x_max - x_min) * tick / 5
        x = x_pos(value)
        y_axis = top + plot_h
        parts.append(f'<line x1="{x}" y1="{y_axis}" x2="{x}" y2="{y_axis + 6}" stroke="#333"/>')
        parts.append(
            f'<text x="{x}" y="{y_axis + 26}" text-anchor="middle" font-size="13" font-family="Arial">{value:.0f}</text>'
        )
    for tick in range(0, 6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_pos(value)
        parts.append(f'<line x1="{left - 6}" y1="{y}" x2="{left}" y2="{y}" stroke="#333"/>')
        parts.append(
            f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#e6e6e6"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" font-size="12" font-family="Arial">{value:.2f}</text>'
        )

    for index, scale in enumerate(scales):
        scale_rows = sorted(by_scale[scale], key=lambda row: float(row[x_key]))
        points = [
            f'{x_pos(float(row[x_key])):.2f},{y_pos(float(row[y_key])):.2f}'
            for row in scale_rows
        ]
        color = colors[index % len(colors)]
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>'
        )
        legend_y = top + 25 + index * 24
        legend_x = left + plot_w + 35
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{legend_x + 36}" y="{legend_y + 5}" font-size="14" font-family="Arial">{escape(scale)}</text>'
        )

    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 25}" text-anchor="middle" font-size="15" font-family="Arial">{escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="22" y="{top + plot_h / 2}" text-anchor="middle" font-size="15" font-family="Arial" transform="rotate(-90 22 {top + plot_h / 2})">{escape(y_label)}</text>'
    )
    parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    case_check_rows, valid_cases = _discover_cases(args)
    history_rows: List[Dict] = []
    run_summary_rows: List[Dict] = []
    failure_rows: List[Dict] = []

    for case in valid_cases:
        for seed in range(args.seeds):
            print(
                f"Running convergence curve: {case['scale']} {case['file']} seed={seed}",
                flush=True,
            )
            try:
                case_history, run_summary = _run_case_seed(case, seed, args)
                history_rows.extend(case_history)
                run_summary_rows.append(run_summary)
            except Exception as exc:  # noqa: BLE001 - keep batch running.
                failure_rows.append(
                    {
                        "scale": case["scale"],
                        "instance_id": case["instance_id"],
                        "file": case["file"],
                        "seed": seed,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    curve_summary_rows = _build_curve_summary(history_rows)
    reward_curve_rows = _build_reward_curve_rows(curve_summary_rows)
    if not failure_rows:
        failure_rows = [{"scale": "", "instance_id": "", "file": "", "seed": "", "error": "no failures", "traceback": ""}]

    return {
        "metadata": _metadata_rows(args, len(valid_cases)),
        "case_check": case_check_rows,
        "run_summary": run_summary_rows,
        "raw_history": history_rows,
        "reward_curve": reward_curve_rows,
        "curve_summary": curve_summary_rows,
        "failures": failure_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record RQL convergence curves for paper figures."
    )
    parser.add_argument("--cases-dir", default="cases")
    parser.add_argument("--scales", type=_parse_int_csv, default=DEFAULT_SCALES)
    parser.add_argument("--max-cases-per-scale", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--max-rollouts-by-scale",
        type=_parse_scale_rollouts,
        default=dict(DEFAULT_MAX_ROLLOUTS_BY_SCALE),
        help="Comma-separated scale:max_rollouts map, e.g. 10:1000,20:2000,30:3000.",
    )
    parser.add_argument("--history-decode-interval", type=int, default=50)
    parser.add_argument("--reward-moving-window", type=int, default=5)
    parser.add_argument("--max-rollouts", type=int, default=TRAINING_PARAMS["max_rollouts"])
    parser.add_argument(
        "--rql-convergence-profile",
        choices=["scale_specific", *RQL_CONVERGENCE_PROFILES.keys()],
        default="scale_specific",
    )
    parser.add_argument("--rql-checkpoint-interval", type=int, default=100)
    parser.add_argument("--rql-max-rollouts-cap", type=int, default=0)
    parser.add_argument("--output-xlsx", default="results/paper_convergence_curve.xlsx")
    parser.add_argument("--output-prefix", default="results/paper_convergence_curve")
    parser.add_argument("--no-svg", action="store_true")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sheets = run(args)
    write_xlsx(args.output_xlsx, sheets)
    print(f"XLSX written to: {Path(args.output_xlsx).resolve()}")

    if not args.no_svg:
        prefix = Path(args.output_prefix)
        _write_svg_curve(
            sheets["reward_curve"],
            prefix.with_name(prefix.name + "_reward.svg"),
            "training_step",
            "avg_smoothed_reward",
            "RQL reward convergence curve",
            "Training steps",
            "Average smoothed reward",
        )
        _write_svg_curve(
            sheets["curve_summary"],
            prefix.with_name(prefix.name + "_penalty.svg"),
            "training_step",
            "avg_best_so_far_penalty",
            "RQL penalty convergence curve",
            "Training steps",
            "Average best-so-far penalty",
        )
        _write_svg_curve(
            sheets["curve_summary"],
            prefix.with_name(prefix.name + "_qdiff.svg"),
            "training_step",
            "avg_log_max_q_diff",
            "RQL Q-table convergence",
            "Training steps",
            "Mean log10(max Q diff)",
        )
        print(f"SVG written to: {prefix.with_name(prefix.name + '_reward.svg').resolve()}")
        print(f"SVG written to: {prefix.with_name(prefix.name + '_penalty.svg').resolve()}")
        print(f"SVG written to: {prefix.with_name(prefix.name + '_qdiff.svg').resolve()}")

    failure_rows = sheets["failures"]
    failure_count = 0 if failure_rows and failure_rows[0].get("error") == "no failures" else len(failure_rows)
    print(f"history_rows={len(sheets['raw_history'])} failures={failure_count}")


if __name__ == "__main__":
    main()
