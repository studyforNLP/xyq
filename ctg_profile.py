from __future__ import annotations

import argparse
import csv
import random
import re
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np

from cost_to_go import clear_cost_to_go_cache, get_cost_to_go_cache_info
from parameters import TRAINING_PARAMS
from parser import parse_custom_file
from qlearning import init_q_table
from state_space import ACTIONS, generate_state_space
from training import sample_training_loop


def _parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _natural_key(value: str) -> List[object]:
    parts = re.split(r"(\d+)", value.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def _case_files(cases_dir: Path, scale: int, max_cases: int) -> List[Path]:
    scale_dir = cases_dir / f"j{scale}.mm"
    files = sorted(scale_dir.glob("*.mm"), key=lambda path: _natural_key(path.name))
    if max_cases > 0:
        files = files[:max_cases]
    return files


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _profile_case(case_file: Path, scale: int, seed: int, args: argparse.Namespace) -> dict:
    parsed = parse_custom_file(case_file)
    state_space = generate_state_space(parsed.people_num)
    q_table = init_q_table(state_space, ACTIONS)
    _set_seed(seed)
    clear_cost_to_go_cache()

    started = time.perf_counter()
    _q_table, metrics = sample_training_loop(
        Q_table=q_table,
        parsed=parsed,
        max_rollouts=args.max_rollouts,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        convergence_threshold=args.convergence_threshold,
        global_reward_scale=args.global_reward_scale,
        local_reward_scale=args.local_reward_scale,
        immediate_reward_weight=args.immediate_reward_weight,
        lookahead_reward_weight=args.lookahead_reward_weight,
        normalize_immediate_reward=not args.disable_reward_normalization,
        lookahead_clip=args.lookahead_clip,
        verbose=False,
        return_metrics=True,
        training_mode="rql",
        return_history=False,
    )
    elapsed = time.perf_counter() - started
    info = get_cost_to_go_cache_info()
    hits = int(info["hits"])
    misses = int(info["misses"])
    calls = int(info["ctg_calls"])
    call_time = float(info["ctg_call_time"])
    simulation_calls = int(info["ctg_simulation_calls"])
    simulation_time = float(info["ctg_simulation_time"])

    return {
        "scale": f"J{scale}",
        "file": case_file.name,
        "seed": seed,
        "max_rollouts": args.max_rollouts,
        "completed_rollouts": metrics["rollouts"],
        "elapsed_time": elapsed,
        "ctg_calls": calls,
        "ctg_call_time": call_time,
        "avg_ctg_time": float(info["avg_ctg_time"]),
        "ctg_simulation_calls": simulation_calls,
        "ctg_simulation_time": simulation_time,
        "avg_ctg_simulation_time": float(info["avg_ctg_simulation_time"]),
        "ctg_time_ratio": _safe_ratio(call_time, elapsed),
        "ctg_simulation_time_ratio": _safe_ratio(simulation_time, elapsed),
        "ctg_cache_hits": hits,
        "ctg_cache_misses": misses,
        "ctg_cache_hit_rate": _safe_ratio(hits, hits + misses),
        "ctg_cache_size": int(info["size"]),
        "converged": metrics["converged"],
        "max_q_diff": metrics["max_q_diff"],
    }


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile cost-to-go runtime with a very small RQL training run."
    )
    parser.add_argument("--cases-dir", default="cases")
    parser.add_argument("--scales", type=_parse_int_csv, default=[10])
    parser.add_argument("--max-cases-per-scale", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--max-rollouts", type=int, default=3)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--alpha", type=float, default=TRAINING_PARAMS["alpha"])
    parser.add_argument("--gamma", type=float, default=TRAINING_PARAMS["gamma"])
    parser.add_argument("--epsilon", type=float, default=TRAINING_PARAMS["epsilon"])
    parser.add_argument("--epsilon-decay", type=float, default=TRAINING_PARAMS["epsilon_decay"])
    parser.add_argument("--epsilon-min", type=float, default=TRAINING_PARAMS["epsilon_min"])
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=TRAINING_PARAMS["convergence_threshold"],
    )
    parser.add_argument(
        "--global-reward-scale",
        type=float,
        default=TRAINING_PARAMS["global_reward_scale"],
    )
    parser.add_argument(
        "--local-reward-scale",
        type=float,
        default=TRAINING_PARAMS["local_reward_scale"],
    )
    parser.add_argument(
        "--immediate-reward-weight",
        type=float,
        default=TRAINING_PARAMS["immediate_reward_weight"],
    )
    parser.add_argument(
        "--lookahead-reward-weight",
        type=float,
        default=TRAINING_PARAMS["lookahead_reward_weight"],
    )
    parser.add_argument("--lookahead-clip", type=float, default=TRAINING_PARAMS["lookahead_clip"])
    parser.add_argument("--disable-reward-normalization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_dir = Path(args.cases_dir)
    rows = []

    for scale in args.scales:
        files = _case_files(cases_dir, scale, args.max_cases_per_scale)
        if not files:
            print(f"No cases found for J{scale}: {cases_dir / f'j{scale}.mm'}")
            continue
        for case_file in files:
            for seed in range(args.seeds):
                row = _profile_case(case_file, scale, seed, args)
                rows.append(row)
                print(
                    " ".join(
                        [
                            f"{row['scale']}/{row['file']}",
                            f"seed={seed}",
                            f"elapsed={row['elapsed_time']:.4f}s",
                            f"ctg_calls={row['ctg_calls']}",
                            f"ctg_time={row['ctg_call_time']:.4f}s",
                            f"avg_ctg={row['avg_ctg_time']:.6f}s",
                            f"sim_time={row['ctg_simulation_time']:.4f}s",
                            f"hit_rate={row['ctg_cache_hit_rate']:.2%}",
                        ]
                    )
                )

    if args.output_csv:
        output_path = Path(args.output_csv)
        _write_csv(output_path, rows)
        print(f"CSV written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
