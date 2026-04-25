import argparse
import csv
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np

from config import DEFAULT_TARGET_FILE
from cost_to_go import (
    calculate_baseline_cost_to_go,
    clear_cost_to_go_cache,
    get_cost_to_go_cache_info,
)
from environment import init_environment
from parameters import FINAL_SOLUTION_PARAMS, TRAINING_PARAMS
from parser import parse_custom_file
from qlearning import init_q_table
from state_space import ACTIONS, generate_state_space
from training import build_final_solution, sample_training_loop


DEFAULT_PARAMS = {**TRAINING_PARAMS, **FINAL_SOLUTION_PARAMS}


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
        "last_action_time": max((item["time"] for item in solution), default=0),
    }


def run_experiment(args: argparse.Namespace) -> List[Dict]:
    target_file = Path(args.target_file)
    parsed = parse_custom_file(target_file)
    max_skill_level = _max_skill_level(parsed.people_num)
    state_space = generate_state_space(parsed.people_num)
    baseline_cost = calculate_baseline_cost_to_go(parsed, init_environment(parsed))

    rows: List[Dict] = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        clear_cost_to_go_cache()
        _set_seed(seed)
        q_table = init_q_table(state_space, ACTIONS)

        started_at = time.perf_counter()
        q_table, train_metrics = sample_training_loop(
            Q_table=q_table,
            parsed=parsed,
            max_rollouts=args.max_rollouts,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
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
        )
        solution, penalty = build_final_solution(
            q_table,
            parsed,
            max_skill_level,
            q_distinction_threshold=args.q_distinction_threshold,
            beam_width=args.beam_width,
            beam_branch_limit=args.beam_branch_limit,
            beam_improvement_margin=args.beam_improvement_margin,
        )
        runtime_seconds = time.perf_counter() - started_at
        cache_info = get_cost_to_go_cache_info()

        row = {
            "seed": seed,
            "penalty": penalty,
            "baseline_cost": baseline_cost,
            "improvement_vs_baseline": baseline_cost - penalty,
            "converged": train_metrics["converged"],
            "rollouts": train_metrics["rollouts"],
            "max_q_diff": train_metrics["max_q_diff"],
            "runtime_seconds": runtime_seconds,
            "ctg_cache_size": cache_info["size"],
            "ctg_cache_hits": cache_info["hits"],
            "ctg_cache_misses": cache_info["misses"],
            **_solution_stats(solution),
        }
        rows.append(row)
        print(
            f"seed={seed} penalty={penalty:.2f} "
            f"rollouts={train_metrics['rollouts']} "
            f"converged={train_metrics['converged']} "
            f"time={runtime_seconds:.2f}s"
        )

    return rows


def write_csv(rows: List[Dict], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: List[Dict]) -> None:
    penalties = [float(row["penalty"]) for row in rows]
    rollouts = [int(row["rollouts"]) for row in rows]
    runtimes = [float(row["runtime_seconds"]) for row in rows]
    converged_count = sum(1 for row in rows if row["converged"])

    print("\nExperiment summary")
    print(f"runs: {len(rows)}")
    print(f"converged: {converged_count}/{len(rows)}")
    print(f"penalty avg/min/max/std: {mean(penalties):.4f}/{min(penalties):.4f}/{max(penalties):.4f}/{pstdev(penalties):.4f}")
    print(f"rollouts avg: {mean(rollouts):.2f}")
    print(f"runtime avg seconds: {mean(runtimes):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed Rollout-Q-learning experiments.")
    parser.add_argument("--target-file", default=str(DEFAULT_TARGET_FILE))
    parser.add_argument("--output", default="experiment_results.csv")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--max-rollouts", type=int, default=DEFAULT_PARAMS["max_rollouts"])
    parser.add_argument("--alpha", type=float, default=DEFAULT_PARAMS["alpha"])
    parser.add_argument("--gamma", type=float, default=DEFAULT_PARAMS["gamma"])
    parser.add_argument("--epsilon", type=float, default=DEFAULT_PARAMS["epsilon"])
    parser.add_argument("--convergence-threshold", type=float, default=DEFAULT_PARAMS["convergence_threshold"])
    parser.add_argument("--global-reward-scale", type=float, default=DEFAULT_PARAMS["global_reward_scale"])
    parser.add_argument("--local-reward-scale", type=float, default=DEFAULT_PARAMS["local_reward_scale"])
    parser.add_argument("--immediate-reward-weight", type=float, default=DEFAULT_PARAMS["immediate_reward_weight"])
    parser.add_argument("--lookahead-reward-weight", type=float, default=DEFAULT_PARAMS["lookahead_reward_weight"])
    parser.add_argument("--lookahead-clip", type=float, default=DEFAULT_PARAMS["lookahead_clip"])
    parser.add_argument("--q-distinction-threshold", type=float, default=DEFAULT_PARAMS["q_distinction_threshold"])
    parser.add_argument("--beam-width", type=int, default=DEFAULT_PARAMS["beam_width"])
    parser.add_argument("--beam-branch-limit", type=int, default=DEFAULT_PARAMS["beam_branch_limit"])
    parser.add_argument("--beam-improvement-margin", type=float, default=DEFAULT_PARAMS["beam_improvement_margin"])
    parser.add_argument("--disable-reward-normalization", action="store_true")
    parser.add_argument("--verbose-training", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_experiment(args)
    output_path = Path(args.output)
    write_csv(rows, output_path)
    print_summary(rows)
    print(f"CSV written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
