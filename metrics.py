from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple


def _as_float(value) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _safe_pstdev(values: Iterable[float]) -> float:
    values = list(values)
    return pstdev(values) if len(values) > 1 else 0.0


def build_instance_summary(raw_rows: List[Dict]) -> List[Dict]:
    best_known: Dict[Tuple[str, str], float] = {}
    for row in raw_rows:
        key = (str(row["scale"]), str(row["instance_id"]))
        penalty = _as_float(row["penalty"])
        best_known[key] = min(best_known.get(key, penalty), penalty)

    grouped: Dict[Tuple[str, str, str, str], List[Dict]] = defaultdict(list)
    for row in raw_rows:
        key = (
            str(row["scale"]),
            str(row["instance_id"]),
            str(row["file"]),
            str(row["method"]),
        )
        grouped[key].append(row)

    summary_rows: List[Dict] = []
    for (scale, instance_id, file_name, method), rows in sorted(grouped.items()):
        penalties = [_as_float(row["penalty"]) for row in rows]
        times = [_as_float(row["total_time"]) for row in rows]
        best_value = best_known[(scale, instance_id)]
        if method == "RQL":
            mean_penalty = min(penalties) if penalties else 0.0
        else:
            mean_penalty = _safe_mean(penalties)
        arpd = ((mean_penalty - best_value) / max(best_value, 1.0)) * 100.0
        summary_rows.append(
            {
                "scale": scale,
                "instance_id": instance_id,
                "file": file_name,
                "method": method,
                "runs": len(rows),
                "mean_penalty": mean_penalty,
                "best_penalty": min(penalties) if penalties else 0.0,
                "std_penalty": _safe_pstdev(penalties),
                "mean_time": _safe_mean(times),
                "best_known_penalty": best_value,
                "arpd": arpd,
            }
        )
    return summary_rows


def build_scale_summary(instance_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for row in instance_rows:
        grouped[(str(row["scale"]), str(row["method"]))].append(row)

    summary_rows: List[Dict] = []
    for (scale, method), rows in sorted(grouped.items(), key=lambda item: (_scale_sort_key(item[0][0]), item[0][1])):
        mean_penalties = [_as_float(row["mean_penalty"]) for row in rows]
        best_penalties = [_as_float(row["best_penalty"]) for row in rows]
        arpds = [_as_float(row["arpd"]) for row in rows]
        mean_times = [_as_float(row["mean_time"]) for row in rows]
        summary_rows.append(
            {
                "scale": scale,
                "method": method,
                "instances": len(rows),
                "AOTV": _safe_mean(mean_penalties),
                "Best": _safe_mean(best_penalties),
                "Std": _safe_pstdev(mean_penalties),
                "ARPD": _safe_mean(arpds),
                "Time": _safe_mean(mean_times),
            }
        )
    return summary_rows


def build_method_rank(scale_rows: List[Dict]) -> List[Dict]:
    by_scale: Dict[str, List[Dict]] = defaultdict(list)
    for row in scale_rows:
        by_scale[str(row["scale"])].append(row)

    rank_rows: List[Dict] = []
    for scale, rows in sorted(by_scale.items(), key=lambda item: _scale_sort_key(item[0])):
        aotv_order = {
            row["method"]: index + 1
            for index, row in enumerate(sorted(rows, key=lambda row: (_as_float(row["AOTV"]), str(row["method"]))))
        }
        arpd_order = {
            row["method"]: index + 1
            for index, row in enumerate(sorted(rows, key=lambda row: (_as_float(row["ARPD"]), str(row["method"]))))
        }
        for row in sorted(rows, key=lambda row: aotv_order[row["method"]]):
            rank_rows.append(
                {
                    "scale": scale,
                    "method": row["method"],
                    "AOTV": row["AOTV"],
                    "ARPD": row["ARPD"],
                    "AOTV_rank": aotv_order[row["method"]],
                    "ARPD_rank": arpd_order[row["method"]],
                }
            )
    return rank_rows


def _scale_sort_key(scale: str) -> int:
    digits = "".join(ch for ch in scale if ch.isdigit())
    return int(digits) if digits else 0


def build_all_summaries(raw_rows: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    instance_summary = build_instance_summary(raw_rows)
    scale_summary = build_scale_summary(instance_summary)
    method_rank = build_method_rank(scale_summary)
    return instance_summary, scale_summary, method_rank
