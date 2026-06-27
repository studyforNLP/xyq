from __future__ import annotations

import csv
import math
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper"
FIG_DIR = PAPER_DIR / "figures"
DATA_DIR = PAPER_DIR / "data"
MAIN_RESULTS_XLSX = ROOT / "results" / "结果对比.xlsx"
CONVERGENCE_XLSX = ROOT / "results" / "paper_convergence_curve.xlsx"

SCALES = ["J10", "J20", "J30", "J60"]
METHODS = ["RQL", "RH", "MXS+MF", "MPV-SLK-EFT", "QL"]
METHOD_COLORS = {
    "RQL": "#0F4D92",
    "RH": "#42949E",
    "MXS+MF": "#8BCF8B",
    "MPV-SLK-EFT": "#767676",
    "QL": "#B64342",
}
SCALE_COLORS = {
    "J10": "#0F4D92",
    "J20": "#42949E",
    "J30": "#9A4D8E",
    "J60": "#B64342",
}


def _xlsx_rows(path: Path) -> dict[str, list[list[str]]]:
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with zipfile.ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in item.findall(".//m:t", ns)))

        def col_index(cell_ref: str) -> int:
            letters = "".join(ch for ch in cell_ref if ch.isalpha())
            result = 0
            for ch in letters:
                result = result * 26 + ord(ch.upper()) - 64
            return result

        def cell_value(cell: ET.Element) -> str:
            cell_type = cell.attrib.get("t")
            value = cell.find("m:v", ns)
            if cell_type == "s" and value is not None:
                return shared[int(value.text or "0")]
            if cell_type == "inlineStr":
                return "".join(t.text or "" for t in cell.findall(".//m:t", ns))
            return value.text if value is not None else ""

        sheets: dict[str, list[list[str]]] = {}
        for sheet in workbook.find("m:sheets", ns) or []:
            name = sheet.attrib["name"]
            rel_id = sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            target = relmap[rel_id]
            if not target.startswith("worksheets/"):
                target = "worksheets/" + target.split("/")[-1]
            root = ET.fromstring(zf.read("xl/" + target))
            rows: list[list[str]] = []
            for row in root.findall(".//m:row", ns):
                values: list[str] = []
                last_col = 0
                for cell in row.findall("m:c", ns):
                    idx = col_index(cell.attrib.get("r", "A1"))
                    while last_col + 1 < idx:
                        values.append("")
                        last_col += 1
                    values.append(cell_value(cell))
                    last_col = idx
                rows.append(values)
            sheets[name] = rows
        return sheets


def _to_float(value: str) -> float:
    return float(value) if value not in {"", None} else float("nan")


def load_main_results() -> dict[str, dict[str, dict[str, float]]]:
    sheets = _xlsx_rows(MAIN_RESULTS_XLSX)
    results: dict[str, dict[str, dict[str, float]]] = {}
    metric_map = {
        "AOTV": "AOTV",
        "标准差": "Std",
        "ARPD(%)": "ARPD",
        "运行时间": "Time",
    }
    for rows in sheets.values():
        if len(rows) < 3:
            continue
        header_idx = None
        for i, row in enumerate(rows):
            if len(row) >= 2 and row[0] == "规模" and row[1] == "指标":
                header_idx = i
                break
        if header_idx is None:
            continue
        header = rows[header_idx]
        method_cols = {name: idx for idx, name in enumerate(header) if name in METHODS}
        scale = ""
        for row in rows[header_idx + 1 :]:
            if not row:
                continue
            if len(row) > 0 and row[0]:
                scale = row[0]
            if scale not in SCALES or len(row) < 2:
                continue
            metric = metric_map.get(row[1])
            if not metric:
                continue
            results.setdefault(scale, {})
            for method, col in method_cols.items():
                if col < len(row):
                    results[scale].setdefault(method, {})[metric] = _to_float(row[col])
    return results


def load_convergence() -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    sheets = _xlsx_rows(CONVERGENCE_XLSX)

    def dict_rows(sheet_name: str) -> list[dict[str, float | str]]:
        rows = sheets.get(sheet_name, [])
        if not rows:
            return []
        header = rows[0]
        output: list[dict[str, float | str]] = []
        for row in rows[1:]:
            item: dict[str, float | str] = {}
            for i, key in enumerate(header):
                raw = row[i] if i < len(row) else ""
                if key in {
                    "training_step",
                    "progress",
                    "runs",
                    "avg_episode_reward",
                    "std_episode_reward",
                    "avg_smoothed_reward",
                    "std_smoothed_reward",
                    "avg_best_so_far_reward",
                    "std_best_so_far_reward",
                    "avg_decoded_penalty",
                    "std_decoded_penalty",
                    "avg_best_so_far_penalty",
                    "std_best_so_far_penalty",
                    "avg_max_q_diff",
                    "avg_log_max_q_diff",
                }:
                    item[key] = _to_float(raw)
                else:
                    item[key] = raw
            output.append(item)
        return output

    return dict_rows("reward_curve"), dict_rows("curve_summary")


def write_source_data(results: dict[str, dict[str, dict[str, float]]], reward_rows, curve_rows) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "main_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scale", "method", "AOTV", "Std", "ARPD", "Time"])
        for scale in SCALES:
            for method in METHODS:
                item = results[scale][method]
                writer.writerow([scale, method, item["AOTV"], item["Std"], item["ARPD"], item["Time"]])
    with (DATA_DIR / "convergence_reward_curve.csv").open("w", newline="", encoding="utf-8") as f:
        if reward_rows:
            writer = csv.DictWriter(f, fieldnames=list(reward_rows[0].keys()))
            writer.writeheader()
            writer.writerows(reward_rows)
    with (DATA_DIR / "convergence_curve_summary.csv").open("w", newline="", encoding="utf-8") as f:
        if curve_rows:
            writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
            writer.writeheader()
            writer.writerows(curve_rows)


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#FFFFFF"/>',
            '<style>text{font-family:Arial,DejaVu Sans,Liberation Sans,sans-serif;} .small{font-size:13px;} .axis{stroke:#272727;stroke-width:1;} .grid{stroke:#E6E6E6;stroke-width:1;} .label{font-size:15px;fill:#272727;} .panel{font-size:19px;font-weight:bold;fill:#272727;} .title{font-size:20px;font-weight:bold;fill:#272727;}</style>',
        ]

    def line(self, x1, y1, x2, y2, color="#272727", width=1, dash=None):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'
        )

    def rect(self, x, y, w, h, fill="#FFFFFF", stroke="#272727", width=1, rx=0, opacity=1):
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'
        )

    def text(self, x, y, text, size=14, color="#272727", weight="normal", anchor="start", rotate=None, klass=None):
        transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate else ""
        cls = f' class="{klass}"' if klass else ""
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}"{transform}{cls}>{escape(str(text))}</text>'
        )

    def arrow(self, x1, y1, x2, y2, color="#4D4D4D", width=1.6):
        self.line(x1, y1, x2, y2, color, width)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 8
        a1 = angle + math.pi * 0.82
        a2 = angle - math.pi * 0.82
        p1 = (x2 + size * math.cos(a1), y2 + size * math.sin(a1))
        p2 = (x2 + size * math.cos(a2), y2 + size * math.sin(a2))
        self.parts.append(
            f'<path d="M {x2:.2f},{y2:.2f} L {p1[0]:.2f},{p1[1]:.2f} L {p2[0]:.2f},{p2[1]:.2f} Z" fill="{color}"/>'
        )

    def polyline(self, points, color="#0F4D92", width=2.5):
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}"/>')

    def circle(self, x, y, r, fill, stroke="#FFFFFF", width=1):
        self.parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def panel_label(svg: Svg, x, y, label):
    svg.text(x, y, label, size=20, weight="bold")


def draw_axes(svg: Svg, x, y, w, h, y_min, y_max, y_ticks=5, y_label="", x_label=""):
    svg.line(x, y + h, x + w, y + h, "#272727", 1.1)
    svg.line(x, y, x, y + h, "#272727", 1.1)
    for i in range(y_ticks + 1):
        val = y_min + (y_max - y_min) * i / y_ticks
        yy = y + h - (val - y_min) / (y_max - y_min) * h
        svg.line(x, yy, x + w, yy, "#E8E8E8", 0.8)
        svg.text(x - 8, yy + 4, f"{val:.0f}" if abs(val) >= 10 else f"{val:.1f}", 11, anchor="end")
    if y_label:
        svg.text(x - 55, y + h / 2, y_label, 13, anchor="middle", rotate=-90)
    if x_label:
        svg.text(x + w / 2, y + h + 42, x_label, 13, anchor="middle")


def draw_grouped_bars(svg: Svg, x, y, w, h, categories, series, y_label, y_max, log=False):
    if log:
        values = [max(1e-4, v) for vals in series.values() for v in vals]
        y_min = math.log10(min(values) * 0.7)
        y_top = math.log10(y_max)
        svg.line(x, y + h, x + w, y + h, "#272727", 1.1)
        svg.line(x, y, x, y + h, "#272727", 1.1)
        tick_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        for tv in tick_values:
            if y_min <= math.log10(tv) <= y_top:
                yy = y + h - (math.log10(tv) - y_min) / (y_top - y_min) * h
                svg.line(x, yy, x + w, yy, "#E8E8E8", 0.8)
                svg.text(x - 8, yy + 4, f"{tv:g}", 11, anchor="end")

        def y_pos(v):
            return y + h - (math.log10(max(1e-4, v)) - y_min) / (y_top - y_min) * h

    else:
        y_min = 0.0
        y_top = y_max
        draw_axes(svg, x, y, w, h, y_min, y_top, y_label=y_label)

        def y_pos(v):
            return y + h - (v - y_min) / (y_top - y_min) * h

    methods = list(series)
    group_w = w / len(categories)
    bar_w = group_w * 0.72 / len(methods)
    for ci, cat in enumerate(categories):
        cx = x + ci * group_w + group_w * 0.14
        for mi, method in enumerate(methods):
            val = series[method][ci]
            bx = cx + mi * bar_w
            by = y_pos(val)
            svg.rect(bx, by, bar_w * 0.88, y + h - by, fill=METHOD_COLORS[method], stroke="none")
        svg.text(x + ci * group_w + group_w / 2, y + h + 22, cat, 12, anchor="middle")
    svg.text(x - 55, y + h / 2, y_label, 13, anchor="middle", rotate=-90)


def draw_line_chart(svg: Svg, x, y, w, h, rows_by_scale, y_key, y_label, title, invert=False):
    all_x = [float(row["training_step"]) for rows in rows_by_scale.values() for row in rows]
    all_y = [float(row[y_key]) for rows in rows_by_scale.values() for row in rows]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    pad = (y_max - y_min) * 0.08 or 1.0
    y_min -= pad
    y_max += pad
    svg.text(x, y - 10, title, 15, weight="bold")
    svg.line(x, y + h, x + w, y + h, "#272727", 1.1)
    svg.line(x, y, x, y + h, "#272727", 1.1)
    for i in range(5):
        val = y_min + (y_max - y_min) * i / 4
        yy = y + h - (val - y_min) / (y_max - y_min) * h
        svg.line(x, yy, x + w, yy, "#E8E8E8", 0.8)
        svg.text(x - 8, yy + 4, f"{val:.1f}", 11, anchor="end")
    for i in range(5):
        val = x_min + (x_max - x_min) * i / 4
        xx = x + (val - x_min) / (x_max - x_min) * w
        svg.line(xx, y + h, xx, y + h + 5, "#272727", 0.9)
        svg.text(xx, y + h + 21, f"{val:.0f}", 11, anchor="middle")

    def pos(row):
        xx = x + (float(row["training_step"]) - x_min) / (x_max - x_min) * w
        yy = y + h - (float(row[y_key]) - y_min) / (y_max - y_min) * h
        return xx, yy

    for scale in SCALES:
        rows = sorted(rows_by_scale.get(scale, []), key=lambda r: float(r["training_step"]))
        if not rows:
            continue
        color = SCALE_COLORS[scale]
        points = [pos(row) for row in rows]
        svg.polyline(points, color, 2.2)
        for px, py in points[:: max(1, len(points) // 6)]:
            svg.circle(px, py, 3.0, color, stroke="white", width=0.8)
        lx, ly = points[-1]
        svg.text(lx + 5, ly + 4, scale, 11, color=color)
    svg.text(x - 55, y + h / 2, y_label, 13, anchor="middle", rotate=-90)
    svg.text(x + w / 2, y + h + 42, "Training steps", 13, anchor="middle")


def figure_1_framework() -> None:
    svg = Svg(1200, 680)
    svg.text(55, 55, "Fig. 1 | Rollout-Q-learning framework for interruptible multi-skill project scheduling", 23, weight="bold")
    panel_label(svg, 55, 105, "a")
    svg.text(88, 105, "Scheduling problem", 18, weight="bold")
    boxes = [
        (80, 140, 220, 80, "Project network", "precedence + milestones"),
        (80, 255, 220, 80, "Multi-skill workforce", "skill-dependent durations"),
        (80, 370, 220, 80, "Interruptible execution", "resume by original worker"),
        (80, 485, 220, 80, "Objective", "weighted milestone delay"),
    ]
    for x, y, w, h, title, subtitle in boxes:
        svg.rect(x, y, w, h, fill="#F5F7FA", stroke="#D0D5DD", rx=12)
        svg.text(x + 18, y + 32, title, 16, weight="bold")
        svg.text(x + 18, y + 58, subtitle, 13, color="#4D4D4D")

    panel_label(svg, 365, 105, "b")
    svg.text(398, 105, "Learning and rollout-assisted decoding", 18, weight="bold")
    pipeline = [
        (385, 175, 180, 75, "6-D state", "NLF, SBI, MUR, RUR, CRT, ITN"),
        (625, 175, 180, 75, "Action tuple", "type, task, person"),
        (865, 175, 190, 75, "Q update", "local + global reward"),
        (625, 330, 180, 75, "Cost-to-go", "rollout lookahead"),
        (865, 330, 190, 75, "Beam decoding", "Q + CTG fallback"),
        (865, 485, 190, 75, "Schedule", "minimum delay penalty"),
    ]
    for x, y, w, h, title, subtitle in pipeline:
        fill = "#EAF2FA" if "Q" in title or "Cost" in title else "#F7F7F7"
        svg.rect(x, y, w, h, fill=fill, stroke="#AEB7C2", rx=12)
        svg.text(x + w / 2, y + 30, title, 16, weight="bold", anchor="middle")
        svg.text(x + w / 2, y + 55, subtitle, 12, color="#4D4D4D", anchor="middle")
    svg.arrow(565, 212, 625, 212)
    svg.arrow(805, 212, 865, 212)
    svg.arrow(715, 250, 715, 330)
    svg.arrow(805, 367, 865, 367)
    svg.arrow(960, 405, 960, 485)
    svg.arrow(960, 250, 960, 330)
    svg.arrow(300, 330, 385, 212)
    svg.text(450, 610, "Training learns action preferences; rollout cost-to-go supplies short-horizon evaluation when Q values are ambiguous.", 15, color="#4D4D4D")
    svg.save(FIG_DIR / "fig1_rql_framework.svg")


def figure_2_quality(results: dict[str, dict[str, dict[str, float]]]) -> None:
    svg = Svg(1200, 760)
    svg.text(55, 50, "Fig. 2 | RQL reduces weighted milestone delay across J10-J60", 23, weight="bold")
    panel_label(svg, 55, 95, "a")
    aotv = {m: [results[s][m]["AOTV"] for s in SCALES] for m in METHODS}
    draw_grouped_bars(svg, 95, 125, 470, 260, SCALES, aotv, "AOTV (log scale)", 80, log=True)
    panel_label(svg, 640, 95, "b")
    arpd = {m: [results[s][m]["ARPD"] for s in SCALES] for m in METHODS}
    draw_grouped_bars(svg, 680, 125, 445, 260, SCALES, arpd, "ARPD (%)", 760)

    panel_label(svg, 55, 455, "c")
    svg.text(95, 455, "RQL improvement relative to RH", 16, weight="bold")
    x, y, w, h = 95, 485, 470, 180
    improvements = [(results[s]["RH"]["AOTV"] - results[s]["RQL"]["AOTV"]) / results[s]["RH"]["AOTV"] * 100 for s in SCALES]
    draw_axes(svg, x, y, w, h, 0, 20, y_label="AOTV reduction (%)")
    bar_w = w / len(SCALES) * 0.55
    for i, (scale, val) in enumerate(zip(SCALES, improvements)):
        bx = x + (i + 0.25) * w / len(SCALES)
        by = y + h - val / 20 * h
        svg.rect(bx, by, bar_w, y + h - by, fill="#0F4D92", stroke="none")
        svg.text(bx + bar_w / 2, by - 7, f"{val:.1f}%", 12, anchor="middle")
        svg.text(bx + bar_w / 2, y + h + 22, scale, 12, anchor="middle")

    panel_label(svg, 640, 455, "d")
    svg.text(680, 455, "Method legend", 16, weight="bold")
    lx, ly = 690, 500
    for i, method in enumerate(METHODS):
        yy = ly + i * 30
        svg.rect(lx, yy - 13, 24, 14, fill=METHOD_COLORS[method], stroke="none")
        svg.text(lx + 36, yy, method, 14)
    svg.text(680, 690, "AOTV is the average total weighted delay penalty. ARPD is computed against the best result obtained on each instance.", 13, color="#4D4D4D")
    svg.save(FIG_DIR / "fig2_solution_quality.svg")


def figure_3_runtime(results: dict[str, dict[str, dict[str, float]]]) -> None:
    svg = Svg(1200, 620)
    svg.text(55, 50, "Fig. 3 | Solution quality comes with a clear training-time cost", 23, weight="bold")
    panel_label(svg, 55, 95, "a")
    times = {m: [results[s][m]["Time"] for s in SCALES] for m in METHODS}
    draw_grouped_bars(svg, 100, 125, 470, 310, SCALES, times, "Total time (s, log scale)", 1500, log=True)
    svg.text(100, 470, "The log axis keeps heuristic and learning methods visible in one panel.", 13, color="#4D4D4D")

    panel_label(svg, 665, 95, "b")
    svg.text(705, 95, "Quality-time frontier", 16, weight="bold")
    x, y, w, h = 705, 125, 380, 310
    all_time = [results[s][m]["Time"] for s in SCALES for m in METHODS]
    all_aotv = [results[s][m]["AOTV"] for s in SCALES for m in METHODS]
    xmin, xmax = math.log10(min(all_time) * 0.7), math.log10(max(all_time) * 1.3)
    ymin, ymax = 0, max(all_aotv) * 1.08
    svg.line(x, y + h, x + w, y + h, "#272727", 1.1)
    svg.line(x, y, x, y + h, "#272727", 1.1)
    for tv in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        lv = math.log10(tv)
        if xmin <= lv <= xmax:
            xx = x + (lv - xmin) / (xmax - xmin) * w
            svg.line(xx, y + h, xx, y + h + 5, "#272727", 0.8)
            svg.text(xx, y + h + 21, f"{tv:g}", 11, anchor="middle")
    for i in range(5):
        val = ymin + (ymax - ymin) * i / 4
        yy = y + h - (val - ymin) / (ymax - ymin) * h
        svg.line(x, yy, x + w, yy, "#E8E8E8", 0.8)
        svg.text(x - 8, yy + 4, f"{val:.0f}", 11, anchor="end")
    for scale in SCALES:
        for method in METHODS:
            xx = x + (math.log10(results[scale][method]["Time"]) - xmin) / (xmax - xmin) * w
            yy = y + h - (results[scale][method]["AOTV"] - ymin) / (ymax - ymin) * h
            r = 5.5 if method == "RQL" else 3.8
            svg.circle(xx, yy, r, METHOD_COLORS[method], stroke="#FFFFFF", width=0.8)
    svg.text(x + w / 2, y + h + 45, "Total time (s, log scale)", 13, anchor="middle")
    svg.text(x - 55, y + h / 2, "AOTV", 13, anchor="middle", rotate=-90)
    svg.text(705, 495, "RQL occupies the low-penalty but high-cost region, consistent with rollout-assisted training.", 13, color="#4D4D4D")
    svg.save(FIG_DIR / "fig3_runtime_tradeoff.svg")


def figure_4_convergence(reward_rows, curve_rows) -> None:
    svg = Svg(1280, 760)
    svg.text(55, 50, "Fig. 4 | RQL training exhibits scale-dependent convergence behaviour", 23, weight="bold")
    reward_by_scale: dict[str, list[dict[str, float | str]]] = {s: [] for s in SCALES}
    curve_by_scale: dict[str, list[dict[str, float | str]]] = {s: [] for s in SCALES}
    for row in reward_rows:
        if row.get("scale") in reward_by_scale:
            reward_by_scale[str(row["scale"])].append(row)
    for row in curve_rows:
        if row.get("scale") in curve_by_scale:
            curve_by_scale[str(row["scale"])].append(row)

    panel_label(svg, 55, 95, "a")
    draw_line_chart(svg, 105, 130, 470, 230, reward_by_scale, "avg_smoothed_reward", "Smoothed reward", "Episode reward")
    panel_label(svg, 665, 95, "b")
    draw_line_chart(svg, 715, 130, 470, 230, curve_by_scale, "avg_best_so_far_penalty", "Best penalty", "Best decoded penalty")
    panel_label(svg, 55, 455, "c")
    draw_line_chart(svg, 105, 490, 470, 210, curve_by_scale, "avg_log_max_q_diff", "log10(max Q diff)", "Q-table change")
    svg.text(715, 500, "Interpretation", 17, weight="bold")
    notes = [
        "J10 and J20 approach stable rewards and Q-table changes near 10^-3.",
        "J30 improves but retains visible variance across cases and seeds.",
        "J60 improves in reward but does not meet the current Q-difference threshold.",
        "Checkpoint selection is therefore essential for larger-scale decoding.",
    ]
    for i, note in enumerate(notes):
        svg.circle(725, 535 + i * 38, 4, "#0F4D92")
        svg.text(742, 540 + i * 38, note, 14, color="#4D4D4D")
    svg.save(FIG_DIR / "fig4_convergence.svg")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = load_main_results()
    reward_rows, curve_rows = load_convergence()
    write_source_data(results, reward_rows, curve_rows)
    figure_1_framework()
    figure_2_quality(results)
    figure_3_runtime(results)
    figure_4_convergence(reward_rows, curve_rows)
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote source data to {DATA_DIR}")


if __name__ == "__main__":
    main()
