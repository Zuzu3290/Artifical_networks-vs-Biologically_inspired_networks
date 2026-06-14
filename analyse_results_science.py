"""
analyse_json_results.py

Purpose:
    Read JSON result files directly.
    Do not depend on existing CSV files.
    Always export the raw JSON contents first, then extract likely model metrics.

Outputs:
    analysis_results/
        raw_json_values.csv
        metric_candidates.csv
        model_metrics_wide.csv
        file_inventory.csv

Run:
    python analyse_json_results.py

Optional:
    python analyse_json_results.py --json path/to/results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


OUT_DIR_NAME = "analysis_results"

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "analysis_results",
    "figures_separate",
}


KNOWN_MODELS = {
    "MLP",
    "CNN",
    "SNN_MLP",
    "SNN_CNN",
    "EventCNN",
    "EventSNN",
    "LSTM",
    "GRU",
    "TCN",
    "SNN_RNN",
    "SNN_TCN",
}


METRIC_NAME_MAP = {
    "accuracy": "accuracy",
    "acc": "accuracy",
    "test_accuracy": "accuracy",
    "test_acc": "accuracy",
    "val_accuracy": "accuracy",

    "f1": "f1_macro",
    "f1_macro": "f1_macro",
    "macro_f1": "f1_macro",
    "f1_score": "f1_macro",

    "latency": "latency_ms_per_sample",
    "latency_ms": "latency_ms_per_sample",
    "inference_latency_ms": "latency_ms_per_sample",
    "latency_ms_per_sample": "latency_ms_per_sample",

    "train_time_s": "train_time_s",
    "training_time_s": "train_time_s",
    "train_seconds": "train_time_s",
    "training_seconds": "train_time_s",

    "gpu_mem_mb": "memory_mb",
    "gpu_memory_mb": "memory_mb",
    "memory_mb": "memory_mb",
    "torch_alloc_mb_peak": "memory_mb",
    "peak_memory_mb": "memory_mb",

    "energy": "energy_joules",
    "energy_j": "energy_joules",
    "energy_joules": "energy_joules",

    "power": "power_draw_w_mean",
    "power_w": "power_draw_w_mean",
    "power_draw_w_mean": "power_draw_w_mean",

    "spike_rate": "spike_rate",
    "real_time_factor": "real_time_factor",
}


MAIN_METRICS = [
    "accuracy",
    "f1_macro",
    "latency_ms_per_sample",
    "train_time_s",
    "energy_joules",
    "power_draw_w_mean",
    "memory_mb",
    "spike_rate",
    "real_time_factor",
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            clean = {}
            for col in columns:
                value = row.get(col, "")
                if value is None:
                    value = ""
                elif isinstance(value, float):
                    if math.isnan(value) or math.isinf(value):
                        value = ""
                    else:
                        value = f"{value:.8g}"
                clean[col] = value
            writer.writerow(clean)


def load_json(path: Path) -> tuple[Any | None, str]:
    for enc in ("utf-8", "utf-8-sig"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f), "loaded"
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            return None, f"error: {exc}"

    return None, "error: could not decode"


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def to_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        x = float(value)
    except Exception:
        return None

    if math.isnan(x) or math.isinf(x):
        return None

    return x


def flatten_json(
    obj: Any,
    source_file: Path,
    rows: list[dict[str, Any]],
    path: str = "$",
) -> None:
    if isinstance(obj, dict):
        if not obj:
            rows.append({
                "source_file": str(source_file),
                "json_path": path,
                "key": "",
                "value": "",
                "value_type": "empty_dict",
            })
            return

        for key, value in obj.items():
            next_path = f"{path}.{key}"
            if is_scalar(value):
                rows.append({
                    "source_file": str(source_file),
                    "json_path": next_path,
                    "key": str(key),
                    "value": value,
                    "value_type": value_type(value),
                })
            else:
                flatten_json(value, source_file, rows, next_path)

    elif isinstance(obj, list):
        if not obj:
            rows.append({
                "source_file": str(source_file),
                "json_path": path,
                "key": "",
                "value": "",
                "value_type": "empty_list",
            })
            return

        for i, value in enumerate(obj):
            next_path = f"{path}[{i}]"
            if is_scalar(value):
                rows.append({
                    "source_file": str(source_file),
                    "json_path": next_path,
                    "key": f"[{i}]",
                    "value": value,
                    "value_type": value_type(value),
                })
            else:
                flatten_json(value, source_file, rows, next_path)

    else:
        rows.append({
            "source_file": str(source_file),
            "json_path": path,
            "key": "",
            "value": obj,
            "value_type": value_type(obj),
        })


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def find_json_files(root: Path) -> list[Path]:
    files = []

    for path in root.rglob("*.json"):
        if path.is_file() and not should_skip(path):
            files.append(path)

    return sorted(files)


def normalize_metric_key(key: str, json_path: str) -> str | None:
    key_clean = key.lower().strip()

    if key_clean in METRIC_NAME_MAP:
        return METRIC_NAME_MAP[key_clean]

    path_lower = json_path.lower()

    for raw, normalized in METRIC_NAME_MAP.items():
        if raw in path_lower:
            return normalized

    return None


def guess_model_from_path(json_path: str) -> str:
    path_text = json_path.replace("[", ".").replace("]", ".")
    parts = [p for p in path_text.split(".") if p]

    for part in parts:
        clean = part.strip().strip('"').strip("'")
        if clean in KNOWN_MODELS:
            return clean

    upper_path = json_path.upper()
    for model in KNOWN_MODELS:
        if model.upper() in upper_path:
            return model

    return "UNKNOWN_MODEL"


def guess_study(source_file: str, json_path: str, model: str) -> str:
    text = f"{source_file} {json_path} {model}".lower()

    if "event" in text:
        return "Event vision"

    if "temporal" in text or model in {"LSTM", "GRU", "TCN", "SNN_RNN", "SNN_TCN"}:
        return "Temporal"

    if "static" in text or "frame" in text or model in {"MLP", "CNN", "SNN_MLP", "SNN_CNN"}:
        return "Static frame"

    return "Unknown"


def build_metric_candidates(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []

    for row in raw_rows:
        number = to_number(row.get("value"))
        if number is None:
            continue

        key = str(row.get("key", ""))
        json_path = str(row.get("json_path", ""))

        metric = normalize_metric_key(key, json_path)
        if metric is None:
            continue

        model = guess_model_from_path(json_path)
        study = guess_study(str(row["source_file"]), json_path, model)

        candidates.append({
            "study": study,
            "model": model,
            "metric": metric,
            "value": number,
            "source_file": row["source_file"],
            "json_path": json_path,
            "raw_key": key,
        })

    return candidates


def build_wide_metrics(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in candidates:
        key = (
            str(row["source_file"]),
            str(row["study"]),
            str(row["model"]),
        )

        if key not in grouped:
            grouped[key] = {
                "source_file": row["source_file"],
                "study": row["study"],
                "model": row["model"],
            }

        metric = str(row["metric"])
        value = row["value"]

        if metric not in grouped[key]:
            grouped[key][metric] = value

    rows = list(grouped.values())

    def sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(row.get("study", "")),
            str(row.get("model", "")),
            str(row.get("source_file", "")),
        )

    return sorted(rows, key=sort_key)


def clean_output(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for suffix in ("*.csv", "*.md"):
        for path in out_dir.glob(suffix):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project folder to scan. Default: current folder.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional single JSON file to analyse.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder. Default: analysis_results.",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (args.out or root / OUT_DIR_NAME).resolve()

    clean_output(out_dir)

    if args.json:
        json_files = [args.json.resolve()]
    else:
        json_files = find_json_files(root)

    inventory = []
    raw_rows = []

    for path in json_files:
        if not path.exists():
            inventory.append({
                "file": str(path),
                "status": "missing",
                "message": "file does not exist",
            })
            continue

        data, status = load_json(path)

        if data is None:
            inventory.append({
                "file": str(path),
                "status": status,
                "message": "could not load JSON",
            })
            continue

        before = len(raw_rows)
        flatten_json(data, path, raw_rows)
        after = len(raw_rows)

        inventory.append({
            "file": str(path),
            "status": "loaded",
            "message": f"raw values extracted: {after - before}",
        })

    metric_candidates = build_metric_candidates(raw_rows)
    wide_rows = build_wide_metrics(metric_candidates)

    write_csv(
        out_dir / "file_inventory.csv",
        inventory,
        ["file", "status", "message"],
    )

    write_csv(
        out_dir / "raw_json_values.csv",
        raw_rows,
        ["source_file", "json_path", "key", "value", "value_type"],
    )

    write_csv(
        out_dir / "metric_candidates.csv",
        metric_candidates,
        ["study", "model", "metric", "value", "source_file", "json_path", "raw_key"],
    )

    write_csv(
        out_dir / "model_metrics_wide.csv",
        wide_rows,
        ["source_file", "study", "model", *MAIN_METRICS],
    )

    print("JSON analysis complete.")
    print(f"Output folder: {out_dir}")
    print(f"JSON files scanned: {len(json_files)}")
    print(f"Raw JSON values written: {len(raw_rows)}")
    print(f"Metric candidates found: {len(metric_candidates)}")
    print(f"Wide model rows written: {len(wide_rows)}")

    if len(raw_rows) == 0:
        print("")
        print("No raw JSON values were written. That means the JSON files were missing, empty, or invalid.")
    elif len(metric_candidates) == 0:
        print("")
        print("Raw JSON values were written, but metric names were not recognized.")
        print("Open analysis_results/raw_json_values.csv and check the exact metric names used in your JSON.")
        print("Then add those names to METRIC_NAME_MAP near the top of this script.")


if __name__ == "__main__":
    main()