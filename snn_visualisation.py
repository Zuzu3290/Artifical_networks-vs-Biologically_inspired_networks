"""
visualise_results_separate_figures.py

Reads the JSON result files produced by the three baseline studies and
exports each requested visualization as a separate, presentation-ready image.

Expected file locations, relative to this script:
    static_frame_based/results/static_results.json
    event_vision_baseline/results/event_results.json
    temporal_baseline/results/temporal_results.json

Output directory:
    figures_separate/

Generated outputs:
    Figure 1: Accuracy & F1
    Figure 2: Latency
    Figure 3: Training Time
    Figure 4: Energy
    Figure 5: GPU Power
    Figure 6: GPU Memory
    Figure 7: Spike Rate
    Figure 8: Confusion Matrix, one image per model
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ── palette ───────────────────────────────────────────────────────
BG    = "#0d1117"
CL    = "#161b22"
C_ANN = "#4a9eff"
C_SNN = "#22c55e"
C_TXT = "#e6edf3"
C_DIM = "#8b949e"
C_ACC = "#f97316"
C_F1  = "#a78bfa"
C_LAT = "#e879f9"
C_EN  = "#fb923c"
C_PWR = "#facc15"
C_SPK = "#22d3ee"

ANN_MODELS = {"MLP", "CNN", "LSTM", "GRU", "TCN", "EventCNN"}
SNN_MODELS = {"SNN_MLP", "SNN_CNN", "SNN_RNN", "SNN_TCN", "EventSNN"}

# ── paths ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
OUT_DIR = ROOT / "figures_separate"

PATHS = {
    "static":   ROOT / "static_frame_based"   / "results" / "static_results.json",
    "event":    ROOT / "event_vision_baseline" / "results" / "event_results.json",
    "temporal": ROOT / "temporal_baseline"     / "results" / "temporal_results.json",
}

STUDY_TITLES = {
    "static": "Static Frame Baseline",
    "event": "Event Vision Baseline",
    "temporal": "Temporal Baseline",
}


# ══════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════
def load_json(path: Path) -> dict[str, dict[str, Any]] | None:
    if not path.exists():
        print(f"[WARN] not found: {path}  →  using synthetic demo data")
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rng_confusion(n_classes: int = 10, samples_per_class: int = 100) -> list[list[int]]:
    rng = np.random.default_rng(7)
    return rng.multinomial(samples_per_class, [1 / n_classes] * n_classes, n_classes).tolist()


def demo_static() -> dict[str, dict[str, Any]]:
    return {
        "MLP": {
            "accuracy": 0.961, "f1_macro": 0.960,
            "latency_ms_per_sample": 0.18, "train_time_s": 28.0,
            "gpu_mem_mb": 210, "gpu_metrics": {
                "energy_joules": 14.2,
                "power_draw_w_mean": 52.0,
                "torch_alloc_mb_peak": 210,
            },
            "confusion_matrix": rng_confusion(),
        },
        "CNN": {
            "accuracy": 0.982, "f1_macro": 0.981,
            "latency_ms_per_sample": 0.22, "train_time_s": 35.0,
            "gpu_mem_mb": 280, "gpu_metrics": {
                "energy_joules": 18.1,
                "power_draw_w_mean": 55.0,
                "torch_alloc_mb_peak": 280,
            },
            "confusion_matrix": rng_confusion(),
        },
        "SNN_MLP": {
            "accuracy": 0.935, "f1_macro": 0.933,
            "latency_ms_per_sample": 0.55, "train_time_s": 72.0,
            "gpu_mem_mb": 340, "spike_rate": 0.062, "gpu_metrics": {
                "energy_joules": 38.4,
                "power_draw_w_mean": 57.0,
                "torch_alloc_mb_peak": 340,
            },
            "confusion_matrix": rng_confusion(),
        },
        "SNN_CNN": {
            "accuracy": 0.958, "f1_macro": 0.956,
            "latency_ms_per_sample": 0.68, "train_time_s": 90.0,
            "gpu_mem_mb": 410, "spike_rate": 0.041, "gpu_metrics": {
                "energy_joules": 48.2,
                "power_draw_w_mean": 58.0,
                "torch_alloc_mb_peak": 410,
            },
            "confusion_matrix": rng_confusion(),
        },
    }


def demo_event() -> dict[str, dict[str, Any]]:
    return {
        "EventCNN": {
            "accuracy": 0.874, "f1_macro": 0.871,
            "latency_ms_per_sample": 0.31, "train_time_s": 44.0,
            "gpu_mem_mb": 320, "gpu_metrics": {
                "energy_joules": 22.6,
                "power_draw_w_mean": 54.0,
                "torch_alloc_mb_peak": 320,
            },
            "confusion_matrix": rng_confusion(),
        },
        "EventSNN": {
            "accuracy": 0.856, "f1_macro": 0.852,
            "latency_ms_per_sample": 0.78, "train_time_s": 98.0,
            "gpu_mem_mb": 490, "spike_rate": 0.038, "gpu_metrics": {
                "energy_joules": 52.4,
                "power_draw_w_mean": 58.0,
                "torch_alloc_mb_peak": 490,
            },
            "confusion_matrix": rng_confusion(),
        },
    }


def demo_temporal() -> dict[str, dict[str, Any]]:
    return {
        "LSTM": {
            "accuracy": 0.921, "f1_macro": 0.919,
            "latency_ms_per_sample": 0.24, "train_time_s": 38.0,
            "gpu_mem_mb": 180, "gpu_metrics": {
                "energy_joules": 19.8,
                "power_draw_w_mean": 53.0,
                "torch_alloc_mb_peak": 180,
            },
            "confusion_matrix": rng_confusion(),
        },
        "GRU": {
            "accuracy": 0.914, "f1_macro": 0.912,
            "latency_ms_per_sample": 0.20, "train_time_s": 33.0,
            "gpu_mem_mb": 162, "gpu_metrics": {
                "energy_joules": 17.2,
                "power_draw_w_mean": 52.0,
                "torch_alloc_mb_peak": 162,
            },
            "confusion_matrix": rng_confusion(),
        },
        "TCN": {
            "accuracy": 0.908, "f1_macro": 0.905,
            "latency_ms_per_sample": 0.14, "train_time_s": 29.0,
            "gpu_mem_mb": 145, "gpu_metrics": {
                "energy_joules": 15.0,
                "power_draw_w_mean": 51.0,
                "torch_alloc_mb_peak": 145,
            },
            "confusion_matrix": rng_confusion(),
        },
        "SNN_RNN": {
            "accuracy": 0.887, "f1_macro": 0.884,
            "latency_ms_per_sample": 0.61, "train_time_s": 84.0,
            "gpu_mem_mb": 380, "spike_rate": 0.071, "gpu_metrics": {
                "energy_joules": 44.2,
                "power_draw_w_mean": 57.0,
                "torch_alloc_mb_peak": 380,
            },
            "confusion_matrix": rng_confusion(),
        },
        "SNN_TCN": {
            "accuracy": 0.879, "f1_macro": 0.876,
            "latency_ms_per_sample": 0.48, "train_time_s": 71.0,
            "gpu_mem_mb": 295, "spike_rate": 0.055, "gpu_metrics": {
                "energy_joules": 37.8,
                "power_draw_w_mean": 56.0,
                "torch_alloc_mb_peak": 295,
            },
            "confusion_matrix": rng_confusion(),
        },
    }


def load_all() -> list[dict[str, Any]]:
    demos = {
        "static": demo_static,
        "event": demo_event,
        "temporal": demo_temporal,
    }

    records: list[dict[str, Any]] = []
    for study_key in ["static", "event", "temporal"]:
        data = load_json(PATHS[study_key]) or demos[study_key]()
        for model_name, metrics in data.items():
            records.append({
                "study_key": study_key,
                "study_title": STUDY_TITLES[study_key],
                "model": model_name,
                "metrics": metrics,
            })
    return records


# ══════════════════════════════════════════════════════════════════
# Plot helpers
# ══════════════════════════════════════════════════════════════════
def model_color(model_name: str) -> str:
    return C_ANN if model_name in ANN_MODELS else C_SNN


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_").lower()


def gpu_val(metrics: dict[str, Any], key: str, default: float | None = None) -> float | None:
    gpu_metrics = metrics.get("gpu_metrics") or {}
    value = gpu_metrics.get(key, default)
    return None if value is None else float(value)


def metric_from_path(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def gpu_metric_getter(key: str, fallback_key: str | None = None) -> Callable[[dict[str, Any]], float | None]:
    def getter(metrics: dict[str, Any]) -> float | None:
        fallback = metrics.get(fallback_key) if fallback_key else None
        return gpu_val(metrics, key, fallback)
    return getter


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(CL)
    for spine in ax.spines.values():
        spine.set_color(C_DIM)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=C_DIM, labelsize=9)
    ax.xaxis.label.set_color(C_DIM)
    ax.yaxis.label.set_color(C_DIM)
    ax.title.set_color(C_TXT)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)


def save(fig: plt.Figure, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


def x_labels(records: list[dict[str, Any]]) -> list[str]:
    return [f"{r['model']}\n{r['study_key']}" for r in records]


def add_study_separators(ax: plt.Axes, records: list[dict[str, Any]], top_y: float) -> None:
    previous = records[0]["study_key"]
    for idx, record in enumerate(records[1:], start=1):
        current = record["study_key"]
        if current != previous:
            ax.axvline(idx - 0.5, color=C_DIM, linewidth=0.9, linestyle=":", alpha=0.65)
            previous = current

    seen: set[str] = set()
    for idx, record in enumerate(records):
        study = record["study_key"]
        if study not in seen:
            indices = [i for i, r in enumerate(records) if r["study_key"] == study]
            center = sum(indices) / len(indices)
            ax.text(center, top_y, STUDY_TITLES[study], color=C_DIM, fontsize=8,
                    ha="center", va="bottom")
            seen.add(study)


def annotate_bars(ax: plt.Axes, bars, values: list[float], formatter: Callable[[float], str]) -> None:
    _, y_max = ax.get_ylim()
    offset = y_max * 0.018
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            formatter(value),
            ha="center",
            va="bottom",
            color=C_TXT,
            fontsize=8,
            fontweight="bold",
        )


def single_metric_figure(
    records: list[dict[str, Any]],
    title: str,
    ylabel: str,
    getter: Callable[[dict[str, Any]], float | None],
    filename: str,
    value_color: str,
    formatter: Callable[[float], str] = lambda v: f"{v:.3f}",
) -> None:
    rows = []
    for record in records:
        value = getter(record["metrics"])
        if value is not None:
            rows.append((record, value))

    if not rows:
        print(f"[SKIP] {title}: no values found")
        return

    plot_records = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [model_color(record["model"]) for record in plot_records]

    width = max(11.5, 1.05 * len(values) + 2)
    fig, ax = plt.subplots(figsize=(width, 6.4), facecolor=BG)
    fig.suptitle(title, color=C_TXT, fontsize=16, fontweight="bold", y=0.98)

    bars = ax.bar(np.arange(len(values)), values, color=colors, alpha=0.90,
                  edgecolor="white", linewidth=0.45)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(x_labels(plot_records), rotation=35, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("Separate frame", fontsize=10, color=C_DIM, pad=8)

    y_max = max(values) * 1.32 if max(values) > 0 else 1
    ax.set_ylim(0, y_max)
    style_axis(ax)
    annotate_bars(ax, bars, values, formatter)
    add_study_separators(ax, plot_records, y_max * 0.96)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_ANN, label="ANN / non-spiking"),
        plt.Rectangle((0, 0), 1, 1, color=C_SNN, label="SNN"),
    ]
    ax.legend(handles=legend_handles, facecolor=BG, edgecolor=C_DIM,
              labelcolor=C_TXT, fontsize=9, loc="upper right")

    save(fig, filename)


def accuracy_f1_figure(records: list[dict[str, Any]]) -> None:
    plot_records = [
        record for record in records
        if record["metrics"].get("accuracy") is not None and record["metrics"].get("f1_macro") is not None
    ]
    if not plot_records:
        print("[SKIP] Figure 1: no accuracy/F1 values found")
        return

    labels = x_labels(plot_records)
    accuracy = [float(record["metrics"]["accuracy"]) for record in plot_records]
    f1 = [float(record["metrics"]["f1_macro"]) for record in plot_records]
    x = np.arange(len(plot_records))
    width = 0.36

    fig_width = max(12, 1.1 * len(plot_records) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, 6.6), facecolor=BG)
    fig.suptitle("Figure 1 — Accuracy & F1", color=C_TXT, fontsize=16,
                 fontweight="bold", y=0.98)

    bars_acc = ax.bar(x - width / 2, accuracy, width=width, color=C_ACC,
                      alpha=0.92, edgecolor="white", linewidth=0.45,
                      label="Accuracy")
    bars_f1 = ax.bar(x + width / 2, f1, width=width, color=C_F1,
                     alpha=0.92, edgecolor="white", linewidth=0.45,
                     label="F1 Macro")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.17)
    style_axis(ax)
    annotate_bars(ax, bars_acc, accuracy, lambda v: f"{v * 100:.1f}%")
    annotate_bars(ax, bars_f1, f1, lambda v: f"{v * 100:.1f}%")
    add_study_separators(ax, plot_records, 1.11)
    ax.legend(facecolor=BG, edgecolor=C_DIM, labelcolor=C_TXT, fontsize=10,
              loc="lower right")

    save(fig, "figure_01_accuracy_f1.png")


def spike_rate_figure(records: list[dict[str, Any]]) -> None:
    single_metric_figure(
        records=records,
        title="Figure 7 — Spike Rate",
        ylabel="Spike rate",
        getter=lambda m: metric_from_path(m, "spike_rate"),
        filename="figure_07_spike_rate.png",
        value_color=C_SPK,
        formatter=lambda v: f"{v:.3f}",
    )


def confusion_matrix_figures(records: list[dict[str, Any]]) -> None:
    saved_any = False
    cmap = LinearSegmentedColormap.from_list("confusion", [BG, C_SNN])

    for record in records:
        model = record["model"]
        study = record["study_key"]
        matrix = record["metrics"].get("confusion_matrix")
        if matrix is None:
            continue

        cm = np.asarray(matrix)
        if cm.ndim != 2:
            print(f"[SKIP] confusion matrix for {model}: expected 2D matrix")
            continue

        fig_size = max(6.5, min(11, 0.62 * max(cm.shape) + 4))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), facecolor=BG)
        fig.suptitle(
            f"Figure 8 — Confusion Matrix: {model} ({STUDY_TITLES[study]})",
            color=C_TXT,
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        im = ax.imshow(cm, cmap=cmap, aspect="equal")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.tick_params(labelsize=8)

        # Keep annotations readable for small/medium class counts.
        if cm.size <= 144:
            threshold = cm.max() * 0.55 if cm.max() > 0 else 0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    color = BG if cm[i, j] > threshold else C_TXT
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color=color, fontsize=7)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, colors=C_DIM)
        style_axis(ax)

        filename = f"figure_08_confusion_matrix_{safe_name(study)}_{safe_name(model)}.png"
        save(fig, filename)
        saved_any = True

    if not saved_any:
        print("[SKIP] Figure 8: no confusion matrices found")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main() -> None:
    records = load_all()

    # Figure 1: one standalone frame.
    accuracy_f1_figure(records)

    # Figures 2-6: one standalone frame per metric.
    single_metric_figure(
        records=records,
        title="Figure 2 — Latency",
        ylabel="ms / sample",
        getter=lambda m: metric_from_path(m, "latency_ms_per_sample"),
        filename="figure_02_latency.png",
        value_color=C_LAT,
        formatter=lambda v: f"{v:.3f}",
    )

    single_metric_figure(
        records=records,
        title="Figure 3 — Training Time",
        ylabel="seconds",
        getter=lambda m: metric_from_path(m, "train_time_s"),
        filename="figure_03_training_time.png",
        value_color=C_ACC,
        formatter=lambda v: f"{v:.1f}",
    )

    single_metric_figure(
        records=records,
        title="Figure 4 — Energy",
        ylabel="Joules",
        getter=gpu_metric_getter("energy_joules"),
        filename="figure_04_energy.png",
        value_color=C_EN,
        formatter=lambda v: f"{v:.1f}",
    )

    single_metric_figure(
        records=records,
        title="Figure 5 — GPU Power",
        ylabel="Watts",
        getter=gpu_metric_getter("power_draw_w_mean"),
        filename="figure_05_gpu_power.png",
        value_color=C_PWR,
        formatter=lambda v: f"{v:.1f}",
    )

    single_metric_figure(
        records=records,
        title="Figure 6 — GPU Memory",
        ylabel="MiB, PyTorch peak allocation",
        getter=gpu_metric_getter("torch_alloc_mb_peak", fallback_key="gpu_mem_mb"),
        filename="figure_06_gpu_memory.png",
        value_color=C_ANN,
        formatter=lambda v: f"{v:.0f}",
    )

    # Figure 7: SNN-only spike-rate frame.
    spike_rate_figure(records)

    # Figure 8: one confusion-matrix frame per model.
    confusion_matrix_figures(records)

    print(f"\nDone. Figures written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
