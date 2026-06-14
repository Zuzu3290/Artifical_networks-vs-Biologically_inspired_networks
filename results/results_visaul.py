from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

def find_results_dir() -> Path:
    """
    Expected structure:

        project_folder/
        ├── generate_linear_report.py
        └── reports/
            └── results/
                ├── event_results.json
                ├── static_results.json
                └── temporal_results.json

    The script also works if you run it from inside the project folder.
    """

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()

    candidates = [
        script_dir / "reports" / "results",
        cwd / "reports" / "results",
        script_dir,
        cwd,
    ]

    for candidate in candidates:
        if (
            (candidate / "event_results.json").exists()
            and (candidate / "static_results.json").exists()
            and (candidate / "temporal_results.json").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find result files. Expected:\n"
        "reports/results/event_results.json\n"
        "reports/results/static_results.json\n"
        "reports/results/temporal_results.json"
    )


RESULTS_DIR = find_results_dir()
OUTPUT_DIR = RESULTS_DIR / "linear_graph_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RESULT_FILES = {
    "Event-based": RESULTS_DIR / "event_results.json",
    "Static": RESULTS_DIR / "static_results.json",
    "Temporal": RESULTS_DIR / "temporal_results.json",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def flatten_results() -> tuple[pd.DataFrame, dict]:
    rows = []
    confusion_matrices = {}

    for group_name, file_path in RESULT_FILES.items():
        data = load_json(file_path)

        for model_name, result in data.items():
            gpu = result.get("gpu_metrics", {}) or {}
            flops = result.get("flops", {}) or {}

            row = {
                "group": group_name,
                "model": model_name,

                "accuracy": result.get("accuracy"),
                "f1_macro": result.get("f1_macro"),

                "latency_ms_per_sample": result.get("latency_ms_per_sample"),
                "train_time_s": result.get("train_time_s"),

                "gpu_mem_mb": result.get("gpu_mem_mb"),
                "num_params": result.get("num_params"),

                "energy_joules": gpu.get("energy_joules"),
                "power_draw_w_mean": gpu.get("power_draw_w_mean"),
                "power_draw_w_peak": gpu.get("power_draw_w_peak"),
                "gpu_util_pct_mean": gpu.get("gpu_util_pct_mean"),
                "temp_c_mean": gpu.get("temp_c_mean"),
                "vram_used_mb_peak": gpu.get("vram_used_mb_peak"),

                "flops_total": flops.get("flops_total"),
                "macs_total": flops.get("macs_total"),

                "spike_rate": result.get("spike_rate"),
                "timesteps": result.get("timesteps"),
                "real_time_factor": result.get("real_time_factor"),
                "input_type": result.get("input_type"),
            }

            rows.append(row)

            if "confusion_matrix" in result:
                confusion_matrices[model_name] = np.array(result["confusion_matrix"])

    df = pd.DataFrame(rows)

    group_order = {
        "Static": 1,
        "Temporal": 2,
        "Event-based": 3,
    }

    df["group_order"] = df["group"].map(group_order)
    df = df.sort_values(["group_order", "model"]).reset_index(drop=True)

    return df, confusion_matrices


df, confusion_matrices = flatten_results()
df.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)


# ============================================================
# GENERAL PLOT FORMATTING
# ============================================================

def setup_model_axis(ax, plot_df: pd.DataFrame):
    """
    Creates a clean model-based x-axis and draws vertical separators
    between Static, Temporal, and Event-based sections.
    """

    x_positions = np.arange(len(plot_df))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["model"], rotation=35, ha="right")

    previous_group = None
    group_start = 0

    for idx, group in enumerate(plot_df["group"]):
        if previous_group is None:
            previous_group = group
            group_start = idx

        elif group != previous_group:
            separator_x = idx - 0.5
            ax.axvline(separator_x, linestyle="--", linewidth=1)

            center = (group_start + idx - 1) / 2
            ax.text(
                center,
                1.02,
                previous_group,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

            previous_group = group
            group_start = idx

    if previous_group is not None:
        center = (group_start + len(plot_df) - 1) / 2
        ax.text(
            center,
            1.02,
            previous_group,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def save_plot(filename: str):
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def clean_metric_df(metric_names: list[str]) -> pd.DataFrame:
    needed = ["group", "model"] + metric_names
    return df[needed].dropna(how="all", subset=metric_names).copy()


# ============================================================
# LINE GRAPH 1: ACCURACY + F1
# ============================================================

def plot_accuracy_f1_line():
    plot_df = clean_metric_df(["accuracy", "f1_macro"])
    x = np.arange(len(plot_df))

    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    ax.plot(x, plot_df["accuracy"] * 100, marker="o", linewidth=2, label="Accuracy")
    ax.plot(x, plot_df["f1_macro"] * 100, marker="o", linewidth=2, label="Macro F1")

    setup_model_axis(ax, plot_df)

    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score [%]")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    save_plot("01_performance_accuracy_f1_line.png")


# ============================================================
# LINE GRAPH 2: LATENCY
# ============================================================

def plot_latency_line():
    plot_df = clean_metric_df(["latency_ms_per_sample"])
    x = np.arange(len(plot_df))

    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    ax.plot(
        x,
        plot_df["latency_ms_per_sample"],
        marker="o",
        linewidth=2,
        label="Latency",
    )

    setup_model_axis(ax, plot_df)

    ax.set_title("Inference Latency Comparison")
    ax.set_ylabel("Latency [ms/sample]")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    save_plot("02_latency_line_log_scale.png")


# ============================================================
# LINE GRAPH 3: TRAINING TIME
# ============================================================

def plot_training_time_line():
    plot_df = clean_metric_df(["train_time_s"])
    x = np.arange(len(plot_df))

    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    ax.plot(
        x,
        plot_df["train_time_s"],
        marker="o",
        linewidth=2,
        label="Training time",
    )

    setup_model_axis(ax, plot_df)

    ax.set_title("Training Time Comparison")
    ax.set_ylabel("Training time [s]")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    save_plot("03_training_time_line_log_scale.png")


# ============================================================
# LINE GRAPH 4: GPU ENERGY
# ============================================================

def plot_energy_line():
    plot_df = clean_metric_df(["energy_joules"])
    x = np.arange(len(plot_df))

    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    ax.plot(
        x,
        plot_df["energy_joules"],
        marker="o",
        linewidth=2,
        label="Training energy",
    )

    setup_model_axis(ax, plot_df)

    ax.set_title("GPU Energy Consumption During Training")
    ax.set_ylabel("Energy [J]")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    save_plot("04_gpu_energy_line_log_scale.png")


# ============================================================
# LINE GRAPH 5: MEMORY USAGE
# ============================================================

def plot_memory_line():
    plot_df = clean_metric_df(["gpu_mem_mb", "vram_used_mb_peak"])
    x = np.arange(len(plot_df))

    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    ax.plot(
        x,
        plot_df["gpu_mem_mb"],
        marker="o",
        linewidth=2,
        label="Torch allocated GPU memory",
    )

    if "vram_used_mb_peak" in plot_df and plot_df["vram_used_mb_peak"].notna().any():
        ax.plot(
            x,
            plot_df["vram_used_mb_peak"],
            marker="o",
            linewidth=2,
            label="Peak VRAM used",
        )

    setup_model_axis(ax, plot_df)

    ax.set_title("GPU Memory Usage Comparison")
    ax.set_ylabel("Memory [MB]")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    save_plot("05_gpu_memory_line_log_scale.png")


# ============================================================
# LINE GRAPH 6: PARAMETERS
# ============================================================

def plot_parameters_line():
    plot_df = clean_metric_df(["num_params"])
    x = np.arange(len(plot_df))

    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    ax.plot(
        x,
        plot_df["num_params"],
        marker="o",
        linewidth=2,
        label="Parameters",
    )

    setup_model_axis(ax, plot_df)

    ax.set_title("Model Size Comparison")
    ax.set_ylabel("Number of parameters")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    save_plot("06_parameters_line_log_scale.png")


# ============================================================
# LINE GRAPH 7: FLOPS / MACS
# ============================================================

def plot_flops_macs_line():
    plot_df = clean_metric_df(["flops_total", "macs_total"])
    x = np.arange(len(plot_df))

    plt.figure(figsize=(13, 6))
    ax = plt.gca()

    ax.plot(
        x,
        plot_df["flops_total"],
        marker="o",
        linewidth=2,
        label="FLOPs",
    )

    ax.plot(
        x,
        plot_df["macs_total"],
        marker="o",
        linewidth=2,
        label="MACs",
    )

    setup_model_axis(ax, plot_df)

    ax.set_title("Computational Cost Comparison")
    ax.set_ylabel("Operations")
    ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    save_plot("07_flops_macs_line_log_scale.png")


# ============================================================
# LINE GRAPH 8: SNN-SPECIFIC SPIKING METRICS
# ============================================================

def plot_spiking_metrics_line():
    plot_df = clean_metric_df(["spike_rate", "real_time_factor"])
    plot_df = plot_df[
        plot_df["spike_rate"].notna() | plot_df["real_time_factor"].notna()
    ].copy()

    if plot_df.empty:
        print("No spiking metrics found. Skipping spiking metric plot.")
        return

    x = np.arange(len(plot_df))

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    if plot_df["spike_rate"].notna().any():
        ax.plot(
            x,
            plot_df["spike_rate"],
            marker="o",
            linewidth=2,
            label="Spike rate",
        )

    if plot_df["real_time_factor"].notna().any():
        ax.plot(
            x,
            plot_df["real_time_factor"],
            marker="o",
            linewidth=2,
            label="Real-time factor",
        )

    setup_model_axis(ax, plot_df)

    ax.set_title("SNN-Specific Runtime Metrics")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    save_plot("08_snn_spiking_metrics_line.png")


# ============================================================
# CONFUSION MATRIX PLOTS
# ============================================================

def plot_confusion_matrix(model_name: str, matrix: np.ndarray):
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    image = ax.imshow(matrix)

    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))

    plt.colorbar(image, ax=ax, label="Count")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                fontsize=7,
            )

    safe_name = "".join(
        char if char.isalnum() or char in ["_", "-"] else "_"
        for char in model_name
    )

    save_plot(f"cm_{safe_name}.png")


def plot_all_confusion_matrices():
    for model_name, matrix in confusion_matrices.items():
        plot_confusion_matrix(model_name, matrix)


# ============================================================
# SIMPLE HTML REPORT
# ============================================================

def create_html_report():
    image_files = sorted(
        file.name
        for file in OUTPUT_DIR.glob("*.png")
    )

    table_html = df.drop(columns=["group_order"]).to_html(index=False)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Linear Graph Model Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 36px;
            line-height: 1.5;
        }}

        h1, h2 {{
            margin-top: 32px;
        }}

        img {{
            width: 100%;
            max-width: 1200px;
            display: block;
            margin: 20px 0 42px 0;
            border: 1px solid #ddd;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
            margin-bottom: 40px;
        }}

        th, td {{
            border: 1px solid #ccc;
            padding: 6px 8px;
            text-align: right;
        }}

        th:nth-child(1),
        th:nth-child(2),
        td:nth-child(1),
        td:nth-child(2) {{
            text-align: left;
        }}

        .note {{
            border-left: 4px solid #777;
            padding-left: 12px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>

<h1>Linear Graph Model Comparison Report</h1>

<div class="note">
This report uses connected line-graph formatting across the model list.
The x-axis is grouped into Static, Temporal, and Event-based sections.
Log scaling is used for latency, training time, energy, memory, parameters, and FLOPs because the values differ by large orders of magnitude.
</div>

<h2>Summary Metrics</h2>
{table_html}

<h2>Generated Graphs</h2>
"""

    for image_file in image_files:
        html += f"""
<h3>{image_file}</h3>
<img src="{image_file}" alt="{image_file}">
"""

    html += """
</body>
</html>
"""

    html_path = OUTPUT_DIR / "linear_graph_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Saved: {html_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Reading result files from: {RESULTS_DIR}")
    print(f"Saving output to: {OUTPUT_DIR}")

    plot_accuracy_f1_line()
    plot_latency_line()
    plot_training_time_line()
    plot_energy_line()
    plot_memory_line()
    plot_parameters_line()
    plot_flops_macs_line()
    plot_spiking_metrics_line()
    plot_all_confusion_matrices()

    create_html_report()

    print("\nDone.")
    print(f"Open this file in your browser:")
    print(OUTPUT_DIR / "linear_graph_report.html")


if __name__ == "__main__":
    main()