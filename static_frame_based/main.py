"""
static_frame_based/main.py

Main runner for the Static / Frame-Based baseline study.

Orchestrates:
  1. conventional_nn.run_conventional_baselines()  → MLP, CNN
  2. snn_model.run_snn_baselines()                 → SNN_MLP, SNN_CNN

Produces:
  • Console summary table
  • results/static_results.json   – raw metrics for downstream use
  • results/static_report.txt     – human-readable statistical comparison

Comparison pairs (document specification):
  MLP  vs  SNN_MLP  – simple reference baseline
  CNN  vs  SNN_CNN  – standard visual processing vs spiking counterpart
"""

import json
import os
import time
from pathlib import Path

import torch

from conventional_nn import run_conventional_baselines
from snn_model import run_snn_baselines
from gpu_profiler import format_gpu_metrics_block


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CFG = dict(
    epochs     = 5,
    batch_size = 64,
    lr         = 1e-3,
    timesteps  = 25,       # SNN simulation steps
    data_root  = "./data",
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────

def delta(conventional: float, snn: float) -> str:
    """Return signed percentage difference (SNN − conventional)."""
    if conventional == 0:
        return "N/A"
    diff = (snn - conventional) / abs(conventional) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}%"


def format_row(label: str, conv_val, snn_val, higher_better: bool = True) -> str:
    if isinstance(conv_val, float):
        d = delta(conv_val, snn_val)
        arrow = ""
        if d != "N/A":
            numeric = float(d.replace("%", "").replace("+", ""))
            arrow = " ✓" if (numeric > 0) == higher_better else " ✗"
        return f"  {label:<35} {conv_val:>10.4f}  {snn_val:>10.4f}  {d:>10}{arrow}"
    return f"  {label:<35} {str(conv_val):>10}  {str(snn_val):>10}"


def compare_pair(
    conv_name: str,
    snn_name: str,
    conv_metrics: dict,
    snn_metrics: dict,
) -> str:
    """Build a formatted comparison block for one pair."""
    lines = [
        f"\n{'═' * 75}",
        f"  Comparison: {conv_name}  vs  {snn_name}",
        f"{'─' * 75}",
        f"  {'Metric':<35} {'Conventional':>10}  {'SNN':>10}  {'Delta':>10}",
        f"{'─' * 75}",
        format_row("Accuracy",                conv_metrics["accuracy"],               snn_metrics["accuracy"],               higher_better=True),
        format_row("F1 Macro",                conv_metrics["f1_macro"],               snn_metrics["f1_macro"],               higher_better=True),
        format_row("Latency ms/sample",       conv_metrics["latency_ms_per_sample"],  snn_metrics["latency_ms_per_sample"],  higher_better=False),
        format_row("Training time (s)",       conv_metrics["train_time_s"],           snn_metrics["train_time_s"],           higher_better=False),
        format_row("GPU memory (MB)",         conv_metrics["gpu_mem_mb"],             snn_metrics["gpu_mem_mb"],             higher_better=False),
        format_row("Parameters",              float(conv_metrics["num_params"]),      float(snn_metrics["num_params"]),      higher_better=False),
    ]

    if "spike_rate" in snn_metrics:
        lines.append(f"  {'Spike rate (SNN only)':<35} {'N/A':>10}  {snn_metrics['spike_rate']:>10.4f}")
        lines.append(f"  {'Timesteps (SNN only)':<35} {'N/A':>10}  {snn_metrics['timesteps']:>10}")

    lines += [
        f"{'─' * 75}",
        f"  ✓ = SNN improvement  ✗ = SNN regression  (direction depends on metric)",
        f"{'═' * 75}",
    ]
    return "\n".join(lines)


def diagnosis(conv_m: dict, snn_m: dict, pair: str) -> str:
    """Produce a short qualitative diagnosis."""
    acc_gap = snn_m["accuracy"] - conv_m["accuracy"]
    lat_ratio = snn_m["latency_ms_per_sample"] / max(conv_m["latency_ms_per_sample"], 1e-9)

    lines = [f"\n  Diagnosis [{pair}]:"]
    if abs(acc_gap) < 0.02:
        lines.append("  • Accuracy within ±2 pp → SNN competitive on static data.")
    elif acc_gap < 0:
        lines.append(f"  • SNN accuracy lower by {abs(acc_gap):.2%} — expected on static frames (rate encoding overhead).")
    else:
        lines.append(f"  • SNN accuracy higher by {acc_gap:.2%} — unexpectedly strong.")

    if lat_ratio < 1.5:
        lines.append("  • Latency within 1.5× — acceptable for non-real-time use.")
    else:
        lines.append(f"  • SNN latency {lat_ratio:.1f}× slower — GPU simulation overhead dominates.")

    if "spike_rate" in snn_m:
        sr = snn_m["spike_rate"]
        if sr < 0.1:
            lines.append(f"  • Spike rate {sr:.3f} → high sparsity, favourable for neuromorphic HW.")
        else:
            lines.append(f"  • Spike rate {sr:.3f} → moderate sparsity; benefit depends on HW target.")

    lines.append(
        "  • Note: static MNIST is NOT the intended strength of SNNs. "
        "These results are a functional reference only."
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print(f"  Static / Frame-Based Baseline Study")
    print(f"  Device : {device}")
    print(f"  Config : {CFG}")
    print(f"{'═'*60}\n")

    # ── Run baselines ──────────────────────────
    print("▶ Training conventional baselines …\n")
    t0 = time.perf_counter()
    conv_results = run_conventional_baselines(
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        lr=CFG["lr"],
        device=device,
        data_root=CFG["data_root"],
    )

    print("▶ Training SNN baselines …\n")
    snn_results = run_snn_baselines(
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        lr=CFG["lr"],
        timesteps=CFG["timesteps"],
        device=device,
        data_root=CFG["data_root"],
    )
    total_time = time.perf_counter() - t0

    # ── Serialise (confusion matrices → lists) ─
    all_results = {**conv_results, **snn_results}
    serialisable = {}
    for k, v in all_results.items():
        serialisable[k] = {
            mk: (mv.tolist() if hasattr(mv, "tolist") else mv)
            for mk, mv in v.items()
        }

    json_path = RESULTS_DIR / "static_results.json"
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n✔ Raw results saved → {json_path}")

    # ── Statistical comparison report ──────────
    comparison_pairs = [
        ("MLP", "SNN_MLP"),
        ("CNN", "SNN_CNN"),
    ]

    report_lines = [
        "=" * 75,
        "  STATIC / FRAME-BASED BASELINE — STATISTICAL COMPARISON REPORT",
        "=" * 75,
        f"\n  Study objective:",
        "  Evaluate whether SNNs are sufficiently promising on static data",
        "  to justify deeper GPU-based investigation.",
        "\n  Note: static images are NOT the natural strength of SNNs.",
        "  Rate encoding introduces simulation overhead; results are a",
        "  functional reference point, not the primary project argument.",
        f"\n  Total wall-clock time: {total_time:.1f} s",
        f"  Device: {device}",
        f"  Timesteps (SNN): {CFG['timesteps']}",
    ]

    for conv_name, snn_name in comparison_pairs:
        block = compare_pair(
            conv_name, snn_name,
            all_results[conv_name], all_results[snn_name],
        )
        diag  = diagnosis(all_results[conv_name], all_results[snn_name], f"{conv_name} vs {snn_name}")
        report_lines.append(block)
        report_lines.append(diag)

        # ── GPU hardware metrics per model ─────────
        for model_name in (conv_name, snn_name):
            gm = all_results[model_name].get("gpu_metrics", {})
            if gm:
                report_lines.append(format_gpu_metrics_block(gm, label=model_name))
            flops = all_results[model_name].get("flops", {})
            if flops.get("profiler_available"):
                report_lines.append(
                    f"  FLOPs [{model_name}]: {flops['flops_total']:,}  "
                    f"MACs: {flops['macs_total']:,}"
                )

    report_lines += [
        "\n" + "=" * 75,
        "  OVERALL VERDICT",
        "=" * 75,
        "  SNNs on static MNIST provide a functional sanity check.",
        "  The primary evaluation strength lies in the Event-Vision and",
        "  Temporal baselines where temporal spike dynamics are meaningful.",
        "=" * 75,
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = RESULTS_DIR / "static_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n✔ Report saved → {report_path}")


if __name__ == "__main__":
    main()
