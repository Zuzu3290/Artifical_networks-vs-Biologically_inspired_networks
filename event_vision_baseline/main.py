"""
event_vision_baseline/main.py

Main runner for the Event-Vision Baseline study.

This is the MOST IMPORTANT comparison in the project:
  EventCNN (event frames)  vs  EventSNN (event spike streams)

Both operate on the same underlying event source.

Comparison produces:
  • Console summary with real-time feasibility verdict
  • results/event_results.json
  • results/event_report.txt
"""

import json
import time
from pathlib import Path

import torch

from conventional_nn import run_event_cnn_baseline
from snn_model import run_event_snn_baseline
from gpu_profiler import format_gpu_metrics_block


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CFG = dict(
    epochs      = 10,
    batch_size  = 64,
    lr          = 1e-3,
    timesteps   = 10,          # number of temporal bins in event window
    num_samples = 3000,
    num_classes = 10,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def delta(a: float, b: float) -> str:
    if a == 0:
        return "N/A"
    diff = (b - a) / abs(a) * 100
    return f"{'+'if diff>=0 else ''}{diff:.1f}%"


def rt_verdict(rt_factor: float) -> str:
    if rt_factor < 1.0:
        return f"✓ real-time capable  (RT factor={rt_factor:.3f} < 1.0)"
    return f"✗ NOT real-time  (RT factor={rt_factor:.3f} ≥ 1.0)"


def format_table_row(label: str, cnn_val, snn_val, higher_better: bool = True) -> str:
    if isinstance(cnn_val, float):
        d = delta(cnn_val, snn_val)
        arrow = ""
        if d not in ("N/A",):
            num = float(d.replace("%","").replace("+",""))
            arrow = " ✓" if (num > 0) == higher_better else " ✗"
        return f"  {label:<40} {cnn_val:>10.4f}  {snn_val:>10.4f}  {d:>10}{arrow}"
    return f"  {label:<40} {str(cnn_val):>10}  {str(snn_val):>10}"


def build_report(cnn_m: dict, snn_m: dict, total_time: float, device: torch.device) -> str:
    lines = [
        "=" * 80,
        "  EVENT-VISION BASELINE — STATISTICAL COMPARISON REPORT",
        "=" * 80,
        "",
        "  This is the most important comparison in the project.",
        "  Both models operate on the same underlying event-camera source.",
        "",
        f"  EventCNN  : processes accumulated event FRAMES  (2-channel ON/OFF)",
        f"  EventSNN  : processes event SPIKE STREAMS       (T×2×H×W tensors)",
        "",
        f"  Total wall-clock time : {total_time:.1f} s",
        f"  Device                : {device}",
        f"  Event timesteps (SNN) : {CFG['timesteps']}",
        f"  Event window (SNN)    : {snn_m['event_window_ms']:.1f} ms",
        "",
        "─" * 80,
        f"  {'Metric':<40} {'EventCNN':>10}  {'EventSNN':>10}  {'Delta':>10}",
        "─" * 80,
        format_table_row("Accuracy",               cnn_m["accuracy"],              snn_m["accuracy"],              higher_better=True),
        format_table_row("F1 Macro",               cnn_m["f1_macro"],              snn_m["f1_macro"],              higher_better=True),
        format_table_row("Latency ms/sample",      cnn_m["latency_ms_per_sample"], snn_m["latency_ms_per_sample"], higher_better=False),
        format_table_row("Training time (s)",      cnn_m["train_time_s"],          snn_m["train_time_s"],          higher_better=False),
        format_table_row("GPU memory (MB)",        cnn_m["gpu_mem_mb"],            snn_m["gpu_mem_mb"],            higher_better=False),
        format_table_row("Parameters",             float(cnn_m["num_params"]),     float(snn_m["num_params"]),     higher_better=False),
        f"  {'Spike rate (SNN only)':<40} {'N/A':>10}  {snn_m['spike_rate']:>10.4f}",
        f"  {'Timesteps (SNN only)':<40} {'N/A':>10}  {snn_m['timesteps']:>10}",
        f"  {'Real-time factor (SNN only)':<40} {'N/A':>10}  {snn_m['real_time_factor']:>10.4f}",
        "─" * 80,
        "  ✓ = SNN improvement   ✗ = SNN regression",
        "",
    ]

    # ── Real-time analysis ──────────────────────
    lines += [
        "=" * 80,
        "  REAL-TIME FEASIBILITY (EVENT-VISION KEY CRITERION)",
        "=" * 80,
        "",
        f"  Event window duration : {snn_m['event_window_ms']:.1f} ms",
        f"  SNN processing time   : {snn_m['latency_ms_per_sample']:.3f} ms/sample",
        "",
        f"  SNN verdict  : {rt_verdict(snn_m['real_time_factor'])}",
        "",
        "  Note: real-time factor = processing_time / event_window_duration.",
        "  Factor < 1.0 means the model keeps up with the sensor stream.",
        "",
    ]

    # ── GPU hardware metrics ────────────────────
    for label, m in [("EventCNN", cnn_m), ("EventSNN", snn_m)]:
        gm = m.get("gpu_metrics", {})
        if gm:
            lines.append(format_gpu_metrics_block(gm, label=label))
        flops = m.get("flops", {})
        if flops.get("profiler_available"):
            lines.append(
                f"  FLOPs [{label}]: {flops['flops_total']:,}  "
                f"MACs: {flops['macs_total']:,}\n"
            )

    # ── Qualitative diagnosis ───────────────────
    acc_gap  = snn_m["accuracy"] - cnn_m["accuracy"]
    lat_ratio = snn_m["latency_ms_per_sample"] / max(cnn_m["latency_ms_per_sample"], 1e-9)
    spike_rt = snn_m["spike_rate"]

    lines += [
        "=" * 80,
        "  DIAGNOSIS",
        "=" * 80,
        "",
    ]

    if abs(acc_gap) < 0.02:
        lines.append("  • Accuracy gap < 2 pp → SNN matches CNN on event classification.")
    elif acc_gap < 0:
        lines.append(f"  • SNN accuracy {abs(acc_gap):.2%} lower → accuracy trade-off for event efficiency.")
    else:
        lines.append(f"  • SNN accuracy {acc_gap:.2%} higher → event stream provides richer temporal cues.")

    if lat_ratio < 1.2:
        lines.append("  • Latency within 1.2× → SNN GPU simulation is competitive.")
    elif lat_ratio < 2.0:
        lines.append(f"  • Latency {lat_ratio:.1f}× slower → moderate GPU simulation overhead.")
    else:
        lines.append(f"  • Latency {lat_ratio:.1f}× slower → significant GPU overhead; consider kernel optimisation.")

    if spike_rt < 0.05:
        lines.append(f"  • Spike rate {spike_rt:.3f} → very sparse; high potential on neuromorphic hardware.")
    elif spike_rt < 0.15:
        lines.append(f"  • Spike rate {spike_rt:.3f} → moderate sparsity; efficiency gains depend on HW target.")
    else:
        lines.append(f"  • Spike rate {spike_rt:.3f} → relatively dense; sparsity benefit limited on GPU.")

    lines += [
        "",
        "  Project implication:",
        "  The event-vision comparison demonstrates whether the SNN's native",
        "  temporal processing offers any advantage over frame-accumulated CNN",
        "  baselines. These results motivate (or constrain) the GPU-based",
        "  scalability investigation.",
        "",
        "=" * 80,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*65}")
    print(f"  Event-Vision Baseline Study")
    print(f"  Device : {device}")
    print(f"  Config : {CFG}")
    print(f"{'═'*65}\n")

    print("▶ Training EventCNN (event frames) …\n")
    t0 = time.perf_counter()
    cnn_res = run_event_cnn_baseline(
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        lr=CFG["lr"],
        device=device,
        num_samples=CFG["num_samples"],
        num_classes=CFG["num_classes"],
    )

    print("▶ Training EventSNN (event spike streams) …\n")
    snn_res = run_event_snn_baseline(
        epochs=CFG["epochs"],
        batch_size=CFG["batch_size"],
        lr=CFG["lr"],
        timesteps=CFG["timesteps"],
        device=device,
        num_samples=CFG["num_samples"],
        num_classes=CFG["num_classes"],
    )
    total_time = time.perf_counter() - t0

    # ── Save raw results ───────────────────────
    all_results = {**cnn_res, **snn_res}
    serialisable = {}
    for k, v in all_results.items():
        serialisable[k] = {
            mk: (mv.tolist() if hasattr(mv, "tolist") else mv)
            for mk, mv in v.items()
        }

    json_path = RESULTS_DIR / "event_results.json"
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"✔ Raw results saved → {json_path}")

    # ── Build and save report ──────────────────
    report = build_report(
        cnn_m=all_results["EventCNN"],
        snn_m=all_results["EventSNN"],
        total_time=total_time,
        device=device,
    )
    print("\n" + report)

    report_path = RESULTS_DIR / "event_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✔ Report saved → {report_path}")


if __name__ == "__main__":
    main()
