"""
gpu_profiler.py  ―  shared across all three baseline study folders

Collects GPU hardware metrics via nvidia-ml-py (pynvml) alongside
PyTorch's own memory counters.  Falls back gracefully when:
  • no NVIDIA GPU is present (CPU-only machine)
  • libnvidia-ml.so is unavailable (container without driver mount)
  • multiple calls from CPU-only unit tests

Metrics captured per profiling session
──────────────────────────────────────
Power / Energy
  • power_draw_w_mean   — mean instantaneous power draw (Watts)
  • power_draw_w_peak   — peak instantaneous power draw (Watts)
  • energy_joules       — ∫ power dt  (trapezoidal, sampled at poll_interval_s)
  • power_limit_w       — board TDP limit for reference

Memory (NVML)
  • vram_used_mb_peak   — peak VRAM used during session (MiB)
  • vram_free_mb_start  — free VRAM at session start (MiB)
  • vram_total_mb       — total VRAM on device (MiB)

Memory (PyTorch allocator)
  • torch_alloc_mb_peak  — peak bytes allocated by PyTorch (MiB)
  • torch_reserved_mb    — bytes reserved (cached) by PyTorch allocator (MiB)

Utilisation
  • gpu_util_pct_mean    — mean GPU compute utilisation (%)
  • gpu_util_pct_peak    — peak GPU compute utilisation (%)
  • mem_util_pct_mean    — mean memory controller utilisation (%)

Throughput / FLOPs (model-level, optional)
  • flops_total          — total FLOPs estimated via torch.profiler (if enabled)
  • macs_total           — total MACs (= flops / 2)

Temperature
  • temp_c_mean          — mean GPU temperature (°C)
  • temp_c_peak          — peak GPU temperature (°C)

Usage
─────
    from gpu_profiler import GPUProfiler

    profiler = GPUProfiler(device_index=0, poll_interval_s=0.05)
    with profiler:
        train(...)
    metrics = profiler.metrics()          # dict ready for JSON serialisation
    profiler.print_summary()
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Optional

import torch

# ── nvidia-ml-py ──────────────────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nvml_handle(device_index: int):
    if not _NVML_OK:
        return None
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception:
        return None


def _query_power(handle) -> Optional[float]:
    """Return instantaneous power in Watts, or None."""
    try:
        mw = pynvml.nvmlDeviceGetPowerUsage(handle)   # milliwatts
        return mw / 1000.0
    except Exception:
        return None


def _query_memory(handle) -> Optional[dict]:
    """Return NVML memory info dict (used/free/total in MiB), or None."""
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            "used_mb":  info.used  / 2**20,
            "free_mb":  info.free  / 2**20,
            "total_mb": info.total / 2**20,
        }
    except Exception:
        return None


def _query_utilisation(handle) -> Optional[dict]:
    """Return GPU and memory controller utilisation (%), or None."""
    try:
        rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {"gpu_pct": rates.gpu, "mem_pct": rates.memory}
    except Exception:
        return None


def _query_temperature(handle) -> Optional[float]:
    try:
        return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return None


def _query_power_limit(handle) -> Optional[float]:
    try:
        return pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0   # W
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GPUProfiler
# ─────────────────────────────────────────────────────────────────────────────

class GPUProfiler:
    """
    Context-manager that polls NVML on a background thread while a workload runs.

    Parameters
    ----------
    device_index    : CUDA device index (0-based)
    poll_interval_s : polling period in seconds (default 50 ms → 20 Hz)

    Example
    -------
    profiler = GPUProfiler(device_index=0, poll_interval_s=0.05)
    with profiler:
        model_output = run_training(model, data)
    print(profiler.metrics())
    """

    def __init__(self, device_index: int = 0, poll_interval_s: float = 0.05):
        self.device_index    = device_index
        self.poll_interval_s = poll_interval_s
        self._handle         = _nvml_handle(device_index)
        self._available      = self._handle is not None

        # sample buffers
        self._timestamps:   list[float] = []
        self._power_samples:list[float] = []
        self._util_samples: list[float] = []
        self._mem_util_s:   list[float] = []
        self._temp_samples: list[float] = []
        self._vram_used_s:  list[float] = []

        self._vram_free_start: float = 0.0
        self._vram_total:      float = 0.0
        self._power_limit_w:   float = 0.0

        self._torch_device = (
            torch.device(f"cuda:{device_index}")
            if torch.cuda.is_available() else None
        )

        self._stop_event = threading.Event()
        self._thread:    Optional[threading.Thread] = None

    # ── polling loop ──────────────────────────────────────────────────────────

    def _poll_loop(self):
        while not self._stop_event.is_set():
            t = time.perf_counter()
            self._timestamps.append(t)

            if self._available:
                pw = _query_power(self._handle)
                if pw is not None:
                    self._power_samples.append(pw)

                util = _query_utilisation(self._handle)
                if util:
                    self._util_samples.append(util["gpu_pct"])
                    self._mem_util_s.append(util["mem_pct"])

                temp = _query_temperature(self._handle)
                if temp is not None:
                    self._temp_samples.append(temp)

                mem = _query_memory(self._handle)
                if mem:
                    self._vram_used_s.append(mem["used_mb"])

            time.sleep(self.poll_interval_s)

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "GPUProfiler":
        # Reset buffers
        self._timestamps.clear()
        self._power_samples.clear()
        self._util_samples.clear()
        self._mem_util_s.clear()
        self._temp_samples.clear()
        self._vram_used_s.clear()

        # Snapshot start state
        if self._available:
            mem_start = _query_memory(self._handle)
            if mem_start:
                self._vram_free_start = mem_start["free_mb"]
                self._vram_total      = mem_start["total_mb"]
            pl = _query_power_limit(self._handle)
            if pl:
                self._power_limit_w = pl

        # Reset PyTorch allocator peak stats
        if self._torch_device is not None:
            torch.cuda.reset_peak_memory_stats(self._torch_device)

        # Start polling thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

        self._t_start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._t_end = time.perf_counter()

    # ── metric aggregation ────────────────────────────────────────────────────

    def _safe_mean(self, lst: list) -> float:
        return float(sum(lst) / len(lst)) if lst else 0.0

    def _safe_peak(self, lst: list) -> float:
        return float(max(lst)) if lst else 0.0

    def _energy_joules(self) -> float:
        """Trapezoidal integration of power over time."""
        if len(self._power_samples) < 2 or len(self._timestamps) < 2:
            return 0.0
        # Use same length (power is sampled inside the loop alongside timestamps)
        n = min(len(self._power_samples), len(self._timestamps))
        ts = self._timestamps[:n]
        pw = self._power_samples[:n]
        energy = 0.0
        for i in range(1, n):
            dt = ts[i] - ts[i - 1]
            energy += 0.5 * (pw[i] + pw[i - 1]) * dt
        return energy

    def metrics(self) -> dict:
        """
        Return a flat dict of all GPU metrics, safe for JSON serialisation.
        All fields are always present; unavailable values are 0.0.
        """
        wall_time = getattr(self, "_t_end", time.perf_counter()) - getattr(self, "_t_start", 0.0)

        # PyTorch allocator stats
        torch_alloc_peak = 0.0
        torch_reserved   = 0.0
        if self._torch_device is not None and torch.cuda.is_available():
            torch_alloc_peak = torch.cuda.max_memory_allocated(self._torch_device) / 2**20
            torch_reserved   = torch.cuda.memory_reserved(self._torch_device)       / 2**20

        return {
            # ── Power / Energy ─────────────────────────────
            "power_draw_w_mean":   self._safe_mean(self._power_samples),
            "power_draw_w_peak":   self._safe_peak(self._power_samples),
            "energy_joules":       self._energy_joules(),
            "power_limit_w":       self._power_limit_w,

            # ── NVML Memory ────────────────────────────────
            "vram_used_mb_peak":   self._safe_peak(self._vram_used_s),
            "vram_free_mb_start":  self._vram_free_start,
            "vram_total_mb":       self._vram_total,

            # ── PyTorch Allocator ──────────────────────────
            "torch_alloc_mb_peak": torch_alloc_peak,
            "torch_reserved_mb":   torch_reserved,

            # ── Compute Utilisation ────────────────────────
            "gpu_util_pct_mean":   self._safe_mean(self._util_samples),
            "gpu_util_pct_peak":   self._safe_peak(self._util_samples),
            "mem_util_pct_mean":   self._safe_mean(self._mem_util_s),

            # ── Temperature ────────────────────────────────
            "temp_c_mean":         self._safe_mean(self._temp_samples),
            "temp_c_peak":         self._safe_peak(self._temp_samples),

            # ── Session metadata ───────────────────────────
            "wall_time_s":         wall_time,
            "poll_interval_s":     self.poll_interval_s,
            "num_samples_polled":  len(self._timestamps),
            "nvml_available":      self._available,
        }

    def print_summary(self, prefix: str = ""):
        m = self.metrics()
        tag = f"[{prefix}] " if prefix else ""
        if not m["nvml_available"]:
            print(f"  {tag}GPU profiler: NVML unavailable (no GPU / driver). "
                  f"PyTorch peak alloc = {m['torch_alloc_mb_peak']:.1f} MiB")
            return

        print(f"\n  {tag}── GPU Hardware Metrics ──────────────────────────────")
        print(f"  {tag}Power  : mean {m['power_draw_w_mean']:.1f} W  |  peak {m['power_draw_w_peak']:.1f} W  "
              f"|  limit {m['power_limit_w']:.1f} W")
        print(f"  {tag}Energy : {m['energy_joules']:.2f} J  (over {m['wall_time_s']:.1f} s)")
        print(f"  {tag}VRAM   : peak {m['vram_used_mb_peak']:.1f} MiB  "
              f"|  total {m['vram_total_mb']:.0f} MiB  "
              f"|  free@start {m['vram_free_mb_start']:.1f} MiB")
        print(f"  {tag}PyTorch: peak alloc {m['torch_alloc_mb_peak']:.1f} MiB  "
              f"|  reserved {m['torch_reserved_mb']:.1f} MiB")
        print(f"  {tag}Util   : GPU mean {m['gpu_util_pct_mean']:.1f}%  peak {m['gpu_util_pct_peak']:.1f}%  "
              f"|  mem ctrl mean {m['mem_util_pct_mean']:.1f}%")
        print(f"  {tag}Temp   : mean {m['temp_c_mean']:.1f}°C  |  peak {m['temp_c_peak']:.1f}°C")
        print(f"  {tag}Samples: {m['num_samples_polled']}  @ {1/m['poll_interval_s']:.0f} Hz")
        print(f"  {tag}────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: FLOPs estimation via torch.profiler
# ─────────────────────────────────────────────────────────────────────────────

def estimate_flops(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Run a single forward pass inside torch.profiler and extract FLOPs / MACs.

    Returns dict with keys: flops_total, macs_total, profiler_available
    Falls back gracefully if torch.profiler is not available or fails.

    Note: torch.profiler FLOPs counting is only accurate for a subset of ops
    (conv, matmul, bmm).  SNN ops (LIF threshold, surrogate gradient) are
    not tracked — this gives a *lower-bound* estimate useful for comparison.
    """
    try:
        from torch.profiler import profile, record_function, ProfilerActivity

        model = model.to(device)
        model.eval()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=True,
        ) as prof:
            with record_function("forward"):
                with torch.no_grad():
                    _ = model(sample_input.to(device))

        total_flops = sum(
            e.flops for e in prof.key_averages() if e.flops and e.flops > 0
        )
        return {
            "flops_total":          total_flops,
            "macs_total":           total_flops // 2,
            "profiler_available":   True,
        }
    except Exception as exc:
        return {
            "flops_total":        0,
            "macs_total":         0,
            "profiler_available": False,
            "profiler_error":     str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Report formatter (used by each main.py)
# ─────────────────────────────────────────────────────────────────────────────

def format_gpu_metrics_block(gpu_metrics: dict, label: str = "") -> str:
    """Return a formatted multi-line string of GPU metrics for inclusion in reports."""
    tag = f" [{label}]" if label else ""
    m = gpu_metrics

    if not m.get("nvml_available"):
        return (
            f"  GPU Metrics{tag}: NVML unavailable (CPU-only run)\n"
            f"    PyTorch peak alloc : {m['torch_alloc_mb_peak']:.1f} MiB\n"
            f"    PyTorch reserved   : {m['torch_reserved_mb']:.1f} MiB\n"
            f"    Wall time          : {m['wall_time_s']:.2f} s\n"
        )

    return (
        f"  GPU Metrics{tag}:\n"
        f"    Power  | mean {m['power_draw_w_mean']:>7.1f} W    peak {m['power_draw_w_peak']:>7.1f} W    "
        f"limit {m['power_limit_w']:.1f} W\n"
        f"    Energy | {m['energy_joules']:>10.3f} J  (over {m['wall_time_s']:.2f} s)\n"
        f"    VRAM   | peak {m['vram_used_mb_peak']:>7.1f} MiB  total {m['vram_total_mb']:.0f} MiB  "
        f"free@start {m['vram_free_mb_start']:.1f} MiB\n"
        f"    PyTorch| alloc peak {m['torch_alloc_mb_peak']:>7.1f} MiB  reserved {m['torch_reserved_mb']:.1f} MiB\n"
        f"    Util   | GPU mean {m['gpu_util_pct_mean']:>5.1f}%  peak {m['gpu_util_pct_peak']:.1f}%  "
        f"mem-ctrl {m['mem_util_pct_mean']:.1f}%\n"
        f"    Temp   | mean {m['temp_c_mean']:>5.1f}°C  peak {m['temp_c_peak']:.1f}°C\n"
        f"    Polls  | {m['num_samples_polled']} samples @ {1/m['poll_interval_s']:.0f} Hz\n"
    )
