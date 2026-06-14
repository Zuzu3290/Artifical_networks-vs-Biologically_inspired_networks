"""
temporal_baseline/snn_model.py

Spiking Neural Network baseline for temporal / time-series data.

The SNN receives multi-channel time-series as a direct input sequence:
  shape: (T, B, input_dim)  — each timestep is a feature vector

Two variants are provided:
  • SpikingRNN   — LIF neurons with recurrent weights (analogue of LSTM/GRU)
  • SpikingTCN   — Convolutional SNN over temporal dimension (analogue of TCN)

Temporal spike dynamics are evaluated against the LSTM, GRU, TCN baselines
to answer: does spike-based temporal processing offer any useful behaviour?
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import surrogate, functional as SF

from sklearn.metrics import f1_score, confusion_matrix

# Re-use dataset from conventional_nn (same task, different model)
from conventional_nn import get_temporal_loaders
from gpu_profiler import GPUProfiler, estimate_flops


# ─────────────────────────────────────────────
# 1.  Architecture definitions
# ─────────────────────────────────────────────

class SpikingRNN(nn.Module):
    """
    Spiking RNN: each timestep of the input is injected via a linear layer,
    then processed by a stack of Leaky Integrate-and-Fire neurons whose
    membrane potentials carry the temporal state.

    This is the SNN analogue of a standard RNN / LSTM.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden: int = 128,
        num_classes: int = 10,
        beta: float = 0.9,
        num_layers: int = 2,
    ):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.num_layers = num_layers

        # Input projection per layer
        self.input_fc  = nn.Linear(input_dim, hidden)
        self.lif_layers = nn.ModuleList([
            snn.Leaky(beta=beta, spike_grad=spike_grad)
            for _ in range(num_layers)
        ])
        self.inter_fc   = nn.ModuleList([
            nn.Linear(hidden, hidden) for _ in range(num_layers - 1)
        ])
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (T, B, input_dim)
        Returns spk_rec (T, B, num_classes), mem_rec (T, B, num_classes)
        """
        mems = [lif.init_leaky() for lif in self.lif_layers]
        spk_rec, mem_rec = [], []

        for t in range(x.shape[0]):
            # First layer: from input
            cur = self.input_fc(x[t])
            spk, mems[0] = self.lif_layers[0](cur, mems[0])

            # Subsequent layers
            for l in range(1, self.num_layers):
                cur  = self.inter_fc[l - 1](spk)
                spk, mems[l] = self.lif_layers[l](cur, mems[l])

            # Output head (no spike — use membrane for soft prediction)
            out = self.head(spk)
            spk_out_lif = snn.Leaky(beta=0.9)   # ephemeral output LIF
            spk_out, _ = spk_out_lif(out, torch.zeros_like(out))

            spk_rec.append(spk_out)
            mem_rec.append(out)

        return torch.stack(spk_rec), torch.stack(mem_rec)


class SpikingTCN(nn.Module):
    """
    Spiking Temporal Convolutional Network.

    Processes (T, B, input_dim) by stepping through time and applying
    a 1-D causal convolution approximated as a sliding-window over
    accumulated spike history.

    Architecture:
      Each timestep feeds into a convolutional SNN block.
      Spike outputs from previous steps are buffered to emulate dilation.
    """

    def __init__(
        self,
        input_dim: int = 8,
        channels: int = 64,
        num_classes: int = 10,
        beta: float = 0.9,
        kernel: int = 3,
        levels: int = 3,
    ):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.kernel = kernel
        self.levels = levels

        # One FC per level (approximates causal conv in step-by-step execution)
        in_dim = input_dim
        self.block_fcs  = nn.ModuleList()
        self.block_lifs = nn.ModuleList()
        for l in range(levels):
            self.block_fcs.append(nn.Linear(in_dim if l == 0 else channels, channels))
            self.block_lifs.append(snn.Leaky(beta=beta, spike_grad=spike_grad))

        self.head = nn.Linear(channels, num_classes)
        self.head_lif = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, B, input_dim)"""
        T = x.shape[0]
        mems = [lif.init_leaky() for lif in self.block_lifs]
        head_mem = self.head_lif.init_leaky()

        spk_rec, mem_rec = [], []
        inp = x[0]  # initialise

        for t in range(T):
            s = x[t]
            for l in range(self.levels):
                s, mems[l] = self.block_lifs[l](self.block_fcs[l](s), mems[l])

            out, head_mem = self.head_lif(self.head(s), head_mem)
            spk_rec.append(out)
            mem_rec.append(head_mem)

        return torch.stack(spk_rec), torch.stack(mem_rec)


# ─────────────────────────────────────────────
# 2.  Training / evaluation helpers
# ─────────────────────────────────────────────

def _temporal_input_to_snn(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Loader returns (B, T, D); SNN expects (T, B, D).
    Also binarise by thresholding at the mean — a simple spike encoding.
    """
    x = x.permute(1, 0, 2).to(device)          # (T, B, D)
    threshold = x.mean()
    return (x > threshold).float()             # binary spike tensor


def _spike_count_loss(spk_rec: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    counts = spk_rec.sum(0)
    return nn.CrossEntropyLoss()(counts, targets)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        spikes = _temporal_input_to_snn(x, device)   # (T, B, D)
        y = y.to(device)
        optimizer.zero_grad()
        spk_rec, _ = model(spikes)
        loss = _spike_count_loss(spk_rec, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * spikes.size(1)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    timesteps: int,
) -> dict:
    model.eval()
    all_preds, all_labels, latencies = [], [], []
    total_spikes = total_nts = 0

    BIN_WIDTH_MS = 2.0                          # ms per time-bin (sensor assumption)
    event_window_ms = timesteps * BIN_WIDTH_MS

    for x, y in loader:
        spikes = _temporal_input_to_snn(x, device)

        t0 = time.perf_counter()
        spk_rec, _ = model(spikes)
        elapsed_ms = (time.perf_counter() - t0) * 1e3

        latencies.append(elapsed_ms / spikes.size(1))

        preds = spk_rec.sum(0).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

        total_spikes += spk_rec.sum().item()
        total_nts    += spk_rec.numel()

    acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1       = f1_score(all_labels, all_preds, average="macro")
    cm       = confusion_matrix(all_labels, all_preds)
    lat      = float(torch.tensor(latencies).mean().item())
    spike_rt = total_spikes / max(total_nts, 1)
    rt_factor = lat / event_window_ms

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "latency_ms_per_sample": lat,
        "spike_rate": spike_rt,
        "timesteps": timesteps,
        "event_window_ms": event_window_ms,
        "real_time_factor": rt_factor,
    }


# ─────────────────────────────────────────────
# 3.  Public entry-point
# ─────────────────────────────────────────────

def run_temporal_snn_baselines(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
    num_samples: int = 3000,
    num_classes: int = 10,
    seq_len: int = 50,
    input_dim: int = 8,
) -> dict:
    """
    Train SpikingRNN and SpikingTCN on the synthetic temporal dataset.

    Returns
    -------
    dict with keys "SNN_RNN" and "SNN_TCN", each containing:
        accuracy, f1_macro, confusion_matrix, latency_ms_per_sample,
        spike_rate, timesteps, event_window_ms, real_time_factor,
        train_time_s, gpu_mem_mb, num_params
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_temporal_loaders(
        batch_size, num_samples, num_classes, seq_len, input_dim
    )
    results = {}

    for name, model in [
        ("SNN_RNN", SpikingRNN(input_dim=input_dim, num_classes=num_classes)),
        ("SNN_TCN", SpikingTCN(input_dim=input_dim, num_classes=num_classes)),
    ]:
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        profiler = GPUProfiler(device_index=device.index or 0, poll_interval_s=0.05)
        with profiler:
            t_start = time.perf_counter()
            for ep in range(1, epochs + 1):
                loss = train_one_epoch(model, train_loader, optimizer, device)
                print(f"  [{name}] epoch {ep}/{epochs}  loss={loss:.4f}")
            train_time = time.perf_counter() - t_start

        gpu_hw = profiler.metrics()
        profiler.print_summary(prefix=name)

        # FLOPs: encode one batch sample and forward
        sample_raw = next(iter(test_loader))[0][:1]        # (1, T, D)
        spk_sample = _temporal_input_to_snn(sample_raw, device)  # (T, 1, D)
        flops_info = estimate_flops(model, spk_sample, device)

        metrics = evaluate(model, test_loader, device, timesteps=seq_len)
        metrics["train_time_s"] = train_time
        metrics["gpu_mem_mb"]   = gpu_hw["torch_alloc_mb_peak"]
        metrics["num_params"]   = sum(p.numel() for p in model.parameters())
        metrics["gpu_metrics"]  = gpu_hw
        metrics["flops"]        = flops_info
        results[name] = metrics

        print(
            f"  [{name}] acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_macro']:.4f}  "
            f"spike_rate={metrics['spike_rate']:.4f}  "
            f"RT_factor={metrics['real_time_factor']:.3f}  "
            f"train={train_time:.1f}s  "
            f"energy={gpu_hw['energy_joules']:.2f}J\n"
        )

    return results


if __name__ == "__main__":
    res = run_temporal_snn_baselines(epochs=10)
    for k, v in res.items():
        print(f"\n=== {k} ===")
        for mk, mv in v.items():
            if mk != "confusion_matrix":
                print(f"  {mk}: {mv}")
