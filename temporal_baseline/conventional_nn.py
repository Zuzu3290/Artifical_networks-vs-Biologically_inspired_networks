"""
temporal_baseline/conventional_nn.py

Conventional sequence-model baselines for temporal / time-series data.

Models:
  • LSTM  — Long Short-Term Memory
  • GRU   — Gated Recurrent Unit
  • TCN   — Temporal Convolutional Network (dilated causal convolutions)

RNN comparison is VALID only for temporal or event-stream data.
This module provides that valid conventional reference.

Simulated task: multi-class temporal pattern classification,
mimicking a sensor time-series (e.g., IMU, event-rate signal).
In production: replace SyntheticTemporalDataset with a real sensor loader.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

from gpu_profiler import GPUProfiler, estimate_flops


# ─────────────────────────────────────────────
# 1.  Synthetic temporal dataset
# ─────────────────────────────────────────────

class SyntheticTemporalDataset(Dataset):
    """
    Generates synthetic multivariate time-series:
      shape  : (T, input_dim)
      classes: class k has a sinusoidal component at frequency k/num_classes
               plus Gaussian noise, making the temporal structure meaningful.

    Mimics multi-channel sensor data (e.g., 6-DoF IMU, event-rate channels).
    """

    def __init__(
        self,
        num_samples: int = 3000,
        num_classes: int = 10,
        seq_len: int = 50,
        input_dim: int = 8,
        seed: int = 42,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.data   = []
        self.labels = []

        t = np.linspace(0, 2 * np.pi, seq_len)

        for i in range(num_samples):
            cls = i % num_classes
            freq = (cls + 1) / num_classes           # class-specific frequency
            # Base signal: sinusoidal + phase shift per channel
            channels = []
            for ch in range(input_dim):
                phase = ch * np.pi / input_dim
                sig = np.sin(freq * t + phase).astype(np.float32)
                sig += rng.normal(0, 0.15, seq_len).astype(np.float32)
                channels.append(sig)
            self.data.append(np.stack(channels, axis=1))    # (T, input_dim)
            self.labels.append(cls)

        self.data = np.array(self.data, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.data[idx]), self.labels[idx]


def get_temporal_loaders(
    batch_size: int = 64,
    num_samples: int = 3000,
    num_classes: int = 10,
    seq_len: int = 50,
    input_dim: int = 8,
    train_split: float = 0.8,
) -> tuple[DataLoader, DataLoader]:
    full_ds = SyntheticTemporalDataset(num_samples, num_classes, seq_len, input_dim)
    split   = int(len(full_ds) * train_split)
    train_ds, test_ds = torch.utils.data.random_split(
        full_ds, [split, len(full_ds) - split],
        generator=torch.Generator().manual_seed(0),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


# ─────────────────────────────────────────────
# 2.  Architecture definitions
# ─────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """Standard LSTM → mean-pooled hidden states → FC classifier."""

    def __init__(self, input_dim: int = 8, hidden: int = 128, num_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim)"""
        out, _ = self.lstm(x)          # (B, T, hidden)
        return self.head(out[:, -1])   # use last timestep


class GRUClassifier(nn.Module):
    """GRU variant — typically faster than LSTM with comparable accuracy."""

    def __init__(self, input_dim: int = 8, hidden: int = 128, num_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.gru  = nn.GRU(input_dim, hidden, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class CausalConv1d(nn.Module):
    """Single causal dilated 1-D convolution layer."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad  = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation, padding=self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[:, :, :-self.pad] if self.pad else self.conv(x)


class TCNClassifier(nn.Module):
    """
    Temporal Convolutional Network with exponentially growing dilation.
    Receptive field covers the entire sequence without recurrence.
    """

    def __init__(self, input_dim: int = 8, channels: int = 64, levels: int = 4, kernel: int = 3, num_classes: int = 10):
        super().__init__()
        layers = []
        for i in range(levels):
            dilation = 2 ** i
            in_ch  = input_dim if i == 0 else channels
            layers += [
                CausalConv1d(in_ch, channels, kernel, dilation),
                nn.ReLU(),
                nn.BatchNorm1d(channels),
            ]
        self.backbone = nn.Sequential(*layers)
        self.head     = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) → permute to (B, C, T) for Conv1d"""
        x = x.permute(0, 2, 1)           # (B, input_dim, T)
        feat = self.backbone(x)           # (B, channels, T)
        return self.head(feat.mean(dim=-1))


# ─────────────────────────────────────────────
# 3.  Training / evaluation helpers
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)      # x: (B, T, D)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds, all_labels, latencies = [], [], []
    for x, y in loader:
        x = x.to(device)
        t0 = time.perf_counter()
        logits = model(x)
        latencies.append((time.perf_counter() - t0) / x.size(0) * 1e3)
        preds = logits.argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1  = f1_score(all_labels, all_preds, average="macro")
    cm  = confusion_matrix(all_labels, all_preds)
    lat = float(torch.tensor(latencies).mean().item())
    return {"accuracy": acc, "f1_macro": f1, "confusion_matrix": cm, "latency_ms_per_sample": lat}


# ─────────────────────────────────────────────
# 4.  Public entry-point
# ─────────────────────────────────────────────

def run_conventional_temporal_baselines(
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
    Train LSTM, GRU, TCN on the synthetic temporal dataset.

    Returns
    -------
    dict with keys "LSTM", "GRU", "TCN", each containing:
        accuracy, f1_macro, confusion_matrix, latency_ms_per_sample,
        train_time_s, gpu_mem_mb, num_params
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_temporal_loaders(
        batch_size, num_samples, num_classes, seq_len, input_dim
    )
    criterion = nn.CrossEntropyLoss()
    results = {}

    models = {
        "LSTM": LSTMClassifier(input_dim=input_dim, num_classes=num_classes),
        "GRU":  GRUClassifier(input_dim=input_dim,  num_classes=num_classes),
        "TCN":  TCNClassifier(input_dim=input_dim,  num_classes=num_classes),
    }

    for name, model in models.items():
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        profiler = GPUProfiler(device_index=device.index or 0, poll_interval_s=0.05)
        with profiler:
            t_start = time.perf_counter()
            for ep in range(1, epochs + 1):
                loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                print(f"  [{name}] epoch {ep}/{epochs}  loss={loss:.4f}")
            train_time = time.perf_counter() - t_start

        gpu_hw = profiler.metrics()
        profiler.print_summary(prefix=name)

        sample = next(iter(test_loader))[0][:1]   # (1, T, D)
        flops_info = estimate_flops(model, sample, device)

        metrics = evaluate(model, test_loader, device)
        metrics["train_time_s"] = train_time
        metrics["gpu_mem_mb"]   = gpu_hw["torch_alloc_mb_peak"]
        metrics["num_params"]   = sum(p.numel() for p in model.parameters())
        metrics["gpu_metrics"]  = gpu_hw
        metrics["flops"]        = flops_info
        results[name] = metrics

        print(
            f"  [{name}] acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_macro']:.4f}  "
            f"train={train_time:.1f}s  "
            f"energy={gpu_hw['energy_joules']:.2f}J\n"
        )

    return results


if __name__ == "__main__":
    res = run_conventional_temporal_baselines(epochs=10)
    for k, v in res.items():
        print(f"\n=== {k} ===")
        for mk, mv in v.items():
            if mk != "confusion_matrix":
                print(f"  {mk}: {mv}")
