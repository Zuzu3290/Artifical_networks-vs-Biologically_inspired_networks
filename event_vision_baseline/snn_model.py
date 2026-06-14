"""
event_vision_baseline/snn_model.py

Spiking Neural Network baseline for event-camera data.

The SNN receives the raw event stream as a *spike tensor*:
    shape (T, B, 2, H, W)  — T timesteps, 2 polarity channels

This is the biologically motivated counterpart to the EventCNN:
  • CNN processes accumulated event FRAMES  (time-collapsed)
  • SNN processes the event STREAM         (time-explicit spikes)

Architecture: Spiking Conv-Net (SCN) with LIF neurons,
designed to mirror the EventCNN depth while preserving temporal dynamics.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import snntorch as snn
from snntorch import surrogate

from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

from gpu_profiler import GPUProfiler, estimate_flops


# ─────────────────────────────────────────────
# 1.  Synthetic event-stream dataset
# ─────────────────────────────────────────────

class SyntheticEventStreamDataset(Dataset):
    """
    Synthetic dataset that mimics a time-windowed event stream.
    Each sample is a binary spike tensor of shape (T, 2, H, W).
    Class-specific spatial patterns ensure the task is learnable.

    In production: replace with tonic.datasets.NMNIST loaded with
    tonic.transforms.ToFrame(n_time_bins=T) → yields (T, 2, H, W) tensors.
    """

    def __init__(
        self,
        num_samples: int = 3000,
        num_classes: int = 10,
        timesteps: int = 10,
        height: int = 34,
        width: int = 34,
        seed: int = 42,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.data   = []
        self.labels = []
        self.T = timesteps

        for i in range(num_samples):
            cls = i % num_classes
            # Class-modulated firing probability (higher class → more ON spikes early)
            p_on_early  = 0.03 + 0.04 * cls / num_classes
            p_off_late  = 0.03 + 0.04 * (num_classes - 1 - cls) / num_classes

            frames = []
            for t in range(timesteps):
                p_on  = p_on_early  if t < timesteps // 2 else 0.02
                p_off = p_off_late  if t >= timesteps // 2 else 0.02
                on_ch  = rng.binomial(1, p_on,  (height, width)).astype(np.float32)
                off_ch = rng.binomial(1, p_off, (height, width)).astype(np.float32)
                frames.append(np.stack([on_ch, off_ch], axis=0))   # (2, H, W)

            self.data.append(np.stack(frames, axis=0))              # (T, 2, H, W)
            self.labels.append(cls)

        self.data   = np.array(self.data, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.data[idx]), self.labels[idx]


def get_event_stream_loaders(
    batch_size: int = 64,
    num_samples: int = 3000,
    num_classes: int = 10,
    timesteps: int = 10,
    height: int = 34,
    width: int = 34,
    train_split: float = 0.8,
) -> tuple[DataLoader, DataLoader]:
    full_ds = SyntheticEventStreamDataset(num_samples, num_classes, timesteps, height, width)
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
# 2.  Architecture
# ─────────────────────────────────────────────

class SpikingEventNet(nn.Module):
    """
    Spiking Convolutional Network for event streams.

    Input:  (T, B, 2, H, W)  — binary spike tensor
    Output: (T, B, num_classes) spike counts per step

    Mirrors EventCNN depth:  3 conv blocks + FC head.
    Each conv layer is followed by a LIF neuron layer.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        beta: float = 0.9,
        height: int = 34,
        width: int = 34,
    ):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Block 1
        self.conv1  = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm2d(32)
        self.pool1  = nn.MaxPool2d(2)
        self.lif1   = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Block 2
        self.conv2  = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm2d(64)
        self.pool2  = nn.MaxPool2d(2)
        self.lif2   = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Block 3
        self.conv3  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3    = nn.BatchNorm2d(128)
        self.gpool  = nn.AdaptiveAvgPool2d(4)
        self.lif3   = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # FC head
        self.flatten = nn.Flatten()
        self.fc1    = nn.Linear(128 * 4 * 4, 256)
        self.lif_f1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2    = nn.Linear(256, num_classes)
        self.lif_f2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def _init_states(self):
        return (
            self.lif1.init_leaky(),
            self.lif2.init_leaky(),
            self.lif3.init_leaky(),
            self.lif_f1.init_leaky(),
            self.lif_f2.init_leaky(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, B, 2, H, W) → returns (spk_rec, mem_rec) each (T, B, num_classes)"""
        m1, m2, m3, mf1, mf2 = self._init_states()
        spk_rec, mem_rec = [], []

        for t in range(x.shape[0]):
            xt = x[t]                                     # (B, 2, H, W)

            s, m1  = self.lif1(self.pool1(self.bn1(self.conv1(xt))), m1)
            s, m2  = self.lif2(self.pool2(self.bn2(self.conv2(s))),  m2)
            s, m3  = self.lif3(self.gpool(self.bn3(self.conv3(s))),  m3)

            s, mf1 = self.lif_f1(self.fc1(self.flatten(s)), mf1)
            s, mf2 = self.lif_f2(self.fc2(s),               mf2)

            spk_rec.append(s)
            mem_rec.append(mf2)

        return torch.stack(spk_rec), torch.stack(mem_rec)


# ─────────────────────────────────────────────
# 3.  Training / evaluation helpers
# ─────────────────────────────────────────────

def _spike_count_loss(spk_rec: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    counts = spk_rec.sum(dim=0)
    return nn.CrossEntropyLoss()(counts, targets)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    loader yields (spike_tensor, label)
    spike_tensor shape: (B, T, 2, H, W)
    We permute to (T, B, 2, H, W) before feeding the SNN.
    """
    model.train()
    total_loss = 0.0

    for x, y in loader:
        # x: (B, T, 2, H, W) → (T, B, 2, H, W)
        x = x.permute(1, 0, 2, 3, 4).to(device)
        y = y.to(device)
        optimizer.zero_grad()
        spk_rec, _ = model(x)
        loss = _spike_count_loss(spk_rec, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(1)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    timesteps: int,
) -> dict:
    """Returns accuracy, f1, confusion matrix, latency, spike rate, real-time factor."""
    model.eval()
    all_preds, all_labels = [], []
    latencies = []
    total_spikes = total_nts = 0

    # Simulated event window duration (ms) = timesteps × bin_width_ms
    BIN_WIDTH_MS = 5.0
    event_window_ms = timesteps * BIN_WIDTH_MS

    for x, y in loader:
        x = x.permute(1, 0, 2, 3, 4).to(device)    # (T, B, 2, H, W)

        t0 = time.perf_counter()
        spk_rec, _ = model(x)
        elapsed_ms = (time.perf_counter() - t0) * 1e3

        latencies.append(elapsed_ms / x.size(1))    # ms per sample

        preds = spk_rec.sum(0).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

        total_spikes += spk_rec.sum().item()
        total_nts    += spk_rec.numel()

    acc       = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1        = f1_score(all_labels, all_preds, average="macro")
    cm        = confusion_matrix(all_labels, all_preds)
    lat       = float(torch.tensor(latencies).mean().item())
    spike_rt  = total_spikes / max(total_nts, 1)

    # Real-time factor: processing_time_per_sample / event_window_ms
    rt_factor = lat / event_window_ms

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "latency_ms_per_sample": lat,
        "spike_rate": spike_rt,
        "timesteps": timesteps,
        "event_window_ms": event_window_ms,
        "real_time_factor": rt_factor,      # < 1.0 → real-time capable
    }


# ─────────────────────────────────────────────
# 4.  Public entry-point
# ─────────────────────────────────────────────

def run_event_snn_baseline(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    timesteps: int = 10,
    device: torch.device | None = None,
    num_samples: int = 3000,
    num_classes: int = 10,
) -> dict:
    """
    Train SpikingEventNet on synthetic event stream data.

    Returns
    -------
    dict with key "EventSNN":
        accuracy, f1_macro, confusion_matrix, latency_ms_per_sample,
        spike_rate, timesteps, event_window_ms, real_time_factor,
        train_time_s, gpu_mem_mb, num_params, input_type
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_event_stream_loaders(
        batch_size=batch_size,
        num_samples=num_samples,
        num_classes=num_classes,
        timesteps=timesteps,
    )
    model     = SpikingEventNet(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    profiler = GPUProfiler(device_index=device.index or 0, poll_interval_s=0.05)
    with profiler:
        t_start = time.perf_counter()
        for ep in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, device)
            print(f"  [EventSNN] epoch {ep}/{epochs}  loss={loss:.4f}")
        train_time = time.perf_counter() - t_start

    gpu_hw = profiler.metrics()
    profiler.print_summary(prefix="EventSNN")

    # FLOPs: pass a single (T, 1, 2, H, W) spike sample
    sample_batch = next(iter(test_loader))[0][:1]        # (1, T, 2, H, W)
    spk_sample   = sample_batch.permute(1, 0, 2, 3, 4)  # (T, 1, 2, H, W)
    flops_info   = estimate_flops(model, spk_sample, device)

    metrics = evaluate(model, test_loader, device, timesteps)
    metrics["train_time_s"] = train_time
    metrics["gpu_mem_mb"]   = gpu_hw["torch_alloc_mb_peak"]
    metrics["num_params"]   = sum(p.numel() for p in model.parameters())
    metrics["input_type"]   = "event_spike_stream"
    metrics["gpu_metrics"]  = gpu_hw
    metrics["flops"]        = flops_info

    print(
        f"\n  [EventSNN] acc={metrics['accuracy']:.4f}  "
        f"F1={metrics['f1_macro']:.4f}  "
        f"spike_rate={metrics['spike_rate']:.4f}  "
        f"RT_factor={metrics['real_time_factor']:.3f}  "
        f"train={train_time:.1f}s  "
        f"energy={gpu_hw['energy_joules']:.2f}J\n"
    )
    return {"EventSNN": metrics}


if __name__ == "__main__":
    res = run_event_snn_baseline(epochs=10, timesteps=10)
    for k, v in res["EventSNN"].items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
