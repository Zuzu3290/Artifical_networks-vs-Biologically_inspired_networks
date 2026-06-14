"""
event_vision_baseline/conventional_nn.py

Conventional CNN baseline for event-camera vision (N-MNIST / NCars / synthetic).

The CNN receives event-camera data as *time-windowed frames* —
i.e., events are accumulated into a 2D histogram over a fixed time window
and treated as a standard image.

This is the most important comparison in the project:
  CNN on event frames  vs  SNN on event/spike streams

Both operate on the same underlying event source, but use different
computational principles.
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
# 1.  Synthetic event-frame dataset
# ─────────────────────────────────────────────
# In production: replace with tonic.datasets.NMNIST or tonic.datasets.NCars.
# We generate a synthetic dataset that mimics the structure so the pipeline
# works out-of-the-box without tonic installed.

class SyntheticEventFrameDataset(Dataset):
    """
    Generates synthetic event-camera frames:
      • shape  : (C=2, H, W)   channel 0 = ON events, channel 1 = OFF events
      • labels : 0 … num_classes-1
      • event counts are drawn from class-specific Poisson distributions
        so a classifier can learn non-trivial structure.
    """

    def __init__(
        self,
        num_samples: int = 3000,
        num_classes: int = 10,
        height: int = 34,
        width: int = 34,
        seed: int = 42,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.frames = []
        self.labels = []

        for i in range(num_samples):
            cls = i % num_classes
            # Class-specific spatial bias
            rate_on  = 0.05 + 0.04 * cls
            rate_off = 0.05 + 0.04 * (num_classes - 1 - cls)
            on_ch  = rng.poisson(rate_on  * np.ones((height, width))).astype(np.float32)
            off_ch = rng.poisson(rate_off * np.ones((height, width))).astype(np.float32)
            # Normalise by max-count across the frame
            max_v = max(on_ch.max(), off_ch.max(), 1.0)
            frame = np.stack([on_ch, off_ch], axis=0) / max_v   # (2, H, W)
            self.frames.append(frame)
            self.labels.append(cls)

        self.frames = np.array(self.frames, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.frames[idx]), self.labels[idx]


def get_event_frame_loaders(
    batch_size: int = 64,
    num_samples: int = 3000,
    num_classes: int = 10,
    height: int = 34,
    width: int = 34,
    train_split: float = 0.8,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for the event-frame dataset."""
    full_ds = SyntheticEventFrameDataset(num_samples, num_classes, height, width)
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

class EventCNN(nn.Module):
    """
    CNN that processes 2-channel event frames (ON / OFF polarity).
    Architecture mirrors what is used in event-camera literature
    (e.g., EST, RED): shallow conv backbone + global-average pool.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 34→17

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # 17→8 (floor)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),                  # fixed 4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# ─────────────────────────────────────────────
# 3.  Training / evaluation
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
        x, y = x.to(device), y.to(device)
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
        preds = logits.argmax(dim=1).cpu()
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

def run_event_cnn_baseline(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
    num_samples: int = 3000,
    num_classes: int = 10,
) -> dict:
    """
    Train EventCNN on synthetic (or real) event frames.

    Returns
    -------
    dict with key "EventCNN":
        accuracy, f1_macro, confusion_matrix, latency_ms_per_sample,
        train_time_s, gpu_mem_mb, num_params, input_type
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_event_frame_loaders(
        batch_size=batch_size,
        num_samples=num_samples,
        num_classes=num_classes,
    )
    criterion = nn.CrossEntropyLoss()
    model = EventCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    profiler = GPUProfiler(device_index=device.index or 0, poll_interval_s=0.05)
    with profiler:
        t_start = time.perf_counter()
        for ep in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            print(f"  [EventCNN] epoch {ep}/{epochs}  loss={loss:.4f}")
        train_time = time.perf_counter() - t_start

    gpu_hw = profiler.metrics()
    profiler.print_summary(prefix="EventCNN")

    sample = next(iter(test_loader))[0][:1]
    flops_info = estimate_flops(model, sample, device)

    metrics = evaluate(model, test_loader, device)
    metrics["train_time_s"] = train_time
    metrics["gpu_mem_mb"]   = gpu_hw["torch_alloc_mb_peak"]
    metrics["num_params"]   = sum(p.numel() for p in model.parameters())
    metrics["input_type"]   = "event_frames_2ch"
    metrics["gpu_metrics"]  = gpu_hw
    metrics["flops"]        = flops_info

    print(
        f"\n  [EventCNN] acc={metrics['accuracy']:.4f}  "
        f"F1={metrics['f1_macro']:.4f}  "
        f"lat={metrics['latency_ms_per_sample']:.3f}ms  "
        f"train={train_time:.1f}s  "
        f"energy={gpu_hw['energy_joules']:.2f}J\n"
    )
    return {"EventCNN": metrics}


if __name__ == "__main__":
    res = run_event_cnn_baseline(epochs=10)
    for k, v in res["EventCNN"].items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
