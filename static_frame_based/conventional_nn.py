"""
static_frame_based/conventional_nn.py

Conventional baseline models for static / frame-based data (MNIST).
Provides an MLP and a small CNN, both returning a unified result dict
that the main runner can compare directly against the SNN results.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, confusion_matrix

from gpu_profiler import GPUProfiler, estimate_flops


# ─────────────────────────────────────────────
# 1.  Architecture definitions
# ─────────────────────────────────────────────

class MLP(nn.Module):
    """Three-layer MLP for flat 28×28 input."""

    def __init__(self, input_dim: int = 784, hidden: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    """Lightweight CNN: two conv blocks + FC head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → 14×14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# 2.  Data loader
# ─────────────────────────────────────────────

def get_mnist_loaders(
    batch_size: int = 64,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Download MNIST and return (train_loader, test_loader)."""
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(data_root, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


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
    """Return accuracy, macro-F1, confusion matrix, and avg latency per batch."""
    model.eval()
    all_preds, all_labels = [], []
    latencies = []

    for x, y in loader:
        x = x.to(device)
        t0 = time.perf_counter()
        logits = model(x)
        latencies.append((time.perf_counter() - t0) / x.size(0) * 1e3)   # ms/sample

        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

    acc  = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1   = f1_score(all_labels, all_preds, average="macro")
    cm   = confusion_matrix(all_labels, all_preds)
    lat  = float(torch.tensor(latencies).mean().item())
    return {"accuracy": acc, "f1_macro": f1, "confusion_matrix": cm, "latency_ms_per_sample": lat}


# ─────────────────────────────────────────────
# 4.  Public entry-point
# ─────────────────────────────────────────────

def run_conventional_baselines(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
    data_root: str = "./data",
) -> dict:
    """
    Train MLP and CNN on MNIST.

    Returns
    -------
    dict with keys "MLP" and "CNN", each containing:
        accuracy, f1_macro, confusion_matrix, latency_ms_per_sample,
        train_time_s, gpu_mem_mb, num_params
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_loaders(batch_size, data_root)
    criterion = nn.CrossEntropyLoss()
    results = {}

    for name, model in [("MLP", MLP()), ("CNN", SmallCNN())]:
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # ── training (wrapped in GPU profiler) ────
        profiler = GPUProfiler(device_index=device.index or 0, poll_interval_s=0.05)
        with profiler:
            t_start = time.perf_counter()
            for ep in range(1, epochs + 1):
                loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                print(f"  [{name}] epoch {ep}/{epochs}  loss={loss:.4f}")
            train_time = time.perf_counter() - t_start

        gpu_hw = profiler.metrics()
        profiler.print_summary(prefix=name)

        # ── FLOPs estimate ─────────────────────────
        sample = next(iter(test_loader))[0][:1]   # single sample
        flops_info = estimate_flops(model, sample, device)

        # ── evaluation ────────────────────────────
        metrics = evaluate(model, test_loader, device)
        metrics["train_time_s"]     = train_time
        # Legacy key kept for report compatibility
        metrics["gpu_mem_mb"]       = gpu_hw["torch_alloc_mb_peak"]
        metrics["num_params"]       = sum(p.numel() for p in model.parameters())
        metrics["gpu_metrics"]      = gpu_hw
        metrics["flops"]            = flops_info
        results[name] = metrics

        print(
            f"  [{name}] acc={metrics['accuracy']:.4f}  "
            f"F1={metrics['f1_macro']:.4f}  "
            f"lat={metrics['latency_ms_per_sample']:.3f}ms  "
            f"train={train_time:.1f}s  "
            f"energy={gpu_hw['energy_joules']:.2f}J\n"
        )

    return results


if __name__ == "__main__":
    res = run_conventional_baselines(epochs=5)
    for model_name, m in res.items():
        print(f"\n=== {model_name} ===")
        for k, v in m.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v}")
