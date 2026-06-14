"""
static_frame_based/snn_model.py

Spiking Neural Network baseline for static / frame-based data (MNIST).
Uses SNNTorch with Leaky Integrate-and-Fire (LIF) neurons.

The image is rate-encoded into T timesteps before being fed to the SNN.
Output: spike-count vote across T steps → class prediction.

Note: static data is not the natural strength of SNNs.
Results here serve as a *functional reference*, not an industrial argument.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate, functional as SF

from sklearn.metrics import f1_score, confusion_matrix

from gpu_profiler import GPUProfiler, estimate_flops


# ─────────────────────────────────────────────
# 1.  Architecture
# ─────────────────────────────────────────────

class SpikingMLP(nn.Module):
    """
    Two-hidden-layer spiking MLP with LIF neurons.
    Input is a rate-coded spike train of shape (T, B, 784).
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden: int = 256,
        num_classes: int = 10,
        beta: float = 0.9,          # membrane decay
        threshold: float = 1.0,
    ):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1  = nn.Linear(input_dim, hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.fc2  = nn.Linear(hidden, hidden // 2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.fc3  = nn.Linear(hidden // 2, num_classes)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (T, B, 784)  rate-coded binary spike tensor

        Returns
        -------
        spk_rec : (T, B, num_classes)  output spikes per timestep
        mem_rec : (T, B, num_classes)  membrane potential per timestep
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec, mem_rec = [], []

        for t in range(x.shape[0]):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk_rec.append(spk3)
            mem_rec.append(mem3)

        return torch.stack(spk_rec), torch.stack(mem_rec)


class SpikingCNN(nn.Module):
    """
    Spiking CNN: conv LIF blocks + spiking FC head.
    Input spike tensor shape: (T, B, 1, 28, 28).
    """

    def __init__(self, num_classes: int = 10, beta: float = 0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1  = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lif_c1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1  = nn.MaxPool2d(2)

        self.conv2  = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif_c2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2  = nn.MaxPool2d(2)

        self.flat   = nn.Flatten()
        self.fc1    = nn.Linear(32 * 7 * 7, 128)
        self.lif_f1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2    = nn.Linear(128, num_classes)
        self.lif_f2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, B, 1, 28, 28)"""
        mc1 = self.lif_c1.init_leaky()
        mc2 = self.lif_c2.init_leaky()
        mf1 = self.lif_f1.init_leaky()
        mf2 = self.lif_f2.init_leaky()

        spk_rec, mem_rec = [], []

        for t in range(x.shape[0]):
            s, mc1 = self.lif_c1(self.pool1(self.conv1(x[t])), mc1)
            s, mc2 = self.lif_c2(self.pool2(self.conv2(s)),    mc2)
            s, mf1 = self.lif_f1(self.fc1(self.flat(s)),       mf1)
            s, mf2 = self.lif_f2(self.fc2(s),                  mf2)
            spk_rec.append(s)
            mem_rec.append(mf2)

        return torch.stack(spk_rec), torch.stack(mem_rec)


# ─────────────────────────────────────────────
# 2.  Rate encoding
# ─────────────────────────────────────────────

def rate_encode(images: torch.Tensor, timesteps: int) -> torch.Tensor:
    """
    Convert pixel intensities in [0,1] to Bernoulli spike trains.

    Parameters
    ----------
    images    : (B, C, H, W) normalised pixel tensor
    timesteps : number of simulation steps T

    Returns
    -------
    spikes : (T, B, C*H*W) or (T, B, C, H, W) depending on flatten flag
    """
    # Clamp to [0,1] after normalisation
    prob = images.clamp(0, 1)
    # Repeat T times and sample Bernoulli
    return torch.bernoulli(prob.unsqueeze(0).expand(timesteps, -1, -1, -1, -1))


# ─────────────────────────────────────────────
# 3.  Data
# ─────────────────────────────────────────────

def get_mnist_loaders(
    batch_size: int = 64,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    tf = transforms.Compose([transforms.ToTensor()])   # keep [0,1] for rate coding
    train_ds = datasets.MNIST(data_root, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(data_root, train=False, download=True, transform=tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


# ─────────────────────────────────────────────
# 4.  Training / evaluation helpers
# ─────────────────────────────────────────────

def _spike_count_loss(spk_rec: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on summed spike counts over T timesteps."""
    spike_counts = spk_rec.sum(dim=0)          # (B, num_classes)
    return nn.CrossEntropyLoss()(spike_counts, targets)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    timesteps: int,
    is_cnn: bool = False,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        spikes = rate_encode(x, timesteps).to(device)    # (T, B, 1, 28, 28)
        if not is_cnn:
            spikes = spikes.flatten(start_dim=2)         # (T, B, 784)

        optimizer.zero_grad()
        spk_rec, _ = model(spikes)
        loss = _spike_count_loss(spk_rec, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    timesteps: int,
    is_cnn: bool = False,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    latencies = []
    total_spikes = 0
    total_neurons_timesteps = 0

    for x, y in loader:
        x = x.to(device)
        spikes = rate_encode(x, timesteps).to(device)
        if not is_cnn:
            spikes = spikes.flatten(start_dim=2)

        t0 = time.perf_counter()
        spk_rec, _ = model(spikes)
        latencies.append((time.perf_counter() - t0) / x.size(0) * 1e3)

        spike_counts = spk_rec.sum(dim=0)              # (B, classes)
        preds = spike_counts.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

        # sparsity metrics
        total_spikes           += spk_rec.sum().item()
        total_neurons_timesteps += spk_rec.numel()

    acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1       = f1_score(all_labels, all_preds, average="macro")
    cm       = confusion_matrix(all_labels, all_preds)
    lat      = float(torch.tensor(latencies).mean().item())
    spike_rt = total_spikes / max(total_neurons_timesteps, 1)

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "latency_ms_per_sample": lat,
        "spike_rate": spike_rt,
        "timesteps": timesteps,
    }


# ─────────────────────────────────────────────
# 5.  Public entry-point
# ─────────────────────────────────────────────

def run_snn_baselines(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    timesteps: int = 25,
    device: torch.device | None = None,
    data_root: str = "./data",
) -> dict:
    """
    Train SpikingMLP and SpikingCNN on rate-coded MNIST.

    Returns
    -------
    dict with keys "SNN_MLP" and "SNN_CNN", each containing:
        accuracy, f1_macro, confusion_matrix, latency_ms_per_sample,
        spike_rate, timesteps, train_time_s, gpu_mem_mb, num_params
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_loaders(batch_size, data_root)
    results = {}

    for name, model, is_cnn in [
        ("SNN_MLP", SpikingMLP(), False),
        ("SNN_CNN", SpikingCNN(), True),
    ]:
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

        profiler = GPUProfiler(device_index=device.index or 0, poll_interval_s=0.05)
        with profiler:
            t_start = time.perf_counter()
            for ep in range(1, epochs + 1):
                loss = train_one_epoch(model, train_loader, optimizer, device, timesteps, is_cnn)
                print(f"  [{name}] epoch {ep}/{epochs}  loss={loss:.4f}")
            train_time = time.perf_counter() - t_start

        gpu_hw = profiler.metrics()
        profiler.print_summary(prefix=name)

        # FLOPs: pass a single rate-encoded sample (T, 1, 784) or (T, 1, 1, 28, 28)
        sample_img = next(iter(test_loader))[0][:1]
        spk_sample = rate_encode(sample_img, timesteps).to(device)
        if not is_cnn:
            spk_sample = spk_sample.flatten(start_dim=2)   # (T, 1, 784)
        flops_info = estimate_flops(model, spk_sample, device)

        metrics = evaluate(model, test_loader, device, timesteps, is_cnn)
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
            f"train={train_time:.1f}s  "
            f"energy={gpu_hw['energy_joules']:.2f}J\n"
        )

    return results


if __name__ == "__main__":
    res = run_snn_baselines(epochs=5, timesteps=25)
    for model_name, m in res.items():
        print(f"\n=== {model_name} ===")
        for k, v in m.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v}")
