import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resdag.layers import ReservoirLayer
from typing import Optional


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class WSResConfig:
    reservoir_size: int = 1000
    feedback_size: int = 28
    spectral_radius: float = 1.3
    leak_rate: float = 0.35
    ws_k: int = 6
    ws_p: float = 0.1
    ws_seed: int = 42


@dataclass
class SparseConfig:
    reservoir_dim: int = 1000
    sparse_dim: int = 2048
    sh_lambda: float = 0.08
    l1_alpha: float = 2e-3
    lr: float = 2e-3
    epochs: int = 10


class SoftShrink(nn.Module):
    def __init__(self, sh_lambda: float):
        super().__init__()
        self.sh_lambda = sh_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.relu(x.abs() - self.sh_lambda)


class SparseHead(nn.Module):
    def __init__(self, res_dim: int, sparse_dim: int, num_classes: int = 10, sh_lambda: float = 0.08):
        super().__init__()
        self.enc = nn.Linear(res_dim, sparse_dim)
        self.shrink = SoftShrink(sh_lambda)
        self.norm = nn.LayerNorm(sparse_dim)
        self.clf = nn.Linear(sparse_dim, num_classes)

    def forward(self, r: torch.Tensor):
        s_raw = self.shrink(self.enc(r))
        s_norm = self.norm(s_raw)
        logits = self.clf(s_norm)
        return logits, s_raw


@torch.no_grad()
def top_k_indices(t: torch.Tensor, k: int = 64) -> torch.Tensor:
    k = min(k, t.size(1))
    _, ind = t.abs().topk(k, dim=1)
    return ind.to(torch.int64)

class MergedModel(nn.Module):
    def __init__(
        self,
        res_config: WSResConfig,
        sparse_config: SparseConfig,
        num_classes: int = 10,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        dev = device if device is not None else torch.device("cpu")

        self.reservoir = ReservoirLayer(
            reservoir_size=res_config.reservoir_size,
            feedback_size=res_config.feedback_size,
            input_size=0,
            topology=("watts_strogatz", {"k": res_config.ws_k, "p": res_config.ws_p, "seed": res_config.ws_seed}),
            spectral_radius=res_config.spectral_radius,
            leak_rate=res_config.leak_rate,
        ).to(dev)

        for param in self.reservoir.parameters():
            param.requires_grad_(False)

        self.head = SparseHead(
            res_dim=sparse_config.reservoir_dim,
            sparse_dim=sparse_config.sparse_dim,
            num_classes=num_classes,
            sh_lambda=sparse_config.sh_lambda,
        ).to(dev)

    @torch.no_grad()
    def reservoir_features(self, rows: torch.Tensor) -> torch.Tensor:
        out = self.reservoir(rows)
        if out.dim() == 3:
            x = out[:, -1, :]
        elif out.dim() == 2:
            x = out
        else:
            raise RuntimeError(f"Unexpected reservoir output shape: {tuple(out.shape)}")
        return x

    def forward(self, rows: torch.Tensor):
        with torch.no_grad():
            x = self.reservoir_features(rows)
        logits, s_raw = self.head(x)
        return logits, s_raw

def train(
    model: MergedModel,
    sparse_config: SparseConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
):
    model.train()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=sparse_config.lr, weight_decay=1e-4)

    for epoch in range(1, sparse_config.epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            rows = x.squeeze(1)

            logits, s_raw = model(rows)

            loss = F.cross_entropy(logits, y)
            loss = loss + sparse_config.l1_alpha * s_raw.abs().mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    return model


@torch.no_grad()
def evaluate(model: MergedModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        rows = x.squeeze(1)
        logits, _ = model(rows)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    model.train()
    return correct / total


@torch.no_grad()
def save_latent_indices(model: MergedModel, loader: DataLoader, device: torch.device, topk: int = 64):
    model.eval()
    all_ind = []
    all_y = []

    for x, y in loader:
        x = x.to(device)
        rows = x.squeeze(1)
        _, s_raw = model(rows)

        ind = top_k_indices(s_raw, k=topk)
        all_ind.append(ind.cpu())
        all_y.append(y.cpu())

    model.train()
    return torch.cat(all_ind, 0), torch.cat(all_y, 0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    topk = 64
    print(f"Device: {device}")
    set_seed(seed)

    res_config = WSResConfig(
        reservoir_size=1000,
        feedback_size=28,
        spectral_radius=1.3,
        leak_rate=0.35,
        ws_k=6,
        ws_p=0.1,
        ws_seed=seed,
    )

    sparse_config = SparseConfig(
        reservoir_dim=res_config.reservoir_size,
        sparse_dim=2048,
        sh_lambda=0.08,
        l1_alpha=2e-3,
        lr=2e-3,
        epochs=10,
    )

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set, batch_size=512, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
    )

    model = MergedModel(res_config, sparse_config, num_classes=10, device=device)
    model = train(model, sparse_config, train_loader, test_loader, device)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "reservoir_cfg": res_config.__dict__,
            "sparse_cfg": sparse_config.__dict__,
            "head_state": model.head.state_dict(),
            "reservoir_state": model.reservoir.state_dict(),
        },
        "checkpoints/reservoir_ws_mnist.pt",
    )
    print("Saved: checkpoints/reservoir_ws_mnist.pt")

    os.makedirs("latents", exist_ok=True)

    train_ind, train_y = save_latent_indices(
        model, DataLoader(train_set, batch_size=512, shuffle=False), device, topk=topk
    )
    test_ind, test_y = save_latent_indices(
        model, DataLoader(test_set, batch_size=512, shuffle=False), device, topk=topk
    )

    torch.save(train_ind, "latents/train_sparse_ind.pt")
    torch.save(train_y, "latents/train_labels.pt")
    torch.save(test_ind, "latents/test_sparse_ind.pt")
    torch.save(test_y, "latents/test_labels.pt")

    print("Saved sparse latents (indices-only):")
    print("latents/train_sparse_ind.pt", train_ind.shape)
    print("latents/train_labels.pt", train_y.shape)
    print("latents/test_sparse_ind.pt", test_ind.shape)
    print("latents/test_labels.pt", test_y.shape)


if __name__ == "__main__":
    main()