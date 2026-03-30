import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import cv2
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from resdag.layers import ReservoirLayer
from typing import Optional, Dict, Any
from torchvision import transforms
from copy import deepcopy
from sklearn.metrics import f1_score, recall_score


def set_seed(seed: int = 42):
    """Mivel több ponton is kell random generálnunk, definiáljuk a seedet"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class ReservoirConfig:
    """A reservoir modell paraméterei"""
    size: int = 1000 # reservoir mag mérete
    feedback_size: int = 28 # visszacsatoló kapcsolatok erőssége
    spectral_radius: float = 1.3 # a belső súlymátrix legnagyobb abszolút sajátértéke, a hálózat memóriáját szabályozza
    leak_rate: float = 0.35 # új értékek milyen gyorsan frissítsék a hálózat neuronjait
    topology_type: str = "watts_strogatz" # reservoir mag gráf struktúrája
    topology_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SparseConfig:
    """Sparse Head és tréning paraméterei"""
    reservoir_dim: int = 1000 # a reservoir mag mérete
    sparse_dim: int = 2048 # a hidden layer dimenziója, nagyobb mint a reservoir mag mérete
    sh_lambda: float = 0.08 # határszint a SoftShrink aktivizációs függvényhez, ezen abszolút érték alatt nulláza az adott jelet
    l1_alpha: float = 2e-3 # L1 regularizáció együtthatója
    lr: float = 2e-3 # learning rate, milyen gyorsan tanul a modell
    epochs: int = 10 # hányszor iteráljon végig a teljes adathalmazon

    # Early stopping paraméterei, tréning terminálása ha egy ideje már nem javul a modell
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    restore_best_weights: bool = True


class SoftShrink(nn.Module):
    """SoftShrink aktivációs függvény definiálása"""
    def __init__(self, sh_lambda: float):
        super().__init__()
        self.sh_lambda = sh_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # levonja a lambdát az abszolút értékből és kinulláza a negatív értékeket (ReLu), majd visszakapja az eredeti előjelét
        return torch.sign(x) * torch.relu(x.abs() - self.sh_lambda)

class SparseHead(nn.Module):
    """A modell tanítható osztályozó feje"""
    def __init__(self, cfg: SparseConfig, num_classes: int = 10):
        super().__init__()
        self.enc = nn.Linear(cfg.reservoir_dim, cfg.sparse_dim) # reservoir dimenziója -> sparse fej dimenziója
        self.shrink = SoftShrink(cfg.sh_lambda) # ritkítás, csak a legfontosabb aktivációk megtartása
        self.norm = nn.LayerNorm(cfg.sparse_dim) # normalizálás
        self.clf = nn.Linear(cfg.sparse_dim, num_classes) # osztályozó réteg

    def forward(self, r: torch.Tensor):
        # r: a reservoir kimenete
        s_raw = self.shrink(self.enc(r)) # enkódolás és ritkítás
        s_norm = self.norm(s_raw) # normalizálás
        logits = self.clf(s_norm) # osztályozás
        return logits, s_raw

class ReservoirModel(nn.Module):
    """A reservoir mag és Sparse fej összefűzése"""
    def __init__(self, res_cfg: ReservoirConfig, sparse_cfg: SparseConfig, num_classes: int = 10):
        super().__init__()

        # Reservoir réteg inicializálása
        self.reservoir = ReservoirLayer(
            reservoir_size=res_cfg.size,
            feedback_size=res_cfg.feedback_size,
            input_size=0,
            topology=(res_cfg.topology_type, res_cfg.topology_params),
            spectral_radius=res_cfg.spectral_radius,
            leak_rate=res_cfg.leak_rate,
        )

        # Reservoir súlyok befagyasztása
        for param in self.reservoir.parameters():
            param.requires_grad_(False)

        # Tanítható fej hozzáadása
        self.head = SparseHead(sparse_cfg, num_classes)

    def forward(self, rows: torch.Tensor):
        # Reservoir futtatása
        with torch.no_grad():
            out = self.reservoir(rows)
            x = out[:, -1, :] if out.dim() == 3 else out

        return self.head(x)


class ReservoirTrainer:
    """Modell tréningelése, tesztelése, adatok kimentése"""
    def __init__(self, model: ReservoirModel, sparse_config: SparseConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = sparse_config
        self.device = device

        # Optimalizáció a tanítható fejhez
        self.optimizer = torch.optim.AdamW(
            self.model.head.parameters(),
            lr=self.cfg.lr,
            weight_decay=1e-4
        )

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        """Tréning ciklus L1 regularizációval és early stopinggal"""
        best_test_acc = -1.0
        best_epoch = 0
        epochs_without_improvement = 0
        best_state_dict = None

        self.model.train()

        for epoch in range(1, self.cfg.epochs + 1):
            total_loss, correct, total = 0.0, 0, 0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                rows = x.squeeze(1)

                logits, s_raw = self.model(rows)

                # Cross-entropy + L1 regularizáció
                loss = F.cross_entropy(logits, y)
                loss += self.cfg.l1_alpha * s_raw.abs().mean()

                # Gradiensek nullázása, backpropagation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * x.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += x.size(0)

            # Epoch végén validáció
            metrics = self.evaluate(test_loader)
            test_acc = metrics["accuracy"]
            train_loss = total_loss / total
            train_acc = correct / total

            print(
                f"Epoch {epoch:02d} | "
                f"Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f}% | "
                f"Test Acc: {test_acc:.4f}% | "
                f"F1: {metrics['f1']:.4f} | "
                f"Recall: {metrics['recall']:.4f}"
            )

            # Early stopping mechanizmusa
            improved = test_acc > (best_test_acc + self.cfg.early_stopping_min_delta)

            if improved:
                best_test_acc = test_acc
                best_metrics = metrics
                best_epoch = epoch
                epochs_without_improvement = 0
                if self.cfg.restore_best_weights:
                    best_state_dict = deepcopy(self.model.state_dict()) # jelenlegi legjobb állapot elmentése
            else:
                epochs_without_improvement += 1

            # Tréning leállítása ha sok ideje nem javultak az eredmények
            if epochs_without_improvement >= self.cfg.early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch: {best_epoch}, best test_acc: {best_test_acc:.4f}"
                )
                break

        if self.cfg.restore_best_weights and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(f"Restored best model weights from epoch {best_epoch}")

        return {
            "best_metrics": best_metrics,
            "best_epoch": best_epoch,
            "stopped_early": epochs_without_improvement >= self.cfg.early_stopping_patience,
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Tesztelési fázis"""
        self.model.eval()
        all_preds = []
        all_labels = []
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x.squeeze(1))
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        accuracy = (np.array(all_preds) == np.array(all_labels)).mean() * 100.0
        f1 = f1_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')

        print(f"Accuracy: {accuracy:.4f}%")
        print(f"F1-score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")

        self.model.train()
        return {
            "accuracy": accuracy,
            "f1": f1,
            "recall": recall
        }

    def load_weights(self, path: str):
        """Elmentett súlyok betöltése"""
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        self.model.eval()
        print(f"Weights loaded from {path}")

    @torch.no_grad()
    def inference(self, image_path: str, topk: int = 64):
        """Inference fázis egy képre és aktív neuronok kimentése"""
        self.model.eval()

        raw_img = cv2.imread(image_path, 0) # kép beolvasása OpenCV-vel, fekete-fehér formátumban
        
        if raw_img is None:
            raise ValueError(f"Error loading in the image for inference: {image_path}")

        # Bemeneti kép feldolgozása
        resized_img = cv2.resize(raw_img, (28, 28), interpolation=cv2.INTER_AREA) # átméretezés
        img_float = resized_img.astype(np.float32) / 255.0 # normalizálás
        img_normalized = (img_float - 0.1307) / 0.3081 # standardizálás
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).to(self.device) # tensorrá alakítás
        logits, s_raw = self.model(img_tensor)

        probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()

        _, top_indices = s_raw.abs().topk(min(topk, s_raw.size(1)), dim=1) # legaktívabb neuronok kiválasztása (k)

        return {
            "prediction": pred_label,
            "latent_indices": top_indices.squeeze(0).cpu().numpy(),
        }

    @torch.no_grad()
    def save_latent_indices_with_images(self, loader: DataLoader, model_type: str, topk: int = 64):
        """Latent vektorok kinyerése címkékkel és képekkel"""
        self.model.eval()
        all_ind, all_y, all_imgs = [], [], []

        for x, y in loader:
            _, s_raw = self.model(x.to(self.device).squeeze(1))

            # Indexek kinyerése
            k_val = min(topk, s_raw.size(1))
            _, ind = s_raw.abs().topk(k_val, dim=1)

            # 2. Képek visszaalakítása
            imgs = (x * 255).clamp(0, 255).to(torch.uint8)

            all_ind.append(ind.cpu())
            all_y.append(y.cpu())
            all_imgs.append(imgs.cpu())

        return {
            "vectors": torch.cat(all_ind, 0),
            "labels": torch.cat(all_y, 0),
            "images": torch.cat(all_imgs, 0),
            "model_type": model_type
        }