import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from reservoir_models import (
    ReservoirConfig,
    SparseConfig,
    ReservoirModel,
    ReservoirTrainer
)

MODEL_WEIGHT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_watts_strogatz")

def load_best_model():
    # Modell fájlok beolvasása
    best_params_path = os.path.join(MODEL_WEIGHT_DIR, "best_params.json")
    best_model_path = os.path.join(MODEL_WEIGHT_DIR, "best_model.pt")

    if not os.path.exists(best_params_path):
        raise FileNotFoundError(f"Missing file: {best_params_path}")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Missing file: {best_model_path}")

    # Legjobb paraméterek beolvasása json fájlból
    with open(best_params_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    print("Loaded best params:")
    print(json.dumps(payload, indent=2))

    res_cfg = ReservoirConfig(**payload["res_cfg"])
    sparse_cfg = SparseConfig(**payload["sparse_cfg"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ReservoirModel(res_cfg, sparse_cfg)
    trainer = ReservoirTrainer(model, sparse_cfg, device)

    trainer.load_weights(best_model_path)

    print("\nLoaded model weights from:", best_model_path)

    return trainer, payload["res_cfg"]["topology_type"]
def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    trainer, topology_name = load_best_model()

    topology_map = {
        "watts_strogatz": "ws",
        "barabasi_albert": "ba",
        "erdos_renyi": "er"
    }
    model_tag = topology_map.get(topology_name, "er")

    _, test_loader = load_mnist()

    metrics = trainer.evaluate(test_loader)
    accuracy = metrics["accuracy"]
    print(f"\nMNIST test accuracy: {accuracy:.4f}")

    latent_data = trainer.save_latent_indices_with_images(
        test_loader, 
        model_type=model_tag, 
        topk=64
    )

    latent_save_path = os.path.join(MODEL_WEIGHT_DIR, f"{model_tag}_latent_data.pt")
    torch.save(latent_data, latent_save_path)

    print(f"\nSaved latent test data with signature '{model_tag}' to: {latent_save_path}")
    print("Saved keys:", list(latent_data.keys()))
    print("Internal model_type:", latent_data.get("model_type"))
    print("vectors shape:", latent_data["vectors"].shape)
    print("labels shape:", latent_data["labels"].shape)
    print("images shape:", latent_data["images"].shape)