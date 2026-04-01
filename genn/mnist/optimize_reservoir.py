import torch
import optuna
import mlflow
import os
import json
from dataclasses import asdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from reservoir_models import (
    ReservoirConfig,
    SparseConfig,
    ReservoirModel,
    ReservoirTrainer,
    set_seed,
)

CHOSEN_TOPOLOGY = "barabasi_albert"   # Lehetséges topológiák: "erdos_renyi", "watts_strogatz", "barabasi_albert"
STUDY_NAME = f"reservoir_{CHOSEN_TOPOLOGY}_optimalization"
N_TRIALS = 50 # iterációk száma
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # ha van elérhető GPU akkor azt használjuk

CHECKPOINT_DIR = f"checkpoints_{CHOSEN_TOPOLOGY}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def build_topology_params(trial, topology_type):
    topo_params = {}
    # Topológiák paramétereinek keresési tere
    if topology_type == "erdos_renyi":
        topo_params["p"] = trial.suggest_float("er_p", 0.001, 0.1, log=True)

    elif topology_type == "watts_strogatz":
        topo_params["k"] = trial.suggest_int("ws_k", 2, 20)
        topo_params["p"] = trial.suggest_float("ws_p", 0.01, 0.5)

    elif topology_type == "barabasi_albert":
        topo_params["m"] = trial.suggest_int("ba_m", 1, 10)

    else:
        raise ValueError(f"Unsupported topology: {topology_type}")

    return topo_params


def objective(trial):
    set_seed(42)

    topo_params = build_topology_params(trial, CHOSEN_TOPOLOGY)
    # Paraméterek keresési terének beállítása
    res_cfg = ReservoirConfig(
        size=trial.suggest_int("size", 500, 2000, step=500),
        spectral_radius=trial.suggest_float("spectral_radius", 0.8, 1.5),
        leak_rate=trial.suggest_float("leak_rate", 0.1, 0.9),
        topology_type=CHOSEN_TOPOLOGY,
        topology_params=topo_params,
        feedback_size=28,
    )

    sparse_cfg = SparseConfig(
    reservoir_dim=res_cfg.size,
    sparse_dim=trial.suggest_int("sparse_dim", 1024, 4096, step=1024),
    sh_lambda=trial.suggest_float("sh_lambda", 0.01, 0.2),
    l1_alpha=trial.suggest_float("l1_alpha", 1e-4, 1e-2, log=True),
    lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
    epochs=15,
    early_stopping_patience=4,
    early_stopping_min_delta=1e-4,
    restore_best_weights=True,
)

    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        mlflow.log_param("topology_type", CHOSEN_TOPOLOGY)
        mlflow.log_params(topo_params)

        res_dict = asdict(res_cfg)
        sparse_dict = asdict(sparse_cfg)

        mlflow.log_params(res_dict)
        mlflow.log_params(sparse_dict)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_loader = DataLoader(
            datasets.MNIST("./data", train=True, download=True, transform=transform),
            batch_size=512,
            shuffle=True,
        )
        test_loader = DataLoader(
            datasets.MNIST("./data", train=False, download=True, transform=transform),
            batch_size=1000,
            shuffle=False,
        )

        model = ReservoirModel(res_cfg, sparse_cfg)
        trainer = ReservoirTrainer(model, sparse_cfg, DEVICE)

        trainer.train(train_loader, test_loader)
        metrics = trainer.evaluate(test_loader)

        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1"],
            "recall": metrics["recall"]
        })
        accuracy = metrics["accuracy"]

        is_best = (trial.number == 0) or (accuracy > study.best_value)
        if is_best:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            params_path = os.path.join(CHECKPOINT_DIR, "best_params.json")

            # Modell paraméterek elmentése
            torch.save(model.state_dict(), checkpoint_path)

            best_payload = {
                "topology_type": CHOSEN_TOPOLOGY,
                "metrics": metrics,
                "trial_number": int(trial.number),
                "res_cfg": res_dict,
                "sparse_cfg": sparse_dict,
            }

            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2)

            stable_weight_map = {
                "watts_strogatz": "reservoir_ws_weights.pth",
                "barabasi_albert": "reservoir_ba_weights.pth",
                "erdos_renyi": "reservoir_er_weights.pth",
            }
            stable_json_map = {
                "watts_strogatz": "reservoir_ws_params.json",
                "barabasi_albert": "reservoir_ba_params.json",
                "erdos_renyi": "reservoir_er_params.json",
            }

            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, stable_weight_map[CHOSEN_TOPOLOGY])
            )
            with open(os.path.join(CHECKPOINT_DIR, stable_json_map[CHOSEN_TOPOLOGY]), "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2)

        return accuracy


if __name__ == "__main__":
    mlflow.set_experiment(STUDY_NAME)
    study = optuna.create_study(direction="maximize")

    print(f"Starting separate optimization for: {CHOSEN_TOPOLOGY}")
    study.optimize(objective, n_trials=N_TRIALS)

    with mlflow.start_run(run_name=f"FINAL_SUMMARY_{CHOSEN_TOPOLOGY}"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_accuracy", study.best_value)

    print(f"\n--- {CHOSEN_TOPOLOGY} Results ---")
    print(f"Best Accuracy: {study.best_value * 100:.4f}%")