import torch
import optuna
import mlflow
import os
import json
import medmnist
from medmnist import INFO
from dataclasses import asdict
from torch.utils.data import DataLoader
from torchvision import transforms

from cnn_models import (
    CNNConfig,
    create_cnn_model,
    CNNTrainer,
    set_seed
)

TARGET_DATASET = "breastmnist" 

CHOSEN_MODEL = "vgg" 
STUDY_NAME = f"cnn_{CHOSEN_MODEL}_{TARGET_DATASET}"
N_TRIALS = 30 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = f"checkpoints_cnn_{CHOSEN_MODEL}_{TARGET_DATASET}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def objective(trial):
    set_seed(42)

    # --- MEDMNIST INFO EXTRACT ---
    info = INFO[TARGET_DATASET]
    num_classes = len(info['label'])
    # -----------------------------

    cfg = CNNConfig(
        model_type=CHOSEN_MODEL,
        embedding_dim=128, 
        num_classes=num_classes, 
        base_channels=trial.suggest_categorical("base_channels", [16, 32, 64]),
        dropout_rate=trial.suggest_float("dropout_rate", 0.0, 0.5),
        lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        epochs=30, 
        early_stopping_patience=5,
        early_stopping_min_delta=1e-4,
        restore_best_weights=True
    )

    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        mlflow.log_param("model_type", CHOSEN_MODEL)
        
        cfg_dict = asdict(cfg)
        mlflow.log_params(cfg_dict)

        DataClass = getattr(medmnist, info['python_class'])

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]) 
        ])

        target_transform = transforms.Lambda(lambda y: y[0])

        train_dataset = DataClass(split='train', transform=transform, target_transform=target_transform, download=True)
        test_dataset = DataClass(split='test', transform=transform, target_transform=target_transform, download=True)

        train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

        model = create_cnn_model(cfg)
        trainer = CNNTrainer(model, cfg, DEVICE)

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
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{TARGET_DATASET}_best_model.pt")
            params_path = os.path.join(CHECKPOINT_DIR, f"{TARGET_DATASET}_best_params.json")

            torch.save(model.state_dict(), checkpoint_path)

            best_payload = {
                "dataset": TARGET_DATASET,
                "model_type": CHOSEN_MODEL,
                "metrics": metrics,
                "trial_number": int(trial.number),
                "cnn_cfg": cfg_dict,
            }

            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2)

            stable_weight_name = f"cnn_{CHOSEN_MODEL}_{TARGET_DATASET}_weights.pth"
            stable_json_name = f"cnn_{CHOSEN_MODEL}_{TARGET_DATASET}_params.json"

            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, stable_weight_name)
            )
            with open(os.path.join(CHECKPOINT_DIR, stable_json_name), "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2)

        return accuracy


if __name__ == "__main__":
    mlflow.set_experiment(STUDY_NAME)
    study = optuna.create_study(direction="maximize")

    print(f"Starting separate optimization for: {CHOSEN_MODEL.upper()} on {TARGET_DATASET.upper()}")
    study.optimize(objective, n_trials=N_TRIALS)

    with mlflow.start_run(run_name=f"FINAL_SUMMARY_{CHOSEN_MODEL}"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_accuracy", study.best_value)

    print(f"\n--- {CHOSEN_MODEL.upper()} Results on {TARGET_DATASET.upper()} ---")
    print(f"Best Accuracy: {study.best_value:.4f}%")