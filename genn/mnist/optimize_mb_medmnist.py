import optuna
import mlflow
import numpy as np
import shutil
import os
import json 
from mushroom_body_class import MushroomBodyModel, MBSimulator, MBConfig, load_mnist, load_medmnist

# ==========================================
# 1. SET YOUR TARGET DATASET HERE
# Options: "breastmnist", "organamnist"
# ==========================================
TARGET_DATASET = "breastmnist" 

STUDY_NAME = f"mushroom_body_{TARGET_DATASET}"
N_TRIALS = 150 
CHECKPOINT_DIR = f"checkpoints_mb_{TARGET_DATASET}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def objective(trial):
    # Broad search space for new, unexplored datasets
    params_medmnist = {
        "PRESENT_TIME_MS": trial.suggest_float("PRESENT_TIME_MS", 10.0, 50.0),
        "INPUT_SCALE": trial.suggest_float("INPUT_SCALE", 30.0, 150.0), 
        "NUM_KC": trial.suggest_int("NUM_KC", 10000, 50000, step=5000),
        "PN_KC_FAN_IN": trial.suggest_int("PN_KC_FAN_IN", 10, 40), 
        "PN_KC_WEIGHT": trial.suggest_float("PN_KC_WEIGHT", 0.05, 0.4),
        "PN_KC_TAU": trial.suggest_float("PN_KC_TAU", 2.0, 15.0),
        "PN_REFRAC": trial.suggest_float("PN_REFRAC", 30.0, 150.0),
        "Vthresh": trial.suggest_float("Vthresh", -58.0, -40.0),
        "TauM": trial.suggest_float("TauM", 10.0, 40.0),
        "KC_GGN_WEIGHT": trial.suggest_float("KC_GGN_WEIGHT", 0.5, 3.0),
        "GGN_KC_WEIGHT": trial.suggest_float("GGN_KC_WEIGHT", -15.0, -2.0),
        "GGN_KC_TAU": trial.suggest_float("GGN_KC_TAU", 2.0, 10.0),
        "KC_MBON_TAU": trial.suggest_float("KC_MBON_TAU", 1.0, 8.0),
        "MBON_STIMULUS_CURRENT": trial.suggest_float("MBON_STIMULUS_CURRENT", 1.0, 12.0),
        "eta": trial.suggest_float("eta", 1e-6, 1e-3, log=True), 
        "tauE": trial.suggest_float("tauE", 50.0, 600.0),
        "rho": trial.suggest_float("rho", 0.0005, 0.01),
        "wMax": trial.suggest_float("wMax", 0.01, 0.1),
    }

    # Frissítés az MBConfigban
    MBConfig.PRESENT_TIME_MS = params_medmnist["PRESENT_TIME_MS"]
    MBConfig.INPUT_SCALE = params_medmnist["INPUT_SCALE"]
    MBConfig.NUM_KC = params_medmnist["NUM_KC"]
    MBConfig.PN_KC_FAN_IN = params_medmnist["PN_KC_FAN_IN"]
    MBConfig.LIF_PARAMS["Vthresh"] = params_medmnist["Vthresh"]
    MBConfig.LIF_PARAMS["TauM"] = params_medmnist["TauM"]
    MBConfig.PN_REFRAC = params_medmnist["PN_REFRAC"]
    MBConfig.PN_KC_WEIGHT = params_medmnist["PN_KC_WEIGHT"]
    MBConfig.PN_KC_TAU = params_medmnist["PN_KC_TAU"]
    MBConfig.KC_GGN_WEIGHT = params_medmnist["KC_GGN_WEIGHT"]
    MBConfig.GGN_KC_WEIGHT = params_medmnist["GGN_KC_WEIGHT"]
    MBConfig.GGN_KC_TAU = params_medmnist["GGN_KC_TAU"]
    MBConfig.KC_MBON_TAU = params_medmnist["KC_MBON_TAU"]
    MBConfig.MBON_STIMULUS_CURRENT = params_medmnist["MBON_STIMULUS_CURRENT"]
    MBConfig.KC_MBON_PARAMS.update({
        "eta": params_medmnist["eta"], "tauE": params_medmnist["tauE"], 
        "rho": params_medmnist["rho"], "wMax": params_medmnist["wMax"]
    })

    model_name = f"trial_{trial.number}"
    
    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        mlflow.log_params(params_medmnist)
        
        try:
            # MedMNIST betöltése a beállított változó alapján
            train_imgs, train_labels, test_imgs, test_labels, num_classes = load_medmnist(TARGET_DATASET)
            MBConfig.NUM_MBON = num_classes
            
            train_model = MushroomBodyModel(name=f"{model_name}_train", is_training=True)
            train_model.build_and_load()
            trainer = MBSimulator(train_model)
            trainer.train(train_imgs, train_labels)

            trainer.model_wrapper.kc_mbon.vars["g"].pull_from_device()
            trained_weights = np.copy(trainer.model_wrapper.kc_mbon.vars["g"].view)
            trainer.model_wrapper.pn_kc.pull_connectivity_from_device()
            trained_indices = np.vstack((
                trainer.model_wrapper.pn_kc.get_sparse_pre_inds(), 
                trainer.model_wrapper.pn_kc.get_sparse_post_inds()
            ))

            # Teszt fázis
            eval_model = MushroomBodyModel(
                name=f"{model_name}_eval",
                is_training=False,
                sparse_indices=trained_indices,
                kc_mbon_g=trained_weights,
            )
            eval_model.build_and_load()
            evaluator = MBSimulator(eval_model)
            accuracy, f1, recall = evaluator.evaluate(test_imgs, test_labels)
            
            mlflow.log_metrics({
                "accuracy": accuracy,
                "f1_score": f1,
                "recall": recall
            })

            w_file = f"weights_trial_{trial.number}.npy"
            i_file = f"indices_trial_{trial.number}.npy"
            np.save(w_file, trained_weights)
            np.save(i_file, trained_indices)
            mlflow.log_artifact(w_file)
            mlflow.log_artifact(i_file)

            is_best = False
            try:
                if accuracy > study.best_value:
                    is_best = True
            except ValueError:
                is_best = True

            # Eddigi legjobb paraméterek mentése a dedikált mappába
            if is_best:
                np.save(os.path.join(CHECKPOINT_DIR, f"{TARGET_DATASET}_best_weights.npy"), trained_weights)
                np.save(os.path.join(CHECKPOINT_DIR, f"{TARGET_DATASET}_best_indices.npy"), trained_indices)
                mlflow.log_dict(params_medmnist, f"{TARGET_DATASET}_best_params.json") 

            mlflow.log_metric("accuracy", accuracy)
            return accuracy

        finally:
            shutil.rmtree(f"{model_name}_train_CODE", ignore_errors=True)
            shutil.rmtree(f"{model_name}_eval_CODE", ignore_errors=True)

if __name__ == "__main__":
    mlflow.set_experiment(STUDY_NAME)
    
    study = optuna.create_study(direction="maximize")
    
    print(f"Starting search across {N_TRIALS} trials on {TARGET_DATASET}.")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*30)
    print(f"BEST ACCURACY ON {TARGET_DATASET.upper()}: {study.best_value:.4f}%")

    # Legjobb paraméterek kiíratása jsonbe
    best_params_path = os.path.join(CHECKPOINT_DIR, f"{TARGET_DATASET}_best_params.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)

    print("BEST PARAMETERS:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")