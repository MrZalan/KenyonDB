import optuna
import mlflow
import numpy as np
import shutil
import os
from mushroom_body_class import MushroomBodyModel, MBSimulator, MBConfig, _load_mnist

# --- Configuration ---
STUDY_NAME = "mushroom_body_full_search"
N_TRIALS = 100

def objective(trial):
    # 1. Suggest ALL Parameters from MBConfig (except fixed architecture sizes)
    params = {
        # Simulation & Input
        "PRESENT_TIME_MS": trial.suggest_float("PRESENT_TIME_MS", 10.0, 40.0),
        "INPUT_SCALE": trial.suggest_float("INPUT_SCALE", 50.0, 120.0),
        
        # Architecture / Connectivity
        "NUM_KC": trial.suggest_int("NUM_KC", 10000, 40000, step=5000),
        "PN_KC_FAN_IN": trial.suggest_int("PN_KC_FAN_IN", 10, 30),
        
        # Neuron Dynamics (LIF)
        "Vthresh": trial.suggest_float("Vthresh", -55.0, -45.0),
        "TauM": trial.suggest_float("TauM", 10.0, 30.0),
        "PN_REFRAC": trial.suggest_float("PN_REFRAC", 50.0, 150.0),
        
        # Synaptic Weights & Time Constants
        "PN_KC_WEIGHT": trial.suggest_float("PN_KC_WEIGHT", 0.05, 0.5),
        "PN_KC_TAU": trial.suggest_float("PN_KC_TAU", 1.0, 10.0),
        "KC_GGN_WEIGHT": trial.suggest_float("KC_GGN_WEIGHT", 0.1, 2.0),
        "GGN_KC_WEIGHT": trial.suggest_float("GGN_KC_WEIGHT", -10.0, -1.0),
        "GGN_KC_TAU": trial.suggest_float("GGN_KC_TAU", 2.0, 10.0),
        "KC_MBON_TAU": trial.suggest_float("KC_MBON_TAU", 1.0, 10.0),
        "MBON_STIMULUS_CURRENT": trial.suggest_float("MBON_STIMULUS_CURRENT", 1.0, 10.0),
        
        # R-STDP Learning Params
        "eta": trial.suggest_float("eta", 1e-6, 1e-4, log=True),
        "tauE": trial.suggest_float("tauE", 50.0, 500.0),
        "rho": trial.suggest_float("rho", 0.001, 0.05),
        "wMax": trial.suggest_float("wMax", 0.01, 0.05),
    }

    # 2. Update MBConfig dynamically
    MBConfig.PRESENT_TIME_MS = params["PRESENT_TIME_MS"]
    MBConfig.INPUT_SCALE = params["INPUT_SCALE"]
    MBConfig.NUM_KC = params["NUM_KC"]
    MBConfig.PN_KC_FAN_IN = params["PN_KC_FAN_IN"]
    MBConfig.LIF_PARAMS["Vthresh"] = params["Vthresh"]
    MBConfig.LIF_PARAMS["TauM"] = params["TauM"]
    MBConfig.PN_REFRAC = params["PN_REFRAC"]
    MBConfig.PN_KC_WEIGHT = params["PN_KC_WEIGHT"]
    MBConfig.PN_KC_TAU = params["PN_KC_TAU"]
    MBConfig.KC_GGN_WEIGHT = params["KC_GGN_WEIGHT"]
    MBConfig.GGN_KC_WEIGHT = params["GGN_KC_WEIGHT"]
    MBConfig.GGN_KC_TAU = params["GGN_KC_TAU"]
    MBConfig.KC_MBON_TAU = params["KC_MBON_TAU"]
    MBConfig.MBON_STIMULUS_CURRENT = params["MBON_STIMULUS_CURRENT"]
    MBConfig.KC_MBON_PARAMS.update({
        "eta": params["eta"], "tauE": params["tauE"], 
        "rho": params["rho"], "wMax": params["wMax"]
    })

    model_name = f"trial_{trial.number}"
    
    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        
        try:
            train_imgs, train_labels, test_imgs, test_labels = _load_mnist()

            # --- Training ---
            train_model = MushroomBodyModel(name=f"{model_name}_train", is_training=True)
            train_model.build_and_load()
            trainer = MBSimulator(train_model)
            trainer.train(train_imgs, train_labels)

            # Capture weights/indices
            trainer.mw.kc_mbon.vars["g"].pull_from_device()
            trained_weights = np.copy(trainer.mw.kc_mbon.vars["g"].view)
            trainer.mw.pn_kc.pull_connectivity_from_device()
            trained_indices = np.vstack((
                trainer.mw.pn_kc.get_sparse_pre_inds(), 
                trainer.mw.pn_kc.get_sparse_post_inds()
            ))

            # --- Evaluation ---
            eval_model = MushroomBodyModel(
                name=f"{model_name}_eval",
                is_training=False,
                sparse_indices=trained_indices,
                kc_mbon_g=trained_weights,
            )
            eval_model.build_and_load()
            evaluator = MBSimulator(eval_model)
            accuracy = evaluator.evaluate(test_imgs, test_labels)

            # --- Save Best Logic ---
            if trial.number == 0 or accuracy > study.best_value:
                np.save(f"best_weights_trial_{trial.number}.npy", trained_weights)
                np.save(f"best_indices_trial_{trial.number}.npy", trained_indices)

            mlflow.log_metric("accuracy", accuracy)
            return accuracy

        finally:
            # Clean up compiled C++/CUDA code folders
            shutil.rmtree(f"{model_name}_train_CODE", ignore_errors=True)
            shutil.rmtree(f"{model_name}_eval_CODE", ignore_errors=True)

if __name__ == "__main__":
    mlflow.set_experiment(STUDY_NAME)
    
    # In-memory study
    study = optuna.create_study(direction="maximize")
    
    print(f"Starting search across {N_TRIALS} trials on full MNIST.")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*30)
    print(f"BEST ACCURACY: {study.best_value:.2f}%")
    print("BEST PARAMETERS:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")