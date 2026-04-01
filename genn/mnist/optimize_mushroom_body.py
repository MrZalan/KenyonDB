import optuna
import mlflow
import numpy as np
import shutil
import os
from mushroom_body_class import MushroomBodyModel, MBSimulator, MBConfig, load_mnist

STUDY_NAME = "mushroom_body_full_search_round_3"
N_TRIALS = 150 # iterációk száma

def objective(trial):
    # Keresési tér meghatározása a paraméterekhez
    params_r1 = {
        "PRESENT_TIME_MS": trial.suggest_float("PRESENT_TIME_MS", 10.0, 40.0),
        "INPUT_SCALE": trial.suggest_float("INPUT_SCALE", 50.0, 120.0),
        "NUM_KC": trial.suggest_int("NUM_KC", 10000, 40000, step=5000),
        "PN_KC_FAN_IN": trial.suggest_int("PN_KC_FAN_IN", 10, 30),
        "Vthresh": trial.suggest_float("Vthresh", -55.0, -45.0),
        "TauM": trial.suggest_float("TauM", 10.0, 30.0),
        "PN_REFRAC": trial.suggest_float("PN_REFRAC", 50.0, 150.0),
        "PN_KC_WEIGHT": trial.suggest_float("PN_KC_WEIGHT", 0.05, 0.5),
        "PN_KC_TAU": trial.suggest_float("PN_KC_TAU", 1.0, 10.0),
        "KC_GGN_WEIGHT": trial.suggest_float("KC_GGN_WEIGHT", 0.1, 2.0),
        "GGN_KC_WEIGHT": trial.suggest_float("GGN_KC_WEIGHT", -10.0, -1.0),
        "GGN_KC_TAU": trial.suggest_float("GGN_KC_TAU", 2.0, 10.0),
        "KC_MBON_TAU": trial.suggest_float("KC_MBON_TAU", 1.0, 10.0),
        "MBON_STIMULUS_CURRENT": trial.suggest_float("MBON_STIMULUS_CURRENT", 1.0, 10.0),
        "eta": trial.suggest_float("eta", 1e-6, 1e-4, log=True),
        "tauE": trial.suggest_float("tauE", 50.0, 500.0),
        "rho": trial.suggest_float("rho", 0.001, 0.05),
        "wMax": trial.suggest_float("wMax", 0.01, 0.05),
    }

    # Mlflow adatok alapján szűkített keresési tér második futtatáshoz
    params_r2 = {
        "PRESENT_TIME_MS": trial.suggest_float("PRESENT_TIME_MS", 20.0, 35.0),
        "INPUT_SCALE": trial.suggest_float("INPUT_SCALE", 85.0, 100.0),
        "NUM_KC": trial.suggest_int("NUM_KC", 30000, 50000, step=5000),
        "PN_KC_FAN_IN": trial.suggest_int("PN_KC_FAN_IN", 18, 22),
        "Vthresh": trial.suggest_float("Vthresh", -53.0, -48.0),
        "TauM": trial.suggest_float("TauM", 25.0, 32.0),
        "PN_REFRAC": trial.suggest_float("PN_REFRAC", 65.0, 95.0),
        "PN_KC_WEIGHT": trial.suggest_float("PN_KC_WEIGHT", 0.1, 0.2),
        "PN_KC_TAU": trial.suggest_float("PN_KC_TAU", 2.0, 5.0),
        "KC_GGN_WEIGHT": trial.suggest_float("KC_GGN_WEIGHT", 1.8, 2.2),
        "GGN_KC_WEIGHT": trial.suggest_float("GGN_KC_WEIGHT", -9.0, -6.0),
        "GGN_KC_TAU": trial.suggest_float("GGN_KC_TAU", 6.5, 8.5),
        "KC_MBON_TAU": trial.suggest_float("KC_MBON_TAU", 4.5, 6.0),
        "MBON_STIMULUS_CURRENT": trial.suggest_float("MBON_STIMULUS_CURRENT", 2.5, 5.5),
        "eta": trial.suggest_float("eta", 1e-6, 5e-5, log=True),
        "tauE": trial.suggest_float("tauE", 200.0, 450.0),
        "rho": trial.suggest_float("rho", 0.001, 0.005),
        "wMax": trial.suggest_float("wMax", 0.03, 0.045),
    }

    params_r3 = {
        "PRESENT_TIME_MS": trial.suggest_float("PRESENT_TIME_MS", 15.0, 25.0),
        "INPUT_SCALE": trial.suggest_float("INPUT_SCALE", 70.0, 95.0),
        "NUM_KC": trial.suggest_int("NUM_KC", 35000, 45000, step=5000),
        "PN_KC_FAN_IN": trial.suggest_int("PN_KC_FAN_IN", 10, 32), 
        "PN_KC_WEIGHT": trial.suggest_float("PN_KC_WEIGHT", 0.08, 0.25),
        "PN_KC_TAU": trial.suggest_float("PN_KC_TAU", 6.5, 9.0),
        "PN_REFRAC": trial.suggest_float("PN_REFRAC", 50.0, 120.0),
        "Vthresh": trial.suggest_float("Vthresh", -50.0, -45.0),
        "TauM": trial.suggest_float("TauM", 13.0, 25.0),
        "KC_GGN_WEIGHT": trial.suggest_float("KC_GGN_WEIGHT", 1.4, 2.0),
        "GGN_KC_WEIGHT": trial.suggest_float("GGN_KC_WEIGHT", -11.0, -6.0),
        "GGN_KC_TAU": trial.suggest_float("GGN_KC_TAU", 2.0, 7.0),
        "KC_MBON_TAU": trial.suggest_float("KC_MBON_TAU", 1.0, 4.0),
        "MBON_STIMULUS_CURRENT": trial.suggest_float("MBON_STIMULUS_CURRENT", 3.0, 9.5),
        "eta": trial.suggest_float("eta", 1e-6, 4e-5, log=True),
        "tauE": trial.suggest_float("tauE", 80.0, 500.0),
        "rho": trial.suggest_float("rho", 0.001, 0.005),
        "wMax": trial.suggest_float("wMax", 0.03, 0.05),
    }

    # Paraméterek frissítése az MBConfigban
    MBConfig.PRESENT_TIME_MS = params_r3["PRESENT_TIME_MS"]
    MBConfig.INPUT_SCALE = params_r3["INPUT_SCALE"]
    MBConfig.NUM_KC = params_r3["NUM_KC"]
    MBConfig.PN_KC_FAN_IN = params_r3["PN_KC_FAN_IN"]
    MBConfig.LIF_PARAMS["Vthresh"] = params_r3["Vthresh"]
    MBConfig.LIF_PARAMS["TauM"] = params_r3["TauM"]
    MBConfig.PN_REFRAC = params_r3["PN_REFRAC"]
    MBConfig.PN_KC_WEIGHT = params_r3["PN_KC_WEIGHT"]
    MBConfig.PN_KC_TAU = params_r3["PN_KC_TAU"]
    MBConfig.KC_GGN_WEIGHT = params_r3["KC_GGN_WEIGHT"]
    MBConfig.GGN_KC_WEIGHT = params_r3["GGN_KC_WEIGHT"]
    MBConfig.GGN_KC_TAU = params_r3["GGN_KC_TAU"]
    MBConfig.KC_MBON_TAU = params_r3["KC_MBON_TAU"]
    MBConfig.MBON_STIMULUS_CURRENT = params_r3["MBON_STIMULUS_CURRENT"]
    MBConfig.KC_MBON_PARAMS.update({
        "eta": params_r3["eta"], "tauE": params_r3["tauE"], 
        "rho": params_r3["rho"], "wMax": params_r3["wMax"]
    })

    model_name = f"trial_{trial.number}"
    
    with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
        mlflow.log_params(params_r3)
        
        try:
            # Tréning fázis
            train_imgs, train_labels, test_imgs, test_labels = load_mnist()

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

            try:
                if accuracy > study.best_value:
                    is_best = True
            except ValueError:
                is_best = True

            # Eddigi legjobb paraméterek mentése
            if is_best:
                np.save("best_weights.npy", trained_weights)
                np.save("best_indices.npy", trained_indices)
                mlflow.log_dict(params, "best_params.json")

            mlflow.log_metric("accuracy", accuracy)
            return accuracy

        finally:
            # Modellekhez generált c++ és cuda fájlok feltakarítása
            shutil.rmtree(f"{model_name}_train_CODE", ignore_errors=True)
            shutil.rmtree(f"{model_name}_eval_CODE", ignore_errors=True)

if __name__ == "__main__":
    mlflow.set_experiment(STUDY_NAME)
    
    study = optuna.create_study(direction="maximize")
    
    print(f"Starting search across {N_TRIALS} trials on full MNIST.")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*30)
    print(f"BEST ACCURACY: {study.best_value:.4f}%")

    # Legjobb paraméterek kiíratása jsonbe
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)

    print("BEST PARAMETERS:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")