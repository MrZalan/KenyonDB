import optuna
import subprocess
import os
import sys
import shutil
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    params = {
        # Architecture
        "num_kc": trial.suggest_int("num_kc", 20000, 50000, step=10000),
        "present_ms": trial.suggest_categorical("present_ms", [30.0, 40.0]),
        "input_scale": trial.suggest_float("input_scale", 75.0, 90.0),

        # Neuron Dynamics
        "lif_c": trial.suggest_float("lif_c", 0.22, 0.28),
        "lif_tau_m": trial.suggest_float("lif_tau_m", 20.0, 28.0),
        "lif_v_rest": -60.0,
        "lif_v_reset": -60.0,
        "lif_v_thresh": trial.suggest_float("lif_v_thresh", -53.0, -50.0),
        "lif_ioffset": trial.suggest_float("lif_ioffset", 0.0, 0.02),
        "lif_tau_refrac": trial.suggest_float("lif_tau_refrac", 2.0, 3.0),
        "pn_tau_refrac": trial.suggest_float("pn_tau_refrac", 85.0, 100.0),
        
        # Synaptic Weights & Connectivity
        "pn_kc_weight": trial.suggest_float("pn_kc_weight", 0.16, 0.22),
        "pn_kc_fan_in": trial.suggest_int("pn_kc_fan_in", 14, 20),
        "pn_kc_tau_syn": trial.suggest_float("pn_kc_tau_syn", 2.6, 3.0),
        
        # Inhibition
        "ggn_v_thresh": trial.suggest_float("ggn_v_thresh", 170.0, 195.0),
        "mbon_stimulus_current": trial.suggest_float("mbon_stimulus_current", 2.8, 3.8),
        "kc_mbon_tau_syn": trial.suggest_float("kc_mbon_tau_syn", 3.2, 3.8),
        
        # STDP Params
        "stdp_eta": trial.suggest_float("stdp_eta", 4e-5, 1e-4, log=True),
        "stdp_rho": trial.suggest_float("stdp_rho", 0.01, 0.025, log=True),
        "stdp_tau": trial.suggest_float("stdp_tau", 14.0, 18.0),
        "stdp_wMin": 0.0,
        "stdp_wMax": trial.suggest_float("stdp_wMax", 0.018, 0.025),
    }

    train_script = "my_model/train_params.py"
    eval_script = "my_model/eval_params.py"

    for folder in ["mnist_mb_training_objects", "mnist_mb_testing_objects"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    train_cmd = ["python3", train_script]
    for key, val in params.items():
        train_cmd.extend([f"--{key}", str(val)])
    
    print(f"\n[Trial {trial.number}] Starting Training...")
    try:
        subprocess.run(train_cmd, check=True, timeout=1800) 
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return 0.0

    eval_cmd = ["python3", eval_script]
    for key, val in params.items():
        eval_cmd.extend([f"--{key}", str(val)])
    
    print(f"[Trial {trial.number}] Starting Evaluation...")
    try:
        output = subprocess.check_output(eval_cmd, stderr=subprocess.STDOUT).decode()
        accuracy = float(output.split("Accuracy: ")[1].split("%")[0])
        print(f"[Trial {trial.number}] Resulting Accuracy: {accuracy}%")
        return accuracy
    except Exception:
        return 0.0

if __name__ == "__main__":
    mlflc = MLflowCallback(tracking_uri="file:./mlruns", metric_name="accuracy")
    study = optuna.create_study(direction="maximize", study_name="SNN_Optimization_v4_83PercentAnchor")
    
    study.enqueue_trial({
        "input_scale": 81.801,
        "num_kc": 30000,
        "present_ms": 30.0,
        "lif_c": 0.2459,
        "lif_tau_m": 23.82,
        "lif_v_rest": -60.0,
        "lif_v_reset": -60.0,
        "lif_v_thresh": -51.98,
        "lif_ioffset": 0.0094,
        "lif_tau_refrac": 2.49,
        "pn_tau_refrac": 91.01,
        "pn_kc_weight": 0.1868,
        "pn_kc_fan_in": 16,
        "pn_kc_tau_syn": 2.808,
        "ggn_v_thresh": 190.41,
        "mbon_stimulus_current": 3.21,
        "kc_mbon_tau_syn": 3.42,
        "stdp_eta": 4.293e-05,
        "stdp_rho": 0.0137,
        "stdp_tau": 16.33,
        "stdp_wMin": 0.0,
        "stdp_wMax": 0.0219,
    })

    study.optimize(objective, n_trials=100, callbacks=[mlflc])

    print("\n" + "="*40)
    print(f"BEST ACCURACY FOUND: {study.best_value}%")
    print("BEST PARAMS:", study.best_params)
    print("="*40)