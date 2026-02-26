import optuna
import subprocess
import os
import sys
import shutil
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    # --- 1. Round 3 Search Ranges: Pushing Capacity ---
    params = {
        # Architecture: Testing if 60k-80k KCs push us toward 90%
        "num_kc": trial.suggest_int("num_kc", 50000, 80000, step=10000),
        "present_ms": trial.suggest_categorical("present_ms", [40.0, 50.0]),
        "input_scale": trial.suggest_float("input_scale", 85.0, 110.0),

        # Neuron Dynamics: Centered on Round 2 winners
        "lif_c": trial.suggest_float("lif_c", 0.23, 0.25),
        "lif_tau_m": trial.suggest_float("lif_tau_m", 26.0, 35.0),
        "lif_v_rest": -60.0, 
        "lif_v_reset": -60.0,
        "lif_v_thresh": trial.suggest_float("lif_v_thresh", -52.0, -50.5),
        "lif_ioffset": trial.suggest_float("lif_ioffset", 0.0, 0.02),
        "lif_tau_refrac": trial.suggest_float("lif_tau_refrac", 2.7, 3.5),
        "pn_tau_refrac": trial.suggest_float("pn_tau_refrac", 80.0, 95.0),
        
        # Synapses: Following the trend of slightly lower weights/fan-in
        "pn_kc_weight": trial.suggest_float("pn_kc_weight", 0.14, 0.18),
        "pn_kc_fan_in": trial.suggest_int("pn_kc_fan_in", 16, 22),
        "pn_kc_tau_syn": trial.suggest_float("pn_kc_tau_syn", 2.7, 3.2),
        
        # Inhibition: Exploring even lower thresholds for sharper sparsity
        "ggn_v_thresh": trial.suggest_float("ggn_v_thresh", 150.0, 180.0),
        "mbon_stimulus_current": trial.suggest_float("mbon_stimulus_current", 3.0, 4.0),
        "kc_mbon_tau_syn": trial.suggest_float("kc_mbon_tau_syn", 3.3, 3.8),
        
        # STDP: Pushing learning rate and stability
        "stdp_eta": trial.suggest_float("stdp_eta", 4e-5, 2e-4, log=True),
        "stdp_rho": trial.suggest_float("stdp_rho", 0.012, 0.03, log=True),
        "stdp_tau": trial.suggest_float("stdp_tau", 15.0, 20.0),
        "stdp_wMin": 0.0,
        "stdp_wMax": trial.suggest_float("stdp_wMax", 0.02, 0.03),
    }

    train_script = "my_model/train_params.py"
    eval_script = "my_model/eval_params.py"

    for folder in ["mnist_mb_training_objects", "mnist_mb_testing_objects"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # Execute Training (Timeout increased for 80k neurons)
    train_cmd = ["python3", train_script]
    for key, val in params.items():
        train_cmd.extend([f"--{key}", str(val)])
    
    print(f"\n[Trial {trial.number}] Starting Training...")
    try:
        subprocess.run(train_cmd, check=True, timeout=2400) 
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return 0.0

    # Execute Evaluation
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
    study = optuna.create_study(direction="maximize", study_name="SNN_Optimization_v5_85PercentAnchor")
    
    # --- 2. Anchor Trial (The 85.3% Winner) ---
    study.enqueue_trial({
        "num_kc": 50000,
        "present_ms": 40.0,
        "input_scale": 89.64,
        "lif_c": 0.2372,
        "lif_tau_m": 27.86,
        "lif_v_thresh": -50.80,
        "lif_ioffset": 0.016,
        "lif_tau_refrac": 2.83,
        "pn_tau_refrac": 93.36,
        "pn_kc_weight": 0.1633,
        "pn_kc_fan_in": 19,
        "pn_kc_tau_syn": 2.81,
        "ggn_v_thresh": 182.50,
        "mbon_stimulus_current": 3.21, # From prev winner
        "kc_mbon_tau_syn": 3.42,      # From prev winner
        "stdp_eta": 4.29e-05,
        "stdp_rho": 0.0137,
        "stdp_tau": 16.33,
        "stdp_wMin": 0.0,
        "stdp_wMax": 0.0219
    })

    study.optimize(objective, n_trials=100, callbacks=[mlflc])