import optuna
import subprocess
import os
import sys
import shutil
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    # --- 1. Aggressive High-Accuracy Search Ranges ---
    params = {
        # Architecture: Scaling up to 100k KCs for better feature separation
        "num_kc": trial.suggest_int("num_kc", 30000, 100000, step=10000),
        "present_ms": trial.suggest_categorical("present_ms", [40.0, 60.0, 100.0]),
        "input_scale": trial.suggest_float("input_scale", 70.0, 110.0),

        # Neuron Dynamics: Refined ranges based on previous winners
        "lif_c": trial.suggest_float("lif_c", 0.20, 0.30),
        "lif_tau_m": trial.suggest_float("lif_tau_m", 15.0, 30.0),
        "lif_v_rest": -60.0,
        "lif_v_reset": -60.0,
        "lif_v_thresh": trial.suggest_float("lif_v_thresh", -53.0, -48.0),
        "lif_ioffset": trial.suggest_float("lif_ioffset", 0.0, 0.05),
        "lif_tau_refrac": trial.suggest_float("lif_tau_refrac", 1.0, 4.0),
        "pn_tau_refrac": trial.suggest_float("pn_tau_refrac", 70.0, 120.0),
        
        # Sparsity Control: Reducing fan-in and increasing connectivity specialization
        "pn_kc_weight": trial.suggest_float("pn_kc_weight", 0.10, 0.25),
        "pn_kc_fan_in": trial.suggest_int("pn_kc_fan_in", 8, 16),
        "pn_kc_tau_syn": trial.suggest_float("pn_kc_tau_syn", 2.0, 5.0),
        
        # Inhibition: Lower thresholds = stricter "Winner-Take-All" sparsity
        "ggn_v_thresh": trial.suggest_float("ggn_v_thresh", 120.0, 180.0),
        "mbon_stimulus_current": trial.suggest_float("mbon_stimulus_current", 3.0, 8.0),
        "kc_mbon_tau_syn": trial.suggest_float("kc_mbon_tau_syn", 2.0, 6.0),
        
        # R-STDP Learning: Finer eta and higher rho for better contrast
        "stdp_eta": trial.suggest_float("stdp_eta", 1e-5, 5e-4, log=True),
        "stdp_rho": trial.suggest_float("stdp_rho", 0.01, 0.05, log=True),
        "stdp_tau": trial.suggest_float("stdp_tau", 10.0, 25.0),
        "stdp_wMin": 0.0,
        "stdp_wMax": trial.suggest_float("stdp_wMax", 0.015, 0.035),
        "stdp_tauE": trial.suggest_float("stdp_tauE", 150.0, 500.0),
    }

    train_script = "my_model/train_rstdp_params.py"
    eval_script = "my_model/eval_rstdp_params.py"

    # Clean GeNN generated code to avoid build conflicts between different KC counts
    for folder in ["mnist_mb_training_rstdp_objects", "mnist_mb_testing_objects"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # Execute Training
    train_cmd = ["python3", train_script]
    for key, val in params.items():
        train_cmd.extend([f"--{key}", str(val)])
    
    print(f"\n[Trial {trial.number}] Starting Training...")
    try:
        # Increased timeout for large KC populations and longer presentation times
        subprocess.run(train_cmd, check=True, timeout=3600) 
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
    study = optuna.create_study(direction="maximize", study_name="SNN_Optimization_RSTDP_2")
    
    # --- Seed with the previous 83% winner to ensure progress ---
    study.enqueue_trial({
        "input_scale": 80.0,
        "num_kc": 20000,
        "present_ms": 40.0,
        "lif_c": 0.2,
        "lif_tau_m": 20.0,
        "lif_v_rest": -60.0,
        "lif_v_reset": -60.0,
        "lif_v_thresh": -50.0,
        "lif_ioffset": 0.0,
        "lif_tau_refrac": 2.0,
        "pn_tau_refrac": 100.0,
        "pn_kc_weight": 0.2,
        "pn_kc_fan_in": 20,
        "pn_kc_tau_syn": 3.0,
        "ggn_v_thresh": 200.0,
        "mbon_stimulus_current": 5.0,
        "kc_mbon_tau_syn": 3.0,
        "stdp_eta": 0.00002,
        "stdp_rho": 0.01,
        "stdp_tau": 15.0,
        "stdp_wMin": 0.0,
        "stdp_wMax": 0.0233,
        "stdp_tauE": 200.0
    })

    study.optimize(objective, n_trials=100, callbacks=[mlflc])