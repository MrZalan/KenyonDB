import optuna
import subprocess
import os
import sys
import shutil
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    # --- 2. Refined Search Ranges based on 85.9% Plateau Analysis ---
    params = {
        # Architecture: Testing the 40k-60k sweet spot found in previous winners
        "num_kc": trial.suggest_int("num_kc", 40000, 80000, step=10000),
        "present_ms": 40.0, # 40ms was a consistent winner; fixing to reduce search noise
        "input_scale": trial.suggest_float("input_scale", 70.0, 95.0),

        # Neuron Dynamics: Shorter TauM for higher temporal precision
        "lif_c": trial.suggest_float("lif_c", 0.25, 0.28),
        "lif_tau_m": trial.suggest_float("lif_tau_m", 15.0, 22.0), # Narrowed toward the faster side
        "lif_v_rest": -60.0,
        "lif_v_reset": -60.0,
        "lif_v_thresh": trial.suggest_float("lif_v_thresh", -53.0, -51.0),
        "lif_ioffset": trial.suggest_float("lif_ioffset", 0.001, 0.005),
        "lif_tau_refrac": trial.suggest_float("lif_tau_refrac", 3.0, 4.0),
        "pn_tau_refrac": trial.suggest_float("pn_tau_refrac", 110.0, 120.0),
        
        # Sparsity Control: Pushing toward lower fan-in for higher specificity
        "pn_kc_weight": trial.suggest_float("pn_kc_weight", 0.18, 0.22),
        "pn_kc_fan_in": trial.suggest_int("pn_kc_fan_in", 10, 15), # Lowered from 16-20
        "pn_kc_tau_syn": trial.suggest_float("pn_kc_tau_syn", 2.5, 3.5),
        
        # Inhibition: Lower thresholds to force extreme sparsity
        "ggn_v_thresh": trial.suggest_float("ggn_v_thresh", 120.0, 155.0), # Stricter than previous runs
        "mbon_stimulus_current": trial.suggest_float("mbon_stimulus_current", 3.5, 4.5),
        "kc_mbon_tau_syn": trial.suggest_float("kc_mbon_tau_syn", 4.0, 5.0),
        
        # R-STDP Learning: Stabilized parameters from top 3
        "stdp_eta": trial.suggest_float("stdp_eta", 1.4e-5, 2.0e-5, log=True),
        "stdp_rho": trial.suggest_float("stdp_rho", 0.01, 0.012),
        "stdp_tau": trial.suggest_float("stdp_tau", 19.0, 24.0),
        "stdp_wMin": 0.0,
        "stdp_wMax": trial.suggest_float("stdp_wMax", 0.030, 0.033),
        "stdp_tauE": trial.suggest_float("stdp_tauE", 175.0, 185.0),
    }

    # Paths to your scripts
    train_script = "my_model/train_rstdp_params.py"
    eval_script = "my_model/eval_rstdp_params.py"

    # Clean GeNN generated code
    for folder in ["mnist_mb_training_rstdp_objects", "mnist_mb_testing_objects"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # Execute Training
    train_cmd = ["python3", train_script]
    for key, val in params.items():
        train_cmd.extend([f"--{key}", str(val)])
    
    try:
        subprocess.run(train_cmd, check=True, timeout=3600) 
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return 0.0

    # Execute Evaluation
    eval_cmd = ["python3", eval_script]
    for key, val in params.items():
        eval_cmd.extend([f"--{key}", str(val)])
    
    try:
        output = subprocess.check_output(eval_cmd, stderr=subprocess.STDOUT).decode()
        accuracy = float(output.split("Accuracy: ")[1].split("%")[0])
        return accuracy
    except Exception:
        return 0.0

if __name__ == "__main__":
    mlflc = MLflowCallback(tracking_uri="file:./mlruns", metric_name="accuracy")
    study = optuna.create_study(direction="maximize", study_name="SNN_Optimization_RSTDP_3")
    
    # --- Enqueue the best 85.9% configuration to start from a position of strength ---
    study.enqueue_trial({
        "num_kc": 50000,
        "input_scale": 74.18,
        "lif_c": 0.265,
        "lif_tau_m": 23.84,
        "lif_v_thresh": -52.23,
        "lif_ioffset": 0.0046,
        "lif_tau_refrac": 3.38,
        "pn_tau_refrac": 118.9,
        "pn_kc_weight": 0.196,
        "pn_kc_fan_in": 16,
        "pn_kc_tau_syn": 2.73,
        "ggn_v_thresh": 143.78,
        "mbon_stimulus_current": 3.82,
        "kc_mbon_tau_syn": 4.25,
        "stdp_eta": 1.92e-05,
        "stdp_rho": 0.0102,
        "stdp_tau": 22.83,
        "stdp_wMax": 0.0307,
        "stdp_tauE": 182.43
    })

    study.optimize(objective, n_trials=50, callbacks=[mlflc])