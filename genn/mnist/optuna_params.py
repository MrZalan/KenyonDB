import optuna
import subprocess
import os
import sys
import shutil
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    params = {
        # Architecture
        "num_kc": trial.suggest_int("num_kc", 10000, 30000, step=10000),
        "input_scale": trial.suggest_float("input_scale", 70.0, 100.0),
        "present_ms": trial.suggest_categorical("present_ms", [20.0, 30.0]),

        # Neuron Dynamics
        "lif_c": trial.suggest_float("lif_c", 0.15, 0.25),
        "lif_tau_m": trial.suggest_float("lif_tau_m", 15.0, 25.0),
        "lif_v_rest": trial.suggest_float("lif_v_rest", -60.0, -60.0),
        "lif_v_reset": trial.suggest_float("lif_v_reset", -60.0, -60.0),
        "lif_v_thresh": trial.suggest_float("lif_v_thresh", -52.0, -48.0),
        "lif_ioffset": trial.suggest_float("lif_ioffset", 0.0, 0.05),
        "lif_tau_refrac": trial.suggest_float("lif_tau_refrac", 1.5, 2.5),
        "pn_tau_refrac": trial.suggest_float("pn_tau_refrac", 90.0, 110.0),
        
        # Synaptic Weights & Connectivity
        "pn_kc_weight": trial.suggest_float("pn_kc_weight", 0.15, 0.25),
        "pn_kc_fan_in": trial.suggest_int("pn_kc_fan_in", 15, 25),
        "pn_kc_tau_syn": trial.suggest_float("pn_kc_tau_syn", 2.5, 3.5),

        #Inhibition
        "ggn_v_thresh": trial.suggest_float("ggn_v_thresh", 190.0, 210.0),
        "mbon_stimulus_current": trial.suggest_float("mbon_stimulus_current", 2.5, 3.5),
        "kc_mbon_tau_syn": trial.suggest_float("kc_mbon_tau_syn", 2.5, 3.5),
        
        # STDP Params:
        "stdp_eta": trial.suggest_float("stdp_eta", 1e-5, 5e-5, log=True),
        "stdp_rho": trial.suggest_float("stdp_rho", 0.008, 0.015, log=True),
        "stdp_tau": trial.suggest_float("stdp_tau", 13.0, 17.0),
        "stdp_wMin": trial.suggest_float("stdp_wMin", 0.0, 0.0),
        "stdp_wMax": trial.suggest_float("stdp_wMax", 0.02, 0.03),
    }

    train_script = "my_model/train_params.py"
    eval_script = "my_model/eval_params.py"

    # Resource Cleanup
    for folder in ["mnist_mb_training_objects", "mnist_mb_testing_objects"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    # Training with Timeout
    train_cmd = ["python3", train_script]
    for key, val in params.items():
        train_cmd.extend([f"--{key}", str(val)])
    
    print(f"\n[Trial {trial.number}] Starting Training...")
    try:
        subprocess.run(train_cmd, check=True, timeout=1200) 
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out. Pruning...")
        return 0.0
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        return 0.0

    # Evaluation
    eval_cmd = ["python3", eval_script]
    for key, val in params.items():
        eval_cmd.extend([f"--{key}", str(val)])
    
    print(f"[Trial {trial.number}] Starting Evaluation...")
    try:
        output = subprocess.check_output(eval_cmd, stderr=subprocess.STDOUT).decode()
        accuracy = float(output.split("Accuracy: ")[1].split("%")[0])
        print(f"[Trial {trial.number}] Resulting Accuracy: {accuracy}%")
        return accuracy
    except Exception as e:
        print(f"Evaluation failed. Error: {e}")
        return 0.0

if __name__ == "__main__":
    mlflc = MLflowCallback(tracking_uri="file:./mlruns", metric_name="accuracy")
    
    study = optuna.create_study(direction="maximize", study_name="Optimized_SNN_Sweep_v1")
    
    # Enqueue Baseline
    study.enqueue_trial({
        "input_scale": 80.0, "num_kc": 20000, "present_ms": 20.0,
        "lif_c": 0.2, "lif_tau_m": 20.0, "lif_v_rest": -60.0, "lif_v_reset": -60.0,
        "lif_v_thresh": -50.0, "lif_ioffset": 0.0, "lif_tau_refrac": 2.0,
        "pn_tau_refrac": 100.0, "pn_kc_weight": 0.2, "pn_kc_fan_in": 20,
        "pn_kc_tau_syn": 3.0, "ggn_v_thresh": 200.0, "mbon_stimulus_current": 3.0,
        "kc_mbon_tau_syn": 3.0, "stdp_eta": 0.00002, "stdp_rho": 0.01,
        "stdp_tau": 15.0, "stdp_wMin": 0.0, "stdp_wMax": 0.0233,
    })

    study.optimize(objective, n_trials=50, callbacks=[mlflc])