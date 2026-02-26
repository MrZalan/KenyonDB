import mnist
import numpy as np
import argparse
import mlflow
from copy import copy
from matplotlib import pyplot as plt
from pygenn import (create_current_source_model, create_neuron_model, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, GeNNModel)
from pygenn import create_var_ref, create_out_post_var_ref, create_custom_update_model
from tqdm.auto import tqdm

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
testing_images = mnist.test_images()
testing_images = np.reshape(testing_images, (testing_images.shape[0], -1)).astype(np.float32)
testing_images /= np.sum(testing_images, axis=1)[:, np.newaxis]
testing_labels = mnist.test_labels()

pn_kc_ind = np.load("pn_kc_ind.npy")
kc_mbon_g = np.load("kc_mbon_g.npy")

DT = 0.1

NUM_PN = 784

NUM_MBON = 10

parser = argparse.ArgumentParser()

parser.add_argument('--input_scale', type=float, default=80.0)
parser.add_argument('--num_kc', type=int, default=20000)
parser.add_argument('--present_ms', type=float, default=20.0)

parser.add_argument('--lif_c', type=float, default=0.2)
parser.add_argument('--lif_tau_m', type=float, default=20.0)
parser.add_argument('--lif_v_rest', type=float, default=-60.0)
parser.add_argument('--lif_v_reset', type=float, default=-60.0)
parser.add_argument('--lif_v_thresh', type=float, default=-50.0)
parser.add_argument('--lif_ioffset', type=float, default=0.0)
parser.add_argument('--lif_tau_refrac', type=float, default=2.0)

parser.add_argument('--pn_tau_refrac', type=float, default=100.0)

parser.add_argument('--pn_kc_weight', type=float, default=0.2)
parser.add_argument('--pn_kc_tau_syn', type=float, default=3.0)
parser.add_argument('--pn_kc_fan_in', type=int, default=20)

parser.add_argument('--ggn_v_thresh', type=float, default=200)

parser.add_argument('--mbon_stimulus_current', type=float, default=3.0)

parser.add_argument('--kc_mbon_tau_syn', type=float, default=3.0)
parser.add_argument('--stdp_tau', type=float, default=15.0)
parser.add_argument('--stdp_rho', type=float, default=0.01)
parser.add_argument('--stdp_eta', type=float, default=0.00002)
parser.add_argument('--stdp_wMin', type=float, default=0.0)
parser.add_argument('--stdp_wMax', type=float, default=0.0233)

args = parser.parse_args()

LIF_PARAMS = {
    "C": args.lif_c,
    "TauM": args.lif_tau_m,
    "Vrest": args.lif_v_rest,
    "Vreset": args.lif_v_reset,
    "Vthresh": args.lif_v_thresh,
    "Ioffset": args.lif_ioffset,
    "TauRefrac": args.lif_tau_refrac}

PN_PARAMS = copy(LIF_PARAMS)
PN_PARAMS["TauRefrac"] = args.pn_tau_refrac

GGN_PARAMS = {
    "Vthresh": args.ggn_v_thresh}

KC_MBON_PARAMS = {"tau": args.stdp_tau,
                  "rho": args.stdp_rho,
                  "eta": args.stdp_eta,
                  "wMin": args.stdp_wMin,
                  "wMax": args.stdp_wMax}

cs_model = create_current_source_model(
    "cs_model",
    vars=[("magnitude", "scalar")],
    injection_code="injectCurrent(magnitude);")

if_model = create_neuron_model(
    "IF",
    params=["Vthresh"],
    vars=[("V", "scalar")],
    sim_code=
    """
    V += Isyn;
    """,
    threshold_condition_code=
    """
    V >= Vthresh
    """,
    reset_code=
    """
    V = 0.0;
    """)

model = GeNNModel("float", "mnist_mb_testing")
model.dt = DT

lif_init = {"V": PN_PARAMS["Vreset"], "RefracTime": 0.0}
if_init = {"V": 0.0}
pn = model.add_neuron_population("pn", NUM_PN, "LIF", PN_PARAMS, lif_init)
kc = model.add_neuron_population("kc", args.num_kc, "LIF", LIF_PARAMS, lif_init)
ggn = model.add_neuron_population("ggn", 1, if_model, GGN_PARAMS, if_init)
mbon = model.add_neuron_population("mbon", NUM_MBON, "LIF", LIF_PARAMS, lif_init)

pn.spike_recording_enabled = True
kc.spike_recording_enabled = True
mbon.spike_recording_enabled = True

pn_input = model.add_current_source("pn_input", cs_model, pn , {}, {"magnitude": 0.0})

kc_ggn = model.add_synapse_population("kc_ggn", "DENSE",
                                      kc, ggn,
                                      init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
                                      init_postsynaptic("DeltaCurr"))

ggn_kc = model.add_synapse_population("ggn_kc", "DENSE",
                                      ggn, kc,
                                      init_weight_update("StaticPulseConstantWeight", {"g": -5.0}),
                                      init_postsynaptic("ExpCurr", {"tau": 5.0}))

pn_kc = model.add_synapse_population("pn_kc", "SPARSE",
                                     pn, kc,
                                     init_weight_update("StaticPulseConstantWeight", {"g": args.pn_kc_weight}),
                                     init_postsynaptic("ExpCurr", {"tau": args.pn_kc_tau_syn}))
pn_kc.set_sparse_connections(pn_kc_ind[0], pn_kc_ind[1])

kc_mbon = model.add_synapse_population("kc_mbon", "DENSE",
                                       kc, mbon,
                                       init_weight_update("StaticPulse", {}, {"g": kc_mbon_g}),
                                       init_postsynaptic("ExpCurr", {"tau": args.kc_mbon_tau_syn}))

present_timesteps = int(round(args.present_ms / DT))

reset_neuron_model = create_custom_update_model(
    "reset_neuron",
    var_refs=[("V", "scalar"), ("RefracTime", "scalar")],
    update_code="""
    V = -60.0;
    RefracTime = 0.0;
    """)

reset_neuron_ggn_model = create_custom_update_model(
    "reset_neuron_ggn",
    var_refs=[("V", "scalar")],
    update_code="""
    V = -60.0;
    """)

reset_synapse_model = create_custom_update_model(
    "reset_synapse",
    var_refs=[("out_post", "scalar")],
    update_code="""
    out_post = 0.0;
    """)

reset_neuron_pn = model.add_custom_update(
    "reset_neuron_pn", "reset_group", reset_neuron_model,
    var_refs={"V": create_var_ref(pn, "V"), "RefracTime": create_var_ref(pn, "RefracTime")}
)

reset_neuron_kc = model.add_custom_update(
    "reset_neuron_kc", "reset_group", reset_neuron_model,
    var_refs={"V": create_var_ref(kc, "V"), "RefracTime": create_var_ref(kc, "RefracTime")}
)

reset_neuron_ggn = model.add_custom_update(
    "reset_neuron_ggn", "reset_group", reset_neuron_ggn_model,
    var_refs={"V": create_var_ref(ggn, "V")}
)

reset_neuron_mbon = model.add_custom_update(
    "reset_neuron_mbon", "reset_group", reset_neuron_model,
    var_refs={"V": create_var_ref(mbon, "V"), "RefracTime": create_var_ref(mbon, "RefracTime")}
)

reset_synapse_pn_kc = model.add_custom_update(
    "reset_synapse_pn_kc", "reset_group", reset_synapse_model,
    var_refs={"out_post": create_out_post_var_ref(pn_kc)}
)

reset_synapse_ggn_kc = model.add_custom_update(
    "reset_synapse_ggn_kc", "reset_group", reset_synapse_model,
    var_refs={"out_post": create_out_post_var_ref(ggn_kc)}
)

reset_synapse_kc_mbon = model.add_custom_update(
    "reset_synapse_kc_mbon", "reset_group", reset_synapse_model,
    var_refs={"out_post": create_out_post_var_ref(kc_mbon)}
)

model.build()
model.load(num_recording_timesteps=present_timesteps)  

num_correct = 0
for s in tqdm(range(testing_images.shape[0])):
    pn_input.vars["magnitude"].view[:] = testing_images[s] * args.input_scale
    pn_input.vars["magnitude"].push_to_device()

    for i in range(present_timesteps):
        model.step_time()

    model.custom_update("reset_group")

    model.pull_recording_buffers_from_device()

    mbon_spike_times, mbon_spike_ids = mbon.spike_recording_data[0]
    if len(mbon_spike_times) > 0:
        if mbon_spike_ids[np.argmin(mbon_spike_times)] == testing_labels[s]:
            num_correct += 1

accuracy = (num_correct * 100.0) / testing_images.shape[0]
print(f"Accuracy: {accuracy}%")

# Log to MLflow
if mlflow.active_run():
    mlflow.log_metric("accuracy", accuracy)
else:
    mlflow.set_experiment("Pygenn_MNIST_Eval")
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.log_metric("accuracy", accuracy)