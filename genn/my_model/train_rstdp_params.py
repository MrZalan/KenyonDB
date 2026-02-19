import mnist
import numpy as np
import argparse
import mlflow
from copy import copy
from matplotlib import pyplot as plt
from pygenn import (create_current_source_model, create_neuron_model, create_weight_update_model,
                    init_postsynaptic, init_sparse_connectivity, init_weight_update, GeNNModel)
from tqdm.auto import tqdm

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
training_images = mnist.train_images()
training_images = np.reshape(training_images, (training_images.shape[0], -1)).astype(np.float32)
training_images /= np.sum(training_images, axis=1)[:, np.newaxis]
training_labels = mnist.train_labels()

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
parser.add_argument('--stdp_tauE', type=float, default=200.0)

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
                  "wMax": args.stdp_wMax,
                  "tauE": args.stdp_tauE }

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

rstdp = create_weight_update_model(
    "rstdp",
    params=["tau", "rho", "eta", "wMin", "wMax", "tauE"],
    vars=[("g", "scalar"), ("e", "scalar")],
    extra_global_params=[("reward", "scalar*")],

    # Runs every timestep for every synapse
    synapse_dynamics_code="""
    // eligibility decay
    e *= exp(-dt / tauE);

    // three-factor update (scale by dt for stability)
    const scalar R = reward[id_post];
    const scalar newWeight = g + (eta * R * e * dt);

    // clamp
    g = fmin(wMax, fmax(wMin, newWeight));
    """,

    # Presynaptic spike: deliver current AND update eligibility
    pre_spike_syn_code="""
    // deliver synaptic current to postsynaptic model
    addToPost(g);

    // eligibility increment from relative timing
    const scalar d = t - st_post;
    const scalar timing = exp(-d / tau) - rho;
    e += timing;
    """,

    # Postsynaptic spike: update eligibility
    post_spike_syn_code="""
    const scalar d = t - st_pre;
    const scalar timing = exp(-d / tau) - rho;
    e += timing;
    """
)

def reset_spike_times(pop):
    pop.spike_times.view[:] = -np.finfo(np.float32).max
    pop.spike_times.push_to_device()


def reset_out_post(pop):
    pop.out_post.view[:] = 0.0
    pop.out_post.push_to_device()

def reset_neuron(pop, var_init):
    for var_name, var_val in var_init.items():
        pop.vars[var_name].view[:] = var_val
        pop.vars[var_name].push_to_device()

def reset_synapse_var(syn, var_name, value=0.0):
    syn.vars[var_name].view[:] = value
    syn.vars[var_name].push_to_device()

def set_egp(syn, name, values):
    syn.extra_global_params[name].view[:] = values
    syn.push_extra_global_param_to_device(name)

mlflow.set_experiment("Pygenn_MNIST_Train_RSTDP")
with mlflow.start_run():
    mlflow.log_params(vars(args))

    model = GeNNModel("float", "mnist_mb_training_rstdp", backend="cuda")
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
    mbon_input = model.add_current_source("mbon_input", cs_model, mbon , {}, {"magnitude": 0.0})

    pn_kc = model.add_synapse_population("pn_kc", "SPARSE",
                                        pn, kc,
                                        init_weight_update("StaticPulseConstantWeight", {"g": args.pn_kc_weight}),
                                        init_postsynaptic("ExpCurr", {"tau": args.pn_kc_tau_syn}),
                                        init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": args.pn_kc_fan_in}))

    kc_ggn = model.add_synapse_population("kc_ggn", "DENSE",
                                        kc, ggn,
                                        init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
                                        init_postsynaptic("DeltaCurr"))

    ggn_kc = model.add_synapse_population("ggn_kc", "DENSE",
                                        ggn, kc,
                                        init_weight_update("StaticPulseConstantWeight", {"g": -5.0}),
                                        init_postsynaptic("ExpCurr", {"tau": 5.0}))

    kc_mbon = model.add_synapse_population(
        "kc_mbon", "DENSE",
        kc, mbon,
        init_weight_update(rstdp, KC_MBON_PARAMS, {"g": 0.0, "e": 0.0}),
        init_postsynaptic("ExpCurr", {"tau": args.kc_mbon_tau_syn})
    )

    kc_mbon.extra_global_params["reward"].set_init_values(np.zeros(NUM_MBON, dtype=np.float32))

    present_timesteps = int(round(args.present_ms / DT))

    model.build()
    model.load(num_recording_timesteps=present_timesteps)
    print("Model backend:", model.backend_name)

    for s in tqdm(range(training_images.shape[0])):
        pn_input.vars["magnitude"].view[:] = training_images[s] * args.input_scale
        pn_input.vars["magnitude"].push_to_device()

        

        mbon_input.vars["magnitude"].view[:] = 0
        mbon_input.vars["magnitude"].view[training_labels[s]] = args.mbon_stimulus_current
        mbon_input.vars["magnitude"].push_to_device()

        R = -np.ones(NUM_MBON, dtype=np.float32)
        R[training_labels[s]] = 1.0
        set_egp(kc_mbon, "reward", R)

        for i in range(present_timesteps):
            model.step_time()

        reset_neuron(pn, lif_init)
        reset_neuron(kc, lif_init)
        reset_neuron(ggn, if_init)
        reset_neuron(mbon, lif_init)

        reset_spike_times(kc)
        reset_spike_times(mbon)

        reset_out_post(pn_kc)
        reset_out_post(ggn_kc)
        reset_out_post(kc_mbon)

        reset_synapse_var(kc_mbon, "e", 0.0)

    kc_mbon.vars["g"].pull_from_device()
    kc_mbon_g_view = kc_mbon.vars["g"].view

    np.save("kc_mbon_g.npy", kc_mbon_g_view)

    pn_kc.pull_connectivity_from_device()
    np.save("pn_kc_ind.npy", np.vstack((pn_kc.get_sparse_pre_inds(), pn_kc.get_sparse_post_inds())))
    mlflow.log_artifact("kc_mbon_g.npy")
    mlflow.log_artifact("pn_kc_ind.npy")