import mnist
import numpy as np
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

# Simulation time step
DT = 0.1

# Scaling factor for converting normalised image pixels to input currents (nA)
INPUT_SCALE = 80.0

# Number of Projection Neurons in model (should match image size)
NUM_PN = 784

# Number of Kenyon Cells in model (defines memory capacity)
NUM_KC = 20000

# How long to present each image to model
PRESENT_TIME_MS = 20.0

# Standard LIF neurons parameters
LIF_PARAMS = {
    "C": 0.2,
    "TauM": 20.0,
    "Vrest": -60.0,
    "Vreset": -60.0,
    "Vthresh": -50.0,
    "Ioffset": 0.0,
    "TauRefrac": 2.0}

# We only want PNs to spike once
PN_PARAMS = copy(LIF_PARAMS)
PN_PARAMS["TauRefrac"] = 100.0

# Weight of each synaptic connection
PN_KC_WEIGHT = 0.2

# Time constant of synaptic integration
PN_KC_TAU_SYN = 3.0

# How many projection neurons should be connected to each Kenyon Cell
PN_KC_FAN_IN = 20

GGN_PARAMS = {"Vthresh": 200.0}

NUM_MBON = 10
MBON_STIMULUS_CURRENT = 5.0

KC_MBON_TAU_SYN = 3.0
KC_MBON_PARAMS = {
    "tau": 15.0,
    "rho": 0.01,
    "eta": 0.00002,
    "wMin": 0.0,
    "wMax": 0.0233,
    "tauE": 200.0  # eligibility decay (ms)
}

cs_model = create_current_source_model(
    "cs_model",
    vars=[("magnitude", "scalar")],
    injection_code="injectCurrent(magnitude);"
)

# Minimal integrate and fire neuron model
if_model = create_neuron_model(
    "IF",
    params=["Vthresh"],
    vars=[("V", "scalar")],
    sim_code="""
    V += Isyn;
    """,
    threshold_condition_code="""
    V >= Vthresh
    """,
    reset_code="""
    V = 0.0;
    """
)

# Reward-modulated STDP (R-STDP): eligibility trace + reward signal (per post neuron)
# PyGeNN 5.x: use synapse_dynamics_code for per-timestep logic.
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


# Create model and neuron populations
model = GeNNModel("float", "mnist_mb_third_layer", backend="cuda")
model.dt = DT

lif_init = {"V": PN_PARAMS["Vreset"], "RefracTime": 0.0}
if_init = {"V": 0.0}

pn = model.add_neuron_population("pn", NUM_PN, "LIF", PN_PARAMS, lif_init)
kc = model.add_neuron_population("kc", NUM_KC, "LIF", LIF_PARAMS, lif_init)
ggn = model.add_neuron_population("ggn", 1, if_model, GGN_PARAMS, if_init)

pn.spike_recording_enabled = True
kc.spike_recording_enabled = True

pn_input = model.add_current_source("pn_input", cs_model, pn, {}, {"magnitude": 0.0})

# Create synapse populations
pn_kc = model.add_synapse_population(
    "pn_kc", "SPARSE",
    pn, kc,
    init_weight_update("StaticPulseConstantWeight", {"g": PN_KC_WEIGHT}),
    init_postsynaptic("ExpCurr", {"tau": PN_KC_TAU_SYN}),
    init_sparse_connectivity("FixedNumberPreWithReplacement", {"num": PN_KC_FAN_IN})
)

kc_ggn = model.add_synapse_population(
    "kc_ggn", "DENSE",
    kc, ggn,
    init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
    init_postsynaptic("DeltaCurr")
)

ggn_kc = model.add_synapse_population(
    "ggn_kc", "DENSE",
    ggn, kc,
    init_weight_update("StaticPulseConstantWeight", {"g": -5.0}),
    init_postsynaptic("ExpCurr", {"tau": 5.0})
)

mbon = model.add_neuron_population("mbon", NUM_MBON, "LIF", LIF_PARAMS, lif_init)
mbon.spike_recording_enabled = True

mbon_input = model.add_current_source("mbon_input", cs_model, mbon, {}, {"magnitude": 0.0})

# KC->MBON with R-STDP
kc_mbon = model.add_synapse_population(
    "kc_mbon", "DENSE",
    kc, mbon,
    init_weight_update(rstdp, KC_MBON_PARAMS, {"g": 0.0, "e": 0.0}),
    init_postsynaptic("ExpCurr", {"tau": KC_MBON_TAU_SYN})
)

# Allocate reward array (must happen before build/load)
kc_mbon.extra_global_params["reward"].set_init_values(np.zeros(NUM_MBON, dtype=np.float32))

present_timesteps = int(round(PRESENT_TIME_MS / DT))

# Build
model.build()
model.load(num_recording_timesteps=present_timesteps)
print("Model backend:", model.backend_name)

# -------- Helpers --------
def reset_spike_times(pop):
    # In PyGeNN 5.4.0 some pops (e.g. MBON) may not have last-spike-time buffers -> pop.spike_times is None
    if getattr(pop, "spike_times", None) is None:
        return
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
    # PyGeNN 5.4.0: ExtraGlobalParameter uses .view and push via syn.push_extra_global_param_to_device()
    syn.extra_global_params[name].view[:] = values
    syn.push_extra_global_param_to_device(name)

# -------- Training loop --------
for s in tqdm(range(training_images.shape[0])):
    # image to PN current
    pn_input.vars["magnitude"].view[:] = training_images[s] * INPUT_SCALE
    pn_input.vars["magnitude"].push_to_device()

    # teacher forcing current into target MBON (optional but helpful)
    mbon_input.vars["magnitude"].view[:] = 0.0
    mbon_input.vars["magnitude"].view[training_labels[s]] = MBON_STIMULUS_CURRENT
    mbon_input.vars["magnitude"].push_to_device()

    # reward: +1 for correct class, -1 for others
    R = -np.ones(NUM_MBON, dtype=np.float32)
    R[training_labels[s]] = 1.0
    set_egp(kc_mbon, "reward", R)

    # present stimulus
    for _ in range(present_timesteps):
        model.step_time()

    # reset state between samples
    reset_neuron(pn, lif_init)
    reset_neuron(kc, lif_init)
    reset_neuron(ggn, if_init)
    reset_neuron(mbon, lif_init)

    # last-spike-time buffers (if they exist)
    reset_spike_times(kc)
    reset_spike_times(mbon)  # safe now

    # synaptic currents / out_post buffers
    reset_out_post(pn_kc)
    reset_out_post(ggn_kc)
    reset_out_post(kc_mbon)

    # reset eligibility trace between samples (recommended)
    reset_synapse_var(kc_mbon, "e", 0.0)

# Pull and plot learned weights
kc_mbon.vars["g"].pull_from_device()
kc_mbon_g_view = kc_mbon.vars["g"].view

fig, axis = plt.subplots(figsize=(10, 5))
axis.hist(kc_mbon_g_view, bins=100)
axis.axvline(np.average(kc_mbon_g_view), linestyle="--")
axis.set_xlabel("Weight [nA]")
axis.set_ylabel("Count")

# Saving weights
np.save("kc_mbon_g.npy", kc_mbon_g_view)

pn_kc.pull_connectivity_from_device()
np.save("pn_kc_ind.npy", np.vstack((pn_kc.get_sparse_pre_inds(), pn_kc.get_sparse_post_inds())))
