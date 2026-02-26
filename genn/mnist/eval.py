import mnist
import numpy as np
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

pn_kc_ind = np.load("pn_kc_ind_test.npy")
kc_mbon_g = np.load("kc_mbon_g_test.npy")

DT = 0.1

INPUT_SCALE = 80.0

NUM_PN = 784

NUM_KC = 20000

NUM_MBON = 10

PRESENT_TIME_MS = 20.0

LIF_PARAMS = {
    "C": 0.2,
    "TauM": 20.0,
    "Vrest": -60.0,
    "Vreset": -60.0,
    "Vthresh": -50.0,
    "Ioffset": 0.0,
    "TauRefrac": 2.0}

PN_PARAMS = copy(LIF_PARAMS)
PN_PARAMS["TauRefrac"] = 100.0

PN_KC_WEIGHT = 0.2

PN_KC_TAU_SYN = 3.0

PN_KC_FAN_IN = 20

KC_MBON_TAU_SYN = 3.0

GGN_PARAMS = {
    "Vthresh": 200.0}

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
kc = model.add_neuron_population("kc", NUM_KC, "LIF", LIF_PARAMS, lif_init)
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
                                     init_weight_update("StaticPulseConstantWeight", {"g": PN_KC_WEIGHT}),
                                     init_postsynaptic("ExpCurr", {"tau": PN_KC_TAU_SYN}))
pn_kc.set_sparse_connections(pn_kc_ind[0], pn_kc_ind[1])

kc_mbon = model.add_synapse_population("kc_mbon", "DENSE",
                                       kc, mbon,
                                       init_weight_update("StaticPulse", {}, {"g": kc_mbon_g}),
                                       init_postsynaptic("ExpCurr", {"tau": KC_MBON_TAU_SYN}))

present_timesteps = int(round(PRESENT_TIME_MS / DT))

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

for s in range(4):
    pn_input.vars["magnitude"].view[:] = testing_images[s] * INPUT_SCALE
    pn_input.vars["magnitude"].push_to_device()

    for i in range(present_timesteps):
        model.step_time()

    model.custom_update("reset_group")

    model.pull_recording_buffers_from_device()

    fig, axes = plt.subplots(3, sharex=True)
    pn_spike_times, pn_spike_ids = pn.spike_recording_data[0]
    kc_spike_times, kc_spike_ids = kc.spike_recording_data[0]
    mbon_spike_times, mbon_spike_ids = mbon.spike_recording_data[0]


    axes[0].scatter(pn_spike_times, pn_spike_ids, s=1)
    axes[0].set_ylabel("PN")
    axes[1].scatter(kc_spike_times, kc_spike_ids, s=1)
    axes[1].set_ylabel("KC")
    axes[2].scatter(mbon_spike_times, mbon_spike_ids, s=2)
    axes[2].axhline(testing_labels[s], linestyle="--", color="green", alpha=0.3)
    axes[2].set_ylim((-0.5, 10.5))

    if len(mbon_spike_times) > 0:
        classification = mbon_spike_ids[np.argmin(mbon_spike_times)]
        axes[2].axhline(classification, linestyle="--", color="red", alpha=0.3)
    axes[2].set_ylabel("MBON")

    axes[2].set_xlabel("Time [ms]")

num_correct = 0
for s in tqdm(range(testing_images.shape[0])):
    pn_input.vars["magnitude"].view[:] = testing_images[s] * INPUT_SCALE
    pn_input.vars["magnitude"].push_to_device()

    for i in range(present_timesteps):
        model.step_time()

    model.custom_update("reset_group")

    model.pull_recording_buffers_from_device()

    mbon_spike_times, mbon_spike_ids = mbon.spike_recording_data[0]
    if len(mbon_spike_times) > 0:
        if mbon_spike_ids[np.argmin(mbon_spike_times)] == testing_labels[s]:
            num_correct += 1

print(f"\n{num_correct}/{testing_images.shape[0]} correct ({(num_correct * 100.0) / testing_images.shape[0]} %%)")