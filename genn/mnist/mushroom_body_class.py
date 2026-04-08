import numpy as np
import mnist
import cv2
from copy import copy
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, recall_score
from pygenn import (
    create_current_source_model,
    create_neuron_model,
    create_weight_update_model,
    init_postsynaptic,
    init_sparse_connectivity,
    init_weight_update,
    GeNNModel,
    create_var_ref,
    create_out_post_var_ref,
    create_custom_update_model,
)


class MBConfig:
    """Szükséges hiperparaméterek és konstansok definiálása a spiking neural modellhez"""
    DT = 0.1 # a szimuláció időbeli lépései
    PRESENT_TIME_MS = 30.38590446583062 # mennyi ideig lássa a modell az egyes képeket

    INPUT_SCALE = 92.22746034773357 # skálázási mérték a normalizált pixel értékek átalakításához bemeneti elektromos értékekké

    NUM_PN = 784 # bemeneti (projection) neuronok száma, meg kell egyeznie a kép mérettel (28x28 pixel)
    NUM_KC = 40000 # Kenyon sejtek száma, a memória kapacitást határozza meg
    NUM_MBON = 10 # kimeneti (mushroom body output) neuronok száma, meg kell egyeznie az osztályok számával (számjegyek)

    # Leaky Integrate-and-Fire modell paraméterei
    LIF_PARAMS = {
        "C": 0.2, # kapacitás, mennyi input érték szükséges a feszültség változásához
        "TauM": 23.599264989267002, # a membrán milyen gyorsan térjen vissza a nyugalmi szintjéjez ha nincs input
        "Vrest": -60.0, # nyugalmi potenciál
        "Vreset": -60.0, # visszaállítási potenciál, miután egy neuron tüzelt
        "Vthresh": -45.656505223575984, # a határérték ami felett tüzel a neuron
        "Ioffset": 0.0, # konstans bemeneti érték
        "TauRefrac": 2.0, # regenerálódási idő miután tüzelt egy neuron és újra tüzelőképes lesz
    }

    PN_REFRAC = 103.40271722138783 # bemeneti neuronok visszaállítási ideje miután tüzeltek

    GGN_PARAMS = {"Vthresh": 200.0} # gátló (giant glomerular) neuron határszintje, ami felett csökkenti a Kenyon sejtek hatását

    PN_KC_WEIGHT = 0.1496601666081794 # szinaptikus kapcsolatok súlya
    PN_KC_TAU = 9.143275839422166 # idő konstans a bemeneti neuronok és Kenyon sejtek közötti szinapszisok létrejöttéhez
    PN_KC_FAN_IN = 29 # mindegyik Kenyon sejt ennyi különböző bemeneti neuronhoz van kapcsolva

    KC_GGN_WEIGHT = 1.6651092523970699 # a Kenyon sejtek és a GGN neuron közötti kapcsolatok súlya

    GGN_KC_WEIGHT = -2.1607205735665684 # a GGN gátlási funkciója, ami biztosítja, hogy csak a legerősebb Kenyon sejtek jele maradjon meg
    GGN_KC_TAU = 2.9252874709652446 # idő konstans a GGN neuron és Kenyon sejtek közötti szinapszisok létrejöttéhez

    KC_MBON_TAU = 2.5502387602459486 # idő konstans a Kenyon sejtek és a kimeneti neuronok közötti szinapszisok létrejöttéhez
    MBON_STIMULUS_CURRENT = 9.317904794983713 # tanító szignál ereje tréning során a helyes MBON stimulálásához

    # Tanulási paraméterek
    KC_MBON_PARAMS = {
        "tau": 15.0, # időablak a preszinaptikus és posztszinaptikus tüzelés között
        "rho": 295.9502778626386, # konstans a neuronok stabilizálásához
        "eta": 1.1225781409621203e-06, # learning rate, milyen gyorsan tanul a modell
        "wMin": 0.0, # szinapszis legkisebb ereje
        "wMax": 0.015054612042772107, # szinapszis maximum ereje
        "tauE": 295.9502778626386, # R-STDP-hez szükséges konstans, mennyi ideig legyen aktív a jutalmi szignál
    }


class MushroomBodyModel:
    """A PyGenn spiking neuron modell alkotóelemeinek definiálása és létrehozása"""
    def __init__(
        self,
        name: str,
        backend: str = "single_threaded_cpu",
        is_training: bool = True,
        sparse_indices=None,
        kc_mbon_g=None,
    ):
        self.model = GeNNModel("float", name, backend=backend)
        self.model.dt = MBConfig.DT

        self.is_training = is_training
        self.sparse_indices = sparse_indices
        self.kc_mbon_g = kc_mbon_g

        self.__create_custom_models()
        self.__add_populations()
        self.__add_synapses()
        self.__add_reset_custom_updates()

    def __create_custom_models(self):
        """Modell komponensek definiálása"""
        self.cs_model = create_current_source_model(
            "cs_model",
            vars=[("magnitude", "scalar")],
            injection_code="injectCurrent(magnitude);",
        )

        # Integrate-and-Fire modell
        self.if_model = create_neuron_model(
            "IF",
            params=["Vthresh"],
            vars=[("V", "scalar")],
            sim_code="V += Isyn;",
            threshold_condition_code="V >= Vthresh",
            reset_code="V = 0.0;",
        )

        # R-STDP tanulási mechanizmus
        self.rstdp_model = create_weight_update_model(
            "rstdp",
            params=["tau", "rho", "eta", "wMin", "wMax", "tauE"],
            vars=[("g", "scalar"), ("e", "scalar")],
            extra_global_params=[("reward", "scalar*")],
            synapse_dynamics_code=r"""
                e *= exp(-dt / tauE); // felejtés, a memória halványodása
                const scalar R = reward[id_post]; // jutalom beolvasása ha a hálózat eltalálta a helyes számjegyet
                g = fmin(wMax, fmax(wMin, g + (eta * R * e * dt))); // a kapcsolat megerősödése vagy gyengülése attól függően hogy R pozitív vagy negatív
            """,
            # a kapcsolat fontos ha kevés idő telt el a preszinaptikus és posztszinaptikus tüzelés között
            pre_spike_syn_code="addToPost(g); const scalar d = t - st_post; e += (exp(-d / tau) - rho);",
            post_spike_syn_code="const scalar d = t - st_pre; e += (exp(-d / tau) - rho);",
        )

    def __add_populations(self):
        """neuron populációk definiálása és paraméterek beolvasása"""
        lif_init = {"V": MBConfig.LIF_PARAMS["Vreset"], "RefracTime": 0.0} # minden neuron a nyugalmi feszültségről induljon
        pn_params = copy(MBConfig.LIF_PARAMS)
        pn_params["TauRefrac"] = MBConfig.PN_REFRAC

        self.pn = self.model.add_neuron_population("pn", MBConfig.NUM_PN, "LIF", pn_params, lif_init)
        self.kc = self.model.add_neuron_population("kc", MBConfig.NUM_KC, "LIF", MBConfig.LIF_PARAMS, lif_init)
        self.ggn = self.model.add_neuron_population("ggn", 1, self.if_model, MBConfig.GGN_PARAMS, {"V": 0.0})
        self.mbon = self.model.add_neuron_population("mbon", MBConfig.NUM_MBON, "LIF", MBConfig.LIF_PARAMS, lif_init)

        # mielőtt kimentjük a GPU-ra a hálózatot jelezzük melyik értékeket szeretnénk majd kiolvasni
        self.pn.spike_recording_enabled = True
        self.kc.spike_recording_enabled = True
        self.mbon.spike_recording_enabled = True

        # bemeneti áram feszültségek
        self.pn_input = self.model.add_current_source("pn_input", self.cs_model, self.pn, {}, {"magnitude": 0.0}) # a hálózat itt kapja meg a pixel értékeket
        self.mbon_input = self.model.add_current_source("mbon_input", self.cs_model, self.mbon, {}, {"magnitude": 0.0}) # tanítási fázisban itt erősödik meg a megfelelő MBON a helyes a találat

    def __add_synapses(self):
        """Szinapszisok definiálása"""
        # PN - KC
        # ha tréningelünk akkor a PN - Kenyon sejt kapcsolatokat random és ritka módon inicializáljuk
        if self.sparse_indices is None:
            self.pn_kc = self.model.add_synapse_population(
                "pn_kc",
                "SPARSE",
                self.pn,
                self.kc,
                init_weight_update("StaticPulseConstantWeight", {"g": MBConfig.PN_KC_WEIGHT}),
                init_postsynaptic("ExpCurr", {"tau": MBConfig.PN_KC_TAU}),
                init_sparse_connectivity(
                    "FixedNumberPreWithReplacement",
                    {"num": MBConfig.PN_KC_FAN_IN},
                ),
            )
        else:
            # tesztelési fázisban nem inicializálunk, hanem a tréning során elmenett indexeket töltjük be
            self.pn_kc = self.model.add_synapse_population(
                "pn_kc",
                "SPARSE",
                self.pn,
                self.kc,
                init_weight_update("StaticPulseConstantWeight", {"g": MBConfig.PN_KC_WEIGHT}),
                init_postsynaptic("ExpCurr", {"tau": MBConfig.PN_KC_TAU}),
            )

        if self.sparse_indices is not None:
            self.pn_kc.set_sparse_connections(self.sparse_indices[0], self.sparse_indices[1])
        # KC - GGN
        self.kc_ggn = self.model.add_synapse_population(
            "kc_ggn",
            "DENSE",
            self.kc,
            self.ggn,
            init_weight_update("StaticPulseConstantWeight", {"g": MBConfig.KC_GGN_WEIGHT}),
            init_postsynaptic("DeltaCurr"),
        )

        self.ggn_kc = self.model.add_synapse_population(
            "ggn_kc",
            "DENSE",
            self.ggn,
            self.kc,
            init_weight_update("StaticPulseConstantWeight", {"g": MBConfig.GGN_KC_WEIGHT}),
            init_postsynaptic("ExpCurr", {"tau": MBConfig.GGN_KC_TAU}),
        )

        # KC - MBON
        if not self.is_training:
            # tesztelési fázisban a tréning során elmentett súlyok betöltése
            g_init = self.kc_mbon_g
            if g_init is None:
                g_init = 0.0
            wu, wu_params, wu_vars = "StaticPulse", {}, {"g": g_init}
        else:
            wu, wu_params, wu_vars = self.rstdp_model, MBConfig.KC_MBON_PARAMS, {"g": 0.0, "e": 0.0}
        

        self.kc_mbon = self.model.add_synapse_population(
            "kc_mbon",
            "DENSE",
            self.kc,
            self.mbon,
            init_weight_update(wu, wu_params, wu_vars),
            init_postsynaptic("ExpCurr", {"tau": MBConfig.KC_MBON_TAU}),
        )

        if self.is_training:
            self.kc_mbon.extra_global_params["reward"].set_init_values(
                np.zeros(MBConfig.NUM_MBON, dtype=np.float32)
            )

    def __add_reset_custom_updates(self):
        """Neuronok és szinapszisok visszaállításainak definiálása"""
        reset_neuron_model = create_custom_update_model(
            "reset_neuron",
            var_refs=[("V", "scalar"), ("RefracTime", "scalar")],
            update_code="""
                V = -60.0;
                RefracTime = 0.0;
            """,
        )

        reset_neuron_ggn_model = create_custom_update_model(
            "reset_neuron_ggn",
            var_refs=[("V", "scalar")],
            update_code="""
                V = 0.0;
            """,
        )

        reset_synapse_model = create_custom_update_model(
            "reset_synapse",
            var_refs=[("out_post", "scalar")],
            update_code="""
                out_post = 0.0;
            """,
        )

        self.model.add_custom_update(
            "reset_neuron_pn",
            "reset_group",
            reset_neuron_model,
            var_refs={"V": create_var_ref(self.pn, "V"), "RefracTime": create_var_ref(self.pn, "RefracTime")},
        )
        self.model.add_custom_update(
            "reset_neuron_kc",
            "reset_group",
            reset_neuron_model,
            var_refs={"V": create_var_ref(self.kc, "V"), "RefracTime": create_var_ref(self.kc, "RefracTime")},
        )
        self.model.add_custom_update(
            "reset_neuron_ggn",
            "reset_group",
            reset_neuron_ggn_model,
            var_refs={"V": create_var_ref(self.ggn, "V")},
        )
        self.model.add_custom_update(
            "reset_neuron_mbon",
            "reset_group",
            reset_neuron_model,
            var_refs={"V": create_var_ref(self.mbon, "V"), "RefracTime": create_var_ref(self.mbon, "RefracTime")},
        )

        self.model.add_custom_update(
            "reset_synapse_pn_kc",
            "reset_group",
            reset_synapse_model,
            var_refs={"out_post": create_out_post_var_ref(self.pn_kc)},
        )
        self.model.add_custom_update(
            "reset_synapse_ggn_kc",
            "reset_group",
            reset_synapse_model,
            var_refs={"out_post": create_out_post_var_ref(self.ggn_kc)},
        )
        self.model.add_custom_update(
            "reset_synapse_kc_mbon",
            "reset_group",
            reset_synapse_model,
            var_refs={"out_post": create_out_post_var_ref(self.kc_mbon)},
        )

    def build_and_load(self):
        """Modell buildelése és betöltése"""
        present_timesteps = int(round(MBConfig.PRESENT_TIME_MS / MBConfig.DT))
        self.model.build()
        self.model.load(num_recording_timesteps=present_timesteps)

    def set_kc_mbon_weights(self, kc_mbon_g):
        """Tréningelt súlyok betöltése és GPU-ra helyezése"""
        kc_mbon_g = np.asarray(kc_mbon_g, dtype=np.float32)
        w = self.kc_mbon.vars["g"].view
        if kc_mbon_g.size != w.size:
            raise ValueError(f"kc_mbon_g size mismatch: file={kc_mbon_g.size} vs model={w.size}")
        self.kc_mbon.vars["g"].view[:] = kc_mbon_g.reshape(w.shape)
        self.kc_mbon.vars["g"].push_to_device()


class MBSimulator:
    """Tréning és teszt fázisok futtatása"""
    def __init__(self, model_wrapper: MushroomBodyModel):
        self.model_wrapper = model_wrapper
        self.timesteps = int(round(MBConfig.PRESENT_TIME_MS / MBConfig.DT))

    def run_image(self, img_data, label=None):
        """ Bemenet beolvasása és GPU-ra helyezése """
        self.model_wrapper.pn_input.vars["magnitude"].view[:] = img_data * MBConfig.INPUT_SCALE
        self.model_wrapper.pn_input.vars["magnitude"].push_to_device()

        # Képek között az MBON tanító szignál nullázása
        self.model_wrapper.mbon_input.vars["magnitude"].view[:] = 0.0
        if label is not None:
            self.model_wrapper.mbon_input.vars["magnitude"].view[label] = MBConfig.MBON_STIMULUS_CURRENT
        self.model_wrapper.mbon_input.vars["magnitude"].push_to_device()

        # Idő léptetése a szimulációban
        for _ in range(self.timesteps):
            self.model_wrapper.model.step_time()

    def set_reward(self, label: int):
        """ Jutalom beállítása: rossz találat esetén R -1, ellenkező esetben 1"""
        R = -np.ones(MBConfig.NUM_MBON, dtype=np.float32)
        R[int(label)] = 1.0
        reward_param = self.model_wrapper.kc_mbon.extra_global_params["reward"]
        reward_param.view[:] = R
        reward_param.push_to_device()

    def train(self, images, labels):
        """ Tréning ciklus"""
        for i in tqdm(range(len(images))):
            self.set_reward(labels[i])
            self.run_image(images[i], labels[i])

            # Képek között értékek visszaállítása
            self.model_wrapper.model.custom_update("reset_group")
            self.model_wrapper.kc_mbon.vars["e"].view[:] = 0.0
            self.model_wrapper.kc_mbon.vars["e"].push_to_device()

    def predict(self, img_data):
        """Predikció adott képre"""
        self.run_image(img_data, label=None)

        self.model_wrapper.model.pull_recording_buffers_from_device()
        times, ids = self.model_wrapper.mbon.spike_recording_data[0] # MBON tüzelési adatok letöltése GPU-ról
        pred = int(ids[np.argmin(times)]) if len(times) > 0 else -1 # leghamarabb tüzelő MBON neuron id-ja lesz az eredmény

        # Értékek visszaállítása
        self.model_wrapper.model.custom_update("reset_group")
        return pred

    def evaluate(self, images, labels):
        """Tesztelési fázis"""
        num_correct = 0
        predictions = []
        all_labels = [int(l) for l in labels] # címkék int-té alakítása

        for i in tqdm(range(len(images))):
            prediction = self.predict(images[i])
            predictions.append(prediction)

            if prediction == all_labels[i]:
                num_correct += 1

        # Standard accuracy, F1 és Recall metrikák kiszámítása
        accuracy = (num_correct * 100.0) / len(images)
        f1 = f1_score(all_labels, predictions, average='weighted')
        recall = recall_score(all_labels, predictions, average='weighted')

        print(f"Accuracy: {accuracy:.4f}%")
        print(f"F1-score: {f1:.4f}%")
        print(f"Recall: {recall:.4f}%")
        return accuracy, f1, recall



    def inference(self, image_path):
        """ Inference fázis, bemeneti kép feldolgozása"""
        img_array = cv2.imread(image_path, 0) # kép beolvasása OpenCV-vel, fekete-fehér formátumban

        if img_array is None:
            raise ValueError(f"Error loading in image for inference: {image_path}")

        img_array = img_array.astype(np.float32) # átalakítás float32-vé
        img_flattened = img_array.reshape(-1) # kép kilapítása

        # Kép normalizálása (Sum-scaling)
        pixel_sum = np.sum(img_flattened)

        if pixel_sum > 0:
            img_input = img_flattened / pixel_sum
        else:
            img_input = img_flattened 

        active_kcs, prediction = self.get_latent_vector(img_input) # latent vector kimentése

        return prediction, active_kcs

    def get_latent_vector(self, image):
        """ Belső reprezentáció (latent vektorok) kinyerése a modellből """
        self.run_image(image, label=None)

        # Adatok mentése a GPU-ról
        self.model_wrapper.model.pull_recording_buffers_from_device()

        _, kc_spike_ids = self.model_wrapper.kc.spike_recording_data[0]
        active_kcs = np.unique(kc_spike_ids).astype(np.uint32)

        mbon_spike_times, mbon_spike_ids = self.model_wrapper.mbon.spike_recording_data[0]
        predicted_label = -1
        if len(mbon_spike_times) > 0:
            predicted_label = int(mbon_spike_ids[np.argmin(mbon_spike_times)])

        # Értékek visszaállítása
        self.model_wrapper.model.custom_update("reset_group")
        self.model_wrapper.mbon_input.vars["magnitude"].view[:] = 0.0
        self.model_wrapper.mbon_input.vars["magnitude"].push_to_device()

        return active_kcs, predicted_label

    def extract_all_data(self, images, labels):
        """ Latent vektorok kimentése címkékkel és képekkel együtt"""
        active_kc_ids = []
        predictions = []
        stored_labels = []
        stored_images = []

        for i in tqdm(range(len(images))):
            kc_ids, pred = self.get_latent_vector(images[i])

            active_kc_ids.append(kc_ids)
            predictions.append(pred)
            stored_labels.append(labels[i])
            img = images[i].reshape(28, 28)

            img = img / img.max() if img.max() > 0 else img

            stored_images.append((img * 255).astype(np.uint8))

        return {
            "active_kc_ids": np.array(active_kc_ids, dtype=object),
            "predictions": np.array(predictions, dtype=np.int32),
            "labels": np.array(stored_labels),
            "images": np.array(stored_images),
            "model_type": "mb"
        }

    def save_model_params(self, weights_path="kc_mbon_g.npy", indices_path="pn_kc_ind.npy"):
        """Modell adatok kimentése (súly, sparse indexek) .npy fájlokba"""

        # KC - MBON súlyok elmentése
        self.model_wrapper.kc_mbon.vars["g"].pull_from_device()
        np.save(weights_path, self.model_wrapper.kc_mbon.vars["g"].view)

        # PN - KC indexek elmentése
        self.model_wrapper.pn_kc.pull_connectivity_from_device()
        np.save(indices_path, np.vstack((self.model_wrapper.pn_kc.get_sparse_pre_inds(), self.model_wrapper.pn_kc.get_sparse_post_inds())))

    def inference_with_spikes(self, image_path):
        """Inference fázis latent vektorok kimentése nélkül, tüzelési id-kal és időkkel a raster plothoz (UI)"""
        img_array = cv2.imread(image_path, 0) # kép beolvasása OpenCV-vel, fekete-fehér formátumban

        if img_array is None:
            raise ValueError(f"Error loading in image for inference: {image_path}")

        img_array = img_array.astype(np.float32) # átalakítás float32-vé
        img_flattened = img_array.reshape(-1) # kép kilapítása

        # Kép normalizálása (Sum-scaling)
        pixel_sum = np.sum(img_flattened)

        if pixel_sum > 0:
            img_input = img_flattened / pixel_sum
        else:
            img_input = img_flattened 

        # Tüzelési értékek elmentése
        self.run_image(img_input, label=None)
        self.model_wrapper.model.pull_recording_buffers_from_device()

        pn_times, pn_ids = self.model_wrapper.pn.spike_recording_data[0]
        kc_times, kc_ids = self.model_wrapper.kc.spike_recording_data[0]
        mbon_times, mbon_ids = self.model_wrapper.mbon.spike_recording_data[0]

        active_kcs = np.unique(kc_ids).astype(np.uint32)

        predicted_label = -1
        if len(mbon_times) > 0:
            predicted_label = int(mbon_ids[np.argmin(mbon_times)])

        result = {
            "prediction": predicted_label,
            "active_kcs": active_kcs,
            "spikes": {
                "pn": {
                    "times": np.array(pn_times),
                    "ids": np.array(pn_ids),
                },
                "kc": {
                    "times": np.array(kc_times),
                    "ids": np.array(kc_ids),
                },
                "mbon": {
                    "times": np.array(mbon_times),
                    "ids": np.array(mbon_ids),
                },
            },
        }

        # Értékek visszaállítása
        self.model_wrapper.model.custom_update("reset_group")
        self.model_wrapper.mbon_input.vars["magnitude"].view[:] = 0.0
        self.model_wrapper.mbon_input.vars["magnitude"].push_to_device()

        return result

def load_mnist():
    """MNIST számjegyek betöltése, feldolgozása, train és test halmazok szétválasztása"""
    mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    train_imgs = mnist.train_images().reshape(60000, -1).astype(np.float32)
    train_imgs /= np.sum(train_imgs, axis=1)[:, np.newaxis]
    train_labels = mnist.train_labels()

    test_imgs = mnist.test_images().reshape(10000, -1).astype(np.float32)
    test_imgs /= np.sum(test_imgs, axis=1)[:, np.newaxis]
    test_labels = mnist.test_labels()
    return train_imgs, train_labels, test_imgs, test_labels
