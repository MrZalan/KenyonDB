import os
import json
import numpy as np

from mushroom_body_class import MushroomBodyModel, MBSimulator, MBConfig, load_mnist


MODEL_WEIGHT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_mushroom_body")
latent_path = os.path.join(MODEL_WEIGHT_DIR, "mb_latent_data.npz")

def apply_best_params_to_mbconfig(params):
    # A legjobb paramétereket tartalmazó json fájlból beolvassuk a paramétereket
    MBConfig.PRESENT_TIME_MS = params["PRESENT_TIME_MS"]
    MBConfig.INPUT_SCALE = params["INPUT_SCALE"]
    MBConfig.NUM_KC = params["NUM_KC"]
    MBConfig.PN_KC_FAN_IN = params["PN_KC_FAN_IN"]

    MBConfig.LIF_PARAMS["Vthresh"] = params["Vthresh"]
    MBConfig.LIF_PARAMS["TauM"] = params["TauM"]
    MBConfig.PN_REFRAC = params["PN_REFRAC"]

    MBConfig.PN_KC_WEIGHT = params["PN_KC_WEIGHT"]
    MBConfig.PN_KC_TAU = params["PN_KC_TAU"]
    MBConfig.KC_GGN_WEIGHT = params["KC_GGN_WEIGHT"]
    MBConfig.GGN_KC_WEIGHT = params["GGN_KC_WEIGHT"]
    MBConfig.GGN_KC_TAU = params["GGN_KC_TAU"]
    MBConfig.KC_MBON_TAU = params["KC_MBON_TAU"]
    MBConfig.MBON_STIMULUS_CURRENT = params["MBON_STIMULUS_CURRENT"]

    MBConfig.KC_MBON_PARAMS.update({
        "eta": params["eta"],
        "tauE": params["tauE"],
        "rho": params["rho"],
        "wMax": params["wMax"],
    })


def load_best_model():
    # Modell fájlok beolvasása
    best_params_path = os.path.join(MODEL_WEIGHT_DIR, "best_params.json")
    pn_kc_ind_path = os.path.join(MODEL_WEIGHT_DIR, "best_pn_kc_indices.npy")
    kc_mbon_g_path = os.path.join(MODEL_WEIGHT_DIR, "best_kc_mbon_weights.npy")

    if not os.path.exists(best_params_path):
        raise FileNotFoundError(f"Missing file: {best_params_path}")
    if not os.path.exists(pn_kc_ind_path):
        raise FileNotFoundError(f"Missing file: {pn_kc_ind_path}")
    if not os.path.exists(kc_mbon_g_path):
        raise FileNotFoundError(f"Missing file: {kc_mbon_g_path}")

    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)

    apply_best_params_to_mbconfig(best_params)

    pn_kc_ind = np.load(pn_kc_ind_path)
    kc_mbon_g = np.load(kc_mbon_g_path)

    print("Loaded best params:")
    print(json.dumps(best_params, indent=2))

    print("\nLoaded KC->MBON weights stats:")
    print("shape:", kc_mbon_g.shape)
    print("min:", kc_mbon_g.min())
    print("max:", kc_mbon_g.max())
    print("mean:", kc_mbon_g.mean())
    print("nonzero:", np.count_nonzero(kc_mbon_g))

    model = MushroomBodyModel(
        name="mnist_best_eval",
        is_training=False,
        sparse_indices=pn_kc_ind,
        kc_mbon_g=kc_mbon_g,
    )
    model.build_and_load()
    model.set_kc_mbon_weights(kc_mbon_g)

    return MBSimulator(model)


if __name__ == "__main__":
    simulator = load_best_model()

    _, _, test_imgs, test_labels = load_mnist()

    accuracy, f1, recall = simulator.evaluate(test_imgs, test_labels)
    print(f"\nMNIST test accuracy: {accuracy:.4f}%")

    print("\nExtracting latent data from test set...")
    data_bundle = simulator.extract_all_data(test_imgs, test_labels)

    np.savez_compressed(
        latent_path,
        active_kc_ids=data_bundle["active_kc_ids"],
        predictions=data_bundle["predictions"],
        labels=data_bundle["labels"],
        images=data_bundle["images"],
        model_type=data_bundle.get("model_type", "mb")
    )

    print(f"Saved latent data to: {latent_path}")
    print("Internal model_type:", data_bundle.get("model_type"))
    print("active_kc_ids shape:", data_bundle["active_kc_ids"].shape)
    print("predictions shape:", data_bundle["predictions"].shape)
    print("labels shape:", data_bundle["labels"].shape)
    print("images shape:", data_bundle["images"].shape)