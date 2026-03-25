import os
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
import torch
import umap
from sklearn.metrics import confusion_matrix
from streamlit_drawable_canvas import st_canvas
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from create_database import KenyonDB, NpzStrategy, PtStrategy
from mushroom_body_class import MBConfig, MBSimulator, MushroomBodyModel, load_mnist
from reservoir_models import ReservoirConfig, ReservoirModel, ReservoirTrainer, SparseConfig


@dataclass(frozen=True)
class AppPaths:
    """Az apphoz szükséges fájl útvonalak definiálása"""
    mnist_root: str
    user_images_dir: str
    model_weight_dir: str
    image_path: str
    db_path: str
    upload_dir: str

    @classmethod
    def build(cls) -> "AppPaths":
        mnist_root = os.path.dirname(os.path.abspath(__file__)) # projekt gyökérmappája
        user_images_dir = os.path.join(mnist_root, "user_images") # felhaszáló által rajzolt számjegyek mappája
        model_weight_dir = os.path.join(mnist_root, "model_weights") # mpdell súlyokat tároló mappa
        image_path = os.path.join(user_images_dir, "digit.png") # felhasználó által rajzolt MNIST számjegy
        db_path = os.path.join(mnist_root, "kenyon.db") # SqLite adatbázis fájl
        upload_dir = os.path.join(mnist_root, "uploaded_latent_vectors") # felhasználó által feltöltött latent vektorok mappája

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(user_images_dir, exist_ok=True)

        return cls(
            mnist_root=mnist_root,
            user_images_dir=user_images_dir,
            model_weight_dir=model_weight_dir,
            image_path=image_path,
            db_path=db_path,
            upload_dir=upload_dir,
        )


class AppConfig:
    """Modell adatok definiálása"""
    SIZE = 192
    MODEL_LABELS = {
        "Spiking Neural Network (Mushroom Body)": "mb",
        "Echo State Network (Watts-Strogatz)": "ws",
        "Echo State Network (Barabási-Albert)": "ba",
        "Echo State Network (Erdős-Rényi)": "er",
    }
    MODEL_LABEL_ORDER = list(MODEL_LABELS.keys())
    MODEL_KEY_TO_NAME = {value: key for key, value in MODEL_LABELS.items()}

    @classmethod
    def topology_map(cls) -> Dict[str, Tuple[str, str]]:
        return {
            "Echo State Network (Watts-Strogatz)": (
                os.path.join("checkpoints_watts_strogatz", "best_model.pt"),
                os.path.join("checkpoints_watts_strogatz", "best_params.json"),
            ),
            "Echo State Network (Barabási-Albert)": (
                os.path.join("checkpoints_barabasi_albert", "best_model.pt"),
                os.path.join("checkpoints_barabasi_albert", "best_params.json"),
            ),
            "Echo State Network (Erdős-Rényi)": (
                os.path.join("checkpoints_erdos_renyi", "best_model.pt"),
                os.path.join("checkpoints_erdos_renyi", "best_params.json"),
            ),
        }


@dataclass
class InferencePayload:
    """Felhasználói felület paramétereinek definiálása"""
    selected_model: str
    model_type_key: str
    prediction: Optional[int] = None
    results: Optional[List[Dict[str, Any]]] = None
    umap_fig: Any = None
    cm_fig: Any = None
    fig_overlap: Any = None
    fig_scores: Any = None
    hero_fig: Any = None
    secondary_fig: Any = None
    summary_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.results is None:
            self.results = []
        if self.summary_metrics is None:
            self.summary_metrics = {}


class SessionManager:
    """Session adatainak inicializálása"""
    @staticmethod
    def initialize(model_order: List[str]) -> None:
        if "db_logs" not in st.session_state:
            st.session_state.db_logs = []
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = model_order[0]
        if "result_count" not in st.session_state:
            st.session_state.result_count = 5




class ModelRegistry:
    """Modell hiperparaméterek betöltése"""
    def __init__(self, paths: AppPaths, config: AppConfig) -> None:
        self.paths = paths
        self.config = config

    def get_model_type_key(self, model_name: str) -> str:
        return self.config.MODEL_LABELS[model_name]

    def get_model_name(self, model_key: str) -> str:
        return self.config.MODEL_KEY_TO_NAME[model_key]

    def is_echo_state_model(self, model_name: str) -> bool:
        return model_name in self.config.topology_map()

    @staticmethod
    def apply_best_params_to_mbconfig(params: Dict[str, Any]) -> None:
        """Optuna optimalizáció során elmentett legjobb paraméterek beolvasása json-ből az snn modellhez"""
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

        MBConfig.KC_MBON_PARAMS.update(
            {
                "eta": params["eta"],
                "tauE": params["tauE"],
                "rho": params["rho"],
                "wMax": params["wMax"],
            }
        )

    # Modell cachelése, _self: hash ellenőrzés kihagyása a cacheléshez
    @st.cache_resource(show_spinner=False)
    def load_snn_model(_self) -> MBSimulator:
        """Optuna optimalizáció során elmentett paraméterek, súlyok, indexek betöltése az snn modellhez"""
        pn_kc_ind_path = os.path.join(_self.paths.model_weight_dir, "best_pn_kc_indices.npy") # PN - KC sparse indexek
        kc_mbon_g_path = os.path.join(_self.paths.model_weight_dir, "best_kc_mbon_weights.npy") # KC - MBON súlyok
        best_params_path = os.path.join(_self.paths.model_weight_dir, "best_params.json") # hálózat hiperparaméterei

        # Json beolvasása
        with open(best_params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)

        _self.apply_best_params_to_mbconfig(best_params)

        pn_kc_ind = np.load(pn_kc_ind_path)
        kc_mbon_g = np.load(kc_mbon_g_path)

        # SNN modell létrehozása
        model = MushroomBodyModel(
            name="mnist_eval",
            is_training=False,
            sparse_indices=pn_kc_ind,
            kc_mbon_g=kc_mbon_g,
        )
        model.build_and_load()
        model.set_kc_mbon_weights(kc_mbon_g)
        return MBSimulator(model)

    # Modell cachelése
    @st.cache_resource(show_spinner=False)
    def load_reservoir_model(_self, weight_path: str, params_path: str) -> ReservoirTrainer:
        """Reservoir modell paraméterek betöltése json-ből"""
        with open(params_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        res_cfg = ReservoirConfig(**payload["res_cfg"])
        sparse_cfg = SparseConfig(**payload["sparse_cfg"])
        device = torch.device("cpu")

        # Reservoir modellek létrehozása
        model = ReservoirModel(res_cfg, sparse_cfg)
        trainer = ReservoirTrainer(model, sparse_cfg, device)
        trainer.load_weights(weight_path)
        return trainer

class DatabaseObserver(ABC):
    "Absztrakt osztály az observer tervezési mintához az adatbázis logokhoz"
    @abstractmethod
    def update(self, message: str):
        pass

class DatabaseService:
    """Adatbázist kezelő osztály"""
    def __init__(self, paths: AppPaths, registry: ModelRegistry) -> None:
        self.paths = paths
        self.registry = registry
        self.db = KenyonDB(paths.db_path)
        self.__observers = []

    def attach(self, observer: DatabaseObserver):
        """Új megfigyelő hozzáadása"""
        if observer not in self.__observers:
            self.__observers.append(observer)

    def detach(self, observer: DatabaseObserver):
        """Megfigyelő törlése a listából"""
        self.__observers.remove(observer)

    def notify(self, message: str):
        """Megfigyelők értesítése"""
        for observer in self.__observers:
            observer.update(message)

    def save_uploaded_latent_file(self, uploaded_file) -> str:
        """Feltöltött latent vektor fájlok elmentése"""
        save_path = os.path.join(self.paths.upload_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return save_path

    def populate_from_uploaded_file(self, uploaded_file, selected_model_name: str) -> bool:
        """Adatbázis felpopulálása feltöltött latent vektor fájlból"""
        if uploaded_file is None:
            self.notify(f"[SKIP] No file uploaded for {selected_model_name}.") # hibás fájl esetén logolás
            return False

        model_type = self.registry.get_model_type_key(selected_model_name)
        saved_file_path = self.save_uploaded_latent_file(uploaded_file)

        strategies = {
            ".npz": NpzStrategy(self.db),
            ".pt": PtStrategy(self.db)
        }

        try:
            file_ext = os.path.splitext(saved_file_path)[1].lower()
            if file_ext in strategies:
                strategies[file_ext].import_data(saved_file_path, model_type)
            else:
                self.notify(
                    f"[ERROR] Unsupported file type for {selected_model_name}: {uploaded_file.name}" # hibás fájl esetén logolás
                )
                return False

            # Összesítő logok kiíratása
            summary = self.db.get_database_summary()
            model_summary = summary["models"][model_type]
            self.notify(
                f"[OK] {selected_model_name} <- {uploaded_file.name} | "
                f"images={model_summary['image_count']} vectors={model_summary['vector_count']}"
            )
            return True
        except Exception as exc:
            self.notify(f"[ROLLBACK] Upload failed: {exc}") # exception esetén ROLLBACK az adatbázisban
            return False

    def summary(self) -> Dict[str, Any]:
        """Adatbázis összesítő lekérése"""
        return self.db.get_database_summary()

    def clear_model_data(self, model_key: str) -> None:
        """Kiválasztott modell törlés funkciójának meghívása"""
        self.db.clear_database_by_model(model_key)

    def get_record_by_image_id(self, image_id: int) -> Optional[Dict[str, Any]]:
        """Rekord lekérdezése image id alapján funkció meghívása"""
        return self.db.get_record_by_image_id(image_id)

    def similarity_search(self, query_active_ids: List[int], model_type: str, top_k: int, progress_callback=None):
        """Fő hasonlósági keresés funkciójának meghívása"""
        return self.db.similarity_search(
            query_active_ids=query_active_ids,
            model_type=model_type,
            top_k=top_k
        )

    def get_all_vectors_by_model(self, model_type: str):
        """Összes vektort lekérdező funkció meghívása"""
        return self.db.get_all_vectors_by_model(model_type)
    
class Logger(DatabaseObserver):
    """Frissiíti a Streamlit session eseményeket"""
    def update(self, message):
        if "db_logs" not in st.session_state:
            st.session_state.db_logs = []
        
        st.session_state.db_logs.insert(0, message)
        st.session_state.db_logs = st.session_state.db_logs[:50]


class VisualizationService:
    """Diagramok megjelenítéséért felelős osztály"""
    def __init__(self, paths: AppPaths, db_service: DatabaseService, registry: ModelRegistry) -> None:
        self.paths = paths
        self.db_service = db_service
        self.registry = registry

    @staticmethod
    def prepare_image_for_display(img):
        """Képek előkészítése megjelenítésre"""
        if img is None:
            return None

        # Kép formátumok egyesítése PyTorch és PyGenn modellek esetén
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[0] == 1:
                return img[0]
            if img.ndim == 3 and img.shape[0] in (3, 4):
                return np.transpose(img, (1, 2, 0))
        return img

    @staticmethod
    def plot_similarity_metrics_split(results):
        """Bar plot a hasonlósági metrikák ábzázolására"""
        labels = [f"ID {r['image_id']} ({r['label']})" for r in results]

        overlap = [r["metrics"]["overlap"] for r in results]
        jaccard = [r["metrics"]["jaccard"] for r in results]
        dice = [r["metrics"]["dice"] for r in results]
        overlap_coeff = [r["metrics"]["overlap_coeff"] for r in results]

        fig_overlap = go.Figure()
        fig_overlap.add_bar(x=labels, y=overlap, name="Overlap")
        fig_overlap.update_layout(title="Overlap", xaxis_title="Retrieved Images", yaxis_title="Count", height=350)

        fig_scores = go.Figure()
        fig_scores.add_bar(x=labels, y=jaccard, name="Jaccard")
        fig_scores.add_bar(x=labels, y=dice, name="Dice")
        fig_scores.add_bar(x=labels, y=overlap_coeff, name="Overlap coefficient")
        fig_scores.update_layout(
            title="Normalized Similarity Scores",
            xaxis_title="Retrieved Images",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            barmode="group",
            height=350,
        )
        return fig_overlap, fig_scores

    @staticmethod
    def build_reservoir_graph_from_weight_hh(reservoir, max_nodes: int = 120, threshold: float = 1e-8):
        """Reservoir modellek gráf magjának reprodukciója"""
        weight_hh = reservoir.weight_hh
        if hasattr(weight_hh, "detach"):
            weight_hh = weight_hh.detach().cpu().numpy()

        n = min(weight_hh.shape[0], max_nodes)
        W = weight_hh[:n, :n]
        G = nx.DiGraph() # Irányított gráf létrehozása (networkx)

        for i in range(n):
            G.add_node(i) # Csúcsok hozzáadása

        rows, cols = np.where(np.abs(W) > threshold)
        for i, j in zip(rows, cols):
            if i != j:
                G.add_edge(int(i), int(j), weight=float(W[i, j])) # Élek és súlyok hozzáadása
        return G

    @staticmethod
    def plot_reservoir_graph_core(graph, title: str = "Reservoir Graph Core"):
        """Reservoir gráf magok vizualizációja"""
        pos = nx.spring_layout(graph, seed=42)

        edge_x, edge_y = [], []
        for u, v in graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", name="Edges")

        node_x, node_y, node_text, node_degree = [], [], [], []
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            deg = graph.degree(node)
            node_degree.append(deg)
            node_text.append(f"Node {node}<br>Degree: {deg}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            text=node_text,
            hoverinfo="text",
            marker=dict(size=9, color=node_degree, showscale=True, colorbar=dict(title="Degree"), line=dict(width=1)),
            name="Nodes",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=title,
            height=650,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig

    @staticmethod
    def get_reservoir_graph_stats(reservoir, threshold: float = 1e-8) -> Dict[str, Any]:
        """Reservoir modellek gráf magjának alap adatai"""
        weight_hh = reservoir.weight_hh
        if hasattr(weight_hh, "detach"):
            weight_hh = weight_hh.detach().cpu().numpy()

        num_nodes = weight_hh.shape[0] # Csúcsok száma
        num_edges = int((np.abs(weight_hh) > threshold).sum()) # Élek száma
        density = num_edges / (num_nodes * num_nodes) # Sűrűség
        return {"nodes": num_nodes, "edges": num_edges, "density": density}

    @staticmethod
    def plot_mb_raster(spike_data):
        """Raster plot az SNN modell vizuálására"""
        fig = go.Figure()
        populations = [("pn", "PN"), ("kc", "KC"), ("mbon", "MBON")]
        y_offset = 0

        # Neuronok tüzelésének időbeli elhelyezkedése id-k szerint
        for key, label in populations:
            times = spike_data[key]["times"]
            ids = spike_data[key]["ids"]
            if len(ids) == 0:
                continue

            shifted_ids = ids + y_offset
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=shifted_ids,
                    mode="markers",
                    name=label,
                    marker=dict(size=4),
                    text=[f"{label} {int(i)}" for i in ids],
                    hovertemplate="t=%{x:.2f} ms<br>%{text}<extra></extra>",
                )
            )
            y_offset += int(ids.max()) + 20

        fig.update_layout(
            title="Mushroom Body Spike Raster",
            xaxis_title="Time (ms)",
            yaxis_title="Neuron ID / Population",
            height=600,
            legend_title="Population",
        )
        return fig

    @staticmethod
    def plot_mbon_activity(spike_data):
        """MBON neuronok aktivitásának ábrázolása"""
        mbon_ids = spike_data["mbon"]["ids"]
        counts = np.bincount(mbon_ids, minlength=10) if len(mbon_ids) > 0 else np.zeros(10, dtype=int)

        fig = go.Figure()
        fig.add_bar(x=list(range(10)), y=counts, name="MBON spike count")
        fig.update_layout(title="MBON Activity", xaxis_title="MBON neuron / class", yaxis_title="Spike count", height=350)
        return fig

    @staticmethod
    def build_binary_matrix_from_model_vectors(model_vectors):
        """Vektorok alapján egy bináris mátrix felépítése UMAP-hez"""
        if not model_vectors:
            return None, None, None

        vocab = sorted({int(neuron_id) for row in model_vectors for neuron_id in row["active_ids"]})
        if len(vocab) < 2:
            return None, None, None

        id_to_col = {nid: i for i, nid in enumerate(vocab)}
        X = np.zeros((len(model_vectors), len(vocab)), dtype=np.float32)
        meta = []

        for row_idx, row in enumerate(model_vectors):
            active_ids = sorted(set(int(x) for x in row["active_ids"]))
            for nid in active_ids:
                col = id_to_col.get(nid)
                if col is not None:
                    X[row_idx, col] = 1.0
            meta.append({"image_id": int(row["image_id"]), "label": int(row["label"])})

        return X, pd.DataFrame(meta), vocab

    # UMAP cachelése
    @st.cache_resource(show_spinner=False)
    def compute_cached_global_umap_bundle(_self, db_path: str, model_type: str):
        """Adatbázisban tárolt vektorok UMAP vetítése"""
        db = KenyonDB(db_path)
        model_vectors = db.get_all_vectors_by_model(model_type)

        X, df, vocab = _self.build_binary_matrix_from_model_vectors(model_vectors)
        if X is None or df is None or vocab is None or X.shape[0] < 3:
            return None

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, X.shape[0] - 1)),
            min_dist=0.15,
            metric="jaccard",
            random_state=42,
            transform_seed=42,
        )
        emb = reducer.fit_transform(X)

        df = df.copy()
        df["x"] = emb[:, 0]
        df["y"] = emb[:, 1]
        return {"df": df, "reducer": reducer, "vocab": vocab}

    @staticmethod
    def project_query_with_cached_bundle(query_ids: List[int], bundle):
        """Adott lekérdezés UMAP vetítése"""
        if bundle is None:
            return None

        vocab = bundle["vocab"]
        reducer = bundle["reducer"]
        if not vocab or reducer is None:
            return None

        id_to_col = {nid: i for i, nid in enumerate(vocab)}
        X_query = np.zeros((1, len(vocab)), dtype=np.float32)

        found_any = False
        for nid in sorted(set(int(x) for x in query_ids)):
            col = id_to_col.get(nid)
            if col is not None:
                X_query[0, col] = 1.0
                found_any = True

        if not found_any:
            return None

        query_emb = reducer.transform(X_query)
        return {"x": float(query_emb[0, 0]), "y": float(query_emb[0, 1])}

    @staticmethod
    def plot_cached_global_umap(cached_df, results, query_point, model_name: str):
        """UMAP megejelenítése"""
        if cached_df is None or cached_df.empty:
            return None

        fig = go.Figure()
        for digit_label in sorted(cached_df["label"].unique()):
            subset = cached_df[cached_df["label"] == digit_label]
            fig.add_trace(
                go.Scatter(
                    x=subset["x"],
                    y=subset["y"],
                    mode="markers",
                    name=f"Label {digit_label}",
                    customdata=np.stack([subset["image_id"].to_numpy(), subset["label"].to_numpy()], axis=1),
                    hovertemplate="Image ID: %{customdata[0]}<br>Label: %{customdata[1]}<extra></extra>",
                    marker=dict(size=6, opacity=0.35),
                )
            )

        result_ids = {int(r["image_id"]) for r in results}
        if result_ids:
            result_df = cached_df[cached_df["image_id"].isin(result_ids)].copy()
            rank_lookup = {int(r["image_id"]): i + 1 for i, r in enumerate(results)}
            result_df["rank"] = result_df["image_id"].map(rank_lookup)

            fig.add_trace(
                go.Scatter(
                    x=result_df["x"],
                    y=result_df["y"],
                    mode="markers+text",
                    name="Results",
                    text=result_df["rank"].astype(str),
                    textposition="top center",
                    customdata=np.stack(
                        [result_df["image_id"].to_numpy(), result_df["label"].to_numpy(), result_df["rank"].to_numpy()],
                        axis=1,
                    ),
                    hovertemplate=(
                        "Result rank: %{customdata[2]}<br>"
                        "Image ID: %{customdata[0]}<br>"
                        "Label: %{customdata[1]}<extra></extra>"
                    ),
                    marker=dict(size=14, symbol="diamond", line=dict(width=2)),
                )
            )

        if query_point is not None:
            fig.add_trace(
                go.Scatter(
                    x=[query_point["x"]],
                    y=[query_point["y"]],
                    mode="markers",
                    name="Query",
                    hovertemplate="Current query<extra></extra>",
                    marker=dict(size=18, symbol="star", line=dict(width=2)),
                )
            )

        fig.update_layout(
            title=f"Global UMAP — {model_name}",
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            height=650,
            legend_title="Digit label",
        )
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true: List[int], y_pred: List[int], model_name: str):
        """Konfúziós mátrix megjelenítése"""
        labels = list(range(10))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        annotation_text = [[str(v) for v in row] for row in cm]
        fig = ff.create_annotated_heatmap(
            z=cm_norm,
            x=[str(x) for x in labels],
            y=[str(y) for y in labels],
            annotation_text=annotation_text,
            colorscale="Blues",
            showscale=True,
        )
        fig.update_layout(
            title=f"Confusion Matrix — {model_name}",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=600,
        )
        return fig

class ModelStrategy(ABC):
    """Absztrakt osztály ami definiál egy közös interfészt a modell típusoknak"""

    @abstractmethod
    def load(self):
        """Modell súlyok és paraméterek betöltése"""
        pass

    @abstractmethod
    def run_inference(image_path):
        """Modellek inference metódusait futtatja és visszakapja a predikciót és latent vektorokat"""
        pass

    @abstractmethod
    def plot_results(inf_result, payload, num_results):
        """Inference eredmények vizualizációja"""
        pass

class SnnStrategy(ModelStrategy):
    def __init__(self, registry, visualizer):
        self.registry = registry
        self.model = None
        self.visualizer = visualizer

    def load(self):
        self.model = self.registry.load_snn_model()

    def run_inference(self, image_path):
        result = self.model.inference_with_spikes(image_path)

        return {
            "prediction": int(result["prediction"]),
            "active_kcs": list(result["active_kcs"]),
            "spike_data": result["spikes"]
        }

    def plot_results(self, inf_result, payload, num_results):
        prediction = inf_result["prediction"]
        active_kcs = inf_result["active_kcs"]
        spike_data = inf_result["spike_data"]

        payload.prediction = prediction
        payload.hero_fig = self.visualizer.plot_mb_raster(spike_data)
        payload.secondary_fig = self.visualizer.plot_mbon_activity(spike_data)

        payload.summary_metrics = {
            "PN spikes": len(spike_data["pn"]["ids"]),
            "Active KCs": len(active_kcs),
            "MBON spikes": len(spike_data["mbon"]["ids"]),
            "Retrieved": num_results,
        }

        return active_kcs

class ReservoirStrategy(ModelStrategy):
    def __init__(self, registry, visualizer, config, selected_model):
        self.registry = registry
        self.model = None
        self.visualizer = visualizer
        self.selected_model = selected_model
        self.config = config

    def load(self):
        weight_file, params_file = self.config.topology_map()[self.selected_model]
        self.model = self.registry.load_reservoir_model(weight_file, params_file)

    def run_inference(self, image_path):
        result = self.model.inference(image_path, topk=64) 
        
        return {
            "retrieval_ids": list(result["latent_indices"]),
            "prediction": int(result["prediction"]),
            "reservoir": self.model.model.reservoir
        }

    def plot_results(self, inf_result, payload, num_results):
        prediction = inf_result["prediction"]
        retrieval_ids = inf_result["retrieval_ids"]
        reservoir = inf_result["reservoir"]

        payload.prediction = prediction
        
        try:
            reservoir_graph = self.visualizer.build_reservoir_graph_from_weight_hh(reservoir, max_nodes=120)
            payload.hero_fig = self.visualizer.plot_reservoir_graph_core(reservoir_graph, title=f"{self.selected_model} Reservoir Core")
            
            stats = self.visualizer.get_reservoir_graph_stats(reservoir)
            payload.summary_metrics = {
                "Nodes": stats["nodes"],
                "Edges": stats["edges"],
                "Density": f"{stats['density']:.4f}",
                "Retrieved": num_results,
            }
        except Exception as exc:
            payload.error = f"Graph visualization failed: {exc}"
            
        return retrieval_ids

class ModelFactory:
    @staticmethod
    def create(model_name, registry, config, visualizer):
        if "Mushroom Body" in model_name:
            return SnnStrategy(registry, visualizer)
        else:
            model_type_key = registry.get_model_type_key(model_name)
            return ReservoirStrategy(registry, visualizer, config, model_type_key)


class InferenceService:
    """Inferencek futtatásáért felelős osztály"""
    def __init__(
        self,
        paths: AppPaths,
        config: AppConfig,
        registry: ModelRegistry,
        db_service: DatabaseService,
        visualizer: VisualizationService,
    ) -> None:
        self.paths = paths
        self.config = config
        self.registry = registry
        self.db_service = db_service
        self.visualizer = visualizer


    def run_inference(self, img: np.ndarray, selected_model: str, num_results: int) -> InferencePayload:
        """Teljes polimorf inference pipeline"""
        model_type_key = self.registry.get_model_type_key(selected_model)
        payload = InferencePayload(selected_model=selected_model, model_type_key=model_type_key)

        try:
            cv2.imwrite(self.paths.image_path, img)

            strategy = ModelFactory.create(selected_model, self.registry, self.config, self.visualizer)
            strategy.load()

            inf_result = strategy.run_inference(self.paths.image_path)
            retrieval_ids = strategy.plot_results(inf_result, payload, num_results)

            payload.results = self.db_service.similarity_search(
                query_active_ids=retrieval_ids,
                model_type=model_type_key,
                top_k=num_results,
            )

            umap_bundle = self.visualizer.compute_cached_global_umap_bundle(self.paths.db_path, model_type_key)
            if umap_bundle:
                query_point = self.visualizer.project_query_with_cached_bundle(retrieval_ids, umap_bundle)
                payload.umap_fig = self.visualizer.plot_cached_global_umap(
                    cached_df=umap_bundle["df"],
                    results=payload.results,
                    query_point=query_point,
                    model_name=selected_model,
                )

            y_true, y_pred = self.evaluate_model_confusion(selected_model, num_samples=1000)
            payload.cm_fig = self.visualizer.plot_confusion_matrix(y_true, y_pred, selected_model)
            
            if payload.results:
                payload.fig_overlap, payload.fig_scores = self.visualizer.plot_similarity_metrics_split(payload.results)

        except Exception as exc:
            payload.error = f"Inference Error: {str(exc)}"

        return payload

    @st.cache_data(show_spinner=False)
    def evaluate_model_confusion(_self, selected_model: str, num_samples: int = 1000):
        """Konfúziós mátrixhoz adatok kiszámítása"""
        y_true, y_pred = [], []

        if selected_model == "Spiking Neural Network (Mushroom Body)":
            simulator = _self.registry.load_snn_model()
            _, _, test_imgs, test_labels = load_mnist()

            limit = min(num_samples, len(test_imgs))
            for i in range(limit):
                pred = simulator.predict(test_imgs[i])
                y_pred.append(int(pred))
                y_true.append(int(test_labels[i]))

        else:
            weight_file, params_file = _self.config.topology_map()[selected_model]
            trainer = _self.registry.load_reservoir_model(weight_file, params_file)

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
            test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

            trainer.model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(trainer.device)
                    y = y.to(trainer.device)

                    logits, _ = trainer.model(x.squeeze(1))
                    preds = logits.argmax(dim=1)

                    y_pred.extend(preds.cpu().numpy().tolist())
                    y_true.extend(y.cpu().numpy().tolist())

                    if len(y_true) >= num_samples:
                        break

            trainer.model.train()

        return y_true[:num_samples], y_pred[:num_samples]

    def warm_umap_cache_for_all_models(self) -> None:
        """UMAP kiszámolása és cachelése előre"""
        for model_key in ["mb", "ws", "ba", "er"]:
            try:
                self.visualizer.compute_cached_global_umap_bundle(self.paths.db_path, model_key)
                SessionManager.add_db_log(f"[OK] UMAP cached for {model_key}")
            except Exception as exc:
                SessionManager.add_db_log(f"[WARN] UMAP cache failed for {model_key}: {exc}")


class StreamlitRenderer:
    """Streamlit felület megjelenítéséért felelős osztály"""
    def __init__(
        self,
        paths: AppPaths,
        config: AppConfig,
        registry: ModelRegistry,
        db_service: DatabaseService,
        visualizer: VisualizationService,
        inference: InferenceService,
    ) -> None:
        self.paths = paths
        self.config = config
        self.registry = registry
        self.db_service = db_service
        self.visualizer = visualizer
        self.inference = inference

    def render_database_summary(self) -> None:
        """Adatbázis összesítő adatok megjelenítése"""
        summary = self.db_service.summary()

        st.subheader("Database logs")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total images", summary["total_images"])
        col2.metric("Total vector rows", summary["total_vectors"])
        loaded_models = sum(1 for m in summary["models"].values() if m["image_count"] > 0)
        col3.metric("Models loaded", f"{loaded_models}/4")

        rows = []
        for model_key in ["mb", "ws", "ba", "er"]:
            stats = summary["models"][model_key]
            rows.append(
                {
                    "Model": self.registry.get_model_name(model_key),
                    "Type": model_key,
                    "Images": stats["image_count"],
                    "Vector rows": stats["vector_count"],
                    "Labels": stats["distinct_labels"],
                    "Index range": "-" if stats["min_image_index"] is None else f"{stats['min_image_index']} - {stats['max_image_index']}",
                }
            )

        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Logok megjelenítése
        with st.expander("Recent database activity", expanded=True):
            if st.session_state.db_logs:
                for log_line in st.session_state.db_logs:
                    st.code(log_line)
            else:
                st.caption("No database activity yet.")

    def render_similarity_result_cards(self, results: List[Dict[str, Any]]) -> None:
        """Keresési találatok megjelenítése"""
        st.subheader("Top retrieved matches")
        if not results:
            st.warning("No similar database entries found.")
            return

        cols = st.columns(3)
        for idx, result in enumerate(results):
            record = self.db_service.get_record_by_image_id(result["image_id"])
            # Rank kiírása
            with cols[idx % 3].container(border=True):
                st.caption(f"Rank #{idx + 1}")
                if record and record["image"] is not None:
                    st.image(self.visualizer.prepare_image_for_display(record["image"]), use_container_width=True)
                else:
                    st.info("No preview available")

                # Metaadatok kiírása
                meta1, meta2 = st.columns(2)
                meta1.metric("Label", result["label"])
                meta2.metric("Image ID", result["image_id"])

                # Hasonlósági metrikák kiírása
                st.caption("Similarity")
                m1, m2 = st.columns(2)
                m1.metric("Overlap", int(result["metrics"]["overlap"]))
                m2.metric("Jaccard", f"{result['metrics']['jaccard']:.3f}")
                m3, m4 = st.columns(2)
                m3.metric("Dice", f"{result['metrics']['dice']:.3f}")
                m4.metric("Overlap coeff.", f"{result['metrics']['overlap_coeff']:.3f}")

    def render_input_panel(self) -> Tuple[Optional[np.ndarray], bool]:
        """Canvas az input számjegyek rajzolásához"""
        st.subheader("Input")
        draw_mode = st.toggle("Draw mode", value=True)
        preview_col, canvas_col = st.columns([1, 1])

        with canvas_col:
            canvas_result = st_canvas(
                fill_color="#000000",
                stroke_width=20,
                stroke_color="#FFFFFF",
                background_color="#000000",
                width=self.config.SIZE,
                height=self.config.SIZE,
                drawing_mode="freedraw" if draw_mode else "transform",
                key="canvas",
            )

        img = None
        rescaled = None

        # Kép átméretezése és fekete-fehérré alakítása
        if canvas_result.image_data is not None:
            img = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
            img = cv2.resize(img, (28, 28))
            rescaled = cv2.resize(img, (self.config.SIZE, self.config.SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Feldolgozott kép megjelenítése
        with preview_col:
            st.caption("Model input")
            if rescaled is not None:
                st.image(rescaled, use_container_width=True)
            else:
                st.info("Draw a digit to preview the 28×28 input.")

        # Modell kiválasztása
        st.session_state.selected_model = st.selectbox(
            "Model",
            tuple(self.config.MODEL_LABEL_ORDER),
            index=self.config.MODEL_LABEL_ORDER.index(st.session_state.selected_model),
        )
        # Találati szám kiválasztása
        st.session_state.result_count = st.segmented_control(
            "Retrieved matches",
            options=[3, 5, 7],
            default=st.session_state.result_count,
            selection_mode="single",
        )

        # Program indítása
        run = st.button("Run analysis", use_container_width=True, type="primary")
        clear_cache = st.button("Clear cache", use_container_width=True) # Opcionálisan cache törlése
        if clear_cache:
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

        st.caption("Use the Database tab to upload latent vectors and manage cached data.")
        return img, run

    def render_inference_results(self, payload: InferencePayload) -> None:
        """Inference eredmények betöltése"""
        if payload.error and payload.prediction is None:
            st.error(f"Failed to run analysis: {payload.error}")
            return

        st.subheader("Overview")
        k1, k2, k3= st.columns(3)
        k1.metric("Prediction", payload.prediction)
        k2.metric("Matches", len(payload.results))
        k3.metric("Model", payload.model_type_key.upper())

        st.caption(payload.latent_info or "")

        if payload.summary_metrics:
            cols = st.columns(len(payload.summary_metrics))
            for idx, (label, value) in enumerate(payload.summary_metrics.items()):
                cols[idx].metric(label, value)

        if payload.error and payload.prediction is not None:
            st.warning(payload.error)

        if payload.hero_fig is not None:
            st.plotly_chart(payload.hero_fig, use_container_width=True)

        if payload.secondary_fig is not None:
            with st.container(border=True):
                st.subheader("Secondary activity view")
                st.plotly_chart(payload.secondary_fig, use_container_width=True)

        mid_left, mid_right = st.columns(2)
        with mid_left.container(border=True):
            st.subheader("Global latent space")
            if payload.umap_fig is not None:
                st.plotly_chart(payload.umap_fig, use_container_width=True)
            else:
                st.info("UMAP could not be generated for this model yet.")

        with mid_right.container(border=True):
            st.subheader("Confusion matrix")
            if payload.cm_fig is not None:
                st.plotly_chart(payload.cm_fig, use_container_width=True)

        if payload.fig_overlap is not None and payload.fig_scores is not None:
            metric_left, metric_right = st.columns(2)
            with metric_left.container(border=True):
                st.plotly_chart(payload.fig_overlap, use_container_width=True)
            with metric_right.container(border=True):
                st.plotly_chart(payload.fig_scores, use_container_width=True)

        self.render_similarity_result_cards(payload.results)

    def render_inference_tab(self) -> None:
        """Inference fül létrehozása egy sidebar-ral és a fő lappal"""
        left_col, right_col = st.columns([2, 3])

        with left_col.container(border=True, height="stretch"):
            img, run = self.render_input_panel()

        if run:
            if img is None:
                with right_col.container(border=True, height="stretch"):
                    st.warning("Draw a digit first.")
            else:
                with right_col.container(border=True, height="stretch"):
                    with st.spinner("Running model inference and retrieval..."):
                        payload = self.inference.run_inference(
                            img,
                            st.session_state.selected_model,
                            st.session_state.result_count,
                        )
                    self.render_inference_results(payload)
        else:
            with right_col.container(border=True, height="stretch"):
                st.subheader("Overview")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Prediction", "-")
                p2.metric("Model accuracy", "-")
                p3.metric("Matches", st.session_state.result_count)
                p4.metric("Model", self.registry.get_model_type_key(st.session_state.selected_model).upper())
                st.info("Run the analysis to populate this dashboard with activity charts, confusion matrix, and retrieved matches.")

    def render_database_tab(self) -> None:
        """Adatbázis fül létrehozása"""
        data_left, data_right = st.columns([2, 1])

        # Adatbázis fájlok feltöltése
        with data_left.container(border=True):
            st.subheader("Upload latent vectors")
            st.caption("You can upload one file for each model and populate the database in one pass.")

            uploaded_files = {}
            for model_name in self.config.MODEL_LABEL_ORDER:
                model_key = self.registry.get_model_type_key(model_name)
                uploaded_files[model_name] = st.file_uploader(
                    f"{model_name} ({model_key})",
                    type=["npz", "pt", "pth"],
                    key=f"uploader_{model_key}",
                    help="The file should contain vectors, labels, and optionally images.",
                )
                if uploaded_files[model_name] is not None:
                    st.caption(f"{model_key}: {uploaded_files[model_name].name}")

            # Adatbázis felpopulálása
            up1, up2 = st.columns(2)
            if up1.button("Populate all model files", use_container_width=True):
                success_count = 0
                for model_name in self.config.MODEL_LABEL_ORDER:
                    success_count += int(self.db_service.populate_from_uploaded_file(uploaded_files[model_name], model_name))
                st.cache_resource.clear()
                st.cache_data.clear()
                if success_count:
                    st.success(f"Database update completed for {success_count} model file(s).")
                else:
                    st.warning("No model files were populated.")

            if up2.button("Populate selected model", use_container_width=True):
                model_name = st.session_state.selected_model
                if self.db_service.populate_from_uploaded_file(uploaded_files.get(model_name), model_name):
                    st.success(f"Database update completed for {model_name}.")
                    st.cache_resource.clear()
                    st.cache_data.clear()
                else:
                    st.warning(f"No file populated for {model_name}.")

        # Adatbázis adatok törlése
        with data_right.container(border=True):
            st.subheader("Maintenance")
            model_to_delete = st.selectbox("Model to clear", ["mb", "ws", "ba", "er"])
            confirm_delete = st.checkbox("I understand this will permanently delete data for this model")

            if st.button("Clear model data", use_container_width=True):
                if not confirm_delete:
                    st.warning("Please confirm deletion using the checkbox.")
                else:
                    try:
                        self.db_service.clear_model_data(model_to_delete)
                        st.cache_resource.clear()
                        st.cache_data.clear()
                        st.success(f"Database cleared for model: {model_to_delete}")
                    except Exception as exc:
                        st.error(f"Failed to clear database: {exc}")

        with st.container(border=True):
            self.render_database_summary()


class DigitRecognitionApp:
    """App komponensek inicializálása"""
    def __init__(self) -> None:
        self.paths = AppPaths.build()
        self.config = AppConfig()
        self.registry = ModelRegistry(self.paths, self.config)
        self.db_service = DatabaseService(self.paths, self.registry)
        self.visualizer = VisualizationService(self.paths, self.db_service, self.registry)
        self.inference = InferenceService(self.paths, self.config, self.registry, self.db_service, self.visualizer)
        self.renderer = StreamlitRenderer(
            self.paths,
            self.config,
            self.registry,
            self.db_service,
            self.visualizer,
            self.inference,
        )

        ui_logger = Logger()
        self.db_service.attach(ui_logger)

    def configure_page(self) -> None:
        """Oldal címe, ikonja"""
        st.set_page_config(
            page_title="Digit Recognition Dashboard",
            page_icon=":material/neurology:",
            layout="wide",
        )

    def render_header(self) -> None:
        """Oldal fejléce"""
        st.write("# :material/neurology: MNIST image retrieval dashboard")
        st.write("Compare models by drawing MNIST digits and inspecting similarity results")

    def run(self) -> None:
        """Oldal futtatása"""
        self.configure_page()
        SessionManager.initialize(self.config.MODEL_LABEL_ORDER)
        self.render_header()

        app_tab, data_tab = st.tabs(["Inference", "Database"])
        with app_tab:
            self.renderer.render_inference_tab()
        with data_tab:
            self.renderer.render_database_tab()


if __name__ == "__main__":
    DigitRecognitionApp().run()
