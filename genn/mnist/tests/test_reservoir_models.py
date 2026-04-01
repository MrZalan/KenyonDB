import torch
import cv2
import pytest
import os
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from reservoir_models import SoftShrink, ReservoirModel, ReservoirConfig, SparseConfig, ReservoirTrainer, set_seed

@pytest.fixture
def mock_mnist_loader():
    """Adathalmaz helyettesítése tesztekhez"""
    set_seed(42)
    x = torch.randn(10, 1, 28, 28)
    y = torch.randint(0, 10, (10,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=2)

def test_reproducibility():
    """Modell architektúra reprodukálásának seedelés tesztelése"""
    topo_params = {"seed": 42}
    cfg = ReservoirConfig(size=100, topology_type="watts_strogatz", topology_params=topo_params)
    scfg = SparseConfig(reservoir_dim=100)
    
    set_seed(42)
    model1 = ReservoirModel(cfg, scfg)
    
    set_seed(42)
    model2 = ReservoirModel(cfg, scfg)

    w1 = model1.reservoir.weight_hh
    w2 = model2.reservoir.weight_hh
    torch.testing.assert_close(w1, w2)

def test_soft_shrink_logic():
    """SoftShrink logika tesztelése"""
    sh_lambda = 0.5
    soft_shrink = SoftShrink(sh_lambda=sh_lambda)
    input = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    expected = torch.tensor([-0.5, 0.0, 0.0, 0.0, 0.5])
    output = soft_shrink(input)
    torch.testing.assert_close(output, expected)

def test_reservoir_freeze():
    """Rezervoir befagyasztásának tesztje"""
    res_cfg = ReservoirConfig(size=100)
    sparse_cfg = SparseConfig(reservoir_dim=100)
    model = ReservoirModel(res_cfg, sparse_cfg)
    
    for param in model.reservoir.parameters():
        assert not param.requires_grad
    
    for param in model.head.parameters():
        assert param.requires_grad

def test_inference_output(tmp_path):
    """Inference output helyességének tesztje"""
    res_cfg = ReservoirConfig(size=128)
    sparse_cfg = SparseConfig(reservoir_dim=128, sparse_dim=256)
    model = ReservoirModel(res_cfg, sparse_cfg)
    trainer = ReservoirTrainer(model, sparse_cfg, torch.device("cpu"))
    
    img_path = str(tmp_path / "test.png")
    cv2.imwrite(img_path, np.random.randint(0, 255, (28, 28), dtype=np.uint8))
    
    result = trainer.inference(img_path, topk=10)
    
    assert "prediction" in result
    assert "latent_indices" in result
    assert len(result["latent_indices"]) == 10
    assert isinstance(result["latent_indices"], np.ndarray)

def test_latent_indices_extraction(mock_mnist_loader):
    """Latent vektor indexek kimentésének tesztje"""
    device = torch.device("cpu")
    model = ReservoirModel(ReservoirConfig(size=100), SparseConfig(reservoir_dim=100))
    trainer = ReservoirTrainer(model, SparseConfig(), device)
    
    topk = 64
    result = trainer.save_latent_indices_with_images(mock_mnist_loader, "ba", topk=topk)
    
    assert result["vectors"].shape == (10, topk)
    assert result["images"].shape == (10, 1, 28, 28)
    assert result["images"].dtype == torch.uint8
    assert result["model_type"] == "ba"

def test_inference_topk_boundary(tmp_path):
    """Kevesebb neuron, mint top_k tesztje"""
    device = torch.device("cpu")
    sparse_cfg = SparseConfig(reservoir_dim=100, sparse_dim=50)
    model = ReservoirModel(ReservoirConfig(size=100), sparse_cfg)
    trainer = ReservoirTrainer(model, sparse_cfg, device)

    img_path = str(tmp_path / "test.png")
    cv2.imwrite(img_path, np.zeros((28, 28), dtype=np.uint8))
    
    result = trainer.inference(img_path, topk=100)
    
    assert len(result["latent_indices"]) == 50

def test_inference_on_missing_file():
    """Teszt ha nem létező fileon akarunk inferencet"""
    cfg = SparseConfig(reservoir_dim=16, sparse_dim=7)
    trainer = ReservoirTrainer(
        ReservoirModel(ReservoirConfig(size=16), cfg),
        cfg,
        torch.device("cpu"),
    )
    with pytest.raises(ValueError):
        trainer.inference("test.png", topk=5)


def test_forward_output_shapes():
    """Teszt hogy a forward pass rendben adja vissza a logitokat"""
    batch_size = 4
    num_classes = 10

    res_cfg = ReservoirConfig(size=64)
    sparse_cfg = SparseConfig(reservoir_dim=64, sparse_dim=128)
    model = ReservoirModel(res_cfg, sparse_cfg, num_classes=num_classes)

    x = torch.randn(batch_size, 28, 28)
    logits, s_raw = model(x)

    assert logits.shape == (batch_size, num_classes)
    assert s_raw.shape == (batch_size, 128)


def test_evaluate_returns_expected_keys_and_ranges(mock_mnist_loader):
    """Teszt az evalute()-re hogy helyes kimenettel tér e vissza"""
    device = torch.device("cpu")
    model = ReservoirModel(ReservoirConfig(size=64), SparseConfig(reservoir_dim=64, sparse_dim=128))
    trainer = ReservoirTrainer(model, SparseConfig(reservoir_dim=64, sparse_dim=128), device)

    metrics = trainer.evaluate(mock_mnist_loader)

    assert set(metrics.keys()) == {"accuracy", "f1", "recall"}
    assert 0.0 <= metrics["accuracy"] <= 100.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


def test_evaluate_keeps_model_in_train_mode_after_call(mock_mnist_loader):
    """Teszt, hogy a modell helyesen vált e tréning és teszt módok között"""
    device = torch.device("cpu")
    model = ReservoirModel(ReservoirConfig(size=64), SparseConfig(reservoir_dim=64, sparse_dim=128))
    trainer = ReservoirTrainer(model, SparseConfig(reservoir_dim=64, sparse_dim=128), device)

    trainer.model.train() # Tréning mód
    assert trainer.model.training is True

    _ = trainer.evaluate(mock_mnist_loader) # Teszt mód

    assert trainer.model.training is True


def test_inference_prediction_type_and_bounds(tmp_path):
    """Teszt a predikció helyes formátumára"""
    device = torch.device("cpu")
    sparse_cfg = SparseConfig(reservoir_dim=64, sparse_dim=32)
    model = ReservoirModel(ReservoirConfig(size=64), sparse_cfg)
    trainer = ReservoirTrainer(model, sparse_cfg, device)

    img_path = str(tmp_path / "digit.png")
    cv2.imwrite(img_path, np.random.randint(0, 255, (28, 28), dtype=np.uint8)) # random bemenet

    result = trainer.inference(img_path, topk=8)

    assert isinstance(result["prediction"], int) # int típus
    assert 0 <= result["prediction"] <= 9 # 0 és 9 között
    assert result["latent_indices"].shape == (8,)


def test_inference_resizes_non_28x28_input(tmp_path):
    """Teszt a kép átméretezésre"""
    device = torch.device("cpu")
    sparse_cfg = SparseConfig(reservoir_dim=64, sparse_dim=20)
    model = ReservoirModel(ReservoirConfig(size=64), sparse_cfg)
    trainer = ReservoirTrainer(model, sparse_cfg, device)

    img_path = str(tmp_path / "large.png")
    cv2.imwrite(img_path, np.random.randint(0, 255, (100, 80), dtype=np.uint8)) # nagyobb kép

    result = trainer.inference(img_path, topk=5)

    assert "prediction" in result
    assert "latent_indices" in result
    assert len(result["latent_indices"]) == 5


def test_load_weights_restores_state(tmp_path):
    """Teszt az elmentett súlyok betöltésére és használatára"""
    device = torch.device("cpu")
    sparse_cfg = SparseConfig(reservoir_dim=32, sparse_dim=16)

    model1 = ReservoirModel(ReservoirConfig(size=32), sparse_cfg) # eredeti modell
    trainer1 = ReservoirTrainer(model1, sparse_cfg, device)

    save_path = tmp_path / "weights.pt" # súly elmentése
    torch.save(trainer1.model.state_dict(), save_path)

    model2 = ReservoirModel(ReservoirConfig(size=32), sparse_cfg) # modell 2
    trainer2 = ReservoirTrainer(model2, sparse_cfg, device)

    trainer2.load_weights(str(save_path)) # modell 2 az elmentett súlyokat használja

    for k, v in trainer1.model.state_dict().items():
        torch.testing.assert_close(v, trainer2.model.state_dict()[k])

    assert trainer2.model.training is False


def test_train_returns_expected_structure(mock_mnist_loader):
    """Teszt a tréning összesítő üzenetek meglétére"""
    device = torch.device("cpu")
    sparse_cfg = SparseConfig(
        reservoir_dim=64,
        sparse_dim=32,
        epochs=2,
        early_stopping_patience=5,
        restore_best_weights=True,
    )
    model = ReservoirModel(ReservoirConfig(size=64), sparse_cfg)
    trainer = ReservoirTrainer(model, sparse_cfg, device)

    result = trainer.train(mock_mnist_loader, mock_mnist_loader)

    assert set(result.keys()) == {"best_metrics", "best_epoch", "stopped_early"}
    assert isinstance(result["best_epoch"], int)
    assert isinstance(result["stopped_early"], bool)
    assert set(result["best_metrics"].keys()) == {"accuracy", "f1", "recall"}


def test_train_with_early_stopping_triggered(mock_mnist_loader, monkeypatch):
    """Teszt az early stoppingra"""
    device = torch.device("cpu")
    sparse_cfg = SparseConfig(
        reservoir_dim=64,
        sparse_dim=32,
        epochs=5,
        early_stopping_patience=1,
        early_stopping_min_delta=1e-4,
        restore_best_weights=False,
    )
    model = ReservoirModel(ReservoirConfig(size=64), sparse_cfg)
    trainer = ReservoirTrainer(model, sparse_cfg, device)

    # Nincs fejlődés
    eval_results = [
        {"accuracy": 10.0, "f1": 0.1, "recall": 0.1},
        {"accuracy": 10.0, "f1": 0.1, "recall": 0.1},
        {"accuracy": 10.0, "f1": 0.1, "recall": 0.1},
    ]
    calls = {"i": 0}

    def fake_evaluate(_loader):
        out = eval_results[min(calls["i"], len(eval_results) - 1)]
        calls["i"] += 1
        return out

    monkeypatch.setattr(trainer, "evaluate", fake_evaluate)

    result = trainer.train(mock_mnist_loader, mock_mnist_loader)

    assert result["stopped_early"] is True
    assert result["best_epoch"] == 1


def test_optimizer_targets_head_only():
    """Teszt, hogy az optimizer csak a SparseHeadre működik"""
    sparse_cfg = SparseConfig(reservoir_dim=64, sparse_dim=32)
    model = ReservoirModel(ReservoirConfig(size=64), sparse_cfg)
    trainer = ReservoirTrainer(model, sparse_cfg, torch.device("cpu"))

    opt_params = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    head_params = {id(p) for p in model.head.parameters()}
    reservoir_params = {id(p) for p in model.reservoir.parameters()}

    assert head_params.issubset(opt_params)
    assert opt_params.isdisjoint(reservoir_params)


def test_soft_shrink_zero_lambda_identity():
    """Teszt ha SoftShrink 0-t kap lamndának"""
    layer = SoftShrink(sh_lambda=0.0)
    x = torch.tensor([-2.0, -0.1, 0.0, 0.3, 5.0])
    y = layer(x)
    torch.testing.assert_close(x, y)