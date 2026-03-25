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
    result = trainer.save_latent_indices_with_images(mock_mnist_loader, topk=topk)
    
    assert result["vectors"].shape == (10, topk)
    assert result["images"].shape == (10, 1, 28, 28)
    assert result["images"].dtype == torch.uint8

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