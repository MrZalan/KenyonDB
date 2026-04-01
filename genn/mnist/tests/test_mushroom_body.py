import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from mushroom_body_class import MBConfig, MBSimulator, MushroomBodyModel, load_mnist

@pytest.fixture
def mock_simulator():
    """Helyettesítő osztály a tesztekhez"""
    mock_wrapper = MagicMock()
    mock_reward_array = np.zeros(10, dtype=np.float32)
    mock_wrapper.kc_mbon.extra_global_params = {
        "reward": MagicMock(view=mock_reward_array)
    }
    # 2 Kenyon-sejt (5-ös és 7es MBON), 5-ös tüzel először [(times, ids)]
    mock_wrapper.kc.spike_recording_data = [(np.array([5.0, 7.0]), np.array([100, 200]))]
    mock_wrapper.mbon.spike_recording_data = [(np.array([10.0, 15.0]), np.array([5, 3]))]
    mock_wrapper.pn.spike_recording_data = [(np.array([1.0]), np.array([10]))]
    
    simulator = MBSimulator(mock_wrapper)
    return simulator

def test_get_latent_vector_logic(mock_simulator):
    """Teszt az elsőnek tüzelő MBON felismerésére"""
    active_kcs, prediction = mock_simulator.get_latent_vector(np.zeros(784))
    
    assert prediction == 5  # 5-ös MBON 10ms-kor, 3-as 15ms-kor
    assert len(active_kcs) == 2
    assert 100 in active_kcs and 200 in active_kcs

def test_inference_with_spikes_structure(mock_simulator):
    """Teszt a megfelelő dictionary output formára"""
    # OpenCV helyetttesítése
    with patch("mushroom_body_class.cv2.imread") as mock_imread:
        # 28x28as tömb nullákkal
        mock_imread.return_value = np.zeros((28, 28), dtype=np.uint8)
        
        result = mock_simulator.inference_with_spikes("test.png")
        
        assert "spikes" in result
        assert "pn" in result["spikes"]
        assert "kc" in result["spikes"]
        assert "mbon" in result["spikes"]


def test_model_build_and_load():
    """Teszt a PyGenn modell buildelésére és betöltésére"""
    try:
        # Modell 1000 Kenyon sejttel
        MBConfig.NUM_KC = 1000 
        model = MushroomBodyModel(name="test_build", backend="single_threaded_cpu", is_training=False)
        model.build_and_load()
        assert model.model._built is True
    except Exception as e:
        pytest.fail(f"GeNN Model failed to build: {e}")

def test_weight_assignment():
    """Teszt a KC-MBON súlyok betöltésére"""
    MBConfig.NUM_KC = 1000
    model = MushroomBodyModel(name="test_weights", is_training=False)
    model.build_and_load()
    
    # Random súlyok készítése
    dummy_weights = np.random.rand(MBConfig.NUM_KC * MBConfig.NUM_MBON).astype(np.float32)
    
    # Súlyok a device-ra kerülnek
    model.set_kc_mbon_weights(dummy_weights)
    
    # Súlyok letöltése a device-ról
    model.kc_mbon.vars["g"].pull_from_device()
    pushed_weights = model.kc_mbon.vars["g"].view.flatten()
    np.testing.assert_allclose(dummy_weights, pushed_weights)

def test_mnist_normalization():
    """Sum normalizálás tesztje"""
    train_imgs, _, _, _ = load_mnist()
    
    # Random kép a tréning halmazból
    sample_img = train_imgs[0]

    # Pixel összegnek 1-nek kell lennie
    assert pytest.approx(np.sum(sample_img)) == 1.0
    assert sample_img.shape == (784,)

def test_get_latent_vector_no_spikes(mock_simulator):
    """Teszt ha egy MBON se tüzelt, nincs találat"""
    # Üres tömbök
    mock_simulator.model_wrapper.mbon.spike_recording_data = [(np.array([]), np.array([]))]
    
    active_kcs, prediction = mock_simulator.get_latent_vector(np.zeros(784))
    
    # Eredménynek -1-nek kell lennie
    assert prediction == -1
    assert len(active_kcs) == 2

def test_mbon_tie_break(mock_simulator):
    """Teszt ha 2 MBON ugyanakkor tüzel"""
    # 3-as és 5-ös MBON is 10ms-kor tüzel
    mock_simulator.model_wrapper.mbon.spike_recording_data = [
        (np.array([10.0, 10.0]), np.array([3, 5]))
    ]
    
    _, prediction = mock_simulator.get_latent_vector(np.zeros(784))
    
    # np.argmin az első indexet fogja visszaadni
    assert prediction == 3

def test_inference_black_image(mock_simulator):
    """Teszt üres képre, 0 pixel összeggel"""
    with patch("mushroom_body_class.cv2.imread") as mock_imread:

        # 28x28-as nulla tömb (fekete kép)
        mock_imread.return_value = np.zeros((28, 28), dtype=np.uint8)
        
        # Ha helyes, akkor nem lesz ZeroDivisionError
        prediction, active_kcs = mock_simulator.inference("test.png")
        assert prediction == 5

def test_set_reward_mapping(mock_simulator):
    """Teszt a jutalom helyes beállítására"""

    # Jutalmazzuk a 3-as számjegyet
    mock_simulator.set_reward(3)
    reward_view = mock_simulator.model_wrapper.kc_mbon.extra_global_params["reward"].view
    
    # 3-asra 1.0 minden másra -1
    assert reward_view[3] == 1.0
    assert reward_view[0] == -1.0
    assert reward_view[9] == -1.0

def test_weight_size_mismatch():
    """Teszt ha eltérő számú súlyokat töltünk be"""
    # 100 Kenyon sejt
    MBConfig.NUM_KC = 100
    model_wrapper = MushroomBodyModel(name="test_fail", is_training=False)
    model_wrapper.build_and_load()
    
    # 10 hosszú súly tömb
    wrong_weights = np.zeros(10) 
    
    with pytest.raises(ValueError, match="kc_mbon_g size mismatch"):
        model_wrapper.set_kc_mbon_weights(wrong_weights)

def test_data_extraction_shapes(mock_simulator):
    """Teszt, hogy az adatok helyesen kerülnek kimentésre"""
    images = [np.zeros(784)]
    labels = [5]
    
    result = mock_simulator.extract_all_data(images, labels)
    
    assert len(result["active_kc_ids"]) == 1
    assert result["labels"][0] == 5

def test_inference_invalid_path(mock_simulator):
    """Teszt, ha OpenCV nem találja a bemeneti képet"""
    with patch("mushroom_body_class.cv2.imread", return_value=None):
        with pytest.raises(ValueError, match="Error loading in image"):
            mock_simulator.inference("non_existent.png")

def test_run_image_without_label_keeps_teacher_signal_zero(mock_simulator):
    """Teszt ha nincs címke a modell nem tanul"""
    pn_mag = np.zeros(784, dtype=np.float32)
    mbon_mag = np.ones(10, dtype=np.float32)

    mock_simulator.model_wrapper.pn_input = MagicMock()
    mock_simulator.model_wrapper.pn_input.vars = {
        "magnitude": MagicMock(view=pn_mag)
    }

    mock_simulator.model_wrapper.mbon_input = MagicMock()
    mock_simulator.model_wrapper.mbon_input.vars = {
        "magnitude": MagicMock(view=mbon_mag)
    }

    mock_simulator.model_wrapper.model = MagicMock()

    mock_simulator.run_image(np.zeros(784, dtype=np.float32), label=None)

    # Az MBON tanító szignálnak 0-nak kell maradnia
    assert np.all(mock_simulator.model_wrapper.mbon_input.vars["magnitude"].view == 0.0)

def test_evaluate_returns_metrics_tuple(mock_simulator, monkeypatch):
    """Teszt, hogy evaluate() a megfelelő metrikákkal tér vissza"""
    images = [np.zeros(784), np.zeros(784), np.zeros(784)]
    labels = [5, 3, 5]

    preds = iter([5, 3, -1])
    monkeypatch.setattr(mock_simulator, "predict", lambda _img: next(preds))

    acc, f1, recall = mock_simulator.evaluate(images, labels)

    assert isinstance(acc, float)
    assert isinstance(f1, float)
    assert isinstance(recall, float)
    assert 0.0 <= acc <= 100.0 # accuracy
    assert 0.0 <= f1 <= 1.0 # F1-score
    assert 0.0 <= recall <= 1.0 # Recall

def test_evaluate_perfect_accuracy(mock_simulator, monkeypatch):
    """Teszt tökéletes egyezésnél 100% pontosságra"""
    images = [np.zeros(784), np.zeros(784)]
    labels = [5, 3] # címkék

    preds = iter([5, 3]) # predikciók
    monkeypatch.setattr(mock_simulator, "predict", lambda _img: next(preds))

    acc, _f1, _recall = mock_simulator.evaluate(images, labels)
    assert acc == 100.0


def test_extract_all_data_image_shape_and_dtype(mock_simulator, monkeypatch):
    """Teszt, hogy extract_all_data() helyes formátumban ment e"""
    images = [np.zeros(784, dtype=np.float32), np.ones(784, dtype=np.float32)]
    labels = [1, 2]

    outputs = [
        (np.array([10, 11], dtype=np.uint32), 1),
        (np.array([12, 13], dtype=np.uint32), 2),
    ]
    it = iter(outputs)
    monkeypatch.setattr(mock_simulator, "get_latent_vector", lambda _img: next(it))

    result = mock_simulator.extract_all_data(images, labels)

    assert result["images"].shape == (2, 28, 28)
    assert result["images"].dtype == np.uint8
    assert result["predictions"].tolist() == [1, 2]
    assert result["labels"].tolist() == [1, 2]
    assert result["model_type"] == "mb"

def test_extract_all_data_scales_nonzero_image(mock_simulator, monkeypatch):
    """Teszt, hogy a nem üres képeknek 0 és 255 közé kell leképződniük"""
    img = np.ones(784, dtype=np.float32) * 0.5
    monkeypatch.setattr(
        mock_simulator,
        "get_latent_vector",
        lambda _img: (np.array([1, 2], dtype=np.uint32), 4),
    )

    result = mock_simulator.extract_all_data([img], [4])
    stored = result["images"][0]

    assert stored.shape == (28, 28)
    assert stored.dtype == np.uint8
    assert stored.max() == 255


def test_save_model_params_writes_expected_arrays(mock_simulator, monkeypatch, tmp_path):
    """Teszt a modell paraméterek helyes elmentésére"""
    g_view = np.arange(6, dtype=np.float32).reshape(2, 3)

    mock_simulator.model_wrapper.kc_mbon.vars = {
        "g": MagicMock(view=g_view, pull_from_device=MagicMock())
    }

    mock_simulator.model_wrapper.pn_kc = MagicMock()
    mock_simulator.model_wrapper.pn_kc.get_sparse_pre_inds.return_value = np.array([1, 2, 3])
    mock_simulator.model_wrapper.pn_kc.get_sparse_post_inds.return_value = np.array([4, 5, 6])

    weights_path = tmp_path / "weights.npy" # KC-MBON súlyok
    indices_path = tmp_path / "indices.npy" # PN-KC indexek

    # Paraméterek elmentése
    mock_simulator.save_model_params(str(weights_path), str(indices_path))

    saved_w = np.load(weights_path)
    saved_i = np.load(indices_path)

    np.testing.assert_allclose(saved_w, g_view)
    np.testing.assert_array_equal(saved_i, np.array([[1, 2, 3], [4, 5, 6]]))

def test_load_mnist_shapes_and_labels(monkeypatch):
    """Teszt, hogy a load_mnist() helyes formátummal tér e vissza"""
    train_imgs_raw = np.ones((60000, 28, 28), dtype=np.uint8)
    test_imgs_raw = np.ones((10000, 28, 28), dtype=np.uint8) * 2
    train_labels = np.arange(60000, dtype=np.uint8) % 10
    test_labels = np.arange(10000, dtype=np.uint8) % 10

    monkeypatch.setattr("mushroom_body_class.mnist.train_images", lambda: train_imgs_raw)
    monkeypatch.setattr("mushroom_body_class.mnist.test_images", lambda: test_imgs_raw)
    monkeypatch.setattr("mushroom_body_class.mnist.train_labels", lambda: train_labels)
    monkeypatch.setattr("mushroom_body_class.mnist.test_labels", lambda: test_labels)

    train_imgs, train_y, test_imgs, test_y = load_mnist()

    assert train_imgs.shape == (60000, 784) # tréning halmaz
    assert test_imgs.shape == (10000, 784) # teszt halmaz
    assert train_y.shape == (60000,)
    assert test_y.shape == (10000,)

    # Címkék egyezése
    assert np.all(train_y[:10] == train_labels[:10])
    assert np.all(test_y[:10] == test_labels[:10])