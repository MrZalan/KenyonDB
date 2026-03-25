import numpy as np
import pytest
import torch
import os

from create_database import KenyonDB, NpzStrategy, PtStrategy

@pytest.fixture
def db():
    test_db = KenyonDB(":memory:")
    return test_db

def test_add_new_record(db):
    active_ids = [1, 2, 5, 7]
    db.add_new_record(label=5, original_index=0, active_ids=active_ids, model_type="mb")

    result = db.get_record_by_image_id(1)
    assert result["label"] == 5
    assert result["model_type"] == "mb"

    saved_vectors = db.get_active_ids_by_image_id(1)
    assert saved_vectors == active_ids

def test_image_serialization(db):
    test_original = np.random.randint(0, 255, (28,28), dtype=np.uint8)

    # Mangled name használata a privát metódusok miatt
    serialized = db._KenyonDB__serialize_image(test_original)
    deserialized = db._KenyonDB__deserialize_image(serialized)
    np.testing.assert_array_equal(test_original, deserialized)

def test_npz_strategy(db, tmp_path):
    file_path = tmp_path / "test.npz"
    np.savez(file_path,
             active_kc_ids=np.array([[1, 2], [3, 4]], dtype=object),
             labels=np.array([0, 1]),
             images=np.random.randint(0, 255, (28,28)))
    strategy = NpzStrategy(db)
    strategy.import_data(str(file_path), "mb")

    summary = db.get_database_summary()
    assert summary["models"]["mb"]["image_count"] == 2
    assert summary["total_vectors"] == 4

def test_pt_strategy(db, tmp_path):
    file_path = tmp_path / "test.pt"
    to_save = {
        'vectors': torch.tensor([[1, 2, 3], [4,5,6]]),
        'labels': torch.tensor([9, 3]),
        'images': torch.zeros((2, 28, 28))
    }
    torch.save(to_save, file_path)

    strategy = PtStrategy(db)
    strategy.import_data(str(file_path), "ba")

    summary = db.get_database_summary()
    assert summary["models"]["ba"]["image_count"] == 2
    assert summary["total_vectors"] == 6

def test_similarity_search(db):
    db.add_new_record(label=1, original_index=0, active_ids=[2,3,4], model_type="mb")
    test_query = [1,2,3]
    results = db.similarity_search(test_query, model_type="mb", top_k=1)

    assert len(results) == 1
    metrics = results[0]["metrics"]
    assert metrics["overlap"] == 2
    assert metrics["jaccard"] == 0.5
    assert metrics["dice"] == (2.0 * 2) / (3 + 3)

def test_empty_query(db):
    assert db.similarity_search([], "mb") == []
    assert db.similarity_search(None, "mb") == []

def test_similarity_perfect_match(db):
    ids = [1, 2, 3, 4]
    db.add_new_record(label=1, original_index=0, active_ids=ids, model_type="er")
    
    results = db.similarity_search(ids, model_type="er", top_k=1)
    metrics = results[0]["metrics"]
    assert metrics["jaccard"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["overlap_coeff"] == 1.0

def test_model_types_filter(db):
    ids = [1, 2, 3]
    db.add_new_record(label=1, original_index=1, active_ids=ids, model_type="mb")
    db.add_new_record(label=1, original_index=1, active_ids=ids, model_type="ws")

    results = db.similarity_search(ids, model_type="mb")
    assert len(results) == 1

    record = db.get_record_by_image_id(results[0]["image_id"])
    assert record["model_type"] == "mb"

def test_clear_database(db):
    db.add_new_record(label=5, original_index=0, active_ids=[1, 2], model_type="ba")
    
    db.clear_database_by_model("ba")
    summary = db.get_database_summary()
    
    assert summary["models"]["ba"]["image_count"] == 0
    assert summary["models"]["ba"]["vector_count"] == 0
    assert summary["total_vectors"] == 0

def test_invalid_image_data(db):
    assert db._KenyonDB__serialize_image(None) is None
    
    db.add_new_record(label=0, original_index=0, active_ids=[1], model_type="mb", image_data=None)
    result = db.get_record_by_image_id(1)
    assert result["image"] is None

def test_get_all_vectors(db):
    db.add_new_record(label=1, original_index=0, active_ids=[1, 2], model_type="mb")
    db.add_new_record(label=2, original_index=1, active_ids=[3, 4], model_type="mb")
    db.add_new_record(label=3, original_index=2, active_ids=[5, 6], model_type="mb")
    
    vectors = db.get_all_vectors_by_model("mb")
    assert len(vectors) == 3
    assert vectors[0]["active_ids"] == [1, 2]
    assert vectors[1]["active_ids"] == [3, 4]
    assert vectors[2]["active_ids"] == [5, 6]
    assert vectors[0]["label"] == 1
    assert vectors[1]["label"] == 2
    assert vectors[2]["label"] == 3

def test_similarity_no_overlap(db):
    db.add_new_record(label=1, original_index=0, active_ids=[1, 2, 3], model_type="mb")
    
    results = db.similarity_search([99, 100], model_type="mb")
    assert len(results) == 0

def test_similarity_ranking_order(db):
    db.add_new_record(label=1, original_index=0, active_ids=[1, 2, 3], model_type="mb") 
    db.add_new_record(label=2, original_index=1, active_ids=[1, 2, 3, 4], model_type="mb") 
    
    query = [1, 2, 3, 4]
    results = db.similarity_search(query, model_type="mb", top_k=5)
    
    assert len(results) == 2
    assert results[0]["label"] == 2
    assert results[0]["metrics"]["jaccard"] == 1.0

def test_large_image_serialization(db):
    img = np.random.rand(100, 100).astype(np.float32)
    db.add_new_record(label=9, original_index=0, active_ids=[1], model_type="ba", image_data=img)
    
    record = db.get_record_by_image_id(1)
    np.testing.assert_array_almost_equal(record["image"], img)

def test_get_image_and_label_not_found(db):
    label, img = db.get_image_and_label(999)
    assert label is None
    assert img is None

def test_import_wrong_file_extension(db, tmp_path):
    bad_file = tmp_path / "not_a_torch_file.pt"
    bad_file.write_text("Ez egy teszt fájl hibás fájlformátum tesztelésére")

    strategy = PtStrategy(db)
    
    with pytest.raises(Exception):
        strategy.import_data(str(bad_file), "er")

def test_duplicate_insertion_prevention(db):
    active_ids = [1, 2, 3]
    db.add_new_record(5, 100, active_ids, "mb")
    db.add_new_record(5, 100, [4, 5, 6], "mb") 

    summary = db.get_database_summary()
    assert summary["models"]["mb"]["image_count"] == 1
    assert summary["models"]["mb"]["vector_count"] == 3

def test_similarity_search_zero_division(db):
    db.add_new_record(1, 0, [1], "mb")
    
    results = db.similarity_search([1], "mb")
    assert results[0]["metrics"]["jaccard"] == 1.0