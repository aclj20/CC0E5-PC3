import numpy as np
import pytest
from busqueda_aproximada.lsh import LSH

def test_train_creates_model():
    X = np.random.randn(50, 10).astype(np.float32)
    lsh = LSH(X)
    lsh.train(num_vector=5)
    assert lsh.model is not None
    assert "random_vectors" in lsh.model
    assert "table" in lsh.model

def test_query_returns_k_results():
    X = np.random.randn(100, 16).astype(np.float32)
    lsh = LSH(X).train(num_vector=10)
    result = lsh.query(X[0], k=5, max_search_radius=3)
    assert len(result) == 5
    assert "id" in result.columns
    assert "distance" in result.columns

def test_nearest_neighbor_matches_manual():
    X = np.random.randn(80, 10).astype(np.float32)
    lsh = LSH(X).train(num_vector=8)
    query = X[42]  

    model = lsh.model
    bits = (query @ model["random_vectors"] >= 0).astype(int)
    powers = 1 << np.arange(model["num_vector"] - 1, -1, -1)
    bin_idx = bits.dot(powers)
    candidates = model["table"][bin_idx]

    dists = [np.linalg.norm(query - X[i]) for i in candidates]
    true_idx = candidates[np.argmin(dists)]

    result = lsh.query(query, k=1, max_search_radius=0)
    assert result["id"].iloc[0] == true_idx

def test_search_radius_adds_more_points():
    X = np.random.randn(200, 16).astype(np.float32)
    lsh = LSH(X).train(num_vector=8)
    query = np.random.randn(16).astype(np.float32)

    res0 = lsh.query(query, k=10, max_search_radius=0)
    res1 = lsh.query(query, k=10, max_search_radius=1)
    res2 = lsh.query(query, k=10, max_search_radius=2)

    # Verifica que se acumulen candidatos
    assert len(set(res1["id"])) >= len(set(res0["id"]))
    assert len(set(res2["id"])) >= len(set(res1["id"]))

def test_no_candidates_handled_gracefully():
    X = np.random.randn(10, 5).astype(np.float32)
    lsh = LSH(X).train(num_vector=12)  
    query = np.random.randn(5).astype(np.float32)

    res = lsh.query(query, k=3, max_search_radius=0)
    
    assert len(res) <= 3  

def test_binary_representation_stability():
    X = np.random.randn(10, 3).astype(np.float32)
    lsh = LSH(X).train(num_vector=3)
    vec = X[0]
    bits = (vec @ lsh.model["random_vectors"] >= 0).astype(int)
    powers = 1 << np.arange(2, -1, -1)
    decimal = bits.dot(powers)
    assert 0 <= decimal < 8

def test_self_query_is_nearest():
    X = np.random.randn(50, 8).astype(np.float32)
    lsh = LSH(X)
    lsh.train(num_vector=8)
    i = 7
    query_point = X[i]
    result = lsh.query(query_point, k=3, max_search_radius=5)
    print(result)
    assert i in result["id"].values

def test_query_out_of_distribution():
    X = np.random.randn(100, 10).astype(np.float32)
    query = np.full((10,), 100.0, dtype=np.float32)  
    lsh = LSH(X).train(num_vector=10)
    res = lsh.query(query, k=3, max_search_radius=5)
    assert len(res) > 0

def test_query_out_of_distribution():
    X = np.random.randn(100, 10).astype(np.float32)
    query = np.full((10,), 100.0, dtype=np.float32)  
    lsh = LSH(X).train(num_vector=10)
    res = lsh.query(query, k=3, max_search_radius=5)
    assert len(res) > 0
