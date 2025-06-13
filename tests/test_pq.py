from busqueda_aproximada.pq import ProductQuantizer
import numpy as np
import pytest

def test_pq_initialize():
    pq = ProductQuantizer()
    assert pq.M == 4
    assert pq.Ks == 256
    assert pq.codebooks == []

def test_pq_initialize_errors():
    with pytest.raises(ValueError):
        pq = ProductQuantizer(M=0)
    with pytest.raises(ValueError):
        pq = ProductQuantizer(Ks=0)

def test_fit_dimension_not_divisble():
    pq = ProductQuantizer(M=5, Ks=16)
    X = np.random.randn(10, 13).astype(np.float32)  # 13 % 5 != 0
    with pytest.raises(ValueError):
        pq.fit(X)

def test_fit():
    pq = ProductQuantizer(M=4, Ks=8)
    X = np.random.randn(20, 8).astype(np.float32)  # 8 % 4 == 0
    pq.fit(X)
    # Codebooks tiene M codebooks
    assert len(pq.codebooks) == 4
    # Cada codebook tiene dimensión D / M y hay Ks en total
    assert all(cb.shape == (8, 2) for cb in pq.codebooks)

def test_encode_no_codebook():
    pq = ProductQuantizer()
    X = np.random.randn(10, 16).astype(np.float32)
    with pytest.raises(ValueError):
        pq.encode(X)

def test_encode():
    pq = ProductQuantizer(M=4, Ks=8)
    X = np.random.randn(10, 16).astype(np.float32)
    pq.fit(X)
    codes = pq.encode(X)
    # Hay la misma cantidad de vectores pero con dimensión D / M
    assert codes.shape == (10, 4)
    # Los códigos son enteros
    assert issubclass(codes.dtype.type, np.integer)
    # Los códigos son valores válidos
    assert np.all((codes >= 0) & (codes < 8))

def test_decode_no_codebook():
    pq = ProductQuantizer()
    codes = np.zeros((10, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        pq.decode(codes)

def test_decode():
    pq = ProductQuantizer(M=4, Ks=8)
    X = np.random.randn(10, 16).astype(np.float32)
    pq.fit(X)
    codes = pq.encode(X)
    decoded = pq.decode(codes)
    # Vector decodificado tiene misma dimensión que vector original
    assert decoded.shape == (10, 16)
    # Vectores decodificados están conformados por centroides
    for m in range(4):
        centroids = pq.codebooks[m]
        sub_decoded = decoded[:, m*4:(m+1)*4]
        code_ids = codes[:, m]
        assert np.allclose(sub_decoded, centroids[code_ids])

def test_approximate_knn_no_codebook():
    pq = ProductQuantizer()
    codes = np.zeros((10, 4), dtype=np.uint8)
    X = np.random.randn(1, 16).astype(np.float32)
    with pytest.raises(ValueError):
        pq.approximate_knn(X, codes)

def test_approximate_knn():
    np.random.seed(0)
    X = np.random.randn(100, 16).astype(np.float32)
    query = np.random.randn(16).astype(np.float32)

    pq = ProductQuantizer(M=4, Ks=16)
    pq.fit(X)
    codes = pq.encode(X)

    k = 5
    indices, distances = pq.approximate_knn(query, codes, k=k)

    # Índices y distancias son arreglos
    assert isinstance(indices, np.ndarray)
    assert isinstance(distances, np.ndarray)
    # Tamaño de arreglos es k
    assert indices.shape == (k,)
    assert distances.shape == (k,)
    # Distancias tienen valores válidos
    assert np.all(distances >= 0)
    # Los indices tienen valores válidos
    assert np.all(indices < X.shape[0]) and np.all(indices >= 0)
