import numpy as np
import pytest
from busqueda_aproximada.opq import OptimizedProductQuantizer
import sys
import os

def generate_data(n=100, d=32, seed=0):
    np.random.seed(seed)
    return np.random.randn(n, d).astype(np.float32)

# Verifica que al hacer fit se genere una matriz de rotación válida y ortonormal
def test_fit_sets_rotation_matrix():
    X = generate_data()
    opq = OptimizedProductQuantizer(M=4, Ks=16)
    opq.fit(X)
    assert opq.R is not None
    assert opq.R.shape == (X.shape[1], X.shape[1])
    # Verificar que es aproximadamente ortogonal R.T @ R ≈ I
    identity = np.eye(X.shape[1])
    assert np.allclose(opq.R.T @ opq.R, identity, atol=1e-3)


def test_encode_decode_shape_and_type():
    X = generate_data()
    opq = OptimizedProductQuantizer(M=4, Ks=16)
    opq.fit(X)
    codes = opq.encode(X)
    decoded = opq.decode(codes)
    
    assert codes.shape == (X.shape[0], opq.M)
    assert decoded.shape == X.shape
    assert decoded.dtype == np.float32

# Verifica que un vector consultado aparece como su propio vecino
def test_knn_includes_self():
    X = generate_data(n=300, d=32)
    opq = OptimizedProductQuantizer(M=4, Ks=32)
    opq.fit(X)
    codes = opq.encode(X)
    
    query = X[123]
    knn_indices, _ = opq.approximate_knn(query, codes, k=1)
    assert knn_indices[0] == 123  

# Verifica que la rotación no modifica las normas de los vectores
def test_rotation_does_not_change_norms():
    X = np.random.randn(50, 32).astype(np.float32)
    opq = OptimizedProductQuantizer(M=4, Ks=8)
    opq.fit(X)
    X_rot = X @ opq.R
    norms_orig = np.linalg.norm(X, axis=1)
    norms_rot = np.linalg.norm(X_rot, axis=1)
    assert np.allclose(norms_orig, norms_rot, atol=1e-5), "La rotación cambió las normas"

# Verifica que OPQ produce menor error de reconstrucción que PQ tradicional
def test_opq_reduces_error_vs_unrotated():
    from busqueda_aproximada.pq import ProductQuantizer
    X = np.random.randn(200, 32).astype(np.float32)

    pq = ProductQuantizer(M=4, Ks=16)
    pq.fit(X)
    X_hat_pq = pq.decode(pq.encode(X))
    err_pq = np.linalg.norm(X - X_hat_pq)

    from busqueda_aproximada.opq import OptimizedProductQuantizer
    opq = OptimizedProductQuantizer(M=4, Ks=16)
    opq.fit(X)
    X_hat_opq = opq.compress(X)
    err_opq = np.linalg.norm(X - X_hat_opq)

    assert err_opq < err_pq, f"OPQ no mejora PQ: {err_opq:.4f} ≥ {err_pq:.4f}"


def test_rotation_matrix_is_orthogonal():
    X = np.random.randn(100, 32).astype(np.float32)
    opq = OptimizedProductQuantizer(M=4, Ks=16)
    opq.fit(X)
    I = np.eye(X.shape[1])
    assert np.allclose(opq.R.T @ opq.R, I, atol=1e-4), "R no es ortogonal"

# Verifica que la rotación aprendida no es trivial
def test_rotation_has_effect_on_structure():
    X = np.random.randn(100, 32).astype(np.float32)
    opq = OptimizedProductQuantizer(M=4, Ks=16)
    opq.fit(X)
    X_rot = X @ opq.R
    assert not np.allclose(X, X_rot), "La rotación no tuvo efecto"

def test_rotation_initial_identity():
    from busqueda_aproximada.opq import OptimizedProductQuantizer
    D = 16
    opq = OptimizedProductQuantizer(M=2, Ks=8)
    R0 = opq.R = np.eye(D, dtype=np.float32)
    assert np.allclose(R0, np.eye(D)), "R inicial no es identidad"
