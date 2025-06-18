"""
Driver que ejemplifica Optimized Product Quantization (OPQ) para 2D o 3D.
Muestra rotación, centroides y vecinos más cercanos.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from busqueda_aproximada.opq import OptimizedProductQuantizer
import os

os.makedirs("plots/opq", exist_ok=True)

def generate_data(N, D):
    np.random.seed(42)
    return np.random.randn(N, D).astype(np.float32)

def plot_opq_2d(X, X_rot, centroids, codes, queries, opq):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    axs[0].scatter(X[:, 0], X[:, 1], c="blue", label="Originales")
    axs[0].set_title("Vectores Originales")
    axs[0].grid(True)
    axs[0].legend()

    for i in range(len(X)):
        axs[1].scatter(X_rot[i, 0], X_rot[i, 1], c=f"C{codes[i, 0] % 10}", s=20)
    for j, centroid in enumerate(centroids):
        axs[1].scatter(*centroid, c=f"C{j % 10}", marker="x", s=100)
    axs[1].set_title("Vectores rotados y centroides")
    axs[1].grid(True)

    axs[2].scatter(X_rot[:, 0], X_rot[:, 1], c="lightgray", s=10)
    axs[2].scatter(centroids[:, 0], centroids[:, 1], c="black", marker="x", s=60)
    for i, Q in enumerate(queries):
        knn_idx = opq.approximate_knn(Q[0], codes, k=1)[0][0]
        
        # Añadir marcador al query
        axs[2].scatter(Q[0, 0], Q[0, 1], c=f"C{i}", marker="*", s=200)

        # Centroide asignado
        assigned_centroid_idx = codes[knn_idx, 0]
        assigned_centroid = centroids[assigned_centroid_idx]
        axs[2].scatter(*assigned_centroid, c=f"C{i}", marker="X", s=100)

        # Vecino más cercano real
        real_nn = X[knn_idx]
        axs[2].scatter(real_nn[0], real_nn[1], edgecolors="red", facecolors="none", s=160, linewidths=2, label="Más cercano" if i == 0 else "")

    axs[2].set_title("Consultas y Centroide asignado")
    axs[2].grid(True)

    fig.delaxes(axs[3])
    plt.tight_layout()
    plt.savefig("plots/opq/opq_plot_2d.png")
    plt.show()

def plot_opq_3d(X, X_rot, centroids, codes, queries, opq):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_rot[:, 0], X_rot[:, 1], X_rot[:, 2], c="lightgray", s=10, label="Rotados")
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="black", marker="x", s=60, label="Centroides")
    for i, Q in enumerate(queries):
        knn_idx = opq.approximate_knn(Q[0], codes, k=1)[0][0]
        ax[2].scatter(Q[0, 0], Q[0, 1], c=f"C{i}", marker="*", s=200)

        # Centroide asignado
        assigned_centroid_idx = codes[knn_idx, 0]
        assigned_centroid = centroids[assigned_centroid_idx]
        ax[2].scatter(*assigned_centroid, c=f"C{i}", marker="X", s=100)

        #  Vecino más cercano real
        real_nn = X[knn_idx]
        ax[2].scatter(real_nn[0], real_nn[1], edgecolors="red", facecolors="none", s=160, linewidths=2, label="Más cercano" if i == 0 else "")

    ax.set_title("OPQ en 3D: rotados y consultas")
    plt.tight_layout()
    plt.savefig("plots/opq/opq_plot_3d.png")
    plt.show()

def run_opq_demo(N=200, D=2, M=1, Ks=8, num_iters=20, k=5, show_plot=True):
    X = generate_data(N, D)
    queries = [generate_data(1, D) for _ in range(3)]

    opq = OptimizedProductQuantizer(M=M, Ks=Ks, num_iters=num_iters)
    opq.fit(X)
    X_rot = X @ opq.R
    codes = opq.encode(X)
    centroids = opq.pq.codebooks[0]

    print(" Punto de consulta:")
    print(queries[0][0])

    print(f"\n Vecinos más cercanos (k={k}):")
    idxs, dists = opq.approximate_knn(queries[0][0], codes, k=k)
    for i, (idx, dist) in enumerate(zip(idxs, dists), 1):
        coord = X[idx]
        coords_str = ", ".join(f"{v:.4f}" for v in coord)
        print(f"{i}. idx = {idx:3d} → distancia = {dist:.4f} → coords = ({coords_str})")

    if show_plot:
        if D == 2:
            plot_opq_2d(X, X_rot, centroids, codes, queries, opq)
        elif D == 3:
            plot_opq_3d(X, X_rot, centroids, codes, queries, opq)

def main():
    run_opq_demo(N=150, D=2, M=1, Ks=8, num_iters=10, k=5, show_plot=True)

if __name__ == "__main__":
    main()
