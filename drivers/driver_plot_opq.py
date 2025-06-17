"""
Driver que ejemplifica Optimized Product Quantization (OPQ) para un caso simple bidimensional.
Similar al driver PQ, pero con rotación aplicada antes de cuantizar.

Se generan tres plots:
 - Vectores aleatorios (originales)
 - Vectores codificados y sus centroides asignados
 - Tres consultas y su centroide más cercano
"""

from busqueda_aproximada.opq import OptimizedProductQuantizer  
import numpy as np
import matplotlib.pyplot as plt

COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

def generate_random_vectors(N=100):
    return np.random.randn(N, 2).astype(np.float32)

def opq_create_plots_2d(N=100, D=2, M=1, Ks=8, num_iters=20):
    X = generate_random_vectors(N)
    Q1, Q2, Q3 = generate_random_vectors(1), generate_random_vectors(1), generate_random_vectors(1)
    queries = [Q1, Q2, Q3]

    opq = OptimizedProductQuantizer(M=M, Ks=Ks, num_iters=num_iters)
    opq.fit(X)

    # Obtener datos rotados y centroides
    X_rot = X @ opq.R
    centroids = opq.pq.codebooks[0]
    codes = opq.encode(X)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    axs[0].scatter(X[:, 0], X[:, 1], c="blue", label="Vector original")
    axs[0].set_title("Vectores Originales")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].grid(True)
    axs[0].legend(fontsize="small")
    xlims = axs[0].get_xlim()
    ylims = axs[0].get_ylim()

    # Plot 2: Vectores coloreados por cluster en espacio rotado
    for i in range(N):
        color = COLORS[int(codes[i, 0]) % len(COLORS)]
        axs[1].scatter(X_rot[i, 0], X_rot[i, 1], c=color, s=20)
    for idx, centroid in enumerate(centroids):
        color = COLORS[idx % len(COLORS)]
        axs[1].scatter(centroid[0], centroid[1], c=color, marker="x", s=100, label=f"Centroide {idx}")
    axs[1].set_title("Vectores Rotados y Centroides")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].grid(True)
    axs[1].legend(fontsize="small")

    # Plot 3: Consulta en espacio rotado con centroides asignados
    axs[2].scatter(X_rot[:, 0], X_rot[:, 1], c="lightgray", s=10, label="Vectores rotados")
    axs[2].scatter(centroids[:, 0], centroids[:, 1], c="black", marker="x", s=60, label="Centroides")

    labels_used = set()
    colors = COLORS[:3]

    for i, Q in enumerate(queries):
        knn_idx = opq.approximate_knn(Q[0], codes, k=1)[0][0]
        assigned_centroid_idx = codes[knn_idx, 0]
        assigned_centroid = centroids[assigned_centroid_idx]
        color = colors[i]

        label_query = f"Consulta {i+1}"
        axs[2].scatter(Q[0, 0], Q[0, 1], c=color, marker="*", s=200, label=label_query if label_query not in labels_used else "")
        labels_used.add(label_query)

        label_centroid = f"Centroide asignado {i+1}"
        axs[2].scatter(assigned_centroid[0], assigned_centroid[1], c=color, marker="X", s=100, label=label_centroid if label_centroid not in labels_used else "")
        labels_used.add(label_centroid)

    axs[2].set_title("Consultas y Centroide más Cercano (Espacio Rotado)")
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")
    axs[2].grid(True)
    axs[2].legend(fontsize="small")

    fig.delaxes(axs[3])
    plt.tight_layout()
    plt.savefig("plots/opq_plots.png")

def main():
    opq_create_plots_2d()

if __name__ == "__main__":
    main()
