"""
Driver que ejemplifica Product Quantization para un caso simple bidimensional.
Se generan tres plots:
 - Un plot con 100 vectores aleatorios (por defecto).
 - Un plot con los 100 vectores y sus centroides asignados.
 - Un plot con tres vectores de consultas y su centroide más cercano.
"""

from busqueda_aproximada.pq import ProductQuantizer
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


def pq_create_plots_2d(N=100, D=2, M=1, Ks=8):
    X = generate_random_vectors(N)
    Q1, Q2, Q3 = (
        generate_random_vectors(1),
        generate_random_vectors(1),
        generate_random_vectors(1),
    )
    queries = [Q1, Q2, Q3]

    pq = ProductQuantizer(M, Ks)
    pq.fit(X)
    centroids = pq.codebooks[0]
    codes = pq.encode(X)[:, 0]
    encoded = codes  # alias

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    # Plot 1: Vectores originales
    axs[0].scatter(X[:, 0], X[:, 1], c="blue", label="Vector original")
    axs[0].set_title("Vectores Originales")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].grid(True)
    axs[0].legend(fontsize="small")

    xlims = axs[0].get_xlim()
    ylims = axs[0].get_ylim()

    # Plot 2: Vectores coloreados por cluster
    for i in range(N):
        color = COLORS[codes[i] % len(COLORS)]
        axs[1].scatter(X[i, 0], X[i, 1], c=color, s=20)

    for idx, centroid in enumerate(centroids):
        color = COLORS[idx % len(COLORS)]
        axs[1].scatter(
            centroid[0],
            centroid[1],
            c=color,
            marker="x",
            s=100,
            label=f"Centroide {idx}",
        )

    axs[1].set_title("Vectores con centroides")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_xlim(xlims)
    axs[1].set_ylim(ylims)
    axs[1].grid(True)
    axs[1].legend(fontsize="small")

    # Plot 3: Consulta de vecinos más cercanos (resalta centroides)
    axs[2].scatter(X[:, 0], X[:, 1], c="lightgray", s=10, label="Vectores originales")
    axs[2].scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        marker="x",
        s=60,
        label="Centroides",
    )

    labels_used = set()
    colors = COLORS[:3]

    for i, Q in enumerate(queries):
        nn_idx = pq.approximate_knn(Q, encoded.reshape(-1, 1), k=1)[0][0]
        assigned_centroid_idx = encoded[nn_idx]
        assigned_centroid = centroids[assigned_centroid_idx]
        color = colors[i]

        # Consulta
        label_query = f"Consulta {i+1}"
        axs[2].scatter(
            Q[0, 0],
            Q[0, 1],
            c=color,
            marker="*",
            s=200,
            label=label_query if label_query not in labels_used else "",
        )
        labels_used.add(label_query)

        # Centroide asignado
        label_centroid = f"Centroide asignado {i+1}"
        axs[2].scatter(
            assigned_centroid[0],
            assigned_centroid[1],
            c=color,
            marker="X",
            s=100,
            label=label_centroid if label_centroid not in labels_used else "",
        )
        labels_used.add(label_centroid)

    axs[2].set_title("Consulta y Centroide del Vecino más Cercano")
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")
    axs[2].set_xlim(xlims)
    axs[2].set_ylim(ylims)
    axs[2].grid(True)
    axs[2].legend(fontsize="small")

    fig.delaxes(axs[3])

    plt.tight_layout()
    plt.savefig("plots/pq_plots.png")


def main():
    pq_create_plots_2d()


if __name__ == "__main__":
    main()
