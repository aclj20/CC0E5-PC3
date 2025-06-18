import numpy as np
from sklearn.neighbors import NearestNeighbors
from busqueda_aproximada.pq import ProductQuantizer
from busqueda_aproximada.opq import OptimizedProductQuantizer
from busqueda_aproximada.lsh import LSH
from util.data_manager import DataManager
import matplotlib.pyplot as plt

SIZES = [
    500,
    1000,
    3000,
    4000,
    5000,
    10000,
    100000
]

def get_data(N=100_000, D=64, seed=42):
    np.random.seed(seed)
    return np.random.randn(N, D).astype(np.float32)

def compute_ground_truth(X, queries, k):
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X)
    _, indices = nn.kneighbors(queries)
    return [set(row) for row in indices]

def recall_at_k(true_sets, approx_sets, k):
    correct = 0
    for true, approx in zip(true_sets, approx_sets):
        correct += len(set(approx).intersection(true))
    return correct / (len(true_sets) * k)

def compare_recall():
    pq_recalls = []
    opq_recalls = []
    lsh_recalls = []
    dm = DataManager()
    k = 5
    n_queries = 20

    for size in SIZES:
        N, M, Ks, D = size, 8, 256, 64
        X = dm.load(size)
        queries = get_data(N=n_queries, D=64)

        # K-nn verdadero
        true_sets = compute_ground_truth(X, queries, k)

        # PQ
        pq = ProductQuantizer(M, Ks)
        pq.fit(X)
        codes = pq.encode(X)
        pq_results = [pq.approximate_knn(q, codes, k=k)[0] for q in queries]
        pq_recall = recall_at_k(true_sets, pq_results, k)
        pq_recalls.append(pq_recall)

        # OPQ
        opq = OptimizedProductQuantizer(M, Ks, num_iters=5)
        opq.fit(X)
        codes = opq.encode(X)
        opq_results = [opq.approximate_knn(q, codes, k=k)[0] for q in queries]
        opq_recall = recall_at_k(true_sets, opq_results, k)
        opq_recalls.append(opq_recall)

        # LSH
        lsh = LSH(X)
        lsh.train(num_vector=16)
        lsh_results = []
        for q in queries:
            df = lsh.query(q, k=k, max_search_radius=2)
            if len(df) == 0:
                # If there are no candidates, fill with dummy indices (e.g. -1)
                lsh_results.append([-1] * k)
            else:
                ids = df['id'].tolist()
                # Pad with -1 if fewer than k
                if len(ids) < k:
                    ids = ids + [-1] * (k - len(ids))
                lsh_results.append(ids)
        lsh_recall = recall_at_k(true_sets, lsh_results, k)
        lsh_recalls.append(lsh_recall)

    # Plot results
    x = np.arange(len(SIZES))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, pq_recalls, width, label='PQ')
    plt.bar(x, opq_recalls, width, label='OPQ')
    plt.bar(x + width, lsh_recalls, width, label='LSH')

    plt.xticks(x, [str(s) for s in SIZES])
    plt.xlabel("Tamaño de los datos (# Vectores)")
    plt.ylabel("Recall@5")
    plt.title("Recall@5 para PQ, OPQ y LSH")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("plots/recall_comparison.png")

if __name__ == "__main__":
    compare_recall()
