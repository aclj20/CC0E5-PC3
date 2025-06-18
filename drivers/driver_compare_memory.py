import numpy as np
from pympler import asizeof
from busqueda_aproximada.pq import ProductQuantizer
from busqueda_aproximada.opq import OptimizedProductQuantizer
from busqueda_aproximada.lsh import LSH
from util.data_manager import DataManager
import matplotlib.pyplot as plt

SIZES = [
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000
]

def get_data(N=100_000, D=64, seed=42):
    np.random.seed(seed)
    return np.random.randn(N, D).astype(np.float32)

def lsh_index_memory(lsh_model):
    return asizeof(lsh_model.model['table'])

def compare_index_memory():
    lsh_memory = []
    pq_memory = []
    opq_memory = []
    dm = DataManager()
    for size in SIZES:
        N, M, Ks = size, 8, 256
        X = dm.load(size)

        pq = ProductQuantizer(M, Ks)
        pq.fit(X)
        pq_codes = pq.encode(X)
        pq_size = pq_codes.nbytes
        pq_memory.append(pq_size / (1024 * 1024))

        opq = OptimizedProductQuantizer(M, Ks, num_iters=5)
        opq.fit(X)
        opq_codes = opq.encode(X)
        opq_size = opq_codes.nbytes
        opq_memory.append(opq_size / (1024 * 1024))

        lsh_model = LSH(X)
        lsh_model.train(num_vector=16)
        lsh_size = asizeof.asizeof(lsh_model.model['table'])
        lsh_memory.append(lsh_size / (1024 * 1024))

    x = np.arange(len(SIZES))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, pq_memory, width, label='PQ')
    plt.bar(x, opq_memory, width, label='OPQ')
    plt.bar(x + width, lsh_memory, width, label='LSH')

    plt.xticks(x, [str(s) for s in SIZES])
    plt.xlabel("Tamaño de los datos (# Vectores)")
    plt.ylabel("Memoria de índices (MB)")
    plt.title("Comparación en el uso de memoria para índices")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("plots/index_comparison.png")

if __name__ == "__main__":
    compare_index_memory()
