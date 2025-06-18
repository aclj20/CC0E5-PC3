import numpy as np
import cProfile
import pstats
import io
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

def profile_cprofile(func, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    func(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(10)
    return s.getvalue(), ps.total_tt

def compare_encoding_time():
    dm = DataManager()
    pq_times = []
    opq_times = []
    lsh_times = []

    for size in SIZES:
        N, M, Ks, D = size, 8, 256, 64
        X = dm.load(size)

        # PQ Encoding
        pq = ProductQuantizer(M, Ks)
        pq.fit(X)
        pq_profile, pq_time = profile_cprofile(lambda: pq.encode(X))
        pq_times.append(pq_time)

        # OPQ Encoding
        opq = OptimizedProductQuantizer(M, Ks, num_iters=5)
        opq.fit(X)
        opq_profile, opq_time = profile_cprofile(lambda: opq.encode(X))
        opq_times.append(opq_time)

        lsh = LSH(X)
        lsh_profile, lsh_time = profile_cprofile(lambda: lsh.train(num_vector=16))
        lsh_times.append(lsh_time)

    x = np.arange(len(SIZES))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, pq_times, width, label='PQ')
    plt.bar(x, opq_times, width, label='OPQ')
    plt.bar(x + width, lsh_times, width, label='LSH')

    plt.xticks(x, [str(s) for s in SIZES])
    plt.xlabel("Tamaño de los datos (# Vectores)")
    plt.ylabel("Tiempo (segundos)")
    plt.title("Comparación de tiempo requerido para indexar")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("plots/time_comparison.png")

if __name__ == "__main__":
    compare_encoding_time()
