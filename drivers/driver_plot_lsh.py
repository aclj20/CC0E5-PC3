import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from busqueda_aproximada.lsh import LSH

def bin_bits_to_str(bits):
    return ''.join(str(b) for b in bits.astype(int))

def plot_scene_2d(data, query, nearest_idx, random_vectors, num_vector):
    fig, ax = plt.subplots(figsize=(8, 6))
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)
    bucket_ids = (data @ random_vectors >= 0).astype(int).dot(powers_of_two)
    unique_buckets = sorted(set(bucket_ids))
    bucket_to_color = {b: i for i, b in enumerate(unique_buckets)}
    colors = [bucket_to_color[b] for b in bucket_ids]

    ax.scatter(data[:, 0], data[:, 1], c=colors, cmap='tab20', s=40, label='Datos')
    ax.scatter(query[0], query[1], marker='*', c='black', s=200, label='Consulta')
    nearest_point = data[nearest_idx]
    ax.scatter(nearest_point[0], nearest_point[1], facecolors='none', edgecolors='red',
               s=160, linewidths=2, label='Más cercano')

    xlim = ax.get_xlim()
    for i in range(num_vector):
        a = random_vectors[:, i]
        if a[1] == 0:
            ax.axvline(0, color='blue', linestyle='--')
        else:
            x_vals = np.linspace(*xlim, 200)
            y_vals = -(a[0] / a[1]) * x_vals
            ax.plot(x_vals, y_vals, linestyle='--', color='blue', linewidth=1)

    ax.set_title(f"LSH 2D con {num_vector} vectores")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    margin = 110
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_scene_3d(data, query, nearest_idx, random_vectors, num_vector):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace(-100, 100, 20)
    y = np.linspace(-100, 100, 20)
    X, Y = np.meshgrid(x, y)

    for i in range(num_vector):
        a = random_vectors[:, i]
        if abs(a[2]) < 1e-6:
            continue
        Z = (-a[0] * X - a[1] * Y) / a[2]
        ax.plot_surface(X, Y, Z, alpha=0.1, color='blue', linewidth=0)

    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)
    bucket_ids = (data @ random_vectors >= 0).astype(int).dot(powers_of_two)
    unique_buckets = sorted(set(bucket_ids))
    bucket_to_color = {b: i for i, b in enumerate(unique_buckets)}
    colors = [bucket_to_color[b] for b in bucket_ids]

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap='tab20', s=40, label='Datos')
    ax.scatter(*query, c='black', marker='*', s=200, label='Consulta')
    nearest_point = data[nearest_idx]
    ax.scatter(*nearest_point, facecolors='none', edgecolors='red', s=160, linewidths=2, label='Más cercano')

    ax.set_title(f"LSH 3D con {num_vector} vectores")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    DIMENSION = 3         
    NUM_POINTS = 100
    NUM_VECTOR = 2
    MAX_RADIUS = 1
    K = 5
    SHOW_PLOT = True

    np.random.seed(42)
    data = np.random.uniform(low=-100, high=100, size=(NUM_POINTS, DIMENSION)).astype(np.float32)
    query = np.random.uniform(low=-100, high=100, size=(DIMENSION,)).astype(np.float32)

    lsh = LSH(data)
    lsh.train(num_vector=NUM_VECTOR)

    print("Buckets de cada punto:")
    powers_of_two = 1 << np.arange(NUM_VECTOR - 1, -1, -1)
    random_vectors = lsh.model["random_vectors"]
    for i, vec in enumerate(data):
        bits = (vec @ random_vectors >= 0).astype(int)
        bucket = bits.dot(powers_of_two)
        print(f" idx {i:2d} → bits = {bin_bits_to_str(bits)}, bucket = {bucket}")

    query_bits = (query @ random_vectors >= 0).astype(int)
    query_bin = query_bits.dot(powers_of_two)
    print(f"\nQuery: {query}")
    print(f"Bucket: bits = {bin_bits_to_str(query_bits)} → entero = {query_bin}")

    result = lsh.query(query, k=K, max_search_radius=MAX_RADIUS)

    print(f"\n Punto de consulta:")
    print(f"    Coordenadas: {query}")

    print(f"\n Vecinos más cercanos (k={K}):")
    for i, row in result.iterrows():
        idx = int(row["id"])
        coord = data[idx]
        coord_str = ', '.join([f"{c:.2f}" for c in coord])
        print(f" idx = {idx:2d} → distancia = {row['distance']:.4f} → coords = ({coord_str})")

    if SHOW_PLOT:
        nearest_idx = int(result["id"].iloc[0])
        if DIMENSION == 2:
            plot_scene_2d(data, query, nearest_idx, random_vectors, NUM_VECTOR)
        elif DIMENSION == 3:
            plot_scene_3d(data, query, nearest_idx, random_vectors, NUM_VECTOR)

if __name__ == "__main__":
    main()
