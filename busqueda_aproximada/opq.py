import numpy as np
from busqueda_aproximada.pq import ProductQuantizer  

class OptimizedProductQuantizer:
    def __init__(self, M=4, Ks=256, num_iters=20):
        """
        M: número de subespacios
        Ks: número de centroides por subespacio
        num_iters: número de iteraciones para aprender la rotación óptima
        """
        self.M = M
        self.Ks = Ks
        self.num_iters = num_iters
        self.pq = ProductQuantizer(M, Ks)
        self.R = None  

    def fit(self, X):
        """
        Aprende la matriz de rotación R y entrena los codebooks
        """
        N, D = X.shape
        self.R = np.eye(D, dtype=np.float32)  

        for i in range(self.num_iters):
            # Paso 1: Rotar los datos
            X_rot = X @ self.R

            # Paso 2: Entrenar PQ sobre los datos rotados
            self.pq.fit(X_rot)

            # Paso 3: Reconstruir los datos desde los códigos
            X_codes = self.pq.encode(X_rot)
            X_reconstructed = self.pq.decode(X_codes)

            # Paso 4: Aprender nueva R usando SVD 
            U, _, Vt = np.linalg.svd(X.T @ X_reconstructed)
            self.R = U @ Vt  

        return self

    def encode(self, X):
        """
        Codifica los vectores aplicando la rotación y luego PQ.
        """
        X_rot = X @ self.R
        return self.pq.encode(X_rot)

    def decode(self, codes):
        """
        Reconstruye los vectores y aplica la rotación inversa.
        """
        X_rot_reconstructed = self.pq.decode(codes)
        return X_rot_reconstructed @ self.R.T  

    def compress(self, X):
        """
        Codifica y luego decodifica (aproximación del vector original).
        """
        return self.decode(self.encode(X))

    def approximate_knn(self, query, codes, k=5):
        """
        Aplica approximate KNN a un vector de consulta rotado.
        """
        query_rot = query @ self.R
        return self.pq.approximate_knn(query_rot, codes, k=k)

# Driver básico para probar OptimizedProductQuantizer
if __name__ == "__main__":
    np.random.seed(0)

    # Genera 1000 vectores de dimensión 32
    X = np.random.randn(1000, 32).astype(np.float32)
    query = np.random.randn(32).astype(np.float32)

    # Inicia OPQ con M = 4, Ks = 256
    opq = OptimizedProductQuantizer(M=4, Ks=256, num_iters=20)
    opq.fit(X)

    # Codifica y reconstruye
    codes = opq.encode(X)
    X_hat = opq.decode(codes)

    # Mide el error promedio de reconstrucción
    error = np.mean(np.linalg.norm(X - X_hat, axis=1))
    print(f"Error promedio de reconstrucción: {error:.4f}\n")

    # Consulta k-NN aproximado
    knn_indices, knn_distances = opq.approximate_knn(query, codes, k=5)
    print("5 Vecinos cercanos (approx):")
    for i, d in zip(knn_indices, knn_distances):
        print(f"Indice: {i}, Distancia: {d:.4f}")






