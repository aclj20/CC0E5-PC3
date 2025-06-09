import numpy as np
from sklearn.cluster import KMeans

class ProductQuantizer:
    def __init__(self, M=4, Ks=256):
        """
        M: N° Subvectores
        Ks: Número de centroides para cada codebook
        """
        self.M = M
        self.Ks = Ks
        self.codebooks = []

    def fit(self, X):
        """
        Obtener codebook a partir de arreglo.
        X: arreglo numpy de tamaño (N, D)
        """
        N, D = X.shape
        if not D % self.M == 0:
            raise ValueError(f"Dimensión {D} debe ser divisible por {self.M}.")
        self.Ds = D // self.M  # dimensión de cada subvector

        # Calcular k-means
        self.codebooks = []
        for m in range(self.M):
            # Extraer sub-vectores
            sub_vectors = X[:, m*self.Ds:(m+1)*self.Ds]
            kmeans = KMeans(n_clusters=self.Ks, random_state=42, n_init=10)
            kmeans.fit(sub_vectors)
            self.codebooks.append(kmeans.cluster_centers_)

    def encode(self, X):
        """
        Codificar el vector X en códigos PQ
        Returns: arreglo numpy de indices (N, M) con indices de centroides.
        """
        if self.codebooks is []:
            print("! No se tiene codebook todavia !")
            return
        N, D = X.shape
        codes = np.empty((N, self.M), dtype=np.uint8)
        for m in range(self.M):
            sub_vectors = X[:, m*self.Ds:(m+1)*self.Ds]
            centroids = self.codebooks[m]
            # Distancia a centroides
            distances = np.linalg.norm(sub_vectors[:, None, :] - centroids[None, :, :], axis=2)
            codes[:, m] = np.argmin(distances, axis=1)
        return codes

    def decode(self, codes):
        """
        Decodificar PQ de vuelta a vector aproximado.
        Returns: arreglo numpy de dimension (N, D)
        """
        N = codes.shape[0]
        X_reconstructed = np.zeros((N, self.M * self.Ds), dtype=np.float32)
        for m in range(self.M):
            centroids = self.codebooks[m]
            X_reconstructed[:, m*self.Ds:(m+1)*self.Ds] = centroids[codes[:, m]]
        return X_reconstructed

if __name__ == "__main__":
    X = np.random.randn(1000, 32).astype(np.float32)

    pq = ProductQuantizer(M=4, Ks=256)
    pq.fit(X)

    codes = pq.encode(X)
    # Deberia tener menor dimension que X
    print("PQ codes shape:", codes.shape)

    X_approx = pq.decode(codes)
    # Debería tener mayor dimension que PQ
    print("Reconstructed vectors shape:", X_approx.shape)
