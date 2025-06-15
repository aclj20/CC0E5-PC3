import numpy as np
from sklearn.cluster import KMeans

class ProductQuantizer:
    def __init__(self, M=4, Ks=256):
        """
        M: N° Subvectores
        Ks: Número de centroides para cada codebook
        """
        if M <= 0:
            raise ValueError(f"Valor M debe ser mayor a 0. Recibido {M}")
        if Ks <= 0:
            raise ValueError(f"Valor Ks debe ser mayor a 0. Recibido {M}")
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
            raise ValueError(f"! Dimensión {D} debe ser divisible por {self.M}.")
        self.Ds = D // self.M  # dimensión de cada subvector

        # Calcular k-means
        self.codebooks = []
        for m in range(self.M):
            # Extraer sub-vectores
            sub_vectors = X[:, m*self.Ds:(m+1)*self.Ds]
            # Entrena clústeres aleatorios
            kmeans = KMeans(n_clusters=self.Ks, random_state=42, n_init=10)
            kmeans.fit(sub_vectors)
            # Clústeres generados son guardados como codebooks
            self.codebooks.append(kmeans.cluster_centers_)

    def encode(self, X):
        """
        Codificar el conjunto de vectores X en códigos PQ
        Returns: arreglo numpy de indices (N, M) con indices de centroides.
        """
        if len(self.codebooks) == 0:
            raise ValueError(f"! No se tiene codebook definido")
        N, D = X.shape
        codes = np.empty((N, self.M), dtype=np.uint8)
        for m in range(self.M):
            # Dividir en subvectores para definir subespacios
            sub_vectors = X[:, m*self.Ds:(m+1)*self.Ds]
            # Separar centroides para subespacio
            centroids = self.codebooks[m]
            # Distancia a centroides
            # Se usa broadcasting. Se acomoda la dimensión de las matrices
            # para hallar la distancia rapidamente.
            # Alternativamente, dos bucles for también logran el mismo objetivo.
            distances = np.linalg.norm(sub_vectors[:, None, :] - centroids[None, :, :], axis=2)
            # Se almacen el índice al cluster de menor distancia para cada
            # subvector.
            codes[:, m] = np.argmin(distances, axis=1)
        return codes

    def decode(self, codes):
        """
        Decodificar códigos PQ de vuelta a vectores aproximados.
        Returns: arreglo numpy de dimension (N, D)
        """
        if len(self.codebooks) == 0:
            raise ValueError(f"! No se tiene codebook definido")
        N = codes.shape[0]
        X_reconstructed = np.zeros((N, self.M * self.Ds), dtype=np.float32)
        for m in range(self.M):
            centroids = self.codebooks[m]
            # Los códigos son índices que apuntan a los centroides.
            # Se recuperan los centroides del subespacio usando los códigos.
            # Estos centroides aproximan al vector original.
            X_reconstructed[:, m*self.Ds:(m+1)*self.Ds] = centroids[codes[:, m]]
        return X_reconstructed

    def approximate_knn(self, query, codes, k=5):
        """
        Aproximar K-nn mediante cálculo asimétrico de distancia (ADC)
        Returns: Índices de los vecinos más cercanos y sus distancias
        """
        if len(self.codebooks) == 0:
            raise ValueError(f"! No se tiene codebook definido")
        N = codes.shape[0]
        distances = np.zeros(N, dtype=np.float32)

        for m in range(self.M):
            # Se divide el vector en subvectores.
            query_sub = query[m*self.Ds:(m+1)*self.Ds]
            centroids = self.codebooks[m]
            # Se usa broadcasting de numpy para hallar la distancia rápidamente.
            # lookup almacena la distancia del subvector a todos los centroides
            lookup = np.linalg.norm(centroids - query_sub, axis=1)
            # Solo se necesitan las distancias de los centroides a los que apuntan
            # los códigos.
            # distances guarda la distancia a los centroides del subespacio
            # que representa codes.
            distances += lookup[codes[:, m]]

        # Recuperar los índices de las distancias más cercanas
        knn_indices = np.argsort(distances)[:k]
        # Recuperar las distancias más cercanas con estos índices
        knn_distances = distances[knn_indices]
        return knn_indices, knn_distances

# Driver básico para probar su funcionamiento esencial
if __name__ == "__main__":
    # Genera 1000 vectores de dimensión 32
    np.random.seed(0)
    X = np.random.randn(1000, 32).astype(np.float32)
    query = np.random.randn(32).astype(np.float32)

    # Inicia PQ con M = 4 y K = 2**8
    # 32 % 8 == 0
    pq = ProductQuantizer(M=4, Ks=256)
    pq.fit(X)
    codes = pq.encode(X)

    knn_indices, knn_distances = pq.approximate_knn(query, codes, k=5)
    print("5 Vecinos cercanos:")
    for i, d in zip(knn_indices, knn_distances):
        print(f"Indice: {i}, Distancia: {d:.4f}")
