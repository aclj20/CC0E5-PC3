from copy import copy
from itertools import combinations
import numpy as np
from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances


class LSH:
    def __init__(self, data):
        self.data = data
        self.model = None

    def __generate_random_vectors(self, num_vector, dim):
        return np.random.randn(dim, num_vector)

    def train(self, num_vector, seed=None):
        dim = self.data.shape[1]
        if seed is not None:
            np.random.seed(seed)

        random_vectors = self.__generate_random_vectors(num_vector, dim)
        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

        table = {}

        # Dividir los puntos de datos en buckets
        bin_index_bits = (self.data.dot(random_vectors) >= 0)

        # Codificar los bits del índice de bucket como enteros
        bin_indices = bin_index_bits.dot(powers_of_two)

        # Actualizar la tabla para que table[i] sea la lista de documentos con índice igual a i
        for data_index, bin_index in enumerate(bin_indices):
            if bin_index not in table:
                # Si aún no existe una lista para este bucket, se inicializa vacía
                table[bin_index] = []
            # Se añade el índice del documento al bucket correspondiente
            table[bin_index].append(data_index)

        self.model = {
            'bin_indices': bin_indices,
            'table': table,
            'random_vectors': random_vectors,
            'num_vector': num_vector
        }
        return self

    def __search_nearby_bins(self, query_bin_bits, table, search_radius=2, initial_candidates=set()):
        num_vector = self.model['num_vector']
        powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

        # Permitir al usuario proporcionar un conjunto inicial de candidatos
        candidate_set = copy(initial_candidates)

        for different_bits in combinations(range(num_vector), search_radius):
            alternate_bits = copy(query_bin_bits)
            for i in different_bits:
                alternate_bits[i] = 1 if alternate_bits[i] == 0 else 0

            # Convertir el nuevo vector binario en un índice entero
            nearby_bin = alternate_bits.dot(powers_of_two)

            # Obtener la lista de documentos que pertenecen a este bucket cercano
            if nearby_bin in table:
                candidate_set.update(table[nearby_bin])

        return candidate_set

    def query(self, query_vec, k, max_search_radius, initial_candidates=set()):
        if not self.model:
            print('Modelo aún no entrenado')
            exit(-1)

        data = self.data
        table = self.model['table']
        random_vectors = self.model['random_vectors']

        bin_index_bits = (query_vec.dot(random_vectors) >= 0).flatten()

        candidate_set = set()
        # Buscar en buckets vecinos y recolectar candidatos
        for search_radius in range(max_search_radius + 1):
            candidate_set = self.__search_nearby_bins(
                bin_index_bits,
                table,
                search_radius,
                initial_candidates=initial_candidates
            )

        # Ordenar candidatos por su distancia real al vector de consulta
        nearest_neighbors = DataFrame({'id': list(candidate_set)})
        candidates = data[np.array(list(candidate_set)), :]
        nearest_neighbors['distance'] = pairwise_distances(candidates, query_vec, metric='cosine').flatten()

        return nearest_neighbors.nsmallest(k, 'distance')
