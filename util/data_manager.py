import numpy as np


class DataManager:
    def generate_and_save(self, num_vectors=100_000, dimension=64):
        """
        Genera vectores aleatorios y los guarda en vector_data_<num_vectors>.
        """
        data = np.random.randn(num_vectors, dimension).astype(np.float32)
        filename = "data/vector_data_" + str(num_vectors) + ".npy"
        np.save(filename, data)

    def load(self, num_vectors):
        """
        Lee datos de vector_data_<num_vectors> y los retorna como arreglo de numpy.
        """
        data = np.load("data/vector_data_" + str(num_vectors) + ".npy")
        return data


def main():
    SIZES = [
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        2000,
        3000,
        4000,
        5000,
        10000,
        20000,
        30000,
        40000,
        50000,
        100000,
    ]
    dm = DataManager()
    for size in SIZES:
        dm.generate_and_save(num_vectors=size)


if __name__ == "__main__":
    main()
