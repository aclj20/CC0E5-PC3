# CC0E5-PC3: Locality Sensitive Hashing (LSH) para búsqueda aproximada y Product Quantiation avanzada

## Descripción
Este proyecto contiene implementaciones de algunos algoritmos de búsqueda de vecinos aproximada: Locality Sensitive Hashing (LSH), Product Quantization (PQ) y Optimized Product Quantization (OPQ).

La búsqueda de vecinos cercanos requiere de alto poder computacional para analizar similaridad entre objetos representables en un espacio vectorial. Esta búsqueda se usa en distintos tipos de sistemas y áreas, por lo que los algoritmos de búsqueda de vecinos aproximada son esenciales cuando se tienen millones de datos de alta dimensionalidad. Dos de estos, LSH y PQ (incluyendo OPQ) ofrecen formas de aproximar similitud entre vectores. El primero se basa en funciones hash para detectar similitudes entre vectores rápidamente, mientras que el segundo se basa en la cuantización o compresión de datos para reducir el espacio de búsqueda. La version optimizada de Product Quantization aplica rotaciones de vectores para mejorar la precisión de la búsquedas y consultas.

El proyecto incluye drivers para demostrar del funcionamiento de LSH, PQ y OPQ. Además, incluye comparaciones graficadas entre LSH, PQ y OPQ en su rendimiento temporal su eficiencia en memoria para grande conjuntos de datos.

## Instrucciones de uso

### Software requerido

- Python 3.x
- `pip`

### Instrucciones

1. Clonar el repositorio.

```bash
$ git clone https://github.com/aclj20/CC0E5-PC3.git
$ cd CC0E5-PC3
```

2. Iniciar un ambiente virtual (recomendable) e instalar librerías necesiras.

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

3. Para interactuar con los drivers, ejecutarlos como una aplicación de Python dentro del ambiente virtual desde la raíz.

```bash
$ python -m drivers.driver_pq.py
```

  - Los drivers de comparación suelen demorar unos minutos en ejecutarse.
  - Los plots generados son guardados en el directorio `plots`.

4. Para ejecutar los tests, solo se llama a `pytest` en la raíz del proyecto.

```bash
$ python -m pytest
```

## Estructura del proyecto

```
CC0E5-PC3-main/
├── busqueda_aproximada/
│   ├── lsh.py
│   ├── opq.py
│   └── pq.py
├── plots/
│   └── (salidas gráficas generadas, como .png)
├── tests/
│   └── (tests pytest para LSH, PQ y OPQ)
├── util/
│   └── data_manager.py
├── data/
│   └── (datos sintéticos generados)
├── drivers/
│   ├── driver_compare_memory.py
│   ├── driver_knn_comparison.py
│   ├── driver_indexing_comparison.py
│   ├── driver_time_comparison.py
│   ├── driver_plot_lsh.py
│   ├── driver_plot_opq.py
│   └── driver_plot_pq.py
└── README.md
```

### `busqueda_aproximada/`

Incluye las implementaciones principales LSH, OPQ y PQ en sus respectivas clases y archivos.

### `drivers/`

Incluye los principales archivos de demostración y comparacion de LSH, OPQ y PQ. Los plots generados de estos drivers son almacenados en `plots/`

### `plots/`

Gráficos generados de los drivers que demuestran el uso de LSH, OPQ y PQ. Además, incluye comparaciones en el rendimiento y efectividad entre estas implementaciones.

### `util/`

Incluye el script `data_manager.py` para generar datos aleatorios para ser usados en los drivers.

### `data/`

Incluye datos sintéticos generados por `data_manager.py` almacenados como archivos NPY.

### `tests/`

Incluye las pruebas realizadas en pytest para las implementaciones desarrolladas.

## API pública

### Product Quantization

#### Clase: `ProductQuantizer`

##### Parámetros:

* `M` (`int`): Número de sub-vectores (subespacios) en los que se dividirá cada vector de entrada.
* `Ks` (`int`): Número de centroides (clusters) por cada subespacio.

##### Excepciones:

* `ValueError`: Si `M` o `Ks` son menores o iguales a 0.

#### `fit(X)`

Entrena los codebooks (uno por subespacio) usando K-Means.

##### Parámetros:

* `X` (`np.ndarray`): Matriz de tamaño `(N, D)` con vectores de entrenamiento.

##### Excepciones:

* `ValueError`: Si la dimensión `D` no es divisible por `M`.

#### `encode(X)`

Codifica los vectores en códigos PQ, es decir, índices de centroides por subespacio.

##### Parámetros:

* `X` (`np.ndarray`): Matriz de tamaño `(N, D)` con vectores a codificar.

##### Retorna:

* `np.ndarray`: Matriz de tamaño `(N, M)` con los índices de centroides.

##### Excepciones:

* `ValueError`: Si no se ha entrenado el codebook (`fit` no ha sido llamado).

#### `decode(codes)`

Reconstruye una aproximación de los vectores originales a partir de los códigos PQ.

##### Parámetros:

* `codes` (`np.ndarray`): Matriz de tamaño `(N, M)` con los códigos PQ.

##### Retorna:

* `np.ndarray`: Matriz de tamaño `(N, D)` con los vectores reconstruidos.

##### Excepciones:

* `ValueError`: Si no se ha entrenado el codebook (`fit` no ha sido llamado).

#### `approximate_knn(query, codes, k=5)`

Realiza una búsqueda aproximada de los `k` vecinos más cercanos usando **ADC (Asymmetric Distance Computation)**.

##### Parámetros:

* `query` (`np.ndarray`): Vector de consulta de tamaño `(D,)`.
* `codes` (`np.ndarray`): Códigos PQ de la base de datos, de tamaño `(N, M)`.
* `k` (`int`): Número de vecinos a recuperar. Por defecto `5`.

##### Retorna:

* `tuple`:

  * `np.ndarray`: Índices de los `k` vecinos más cercanos.
  * `np.ndarray`: Distancias aproximadas correspondientes.

##### Excepciones:

* `ValueError`: Si no se ha entrenado el codebook (`fit` no ha sido llamado).

### Optimized Product Quantization

#### `__init__(self, M=4, Ks=256, num_iters=20)`

Inicializa el codificador OPQ.

##### Parámetros:

* `M` (`int`): Número de subespacios (o particiones) en los que se divide cada vector.
* `Ks` (`int`): Número de centroides (clusters) por subespacio.
* `num_iters` (`int`): Número de iteraciones de optimización de la rotación.

#### `fit(self, X)`

Entrena el modelo OPQ aprendiendo una matriz de rotación `R` y los centroides de cada subespacio.

##### Parámetros:

* `X` (`np.ndarray`): Matriz de entrenamiento de forma `(N, D)`.

##### Retorno:

* `self`: El objeto `OptimizedProductQuantizer` entrenado.

##### Excepciones:

* `ValueError`: Si `X.ndim != 2` o si no puede calcular la descomposición SVD.

#### `encode(self, X)`

Codifica un conjunto de vectores aplicando primero la rotación y luego el Product Quantization.

##### Parámetros:

* `X` (`np.ndarray`): Datos a codificar de forma `(N, D)`.

##### Retorno:

* `codes` (`np.ndarray`): Códigos comprimidos de forma `(N, M)` con índices de centroides.

##### Excepciones:

* `RuntimeError`: Si el modelo no ha sido entrenado (`self.R is None`).
* `ValueError`: Si las dimensiones de `X` no coinciden con las del entrenamiento.

#### `decode(self, codes)`

Reconstruye vectores aproximados a partir de los códigos PQ, aplicando la rotación inversa.

##### Parámetros:

* `codes` (`np.ndarray`): Códigos de forma `(N, M)`.

##### Retorno:

* `X_hat` (`np.ndarray`): Aproximación de los vectores originales, forma `(N, D)`.

#### `compress(self, X)`

Realiza compresión completa: codificación seguida de decodificación.

##### Parámetros:

* `X` (`np.ndarray`): Vectores de entrada a comprimir, forma `(N, D)`.

##### Retorno:

* `X_hat` (`np.ndarray`): Vectores reconstruidos, forma `(N, D)`.


#### `approximate_knn(self, query, codes, k=5)`

Calcula los **k vecinos más cercanos aproximados** a un vector de consulta, usando PQ y rotación.

##### Parámetros:

* `query` (`np.ndarray`): Vector de consulta, forma `(D,)`.
* `codes` (`np.ndarray`): Códigos comprimidos de base de datos, forma `(N, M)`.
* `k` (`int`): Número de vecinos a retornar.

##### Retorno:

* `indices` (`List[int]`): Índices de los vecinos más cercanos.
* `distances` (`List[float]`): Distancias aproximadas correspondientes.

##### Excepciones:

* `RuntimeError`: Si el modelo no ha sido entrenado (`self.R is None`).
* `ValueError`: Si `query.shape` no coincide con `D`.

### Locality Sensitive Hashing

#### Clase: `LSH`

Implementación de **Locality Sensitive Hashing** para búsqueda aproximada de vecinos cercanos utilizando **proyecciones aleatorias y partición en buckets**.

##### Parámetros:

* `data` (`np.ndarray`): Matriz de vectores de entrada de tamaño `(N, D)`, donde `N` es el número de vectores y `D` su dimensionalidad.

#### `train(num_vector, seed=None)`

Entrena el índice LSH generando funciones de hash aleatorias y asignando vectores a buckets.

##### Parámetros:

* `num_vector` (`int`): Número de vectores aleatorios (bits de hash) que definen la función hash. Controla la granularidad del particionado.
* `seed` (`int`, opcional): Semilla aleatoria para reproducibilidad.

##### Retorna:

* `LSH`: La instancia entrenada (`self`).

#### `query(query_vec, k, max_search_radius, initial_candidates=set())`

Realiza una búsqueda aproximada de los `k` vecinos más cercanos al vector de consulta usando LSH con expansión de buckets vecinos.

##### Parámetros:

* `query_vec` (`np.ndarray`): Vector de consulta de tamaño `(D,)`.
* `k` (`int`): Número de vecinos más cercanos a recuperar.
* `max_search_radius` (`int`): Radio máximo de buckets alternativos a explorar.
* `initial_candidates` (`set`, opcional): Conjunto de índices candidatos iniciales (por ejemplo, de una consulta previa).

##### Retorna:

* `pandas.DataFrame`: DataFrame con las columnas:

  * `"id"`: Índices de los vecinos encontrados.
  * `"distance"`: Distancias reales (Euclidianas) al vector de consulta.

##### Excepciones:

* `SystemExit`: Si `train()` no ha sido llamado antes (modelo no inicializado).
* Puede lanzar errores de NumPy si `query_vec` tiene una dimensión incompatible.

