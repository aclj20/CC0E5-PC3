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

3. Para interactuar con los drivers, ejecutarlos como una aplicación de Python dentro del ambiente virtual.

```bash
$ python -m drivers.driver_pq.py
```

4. Para ejecutar los tests, solo se llama a `pytest` en la raíz del proyecto.

```bash
$ python -m pytest
```

## Estructura del proyecto

## API pública

### Product Quantization

#### Clase: `ProductQuantizer`

```python
ProductQuantizer(M=4, Ks=256)
```

##### Parámetros:

* `M` (`int`): Número de sub-vectores (subespacios) en los que se dividirá cada vector de entrada.
* `Ks` (`int`): Número de centroides (clusters) por cada subespacio.

##### Excepciones:

* `ValueError`: Si `M` o `Ks` son menores o iguales a 0.

#### `fit(X)`

```python
fit(X: np.ndarray) -> None
```

Entrena los codebooks (uno por subespacio) usando K-Means.

##### Parámetros:

* `X` (`np.ndarray`): Matriz de tamaño `(N, D)` con vectores de entrenamiento.

##### Excepciones:

* `ValueError`: Si la dimensión `D` no es divisible por `M`.

#### `encode(X)`

```python
encode(X: np.ndarray) -> np.ndarray
```

Codifica los vectores en códigos PQ, es decir, índices de centroides por subespacio.

##### Parámetros:

* `X` (`np.ndarray`): Matriz de tamaño `(N, D)` con vectores a codificar.

##### Retorna:

* `np.ndarray`: Matriz de tamaño `(N, M)` con los índices de centroides.

##### Excepciones:

* `ValueError`: Si no se ha entrenado el codebook (`fit` no ha sido llamado).

#### `decode(codes)`

```python
decode(codes: np.ndarray) -> np.ndarray
```

Reconstruye una aproximación de los vectores originales a partir de los códigos PQ.

##### Parámetros:

* `codes` (`np.ndarray`): Matriz de tamaño `(N, M)` con los códigos PQ.

##### Retorna:

* `np.ndarray`: Matriz de tamaño `(N, D)` con los vectores reconstruidos.

##### Excepciones:

* `ValueError`: Si no se ha entrenado el codebook (`fit` no ha sido llamado).

#### `approximate_knn(query, codes, k=5)`

```python
approximate_knn(query: np.ndarray, codes: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]
```

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
