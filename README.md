# MNIST Digit Classification Studio: Decision Tree, Random Forest & k-NN

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c?style=flat)](https://matplotlib.org/)
[![GUI](https://img.shields.io/badge/GUI-Tkinter%20%2F%20ttk-blue?style=flat)](https://docs.python.org/3/library/tkinter.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema integral de experimentación, benchmarking y clasificación interactiva de dígitos manuscritos del conjunto de datos **MNIST** ($28 \times 28$ píxeles, $784$ dimensiones), contrastando tres paradigmas supervisados fundamentales del Machine Learning: **Árboles de Decisión (CART)**, ensambles de **Bosques Aleatorios (Random Forest)** y métodos basados en vecindad de instancias (**k-Nearest Neighbors**).

El proyecto integra un pipeline determinista con muestreo estratificado, verificación formal de invariantes estadísticas, diagnóstico visual de errores de clasificación, un notebook de análisis y una aplicación de escritorio con GUI interactiva para inferencia en tiempo real y dibujo a mano alzada.

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Algoritmos Implementados y Notación Matemática](#algoritmos-implementados-y-notación-matemática)
   - [Árbol de Decisión (Decision Tree - CART)](#1-árbol-de-decisión-decision-tree---cart)
   - [Bosque Aleatorio (Random Forest)](#2-bosque-aleatorio-random-forest)
   - [k-Vecinos más Cercanos (k-Nearest Neighbors - k-NN)](#3-k-vecinos-más-cercanos-k-nearest-neighbors---k-nn)
3. [Funciones Heurísticas y Criterios de Decisión](#funciones-heurísticas-y-criterios-de-decisión)
4. [Control de Solvabilidad, Estratificación y Reproducibilidad](#control-de-solvabilidad-estratificación-y-reproducibilidad)
5. [Arquitectura del Proyecto](#arquitectura-del-proyecto)
6. [Funcionalidades de la Interfaz Gráfica (GUI)](#funcionalidades-de-la-interfaz-gráfica-gui)
7. [Matriz Comparativa de Métricas de Rendimiento](#matriz-comparativa-de-métricas-de-rendimiento)
   - [Resultados Oficiales del Benchmark](#resultados-oficiales-del-benchmark)
   - [Análisis de Confusiones Críticas](#análisis-de-confusiones-críticas)
   - [Compromiso Sesgo-Varianza y Costo Computacional](#compromiso-sesgo-varianza-y-costo-computacional)
8. [Instalación y Guía de Uso](#instalación-y-guía-de-uso)
9. [Licencia](#licencia)

---

## 📌 Descripción del Proyecto

El conjunto de datos **MNIST** (*Modified National Institute of Standards and Technology*) es el estándar de referencia canónico para tareas de visión artificial y reconocimiento de patrones. Cada observación representa una imagen en escala de grises de $28 \times 28$ píxeles correspondiente a un dígito del $0$ al $9$, linealizada en un vector de características en $\mathbb{R}^{784}$.

Este proyecto implementa una solución de Machine Learning que aborda los desafíos centrales del aprendizaje supervisado multiclase:
- **Alta dimensionalidad:** Procesamiento de tensores espaciales en espacios $d = 784$.
- **Generalización vs. Sobreajuste:** Impacto de la profundidad en árboles frente al efecto reductor de varianza de los ensambles (*Bagging*).
- **Compromiso en costo temporal:** Diferencia estructural entre algoritmos de aprendizaje ávido (*eager learning*) con entrenamiento pesado pero inferencia inmediata, frente a algoritmos perezosos (*lazy learning*) con entrenamiento nulo pero búsqueda vecinal computacionalmente costosa en inferencia.

---

## 🧠 Algoritmos Implementados y Notación Matemática

### 1. Árbol de Decisión (Decision Tree - CART)

El algoritmo construye un árbol binario particionando recursivamente el espacio de características $X \subseteq \mathbb{R}^{784}$ mediante divisiones ortogonales a los ejes coordenados.

#### Criterio de Impureza de Gini
En cada nodo interno $m$, que contiene un subconjunto de entrenamiento $D_m$ con $N_m$ muestras, la probabilidad empírica de la clase $k \in \{0, 1, \dots, 9\}$ se define como:

$$p_{mk} = \frac{1}{N_m} \sum_{x_i \in D_m} \mathbb{I}(y_i = k)$$

El índice de impureza de Gini mide la heterogeneidad de la distribución de clases en el nodo:

$$I_G(D_m) = 1 - \sum_{k=0}^{9} p_{mk}^2$$

#### Función de Costo de Partición
Para cada variable de pixel $j \in \{1, \dots, 784\}$ y cada umbral de corte $\theta \in \mathbb{R}$, se divide el conjunto en:

$$D_m^L(\theta) = \{x \in D_m \mid x_j \le \theta\}, \quad D_m^R(\theta) = \{x \in D_m \mid x_j > \theta\}$$

La división óptima $(j^*, \theta^*)$ se selecciona minimizando la impureza ponderada de las particiones resultantes:

$$(j^*, \theta^*) = \arg\min_{j, \theta} \left[ \frac{N_m^L}{N_m} I_G(D_m^L(\theta)) + \frac{N_m^R}{N_m} I_G(D_m^R(\theta)) \right]$$

- **Hiperparámetros de control:** Profundidad máxima acotada a $\text{max\_depth} = 15$ para mitigar el sobreajuste, con inicialización determinista (`random_state=42`).

---

### 2. Bosque Aleatorio (Random Forest)

Random Forest es un metaestimador por ensamble que combina $B$ árboles de decisión independientes $\{T_b\}_{b=1}^B$ mediante dos mecanismos de aleatorización complementarios:

#### Bootstrap Aggregating (Bagging)
Cada árbol $T_b$ se entrena sobre un conjunto $D_b$ de tamaño $N$ generado mediante remuestreo con reemplazo (*bootstrapping*) del conjunto original $D_{ent}$. La probabilidad asintótica de que una muestra particular no sea seleccionada en una bolsa individual (*Out-Of-Bag*) es:

$$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 36.8\%$$

#### Selección Aleatoria de Subespacios de Características
En cada división interna de cada árbol, no se evalúan las $p = 784$ variables, sino un subconjunto aleatorio de tamaño:

$$m = \lfloor \sqrt{p} \rfloor = \lfloor \sqrt{784} \rfloor = 28 \text{ píxeles}$$

Esto descorrelaciona los árboles individuales, reduciendo la varianza del ensamble total sin incrementar sustancialmente el sesgo.

#### Regla de Agregación
La predicción final para una muestra $x$ se obtiene mediante votación mayoritaria (*plurality voting*):

$$\hat{y}_{\text{RF}}(x) = \arg\max_{k \in \{0, \dots, 9\}} \sum_{b=1}^{B} \mathbb{I}\left(T_b(x) = k\right)$$

- **Hiperparámetros:** $B = 100$ estimadores (`n_estimators=100`), profundidad individual $\text{max\_depth} = 15$, paralelización multinúcleo completa (`n_jobs=-1`).

---

### 3. k-Vecinos más Cercanos (k-Nearest Neighbors - k-NN)

k-NN es un algoritmo de aprendizaje no paramétrico y basado en memoria (*instance-based / lazy learning*). Durante la fase de ajuste (`fit`), almacena el conjunto de entrenamiento $D_{ent} = \{(x_i, y_i)\}_{i=1}^{N}$.

#### Métrica de Distancia en Espacio $784$-Dimensional
La proximidad geométrica entre una instancia de consulta $x \in \mathbb{R}^{784}$ y cada punto almacenado $x_i \in \mathbb{R}^{784}$ se cuantifica mediante la norma $L_2$ (distancia euclídea):

$$d(x, x_i) = \|x - x_i\|_2 = \sqrt{\sum_{j=1}^{784} (x_j - x_{i,j})^2}$$

#### Conjunto de Vecinos y Votación
Sea $\mathcal{N}_k(x) \subset D_{ent}$ el conjunto de las $k$ instancias más cercanas a $x$ según $d(x, \cdot)$. La predicción asigna la clase más frecuente entre dichos vecinos con ponderación uniforme:

$$\hat{y}_{\text{kNN}}(x) = \arg\max_{c \in \{0, \dots, 9\}} \sum_{(x_i, y_i) \in \mathcal{N}_k(x)} \mathbb{I}(y_i = c)$$

- **Hiperparámetros:** $k = 5$ vecinos (`n_neighbors=5`), métrica euclídea (`metric='euclidean'`), ponderación uniforme (`weights='uniform'`).

---

## ⚙️ Funciones Heurísticas y Criterios de Decisión

| Algoritmo | Criterio de Selección / Heurística | Función Matemática | Rol en la Decisión |
| :--- | :--- | :--- | :--- |
| **Árbol de Decisión** | Reducción de Impureza de Gini | $\Delta I_G = I_G(D_m) - \left[ \frac{N_L}{N} I_G(D_L) + \frac{N_R}{N} I_G(D_R) \right]$ | Heurística voraz (*greedy*) para elegir el mejor corte de píxel en cada nodo. |
| **Random Forest** | Consenso por Ensamble (*Majority Voting*) | $\hat{y} = \operatorname{mode}(\{T_1(x), T_2(x), \dots, T_B(x)\})$ | Promediado estocástico que reduce el error por varianza $\sigma^2 / B$. |
| **k-NN** | Proximidad Métrica $L_2$ | $\min \sum_{j=1}^{784} (x_j - x_{i,j})^2$ | Heurística de continuidad topológica en el hiperespacio de intensidades. |

---

## 🛡️ Control de Solvabilidad, Estratificación y Reproducibilidad

Para garantizar rigor metodológico, comparabilidad estadística y ausencia de sesgos muestrales, el sistema implementa los siguientes controles formales:

1. **Normalización Espacial del Dominio:**
   Los valores crudos de intensidad de píxeles $x_{crudo} \in [0, 255]$ son reescalados al intervalo continuo $[0.0, 1.0]$:
   $$X = \frac{X_{\text{raw}}}{255.0} \in [0.0, 1.0]^{N \times 784}$$
   Esto evita distorsiones de escala en la norma $L_2$ de k-NN y acelera la evaluación de umbrales.

2. **Muestreo Estratificado Proporcional (*Stratified Sampling*):**
   Se extraen exactamente $N_{ent} = 10,000$ muestras de entrenamiento y $N_{pru} = 2,000$ muestras de prueba preservando la distribución uniforme de los dígitos:
   $$P(y = k \mid D_{ent}) = P(y = k \mid D_{pru}) = \frac{1}{10} = 10.00\%, \quad \forall k \in \{0, \dots, 9\}$$

3. **Verificación Formal de Invariantes Numéricas:**
   En cada ciclo de inferencia se valida algebraicamente la completitud del espacio de hipótesis mediante aserciones unitarias:
   $$\text{Exactitud (\%)} + \text{Error (\%)} = 100.00\% \quad (\text{tolerancia } \varepsilon < 10^{-7})$$

4. **Reproducibilidad Determinista:**
   Se fija la semilla pseudoaleatoria $\text{seed} = 42$ en todas las rutinas estocásticas (partición de datos, selección de características y construcción de árboles).

5. **Capa de Compatibilidad para Python 3.14+:**
   Incluye wrappers de seguridad para el analizador ARFF de `scikit-learn` y manipulación de tipos categóricos en `pandas`, previniendo excepciones por coerción de tipos en versiones modernas del runtime.

---

## 🏗️ Arquitectura del Proyecto

```text
DF-RT-KNN/
├── .gitignore                          # Exclusión exhaustiva de venvs, caché, secretos y binarios
├── LICENSE                             # Licencia MIT (Copyright 2024-2026 Angel Ayuquina)
├── README.md                           # Documentación técnica integral para portafolio
├── clasificacion_mnist.py              # Script principal CLI de benchmarking y generación de figuras
├── gui.py                              # Interfaz gráfica interactiva de escritorio (Tkinter)
├── Taller_MNIST_Arbol_RF_kNN.ipynb     # Jupyter Notebook con el flujo experimental completo
├── figura1_muestras_mnist.png          # Visualización de muestra de dígitos del dataset
├── figura2_matrices_confusion.png      # Matrices de confusión de los tres clasificadores
└── figura3_caso_mal_clasificado.png    # Análisis visual de imagen de prueba con fallo
```

---

## 🖥️ Funcionalidades de la Interfaz Gráfica (GUI)

La aplicación de escritorio implementada en `gui.py` (Tkinter / TTK) proporciona un entorno interactivo de experimentación visual:

- **Visualizador de Muestras 28x28:** Renderizado en alta resolución ($280 \times 280$ píxeles, escala $\times 10$) de los dígitos del conjunto de prueba.
- **Navegación de Muestras:** Controles de desplazamiento secuencial (*Anterior / Siguiente*), selección aleatoria (*🎲 Aleatoria*) y salto directo a casos problemáticos.
- **Inspección de Casos Mal Clasificados (⚠️):** Localización y filtrado automático de muestras donde los modelos emiten predicciones erróneas o divergentes.
- **Lienzo de Dibujo a Mano Alzada:** Permite al usuario dibujar dígitos con el ratón directamente sobre el canvas negro con interpolación bilinear básica para evaluar la robustez de los modelos ante trazos fuera de distribución (*Out-of-Distribution*).
- **Inferencia Multi-Modelo Simultánea:** Clasificación en paralelo con Árbol de Decisión, Random Forest y k-NN, contrastando etiquetas predichas y estados de acierto/error frente al *Ground Truth*.
- **Tabla Oficial de Rendimiento Embebida:** Visualización de métricas consolidadas (Exactitud, Error, Brecha de generalización y Tiempo de entrenamiento) en una tabla estilizada en tiempo real.

---

## 📊 Matriz Comparativa de Métricas de Rendimiento

### Resultados Oficiales del Benchmark

Evaluación efectuada sobre $10,000$ muestras de entrenamiento y $2,000$ muestras de prueba ($10$ clases balanceadas, $200$ instancias por dígito en prueba):

| Modelo | Hiperparámetros Clave | Exactitud Ent. (%) | Error Ent. (%) | Exactitud Prueba (%) | Error Prueba (%) | Brecha Gen. (p.p.) | Tiempo Ajuste (s) | Tiempo Infer. Test (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Árbol de Decisión** | $\text{max\_depth}=15$ | $99.02\%$ | $0.98\%$ | **$81.80\%$** | $18.20\%$ | $+17.22$ | $2.57\text{ s}$ | $< 0.01\text{ s}$ |
| **Random Forest** | $B=100, \text{depth}=15$ | $99.93\%$ | $0.07\%$ | **$94.80\%$** | $5.20\%$ | $+5.13$ | $2.37\text{ s}$ | $0.06\text{ s}$ |
| **k-NN** | $k=5, \text{métrica } L_2$ | $96.51\%$ | $3.49\%$ | **$94.55\%$** | $5.45\%$ | **$+1.96$** | **$0.01\text{ s}$** | $3.87\text{ s}$ |

> [!NOTE]
> La **Brecha de Generalización** se define como $\Delta_{\text{gen}} = \text{Exactitud}_{\text{ent}} - \text{Exactitud}_{\text{pru}}$. Una brecha elevada denota memorización de patrones particulares en desmedro de la capacidad de generalización.

---

### Análisis de Confusiones Críticas

A partir de las matrices de confusión generadas sobre las $2,000$ muestras de prueba:

| Modelo | Confusión Fuera de Diagonal Más Frecuente | Casos Erróneos | Causa Morfológica / Geométrica |
| :--- | :---: | :---: | :--- |
| **Árbol de Decisión** | Dígito **$3$** clasificado como **$5$** | $14$ casos | Ruptura o solapamiento del arco central; los cortes ortogonales individuales no capturan la continuidad de trazos curvilíneos continuos. |
| **Random Forest** | Dígito **$7$** clasificado como **$9$** | $7$ casos | Trazo superior horizontal con curva cerrada imperfecta en el bucle superior del $9$, provocando ambigüedad entre ramas. |
| **k-NN ($k=5$)** | Dígito **$2$** clasificado como **$7$** | $9$ casos | La base horizontal del $2$ escrita con trazo tenue o corto genera distancias $L_2$ menores respecto a centroides de trazo oblicuo del $7$. |

---

### Compromiso Sesgo-Varianza y Costo Computacional

1. **Sobreajuste Severo en Árboles Individuales:**
   El árbol de decisión individual sufre una marcada penalización de varianza ($\Delta_{\text{gen}} = 17.22\text{ p.p.}$). Al limitar la profundidad a $15$, el árbol genera hiperplanos ortogonales rígidos que aíslan ruido muestral.
2. **Eficacia del Ensamble Random Forest:**
   Random Forest logra el mejor rendimiento global ($94.80\%$ exactitud en prueba) reduciendo la brecha a $5.13\text{ p.p.}$. La inyección de aleatoriedad en el muestreo de datos y en la selección de los $28$ píxeles candidatas por nodo atenúa notablemente la varianza.
3. **Estructura Computacional de k-NN (*Lazy vs. Eager*):**
   k-NN presenta la menor brecha de generalización ($1.96\text{ p.p.}$), pero invierte la carga computacional: su entrenamiento toma apenas $0.01\text{ s}$ (almacenamiento en memoria), pero su inferencia sobre $2,000$ vectores requiere calcular $2,000 \times 10,000 = 20,000,000$ distancias euclídeas de dimensión $784$, tardando casi $4\text{ s}$.

---

## 🚀 Instalación y Guía de Uso

### Requisitos Previos

- Python 3.10 o superior (compatible con Python 3.11, 3.12, 3.13 y 3.14).
- Administrador de paquetes `pip`.

### 1. Clonar el Repositorio

```bash
git clone https://github.com/T1Angel4220/DF-RT-KNN.git
cd DF-RT-KNN
```

### 2. Crear y Activar un Entorno Virtual

En Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

En Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalar Dependencias

```bash
pip install numpy pandas scikit-learn matplotlib
```

### 4. Ejecución del Benchmark Completo (CLI)

Ejecuta el pipeline completo de carga, entrenamiento, evaluación formal y exportación de figuras gráficas:

```bash
python clasificacion_mnist.py
```

*Las figuras `figura1_muestras_mnist.png`, `figura2_matrices_confusion.png` y `figura3_caso_mal_clasificado.png` se guardarán automáticamente en la raíz del proyecto.*

### 5. Ejecución de la Interfaz Gráfica (GUI)

Inicia el visualizador de escritorio interactivo para clasificar muestras y probar trazos a mano alzada:

```bash
python gui.py
```

### 6. Exploración en Jupyter Notebook

Para revisar el paso a paso metodológico con celdas ejecutables:

```bash
pip install jupyter
jupyter notebook Taller_MNIST_Arbol_RF_kNN.ipynb
```

---

## 📄 Licencia

Este proyecto se distribuye bajo los términos de la licencia **MIT**. Para más detalles, consulta el archivo [LICENSE](LICENSE).

```text
Copyright (c) 2024-2026 Angel Ayuquina
```
