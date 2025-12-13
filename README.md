# Análisis de emoción y sentimiento en música desde una perspectiva multimodal

Sistema de reconocimiento de emociones musicales utilizando características acústicas y textuales, con modelos de aprendizaje profundo y fusión basada en **stacking**. 

El proyecto incluye pipelines modulares para preprocesamiento, modelado unimodal y fusión multimodal, enfocado en la reproducibilidad y la extensibilidad dentro de la investigación en MIR (Music Information Retrieval).


## Descripción general

Este repositorio forma parte del proyecto final del curso  
**Temas Selectos de Ciencia de Datos: Recuperación de Información Musical (MIR) — 2025. Centro de Investigación en Matemáticas (CIMAT) Unidad Monterrey.**

El objetivo principal es analizar y clasificar **emociones en música** desde una **perspectiva multimodal**, integrando datos que provienen tanto de **audio** como de **letras**.  
El diseño se basa en enfoques recientes de MIR que utilizan arquitecturas de redes profundas y estrategias de fusión por aprendizaje supervisado.



## Objetivos

- Construir un pipeline reproducible para clasificación emocional multimodal.
- Combinar representaciones acústicas y textuales para enriquecer el análisis emocional.
- Implementar y comparar modelos unimodales (audio / letras).
- Diseñar un modelo de fusión por *stacking* para integrar decisiones multimodales.
- Documentar el proceso en formato técnico y académico.

---

## Estructura del proyecto

```bash
music-emotion-multimodal/
│
├── data/ # Conjuntos de datos (crudos / procesados / intermedios)
│ ├── raw/                # Conjunto de datos original (FMA small + metadatos)
│ ├── rawv2/                 # Conjunto de datos versión 2 (YouTube + Spotify(Kaggle) + Genius)
│ └── processed/
│ └── interim/
│
├── src/ # Código fuente del modelo
│ ├── audio/
│ ├── text/
│ ├── fusion/
│ ├── ExtractDataV2/
│ ├── Models/
│ ├── Loaders/
│ ├── ProcessData/
│
├── notebooks/ # Experimentos exploratorios y prototipos
│
├── results/ # Resultados, métricas, gráficas
│
├── reports/ # Recursos para los reportes en LaTeX
│
├── docs/ # Documentación técnica y decisiones de diseño (archivos tipo .md)
│
├── requirements.txt
├── README.md
└── .gitignore
```


---

## Equipo

| Nombre | Responsabilidades adoptadas en el proyecto |
|--------|------------------|
| **Brenda Transito** | - Curación y construcción del conjunto de datos multimodal. |
| **Uziel Luján** | - Estructura del repositorio, diseño del pipeline, documentación técnica. |

---

## Stack tecnológico propuesto

- **Python 3.11+**
- **PyTorch** – modelos de deep learning
- **Librosa** – extracción de características acústicas  
- **Transformers (Hugging Face)** – embeddings textuales  
- **Scikit-learn** – modelos dde machine learning tradicionales y evaluación  
- **Pandas / NumPy** – manipulación y análisis de datos 
- **Matplotlib / Seaborn / Plotly** – visualizaciones

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/UzielLujan/Music-Emotion-Multimodal.git
cd Music-Emotion-Multimodal

# Crear entorno Conda desde el archivo environment.yaml
conda env create -f environment.yaml

# Activar entorno
conda activate mem-env
```

## Descarga y construcción del dataset multimodal

Motivación del rediseño del pipeline de descarga
El pipeline original basado en FMA + Last.fm presentó una limitación crítica: la intersección entre audio disponible, letras recuperables e idioma inglés resultó insuficiente para un enfoque multimodal escalable.
Por esta razón, se diseñó un Pipeline V2, cuyo objetivo fue construir un dataset desde cero, controlando explícitamente cada modalidad.

**Fuentes de datos utilizadas**

El dataset multimodal se construyó integrando tres fuentes principales:

- **Spotify (Kaggle)**
  Proporciona metadatos musicales y etiquetas emocionales continuas:
  - valence
  - arousal
  - label_quadrant (Happy, Angry, Sad, Relaxed)

- **Genius API**
Utilizada para la recuperación automática de letras en inglés, filtrando:
  - Canciones instrumentales
  - Idiomas no deseados
  - Letras incompletas

- **YouTube**  Fuente de los archivos de audio (.mp3), descargados directamente a partir del identificador de cada canción.

```text
Flujo general de descarga (Pipeline V2)
Spotify (Kaggle)
      ↓
metadata_step2_lyrics_clean.csv
      ↓
Genius API (letras)
      ↓
YouTube (audio)
      ↓
Dataset multimodal alineado por spotify_id
```

** Scripts de descarga y su función**

1. Preparación y curación inicial de metadatos
Archivo generado

```bash
data/raw_v2/metadata_step2_lyrics_clean.csv
```
Contiene:
  - spotify_id
  - artista
  - título
  - valence / arousal
  - cuadrante emocional
  - letra cruda (Genius)

Este archivo es el punto de entrada único para todo el pipeline de descarga.

2. Descarga distribuida de audio desde YouTube

Para evitar bloqueos por límite de peticiones, la descarga de audio se realizó de forma distribuida entre los integrantes del equipo.

Script principal

```bash
src/ExtractDataV2/main.py
```

Lógica implementada

- Lee un CSV con metadatos (spotify_id, artista, título).

- Genera automáticamente la consulta de búsqueda en YouTube.

- Descarga el audio en formato .mp3.

- Guarda el archivo usando spotify_id como nombre.

- El proceso es incremental: si un archivo ya existe, se omite.

Ejecución típica

```bash
python src/ExtractDataV2/main.py
```

3. Estrategia de descarga paralela

El archivo maestro se dividió en dos partes:

```bash
data/raw_v2/metadata_part_uzi.csv
data/raw_v2/metadata_part_brenda.csv
```

Cada integrante ejecutó el mismo script (main.py), modificando únicamente la variable de entrada:

```python
CSV_STEP2 = RAW_V2_DIR / "metadata_part_brenda.csv"
```

Esto permitió:

- Reducir tiempo total de descarga

- Evitar bloqueos de IP por YouTube

- Mantener trazabilidad del progreso

4. Organización de los audios descargados

Los audios se almacenan en una estructura plana:

```bash
data/raw_v2/audio/
├── 0AcJ0eX.mp3
├── 3FZxA91.mp3
├── ...
```

Convención:

- Nombre del archivo = spotify_id

- Permite alineación directa con letras y features posteriores.

**Alineación final de modalidades**

Una vez completada la descarga, se ejecuta el script de alineación:

```bash
src/ProcessData/utils/alignment.py
```

Este script:

- Escanea la carpeta de audios descargados.

- Cruza contra metadata_step2_lyrics_clean.csv.

- Conserva únicamente canciones con:

  - Audio disponible

  - Letra válida

- Genera el archivo intermedio:

```bash
data/interim/aligned_metadata.csv
```

Este archivo define el subconjunto final de canciones que entran a la Fase 2 de procesamiento.


## Procesamiento multimodal y construcción del Dataset Maestro

Esta fase corresponde al núcleo de ingeniería de características del proyecto. El objetivo es transformar los datos crudos (audio y letras) en representaciones numéricas organizadas, listas para ser consumidas por modelos de aprendizaje profundo unimodales y posteriormente integradas mediante stacking.

### 2.1 Alineación y validación de datos

- Se ejecuta un proceso de alineación por spotify_id entre:

  - Audios descargados (YouTube).

  - Letras limpias (Genius API).

- Se valida la existencia física de cada archivo requerido.

- El resultado es un archivo intermedio:

  ```bash
   data/interim/aligned_metadata.csv
   ```

Este archivo actúa como lista de control para todas las etapas posteriores, garantizando que solo se procesen canciones con información multimodal completa.

### 2.2 Procesamiento de texto (Rama Textual)

Responsable: Brenda Tránsito

- Limpieza y normalización

- Eliminación de etiquetas estructurales ([Chorus], [Verse], etc.).

- Normalización Unicode y filtrado estricto de caracteres no latinos.

- Generación de dos vistas del texto:

  - clean_lyrics_tfidf: texto plano para métodos estadísticos.

  - clean_lyrics_bert: texto estructurado para modelos Transformer.

Salida intermedia:
```bash
data/interim/lyrics_cleaned.parquet
```

**Representaciones textuales**

**Texto 1D (ligero)**
- Técnica: TF-IDF + selección de características (Chi-cuadrado, solo para el conjunto de entrenamiento).
- Dimensionalidad controlada para evitar sparsity excesiva.

Salida:
```bash
data/processed/features_1d/features_text_1d.parquet
```
**Texto 2D (profundo)**
- Técnica: Embeddings contextuales con DistilBERT.
- Longitud de secuencia fija (max_length=256).
- Almacenamiento eficiente mediante:

  - embeddings_2d.npy (tensor global)

  - embeddings_ids.npy (mapeo por spotify_id)

Ubicación:
```bash
data/processed/features_2d/embeddings/
```

### 2.3 Procesamiento de audio (Rama Acústica)

Responsable: Uziel Luján

- Recorte inteligente: selección automática de los 15 segundos más energéticos.

- Audio 1D:

  - HSFs derivados de MFCCs, Chroma y ZCR.

  - Estadísticos: media y desviación estándar.

- Audio 2D:

  - Mel-Spectrogramas almacenados como matrices NumPy (.npy).

Salidas:
```bash
 data/processed/features_1d/features_audio_1d.csv
data/processed/features_2d/spectrograms/{spotify_id}.npy
```

### 2.4 Ensamblaje del Dataset Maestro

Script clave:

```bash
src/ProcessData/create_master_dataset.py
```

Funcionalidades:

- Unión de metadata, features 1D y rutas a features 2D.

- Corrección explícita del desalineamiento entre embeddings y IDs.

- Split estratificado:

  - Train (70%)

  - Validation (15%)

  - Test (15%)

Salida final:

```bash
data/processed/master_dataset.csv
```

Este archivo es el índice central del proyecto, utilizado por todos los DataLoaders.

#### Modelos y Arquitecturas Utilizadas

El sistema propuesto implementa un enfoque multimodal para el reconocimiento de emociones musicales, donde cada modalidad (audio y texto) es entrenada de forma independiente y posteriormente integrada mediante una estrategia de fusión por stacking.

Todo el flujo se encuentra desacoplado en tres fases: entrenamiento unimodal, generación de meta-features y entrenamiento del meta-learner.

### 1. Rama de Audio

La rama de audio modela información espectral, temporal y estadística de la señal musical. El objetivo es capturar patrones locales y globales relevantes para la percepción emocional.

Archivos principales utilizados:

- Entrenamiento del modelo de audio
  ```bash
  src/Models/main_audio.ipynb
  ```

- Definición de la arquitectura
  ```bash
  src/Models/definitions.py (clase AudioNetwork)
  ```

- Carga de datos multimodales
  ```bash
  src/Models/utils.py (clase MultimodalDataset, función get_dataloaders)
  ```

**Características del entrenamiento**
- Manejo explícito de desbalance de clases mediante class weights.

- Regularización mediante dropout y weight decay.

- Gradient clipping para estabilidad del entrenamiento.

- Early stopping basado en desempeño en validación.

- Exportación de:

  - probabilidades por clase,

  - etiquetas reales,

  - identificadores (spotify_id), para su uso posterior en la fusión.

Los resultados se almacenan en:
  ```bash
  reports/audio_expert/
  ```

### 2. Rama de Texto

La rama textual combina información semántica profunda con representaciones léxicas seleccionadas, permitiendo capturar tanto significado contextual como patrones discriminativos de vocabulario.

**Archivos principales utilizados**

- Entrenamiento del modelo de texto
  ```bash
  src/Models/main_text.ipynb
  ```
- Definición de la arquitectura
  ```bash
  src/Models/definitions.py (clase TextNetwork)
  ```
- Carga de datos y selección de características
  ```bash
  src/Models/utils.py
  ```
    - selección Chi-cuadrada (SelectKBest)

    - embeddings textuales 2D

    - TF-IDF reducido

**Características del entrenamiento**

- Ponderación de clases para tratar el desbalance.

- Regularización L2.

- Learning rate scheduling (ReduceLROnPlateau).

- Early stopping para evitar sobreajuste.

- Exportación de probabilidades por clase y etiquetas reales.

Los resultados se almacenan en:
```bash
reports/text_expert/
```

### 3. Generación de Meta-Features (Fase de Stacking)

Una vez entrenados ambos expertos unimodales, se congelan sus pesos y se utilizan únicamente como extractores de decisiones.

Cada modelo produce probabilidades por clase, las cuales se concatenan para formar un nuevo espacio de características de baja dimensión.

**Archivo utilizado** 

- Generación de meta-features
  ```bash
  generate_meta_features.py
  ```
Este script:

- carga los modelos entrenados,

- evalúa los conjuntos train, val y test,

- concatena las probabilidades de audio y texto,

### 4. Fusión Multimodal (Stacking)

La integración final se realiza mediante fusión tardía, donde el aprendizaje ocurre a nivel de decisión.

Se implementaron dos variantes de meta-learner, con fines comparativos.

**Opción A: Meta-learner neuronal (baseline)**

- Entrenamiento
  ```bash
  src/Models/stacking_fusion.ipynb
  ```
Este modelo sirve como línea base y permite comparar directamente el desempeño frente a métodos más expresivos.

**Opción B: Meta-learner basado en XGBoost**

Para capturar relaciones no lineales entre las predicciones unimodales, se entrenó un modelo de gradient boosting directamente sobre las probabilidades concatenadas.

**Notebook utilizado**

- Entrenamiento y evaluación del stacking
  ```bash
  src/Models/stacking_fusion.ipynb
  ```
En este notebook se realiza:

- carga de predicciones de audio y texto,

- entrenamiento del modelo XGBoost,

- evaluación en train / val / test,

- generación de:

  - classification report,

  - matriz de confusión,

  - análisis de importancia de características.

Los resultados finales se guardan en:

- reports/fusion_model/

5. Justificación del Diseño

Este diseño modular permite:

- evaluar de forma aislada cada modalidad,

- analizar el impacto real de la multimodalidad,

- comparar estrategias de fusión lineales y no lineales,

- y extender el sistema a nuevas modalidades sin rediseñar el pipeline completo.

El uso de stacking facilita una integración interpretable y flexible, alineada con prácticas actuales en Music Information Retrieval (MIR) y aprendizaje multimodal.