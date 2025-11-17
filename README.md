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
├── data/ # Conjuntos de datos (crudos / procesados)
│ ├── raw/
│ └── processed/
│
├── src/ # Código fuente del modelo
│ ├── audio/
│ ├── text/
│ ├── fusion/
│ └── utils/
│
├── notebooks/ # Experimentos exploratorios y prototipos
│
├── results/ # Resultados, métricas, gráficas
│
├── reports/ # Reportes en LaTeX
│
├── docs/ # Documentación técnica y decisiones de diseño
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

- **Python 3.10+**
- **PyTorch / TensorFlow** (por definir)
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


# Procesamiento del Conjunto de Datos FMA (Free Music Archive)
Proyecto MIR — Música, Emociones y Representaciones

Este documento describe de forma detallada cada uno de los pasos realizados para construir el pipeline de datos basado en el conjunto de datos **FMA (Free Music Archive)**, desde la descarga de metadatos y audio hasta la extracción de características y la preparación del conjunto final para el modelado emocional.

---

# 1. Descarga y estructura del conjunto de datos

## 1.1. Descarga de FMA Small y metadatos

Se descargaron dos componentes fundamentales:

- `fma_small.zip` — contiene 8,000 pistas de audio en formato `.mp3`.
- `fma_metadata.zip` — contiene los archivos:
  - `tracks.csv`
  - `genres.csv`
  - `features.csv`
  - `echonest.csv`

# Estructura de Carpetas del Proyecto MIR (FMA)

Este documento describe la estructura completa de directorios utilizada en el proyecto MIR basado en el dataset FMA (Free Music Archive). Incluye archivos de audio, metadatos, scripts, datos procesados y resultados.

# Estructura Actual del Proyecto MIR (FMA)
Esta es la estructura de carpetas generada hasta el punto alcanzado en el procesamiento:
- Descarga de FMA small
- Descarga de metadatos
- Filtrado por idioma
- Extracción de LLDs
- Generación de espectrogramas
- Unión de valence y arousal

---

Proyecto_MIR/
│
├── data/
│   ├── raw/
│   │   ├── fma_small/                 # Audio original (.mp3)
│   │   │   ├── 000/
│   │   │   ├── 001/
│   │   │   ├── ...
│   │   │   └── 007/
│   │   │
│   │   └── fma_metadata/              # Metadatos descargados
│   │       ├── tracks.csv
│   │       ├── features.csv
│   │       ├── genres.csv
│   │       └── echonest.csv
│   │
│   ├── intermediate/
│   │   └── feats_raw/                 # LLDs por track (temporales)
│   │
│   └── processed/
│       ├── features/                  # DataFrame final con características
│       │   └── df_feats.csv
│       │
│       └── spectrograms/              # Espectrogramas log-Mel generados
│           ├── 000001.png
│           ├── 000002.png
│           └── ...
│
├── scripts/
│   ├── scripts01_download_fma.py        # Descarga fma_small + metadata
│   ├── scripts02_filter_english.py      # Filtro por idioma (language_code = 'en')
│   ├── scripts03_extract_audio_features.py   # Extracción LLD + MFCC
│   ├── scripts04_generate_spectrograms.py    # Generación de espectrogramas
│   └── scripts05_merge_valence.py        # Agrega valence + arousal (energy)
└──


Se verificó que `tracks.csv` tiene **dos niveles de encabezado (MultiIndex)**, mientras que `echonest.csv` tiene **tres niveles**.

---

# 2. Identificación del idioma de cada canción

El archivo `tracks.csv` contiene una columna clave: ("track", "language_code") 
que sigue códigos ISO del idioma del artista (en, es, fr, de…).

Se desarrolló el script: scripts02_filter_english.py

Este script:

1. Detecta automáticamente los niveles del encabezado del CSV.
2. Carga el MultiIndex correctamente.
3. Verifica que la columna `language_code` existe.
4. Filtra únicamente los tracks cuyo idioma = `"en"`.

El resultado es una lista: english_ids = [track_id1, track_id2, ...] 
con todas las pistas en inglés.

---

# 3. Extracción de características de audio (LLDs y MFCC)

El script: scripts03_extract_audio_features.py

procesa únicamente los `track_id` en inglés.

Características extraídas:

- RMS
- Zero Crossing Rate
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Chroma STFT
- MFCC (13 coeficientes)
- Delta MFCC
- Delta-Delta MFCC

Cada archivo `.mp3` se cargó usando **librosa**, y se generó un DataFrame `df_feats` donde:

- filas = track_id
- columnas = características de audio

Ejemplo: df_feats.loc[1234, ["mfcc_1", "mfcc_2", "spectral_centroid", ...]]


---

# 4. Generación de espectrogramas (para modelos CNN)

El script:  genera y guarda espectrogramas log-mel para cada pista en inglés.
Los espectrogramas se exportan como imágenes `.png` o matrices `.npy` en: data/processed/spectrograms/{track_id}.png


Parámetros:

- 128 bandas Mel
- Ventana: 2048
- Hop length: 512

Estos espectrogramas se utilizarán posteriormente para modelos de visión profunda.

---

# 5. Obtención de características de valence y arousal desde echonest.csv

El archivo `echonest.csv` incluye múltiples grupos de características, entre ellas:

- `("echonest", "audio_features", "valence")`
- `("echonest", "audio_features", "energy")` (utilizada como proxy de arousal)

Se creó el script: scripts05_merge_valence.py

Este:

1. Carga `echonest.csv` con su MultiIndex de 3 niveles.
2. Extrae únicamente las columnas relevantes:
   - `valence`
   - `energy` → renombrada a `arousal`
3. Une estas columnas al DataFrame `df_feats`.

Resultado final parcial:


Ahora cada canción tiene:

- características de audio (LLDs, MFCC)
- valence
- arousal

---

# 6. Construcción del plano emocional (4 cuadrantes)

Como la literatura del artículo base usa un modelo de cuatro emociones, el proyecto utiliza:

| Cuadrante | Valence | Arousal | Emoción |
|----------|---------|---------|---------|
| Q1       | +       | +       | Happy / Energetic |
| Q2       | -       | +       | Angry / Tense |
| Q3       | -       | -       | Sad |
| Q4       | +       | -       | Calm / Relaxed |

Se implementará en: scripts06_map_emotions.py


Función:

- Normaliza valence/arousal
- Determina el cuadrante
- Asigna una etiqueta emocional

---

# 7. Resultado final

El dataset final contiene, por cada canción:

- `track_id`
- características acústicas (LLDs, MFCCs)
- espectrograma
- `valence`
- `arousal`
- `emotion_4Q` (categoría final)

Este DataFrame está listo para:

- entrenamiento de modelos supervisados
- clasificación emocional
- análisis comparativo de enfoques de audio, espectrograma y emoción

---




















