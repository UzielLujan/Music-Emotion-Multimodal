# 📔 Bitácora de Proyecto: Procesamiento Multimodal y Arquitectura

**Fecha:** 8 de Diciembre, 2024
**Fase:** 2 (Procesamiento de Texto y Construcción de Dataset)
**Autores:** Equipo de Integración (Uziel & Brenda)

---

## Objetivo General
Transformar las letras de canciones crudas (`aligned_metadata.csv`) y los audios procesados en un **Dataset Maestro Multimodal** listo para entrenar una Red Neuronal Híbrida (Audio + Texto).

---

## 1. Limpieza y Normalización de Texto
**Script:** `src/ProcessData/text/cleaning.py`

Se detectó una contaminación significativa de caracteres no latinos (Kanji, Hangul, Cirílico, Emojis) en el dataset original.

### Estrategia Implementada: "Lista Blanca Estricta & Opción Nuclear"
1.  **Auditoría:** Se crearon scripts para identificar filas con caracteres asiáticos y rusos.
2.  **Limpieza Profunda:**
    * **Normalización Unicode:** Se fuerza la conversión a ASCII (`.encode('ascii', 'ignore')`). Esto elimina *físicamente* cualquier caracter chino, coreano o ruso.
    * **Regex Estricto:** `[^a-zA-Z0-9\s.,!?'\"\n-]`. Solo sobreviven letras inglesas, números y puntuación básica.
3.  **Salida Dual:** Se generan dos versiones del texto en `lyrics_cleaned.parquet`:
    * `clean_lyrics_tfidf`: Texto plano, minúsculas, sin stopwords (para conteo de palabras).
    * `clean_lyrics_bert`: Texto con estructura, mayúsculas y puntuación (para contexto semántico).

---

## 2. Extracción de Features de Texto (Feature Engineering)
**Orquestador:** `src/ProcessData/run_text_pipeline.py`

### A. Rama Tabular (1D)
* **Script:** `src/ProcessData/text/features_1d.py`
* **Técnica:** TF-IDF (Term Frequency - Inverse Document Frequency).
* **Configuración:** Vocabulario de 2,000 palabras más frecuentes.
* **Salida:** `data/processed/features_1d/features_text_1d.parquet`

### B. Rama Profunda (2D)
* **Script:** `src/ProcessData/text/embeddings.py`
* **Técnica:** Embeddings contextuales usando **DistilBERT**.
* **Configuración:** `max_length=256`.
* **Salida:** Un tensor gigante unificado para eficiencia.
    * `data/processed/features_2d/embeddings/embeddings_2d.npy` (Matriz NumPy).
    * `data/processed/features_2d/embeddings/embeddings_ids.npy` (IDs correspondientes).

---

## 3. Ensamblaje del Dataset Maestro
**Script:** `src/ProcessData/create_master_dataset.py`

Este script es el corazón de la preparación de datos. Su función es unificar fuentes heterogéneas (CSV, Parquet, NPY, Imágenes) en un índice maestro confiable.

Funcionalidades Clave
1. Unificación de Metadata: Fusiona aligned_metadata.csv con las features tabulares de audio (features_audio_1d.csv) y texto (features_text_1d.parquet).

2. Validación de Integridad: Verifica físicamente que existan los archivos .npy (espectrogramas) para cada canción. Si falta el archivo, la canción se descarta.

3. Corrección de Embeddings (Critical Fix):

    - Problema: Los embeddings de texto (embeddings_2d.npy) son una matriz gigante sin etiquetas.

    - Solución: Se carga un archivo auxiliar embeddings_ids.npy que contiene el orden exacto de los IDs. Se crea un mapa {spotify_id: indice_fila} para que el DataLoader sepa qué fila de la matriz leer para cada canción.

4. Split Estratificado: Divide los datos en Train (70%), Val (15%) y Test (15%) manteniendo la proporción de clases (label_quadrant).

Salida: data/processed/master_dataset.csv
---

##  4. Infraestructura de Carga (PyTorch)
**Script:** `src/Loaders/dataset.py`

Se creó la clase `MultimodalDataset` que maneja la complejidad de leer 4 modalidades simultáneamente.

### Características Clave:
* **Carga Híbrida:**
    * *Audio 2D:* Lee archivos individuales `.npy` (espectrogramas) desde disco.
    * *Texto 2D:* Usa **Memory Mapping** (`mmap_mode='r'`) para leer la matriz gigante de Embeddings sin saturar la RAM.
* **Auto-adaptable:** Detecta automáticamente las columnas de MFCCs y TF-IDF.
* **Filtrado Dinámico (Actualización)**: Ahora acepta una lista specific_text_cols. Si se proporciona, el dataset ignora el resto de columnas TF-IDF, asegurando consistencia entre Train/Val/Test tras la selección de características.
* **Corrección de Dimensiones**: Se ajustaron los tensores en '__getitem__:
    - Audio: Se agrega dimensión de canal (unsqueeze(0)) -> [1, 128, 128].
    - Texto: Se elimina dimensión extra y se transpone para Conv1d -> [768, 256].
* **Corrección de Rutas:** Soluciona automáticamente las diferencias entre rutas relativas del CSV y la ubicación real en `processed/`.

## 5. Pipeline de Carga (Data Loaders)
**Script:**  src/Models/utils.py
Define cómo el sistema alimenta a las redes neuronales durante el entrenamiento e implementa la selección de características estadística.
**Optimización con Chi-Cuadrado ($\chi^2$)**
Se implementó SelectKBest dentro de get_dataloaders para reducir el ruido del TF-IDF.
    - Prevención de Data Leakage: El ajuste (.fit) se realiza exclusivamente sobre el conjunto de TRAIN.
    - Propagación: Las columnas seleccionadas (Top-K features) se pasan como argumento a los datasets de Validación y Test.

📦 Clase MultimodalDatasetEsta clase hereda de torch.utils.data.Dataset y maneja la complejidad de cargar 4 tipos de datos simultáneamente para una sola canción:

| Tipo de Dato | Fuente | Procesamiento en `__getitem__` | Tensor Shape Resultante |
|--------------|--------|----------------------------------|--------------------------|
| Audio 2D | Archivo `.npy` individual | Carga espectrograma → `unsqueeze(0)` | `(1, 128, 128)` |
| Audio 1D | Columna CSV | Normalización (si aplica) | `(34,)` |
| Texto 2D | Matriz en memoria | Busca índice → Transpone de `(Seq, 768)` a `(768, Seq)` | `(768, 256)` |
| Texto 1D | Columna CSV | Carga vector TF-IDF / Metadatos | `(k_features,)` |

## 6. Arquitectura de los Modelos (Deep Learning)
**Script:** src/Models/definitions.py

Se implementó una Arquitectura Híbrida (CRNN + Attention) para ambos expertos. La filosofía es capturar patrones espaciales (CNN), temporales (RNN) y características globales (Dense).

**Modelo de Audio (AudioNetwork)**
1. Rama Espectral (CNN 2D):

    - 4 Bloques de Convolución + Batch Norm + MaxPool.

    - Extrae texturas visuales del espectrograma (ritmo, energía).

2. Rama Temporal (Bi-GRU + Atención):

    - Aplana la salida de la CNN para tratarla como secuencia temporal.

    - Bi-Directional GRU para contexto pasado/futuro.

    - Attention Layer: Decide qué partes de la canción son más relevantes para la emoción.

3. Rama 1D (Dense):

    - Procesa las 34 features de ingeniería (MFCCs, Chroma, ZCR).

4. Fusión: Concatena (Vector Contexto + Vector Tabular) → Clasificador Final.

**Modelo de Texto (TextNetwork) Rama Semántica (CNN 1D):**

1. Entrada: Embeddings (dim 768).

    -  4 Bloques de Convolución 1D. .
    - Reduce la dimensionalidad secuencial progresivamente.

2.  Rama Secuencial (Bi-GRU + Atención):

    -  Similar al audio, analiza la narrativa de la letra a lo largo del tiempo.

4. Rama 1D (Dense):

    - Procesa metadatos y vectores TF-IDF.
    - Cambio: La capa de entrada ya no es fija (26), sino parametrizable (text_1d_dim) para adaptarse al número de features seleccionadas por Chi-cuadrado.

5. Fusión: Concatena las ramas → Clasificador Final.

## 7. Entrenamiento y Exportación (Stacking)
**Scripts:** main_audio.py y main_text.py

Estos scripts orquestan el aprendizaje de los expertos individuales.

Flujo de Trabajo
1. Training Loop:

    - Usa CrossEntropyLoss y optimizador Adam.

    - Barra de progreso con tqdm.

2. Validation Loop:

    - Calcula Accuracy y Loss al final de cada época.

    - Checkpointing: Guarda el modelo (.pth) solo si mejora el Validation Accuracy.

3. Generación de Predicciones (Clave para Stacking):

    - Una vez entrenado, el modelo vuelve a cargar los mejores pesos.

    - Pasa por TODO el dataset (Train, Val, Test).

    - Genera archivos CSV (predicciones_audio_TRAIN.csv, etc.) que contienen:

        - spotify_id

        -  Probabilidades Softmax (prob_Q1, prob_Q2...)

        - Etiqueta Real

**Diagrama de Flujo de Datos**
Fragmento de código

```bash
graph TD
    A[Raw Data] --> B(create_master_dataset.py)
    B --> C{master_dataset.csv}
    C --> D[Utils: MultimodalDataset]
    
    subgraph "Experto Audio"
    D --> E[AudioNetwork]
    E --> F(train_audio.py)
    F --> G[CSVs Probabilidades Audio]
    end
    
    subgraph "Experto Texto"
    D --> H[TextNetwork]
    H --> I(train_text.py)
    I --> J[CSVs Probabilidades Texto]
    end
    
    G --> K[Fusión para Stacking]
    J --> K
```

✅ Estado Actual
[x] Ingesta de datos robusta con corrección de IDs.

[x] DataLoaders configurados para manejo multimodal.

[x] Arquitecturas definidas (CRNN + Attention).

[x] Scripts de entrenamiento listos para generar insumos del Stacking.

Siguiente Paso: Ejecutar los scripts de entrenamiento y proceder a crear el Meta-Learner (el modelo que unirá los CSVs de audio y texto).

###