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

## Bitácora – Implementación del Modelo de Fusión (Stacking) Multimodal
📌 Contexto
El objetivo de esta etapa fue integrar las predicciones de los modelos unimodales:
    - Experto de Audio (espectrogramas 2D + features 1D)

    - Experto de Texto (embeddings 2D + TF-IDF 1D)

para construir un modelo de fusión basado en stacking, similar al planteado en el artículo base, pero adaptado a la infraestructura del proyecto.

## 📂 1. Estructura de archivos utilizada

Las predicciones de los expertos fueron generadas previamente y almacenadas en:
```bash
/Reports/audio_expert/predicciones_audio_TRAIN.csv
/Reports/audio_expert/predicciones_audio_VAL.csv
/Reports/audio_expert/predicciones_audio_TEST.csv

/Reports/text_expert/predicciones_text_TRAIN.csv
/Reports/text_expert/predicciones_text_VAL.csv
/Reports/text_expert/predicciones_text_TEST.csv
```
Cada archivo contiene:
    - spotify_id

    - Probabilidades por cuadrante:

    - audio → prob_audio_Q1 … prob_audio_Q4

    - texto → prob_text_Q1 … prob_text_Q4

    - true_label (0=Q1, 1=Q2, 2=Q3, 3=Q4)

## 🔧 2. Función load_and_merge() — Construcción del dataset para stacking

Se implementó la función load_and_merge(split, paths) que:

1. Carga las predicciones de audio y texto para el split (TRAIN, VAL, TEST).

2. Realiza un merge por spotify_id.

3. Detecta automáticamente la columna correcta del label
(true_label, true_label_audio o true_label_text).

4. Construye el vector de features del meta-modelo concatenando:

```css
[prob_audio_Q1..Q4] + [prob_text_Q1..Q4]
```
→ 8 features por muestra

5. Devuelve:

    - X: matriz de features

    - y: etiquetas verdaderas

    - ids: lista de spotify_id

Esto garantiza un dataset limpio, consistente y sin fuga de información.

## 🧩 3. Definición del modelo de fusión – FusionNet

Se entrenó un modelo sencillo pero eficiente para stacking:

```Python
FusionNet(in_features=8, num_classes=4)
```

Arquitectura:
- Linear(8 → 32)
- ReLU
- Dropout(0.3)
- Linear(32 → 4)
- Softmax implícito a través del criterio CrossEntropyLoss.

Este clasificador aprende relaciones entre:

- Probabilidades acústicas

- Probabilidades semánticas

- Interacciones entre cuadrantes

## 🚀 4. Entrenamiento del meta-modelo

El entrenamiento se realizó sobre X_train, y_train, validando en X_val, y_val.

Detalles:

- Optimizador: Adam

- Learning rate: 1e-3

- Early stopping basado en pérdida de validación

- Entrenamiento rápido (dimensión baja: 8 features)

Una vez completado, se evaluó sobre X_test, y_test.

## 📊 5. Resultados del modelo de fusión

Los valores exactos obtenidos:
```makefile
Accuracy: 75.13%
Macro-F1: 0.7127
Weighted-F1: 0.7475
```
F1 por clase:

Clase	F1-score
Q1 – Happy	0.7279
Q2 – Angry	0.7103
Q3 – Sad	0.8506
Q4 – Relaxed	0.5620
Observaciones relevantes:

- La fusión supera el rendimiento de los modelos unimodales (audio y texto).

- Especialmente mejora el rendimiento en:

    - Q2 (Angry)

    - Q3 (Sad)

    - Q4 (Relaxed), la clase más difícil por desbalance.

- Mantiene un buen desempeño en Q1.

- La meta-representación es estable y captura información complementaria entre modales.

## 🖼 6. Generación y guardado de resultados

Se guardaron los outputs del modelo de fusión en:
```bash
/Reports/fusion_model/
    classification_report.txt
    confusion_matrix.csv
    confusion_matrix.png
```

Incluye:

- Reporte de clasificación completo

- Matriz de confusión tabular

- Heatmap visual en PNG

- Esto asegura reproducibilidad y trazabilidad de experimentos.

## ⭐ 7. Conclusión general del stacking

El stacking permitió:

- Combinar fortalezas de ambos expertos.

- Reducir errores sistemáticos de cada modal.

- Mejorar el desempeño general del sistema.

- Obtener resultados más estables y alineados con el comportamiento esperado en MER multimodal.

La implementación actual usa stacking simple con splits fijos (train/val/test).
En caso de requerir estricta reproducibilidad del artículo, puede extenderse a un esquema 5-fold cross-validated stacking.