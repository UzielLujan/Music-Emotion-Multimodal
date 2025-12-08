# Guía de Transición: Fase 2 (Extracción Distribuida y Procesamiento)

**Fecha:** Noviembre 2025  
**Contexto:** Migración del Pipeline FMA (Legacy) al Pipeline V2 (Spotify/Genius/YouTube).

---

## 1. ¿Por qué cambiamos todo?

El dataset original (FMA) presentaba un problema crítico: la intersección entre "Audio Disponible" + "Idioma Inglés" + "Etiquetas Emocionales" era demasiado pequeña (< 2,000 muestras) y de hecho se volvió cero en cuanto se intentaron recuperar sus letras con Genius.

**La Nueva Estrategia (Pipeline V2):**
En lugar de filtrar lo que hay (buscando datos que cumplieran con todas las condiciones), hemos construido nuestro propio dataset desde cero:
1.  **Semillas:** 30,000+ canciones extraídas de Kaggle (Spotify Tracks), balanceadas por cuadrantes emocionales (Happy, Sad, Angry, Relaxed).
2.  **Letras:** Scrapeadas via Genius API (filtrando instrumentales e idioma no-inglés).
3.  **Audio:** Descarga directa desde YouTube.

**Estado Actual:**
Tenemos un archivo maestro limpio (`metadata_step2_lyrics_clean.csv`) con **~6,500 canciones** perfectamente etiquetadas y con letra. Ahora falta terminar de descargar los audios.

---

## 2. Configuración del Entorno (CRÍTICO ⚠️)

Para que el nuevo pipeline funcione, las dependencias han cambiado significativamente (Python 3.11, soporte nativo de tipos, librerías de audio modernas).

**Acción Requerida:**
Por favor, elimina tu entorno anterior y créalo de nuevo usando el archivo `environment.yaml` actualizado en el repositorio.

```bash
# 1. Desactivar y eliminar el viejo
conda deactivate
conda remove --name mem-env --all

# 2. Crear el nuevo (Asegura tener el environment.yaml actualizado)
conda env create -f environment.yaml

# 3. Activar
conda activate mem-env
```
## 3. Estrategia de descarga distribuida ("Divide y Vencerás")

YouTube bloquea temporalmente las IPs que descargan demasiado rápido. Para evitar esto y acelerar el proceso, dividiremos la carga de trabajo en dos mitades paralelas.

**Paso A: Generar los archivos divididos**

Uzi ejecutará el script de división que genera dos archivos en `data/raw_v2/`:

- `metadata_part_uzi.csv` (Primera mitad)

- `metadata_part_brenda.csv` (Segunda mitad)

**Paso B: Tu Misión de Descarga**

1. Abre el archivo src/ExtractDataV2/main.py.

2. Busca la línea donde se define `CSV_STEP2` dentro de `src/ExtractDataV2/main.py` y modifícala para apuntar a tu parte:

    ```bash
    # --- EN main.py ---
    # CSV_STEP2 = RAW_V2_DIR / "metadata_step2_lyrics_clean.csv"  <-- COMENTAR ESTA
    CSV_STEP2 = RAW_V2_DIR / "metadata_part_brenda.csv"           # <-- USAR ESTA
    ```
3. Ejecuta el pipeline de descarga:

    ```bash
    python src/ExtractDataV2/main.py
    ```
4. El script saltará automáticamente los Pasos 1 y 2, e iniciará la descarga de audios en el Paso 3.

**Nota**: Si YouTube te bloquea (Error "Sign in to confirm..."), detén el script, cambia tu IP (reinicia módem o usa datos) y vuelve a intentar. **El script es incremental, no perderás progreso**.

## 4. Arquitectura de la Fase 2: Procesamiento de los Datos Multimodales

Una vez tengamos descargados todos los audios, entraremos a la fase de **Ingeniería de Características**. Hemos diseñado una estructura modular para trabajar en paralelo sin conflictos.

Para que el modelo multimodal funcione correctamente, necesitamos transformar los datos crudos (MP3 y Texto Raw) en **tensores** y **vectores** organizados. No crearemos un unico archivo gigante que contenga todo, sino una estructura modular de archivos todos vinculados por el `spotify_id`.


El modelo consumirá 4 flujos de datos. Proponemos esta estructura de archivos finales en `data/processed/`:
 
1. `master_dataset.csv`: El cerebro. Contiene `spotify_id`, etiquetas ($Y$: valence, arousal, quadrant), metadatos (artista, título) y rutas a los archivos pesados.
2. `features_1d/` (Carpeta): Aquí vivirán los datos 1D ligeros en formato CSV.
   - `features_audio_1d.csv`: Tabla con los HSFs (media, varianza de MFCCs, etc.) para cada ID.
   - `features_text_1d.csv`: Tabla con vectores TF-IDF o estadísticos para cada ID.
4. `features_2d/` (Carpeta): Aquí vivirán los datos 2D pesados en formato .npy (NumPy binario).
   - `features_2d/spectrograms/{id}.npy`: Matriz del espectrograma (Audio 2D).
   - `features_2d/embeddings/{id}.npy`: Matriz de embeddings BERT/Word2Vec (Texto 2D).

### 4.1. El Roadmap Paso a Paso de la Fase de Procesamiento

El proceso se divide en 3 etapas lógicas para llegar a esta arquitectura:

---
**Etapa 1: La Gran Alineación (The Great Alignment)**
* **Script:** `src/ProcessData/utils/alignment.py` (Ya implementado).
* **Acción:** Escanea la carpeta física `audio/` y la cruza con el CSV limpio de letras `metadata_step2_lyrics_clean.csv`. Es decir, valida cuáles letras de las 6,500 tienen audio descargado.
* **Output Crítico:** `data/interim/aligned_metadata.csv`.

    > - **NOTA PARA BRENDA:** Este archivo es tu **"Lista de Tareas"**. Tus scripts deben leer este CSV para saber qué canciones procesar. No uses los archivos de `raw_v2`.
---
**Etapa 2: Extracción de Características (Paralelo)**
Aquí nos dividimos el trabajo. Ambos leemos la "Lista de Tareas" y generamos archivos en `data/processed/`.

* **Rama Audio (Uzi):** Genera HSFs (**1D**) y Espectrogramas (**2D**).

    > **Nota**: Uzi se acaba de dar cuenta que los HSFs son los que realmente importan para el modelo y no directamente los LLDs, por si acaso tambien compartias esa duda.
* **Rama Texto (Brenda):** Genera TF-IDF/Chi2 (**1D**) y Embeddings BERT 
(**2D**).

    > **Nota**: Puedes evaluar si usar TF-IDF o Chi2 para la representación 1D o incluso una representación más reciente pero ligera. Lo importante es que las representaciones sean generadas en un formato adecuado.
---
**Etapa 3: El Archivo Maestro**

Una vez que ambos hayamos generado nuestras características, se ejecutará un script final que cruce todos los outputs y genere el archivo maestro final. 
Este script final hace Merge de `aligned_metadata.csv` con `features_1d` y `features_2d` usando `spotify_id`.
* **Output:** El archivo generado estará ubicado en `data/processed/master_dataset.csv`.
* **Función:** Es el índice final que validará que, para cada fila (`spotify_id`), existan tanto los archivos de audio como los de texto y tengan correctamente asignados los vectores correspondientes asi como las etiquetas emocionales. Por lo tanto, el modelo leerá este archivo para conectarse a los datos cuando entrene.

### 4.2. Estructura de la fase 2

Para la creacion de los modulos del procesamiento se diseñó la siguiente **Estructura de Carpetas en ProcessData:**

```bash
src/
└── ProcessData/             <-- NUEVA CARPETA FASE 2
    ├── main_processing.py   <-- Orquestador
    │
    ├── utils/   # Funciones compartidas (Carga de archivos, Alineación, etc.)
    │   ├── alignment.py     <-- Paso 1 (Cruza CSV vs Carpeta Audio)
    │   └── io_utils.py      <-- Funciones para guardar/cargar .npy
    │
    ├── audio/      # (Responsable: Uzi) - Recorte, Espectrogramas, HSFs
    │   ├── trimming.py      <-- Lógica de corte (30s -> 15s por Energía)
    │   ├── features_1d.py   <-- Librosa -> HSFs
    │   └── spectrograms.py  <-- Librosa -> MelSpec -> .npy
    │
    └── text/  # (Responsable: Brenda) - Limpieza, TF-IDF, Embeddings
        ├── cleaning.py      <-- Regex y NLTK/Spacy
        ├── features_1d.py   <-- TF-IDF (Scikit-learn)
        └── embeddings.py    <-- Transformers/Gensim -> .npy
```

Una vez generados, los scripts deben depositar los resultados siguiendo esta estructura simétrica para que la integración sea automática y limpia:

```bash
data/processed/
│
├── features_1d/              # Tablas numéricas (CSV)
│   ├── features_audio_1d.csv    # (Uzi) Estadísticas de audio (HSFs)
│   └── features_text_1d.csv   # (Brenda) Vectores TF-IDF/Chi2
│
├── features_2d/              # Tensores pesados (NumPy Binary)
│   ├── spectrograms/         # (Uzi) Matrices .npy de Audio
│   │   ├── 0AcJ0e....npy
│   │   └── ...
│   └── embeddings/           # (Brenda) Matrices .npy de Texto
│       ├── 0AcJ0e....npy
│       └── ...
│
└── master_dataset.csv        # (Final) Índice validado
```
### 4.3 Especificaciones para el desarrollo de la Rama de Audio

Este modulo se encargará de la señal acustica.

#### A (Recorte Inteligente) `trimming.py`: 
Cargar los 30s descargados. Calcular la energía RMS en ventanas deslizantes para encontrar los 15 segundos de mayor intensidad (probablemente el coro) y descartar el resto, de esta forma intentamos capturar la parte **emocionalmente más representativa** de la canción.

#### B (Extracción 1D - HSFs) `features_1d.py`: 
Calcular MFCCs, Chroma, ZCR y sacar sus estadísticas (media, std). Guardar en features_audio_1d.csv.

#### C (Extracción 2D - Espectrogramas) `spectrograms.py`: 
Generar el Mel-Spectrogram de los 15s recortados.

>**Decisión Técnica**: No debemos guardar los espectrogramas como imágenes (.png), sino como matrices numéricas (.npy). Esto evita pérdidas de compresión y facilita la carga en PyTorch/TensorFlow.


### 4.4. Especificaciones para el desarrollo de la Rama de Texto

Mientras Uzi se encarga de procesar los audios (cuando finalmente se descarguen todos), Brenda estará a cargo de la inteligencia del Texto. Es necesario que se desarrollen los siguientes módulos dentro de `src/ProcessData/text/`, siguiendo las mejores prácticas descritas más adelante.

#### A. Limpieza `cleaning.py`:

- Función que reciba el string raw de Genius.

- Elimine etiquetas como `[Chorus]`, `[Verse 1]` y arregle errores de codificación (muy importante).

- Aplique preprocesamiento clásico: elimine caracteres especiales y normalice (lowercase), lemmatice si es necesario, remueva stopwords.

- Devuelva el texto limpio listo para vectorización. Considera formatos adecuados para esto como `json`. Considero que para capturar bien la estrcutrua de una letra, es mejor no eliminar los saltos de línea, sino todo lo contrario, preservarlos es importante para que el modelo entienda la estructura de la canción (verso, coro, puente, etc), estos saltos de linea pueden ser representados en el texto limpio como `\n` y almenos los embeddings generados por `BERT` entienden muy bien esta representación.

> **Nota**: El preprocesamiento es importante en técnicas como *TF-IDF*, pero para `word embeddings` debe ser ligero para no perder contexto emocional y semántico. Considera que `BERT` ya maneja mucho de esto internamente por lo que para los `word embeddings` no es necesario un preprocesamiento agresivo, incluso puede ser contraproducente.

#### B. Features 1D `features_1d.py`:

- **Objetivo**: Generar representaciones estadísticas ligeras.

- **Método**: *TF-IDF*, *Chi2* u otra técnica ligera para convertir el texto limpio en un vector numérico fijo por canción.

- **Salida**: Un archivo `features_text_tfidf.csv` en la carpeta `features_1d`.

    - *Ruta*: `data/processed/features_1d/features_text_tfidf.csv`.

    - *Formato*: CSV o algun otro formato adecuado para este tipo de representaciones.

    - *Columnas*: `spotify_id` (Obligatorio) + columnas del vector.

> **Nota importante:**
El formato CSV es ideal por compatibilidad, pero ineficiente para matrices gigantes.
Como estamos considerando guardar estos vectores en CSV, **es vital limitar la dimensionalidad** para no generar archivos gigantes llenos de ceros (por su sparsity).
>* Configura tu `TfidfVectorizer` con `max_features=1000` (o usa `SelectKBest` con Chi2 para seleccionar los top 1000).
>* *Razón:* Un vector de 1,000 dimensiones generalmente es adecuado para mezclar con el audio. Un vector de 20,000 (vocabulario completo) sería inmanejable en formato CSV. Verifica el tamaño adecuado para que el archivo final no sea demasiado grande pero que aún capture suficiente información.


#### C. Features 2D `embeddings.py`:

Esta es la pieza clave. Necesitamos una función que cargue un modelo Transformer (`BERT`) y convierta el texto limpio en un tensor/vector. Tip: Diseña la función para que reciba el texto y devuelva un `numpy array` (`.npy`).

- **Objetivo**: Generar representaciones semánticas profundas.

- **Método**: Usar un modelo Transformer (ej. `DistilBERT` o `BERT`) para convertir el texto limpio en un tensor, es decir, una matriz de dimensiones (`tokens` x `embedding_size`), donde cada fila representa el embedding de un token y cada columna una dimensión del embedding.

- **Salida**: Un archivo `.npy` individual por cada canción.

    - *Ruta*: `data/processed/features_2d/embeddings/{spotify_id}.npy`.

    - *Formato*: Array de NumPy.

> **Nota Técnica**: Asegúrate de que tus scripts de extracción guarden el `spotify_id` para poder hacer el cruce al final.



## 5. Filosofía de Código para la Fase 2 (Best Practices) 💡
Dado que vamos a integrar tu código de texto con el pipeline de audio para correrlo masivamente en una GPU, necesitamos seguir ciertas pautas de ingeniería de software para que todo encaje como piezas de LEGO.

1. **Adiós a los Notebooks (.ipynb) en Producción**

- Los notebooks son geniales para explorar, pero para el pipeline final necesitamos archivos `.py` pues los archivos `.py` se pueden importar entre sí. Un notebook no.

- **Flujo de trabajo**: Prototipa en Colab/Jupyter Notebooks si quieres, pero el código final debe estar limpio en `src/ProcessData/text/tus_scripts.py`.

2. **Funciones Puras (Modularidad)**

- Evita escribir código que se ejecute "suelto" al inicio del archivo. Todo debe estar dentro de funciones.
- De esta forma, el pipeline puede llamar a tus funciones cuando lo necesite. Por ejemplo, Uzi puede importar `from text.cleaning import limpiar_texto` y aplicarlo a las 6,000 canciones automáticamente.

3. **Rutas Relativas (Pathlib) y Configuración Centralizada**

- Evita usar rutas absolutas como `C:/Users/Brenda/....` Eso romperá el código en la otra computadora.

- Usa siempre `pathlib` basado en la raíz del proyecto (ya configurado en `main.py`).

4. **Preparación para GPU (Código y Dependencias)**

- Parametrización del Device: Cuando diseñes las funciones para BERT/Embeddings, evita "hardcodear" el uso de CPU. Estructura tu función para aceptar un parámetro device:

```Python
def get_embedding(text, model, device='cpu'):
    # El orquestador le pasará 'cuda' cuando Uzi corra el script en la GPU
    # Ejemplo interno: inputs = tokenizer(text, ...).to(device)
    pass
```
- **Verificación de Dependencias (CUDA)**: El archivo `environment.yaml` actual instala transformers, lo cual usualmente instala torch (PyTorch) como dependencia. Sin embargo, a veces los gestores de paquetes descargan la versión **CPU-only** por defecto para ahorrar espacio.

- **Tu Misión**: Investiga y verifica si necesitamos especificar una versión de PyTorch compatible con CUDA (ej. pytorch-cuda en Conda).

- **Acción**: Si encuentras que se necesita una instalación específica para habilitar la GPU, por favor actualiza el `environment.yaml` o agrega una nota en el código, no solo con esa dependencia en especifico, sino con cualquier otra que sea necesaria. Esto es vital para que Uzi pueda correr el pipeline completo en GPU sin problemas con el entorno adecuado.


## 📘 Manual de Integración: Módulo de Texto
De: Brenda Para: Uziel Objetivo: Replicar el pipeline de procesamiento de texto (TF-IDF y BERT Embeddings) en tu máquina local usando GPU.

### 1. Configuración del Entorno (CRÍTICO ⚠️)
Para que el procesamiento de BERT no tarde horas, necesitamos asegurar que PyTorch reconozca tu tarjeta gráfica (GPU).

#### Paso A: Actualizar environment.yaml
Asegúrate de que tu archivo environment.yaml tenga estos canales y dependencias específicas para GPU (reemplaza o verifica lo que tengas):

YAML
```bash
name: mem-env
channels:
  - pytorch           # Canal oficial
  - nvidia            # Drivers CUDA
  - conda-forge
  - defaults

dependencies:
  # ... (resto de librerías: pandas, numpy, librosa, etc.) ...
  - ftfy              # Limpieza de texto
  - pyarrow           # Lectura de parquets
  
  # --- BLOQUE GPU ---
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=12.1 # Fuerza la descarga de la versión compatible con GPU
  
  - pip:
      - transformers
      - accelerate    # Optimización para HuggingFace
``` 
#### Paso B: Reinstalar el Entorno
Ejecuta esto en tu terminal para aplicar los cambios limpios:

```bash
conda activate base
conda env remove -n mem-env
conda env create -f environment.yaml
conda activate mem-env
```
#### Paso C: Verificar GPU
Corre este mini-script en Python para confirmar que estamos listos:

```Python
import torch
print(f"¿Detecta GPU?: {torch.cuda.is_available()}")
# Si dice "True", ¡ya ganamos! 
```

## 2. Ejecución del Pipeline de Texto
No necesitas correr los scripts individuales (cleaning.py, features_1d.py, etc.). He creado un Orquestador Maestro que hace todo en orden.

Comando a ejecutar: Desde la raíz del proyecto:
```Python
python src/ProcessData/run_text_pipeline.py
```
¿Qué va a pasar?

Limpieza: Genera textos limpios para TF-IDF y BERT.

Rama 1D: Crea el vector TF-IDF (vocabulario 3,000).

Rama 2D: Descarga DistilBERT y genera los embeddings (Tensores).

Nota: Si tienes GPU, tardará unos 15-20 mins. Si usas CPU, tardará horas.

## 3. Salidas Generadas (Los Archivos) 
Una vez que termine el script, encontrarás los siguientes archivos en data/. Estos son los insumos para la Fase 3 (Entrenamiento).

- A. Texto Limpio (Intermedio)
    - Ruta: data/interim/lyrics_cleaned.parquet

    - Formato: .parquet (Usamos Parquet porque CSV rompe los textos con comas y saltos de línea).

    - Contenido:

        * clean_lyrics_tfidf: Texto plano, minúsculas, sin stopwords (para DNN).

        * clean_lyrics_bert: Texto con estructura, mayúsculas y puntuación (para CNN).

- B. Features 1D (Estadísticos)
    - Ruta: data/processed/features_1d/features_text_1d.parquet

    - Formato: .parquet

    - Contenido: Matriz dispersa de TF-IDF.

    - Uso: Entrada para el modelo TextDNN (Red Densa).

- C. Features 2D (Embeddings Profundos)
    - Ruta: data/processed/features_2d/embeddings/

    - Archivos:

    1. embeddings_2d.npy: El tensor pesado de datos.

        - Shape: (N_canciones, 256, 768).

        - 256: Longitud de secuencia (Tiempo).

        - 768: Dimensiones de BERT.

    2. embeddings_ids.npy: Lista de IDs.

        - Formato: .npy (NumPy Binary). Es el estándar para guardar tensores multidimensionales.

        - Uso: Entrada para el modelo TextCNN (Red Convolucional).

## 4. Troubleshooting (Si algo falla) 🛠️
- Error: CUDA Out of Memory:

    - Significa que la GPU se llenó.

    - Solución: Ve a src/ProcessData/text/embeddings.py y baja el BATCH_SIZE de 32 a 16 u 8.

- Error: FileNotFoundError: aligned_metadata.csv:

    - Asegúrate de haber corrido el paso de alineación de metadatos antes de correr el de texto.

- Error leyendo Parquet:

    - Asegúrate de haber instalado pyarrow (pip install pyarrow o vía conda).