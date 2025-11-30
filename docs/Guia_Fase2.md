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

## 4. Arquitectura de la Fase 2: Procesamiento (ProcessData)

Una vez tengamos los audios, entraremos a la fase de **Ingeniería de Características**. Hemos diseñado una estructura modular para trabajar en paralelo sin conflictos.

**Estructura de Carpetas de ProcessData:**

```bash
src/ProcessData/
├── audio/            # (Responsable: Uzi) - Recorte, Espectrogramas, HSFs
├── text/             # (Responsable: Brenda) - Limpieza, TF-IDF, Embeddings
└── utils/            # Funciones compartidas (Carga de archivos, Alineación)
```

**Tu Misión de Desarrollo (Rama de Texto):**

Mientras Uzi se encarga de procesar los audios (cuando finalmente se descarguen todos), tú estarás a cargo de la inteligencia del Texto. Necesitamos que desarrolles los siguientes módulos dentro de `src/ProcessData/text/`:

1. `cleaning.py`:

Función que reciba el string raw de Genius.

Elimine etiquetas como [Chorus], [Verse 1].

Elimine caracteres especiales y normalice (lowercase).

2. `embeddings.py`:

Esta es la pieza clave. Necesitamos una función que cargue un modelo Transformer (ej. DistilBERT) y convierta el texto limpio en un tensor/vector.

Tip: Diseña la función para que reciba el texto y devuelva un numpy array.