"""
Pipeline de Procesamiento de Texto (Fase 2)
-------------------------------------------------
Orquestador principal que transforma letras crudas en:

1. Archivo único limpio (Parquet)
2. Features 1D Base (TF-IDF amplio para posterior Chi²)
3. Features 2D (Embeddings BERT para CNN)
"""

import sys
from pathlib import Path
import torch

# Ajuste de paths para acceso a módulos
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- IMPORTACIONES MODULARES ---
from src.ProcessData.text.cleaning import save_clean_lyrics
from src.ProcessData.text.features_1d import generate_text_1d_features
from src.ProcessData.text.embeddings import generate_bert_embeddings # <--- NUEVO

# =======================
# Rutas principales
# =======================

DATA_DIR = PROJECT_ROOT / "data"

# Entrada: Metadatos alineados
INPUT_FILE = DATA_DIR / "interim" / "aligned_metadata.csv"

# Salida Paso 1: Archivo limpio único
OUT_CLEAN = DATA_DIR / "interim" / "lyrics_cleaned.parquet"

# Salida Paso 2: Features 1D
OUT_1D_DIR = DATA_DIR / "processed" / "features_1d"
OUT_FEATURES_1D = OUT_1D_DIR / "features_text_1d.parquet"

# Salida Paso 3: Features 2D (Embeddings) <--- NUEVO
OUT_2D_DIR = DATA_DIR / "processed" / "features_2d" / "embeddings"


def run_pipeline():
    print("INICIANDO PIPELINE DE TEXTO COMPLETO (Fase 2)...\n")
    
    # Detección de Hardware para reporte inicial
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware detectado para Deep Learning: {device.upper()}")
    if device == "cpu":
        print("ADVERTENCIA: Se usará CPU. El paso 3 será lento.")

    # Validación de archivo base
    if not INPUT_FILE.exists():
        print(f"Error: No existe {INPUT_FILE}. Ejecuta el paso de alineación primero.")
        return

    # Crear directorios necesarios
    OUT_1D_DIR.mkdir(parents=True, exist_ok=True)
    OUT_2D_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1. Limpieza de letras (BERT + TF-IDF)
    # -------------------------------------------------
    print(f"\n--- Paso 1/3: Limpieza y Estandarización ---")
    save_clean_lyrics(
        aligned_csv_path=INPUT_FILE,
        output_path=OUT_CLEAN,
        format='parquet'
    )

    # -------------------------------------------------
    # 2. Generación de features 1D (Base para Chi²)
    # -------------------------------------------------
    print(f"\n--- Paso 2/3: Generación de Vector Base (TF-IDF) ---")
    generate_text_1d_features(
        clean_file_path=OUT_CLEAN,
        output_path=OUT_FEATURES_1D,
        max_vocab_size=2000, # Basado en análisis de cobertura (Zipf Law)
        format='parquet'
    )

    # -------------------------------------------------
    # 3. Generación de features 2D (Embeddings BERT)
    # -------------------------------------------------
    print(f"\n--- Paso 3/3: Generación de Embeddings 2D (BERT) ---")
    
    # Parámetros optimizados según nuestro análisis EDA
    MAX_LEN = 256  # Balance ideal entre memoria y cobertura de canción
    BATCH_SIZE = 32 # Seguro para MAX_LEN=256 en GPU T4/RTX
    
    generate_bert_embeddings(
        input_path=OUT_CLEAN,
        output_dir=OUT_2D_DIR,
        model_name="distilbert-base-uncased",
        max_length=MAX_LEN,
        batch_size=BATCH_SIZE,
        device_str='cuda' # Dejamos que el script detecte auto (o pon 'cuda')
    )

    # Reporte final
    print("\n Pipeline de texto completado con éxito.")
    print("Archivos generados:")
    print(f"   1. Texto Limpio:   {OUT_CLEAN.name}")
    print(f"   2. Features 1D:    {OUT_FEATURES_1D.name}")
    print(f"   3. Features 2D:    embeddings_2d.npy (en {OUT_2D_DIR})")
    print("\n¡Listo para la Fase 3 (Entrenamiento)!")


if __name__ == "__main__":
    run_pipeline()