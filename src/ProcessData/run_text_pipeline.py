"""
Pipeline de Procesamiento de Texto (Fase 2)
-------------------------------------------------
Transforma letras crudas en:

1. Archivo único limpio con:
    - clean_lyrics_tfidf
    - clean_lyrics_bert
    - todas las columnas originales

2. Features 1D basados en:
    - TF-IDF
    - Selección Chi² supervisada
    - Reducción LSA (TruncatedSVD) a 100 dimensiones
"""

import sys
import pandas as pd
from pathlib import Path

# Ajuste de paths para acceso a módulos
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Importar funciones del procesamiento de texto
from src.ProcessData.text.cleaning import save_clean_lyrics_single
from src.ProcessData.text.features_1d import generate_text_1d_features


# =======================
# Rutas principales
# =======================

DATA_DIR = PROJECT_ROOT / "data"

INPUT_CSV = DATA_DIR / "interim" / "aligned_metadata.csv"

# Archivo limpio único
OUT_CLEAN = DATA_DIR / "interim" / "aligned_metadata_cleanlyrics.csv"

# Features 1D
OUT_1D_DIR = DATA_DIR / "processed" / "features_1d"
OUT_FEATURES_1D_CSV = OUT_1D_DIR / "features_text_1d.csv"


def run_pipeline():
    print("INICIANDO PIPELINE DE TEXTO...\n")

    # Validación de archivo base
    if not INPUT_CSV.exists():
        print(f"Error: No existe {INPUT_CSV}. Ejecute alignment.py primero.")
        return

    # Crear directorios necesarios
    OUT_1D_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    print(f"Archivo alineado detectado. Registros: {len(df)}")

    # -------------------------------------------------
    # 1. Limpieza de letras → archivo único limpio
    # -------------------------------------------------
    print("\nPaso 1/2: Generando archivo único de letras limpias...")
    save_clean_lyrics_single(
        aligned_csv_path=INPUT_CSV,
        output_csv_path=OUT_CLEAN
    )

    # -------------------------------------------------
    # 2. Generación de features 1D (TF-IDF → Chi² → LSA)
    # -------------------------------------------------
    print("\nPaso 2/2: Generando representación 1D (TF-IDF, Chi², LSA)...")

    generate_text_1d_features(
        clean_tfidf_csv_path=OUT_CLEAN,
        output_csv_path=OUT_FEATURES_1D_CSV,
        tfidf_max_features=5000,  # tamaño máximo del vocabulario TF-IDF
        chi2_dims=500,            # dimensiones tras Chi²
        lsa_dims=100              # dimensiones finales LSA
    )

    # Reporte final
    print("\nPipeline de texto completado.")
    print("Archivos generados:")
    print(f"- Archivo limpio único: {OUT_CLEAN}")
    print(f"- Features 1D:          {OUT_FEATURES_1D_CSV}\n")


if __name__ == "__main__":
    run_pipeline()
