"""
Pipeline de Procesamiento de Texto (Fase 2)
-------------------------------------------------
Transforma letras crudas en:

1. Archivo único limpio (Parquet) con:
    - clean_lyrics_tfidf (texto plano)
    - clean_lyrics_bert (texto con estructura)
    - spotify_id

2. Features 1D Base (Parquet) con:
    - TF-IDF "crudo" (vocabulario amplio, ej. 3000-5000 palabras)
    * NOTA: La selección Chi² se hará durante el entrenamiento
      para evitar Data Leakage.
"""

import sys
from pathlib import Path

# Ajuste de paths para acceso a módulos
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Importar funciones actualizadas
# Asegúrate de que en cleaning.py la función se llame así
from src.ProcessData.text.cleaning import save_clean_lyrics
from src.ProcessData.text.features_1d import generate_text_1d_features

# =======================
# Rutas principales
# =======================

DATA_DIR = PROJECT_ROOT / "data"

# Entrada: Metadatos alineados (CSV o Parquet)
INPUT_FILE = DATA_DIR / "interim" / "aligned_metadata.csv"

# Salida Paso 1: Archivo limpio único (PARQUET)
# Usamos parquet para evitar problemas con saltos de línea y comas
OUT_CLEAN = DATA_DIR / "interim" / "lyrics_cleaned.parquet"

# Salida Paso 2: Features 1D (PARQUET)
OUT_1D_DIR = DATA_DIR / "processed" / "features_1d"
OUT_FEATURES_1D = OUT_1D_DIR / "features_text_1d.parquet"


def run_pipeline():
    print("🚀 INICIANDO PIPELINE DE TEXTO (Fase 2)...\n")

    # Validación de archivo base
    if not INPUT_FILE.exists():
        print(f"❌ Error: No existe {INPUT_FILE}. Ejecuta el paso de alineación primero.")
        return

    # Crear directorios necesarios
    OUT_1D_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1. Limpieza de letras (BERT + TF-IDF)
    # -------------------------------------------------
    print(f"\n--- Paso 1/2: Limpieza y Estandarización ---")
    print(f"Entrada: {INPUT_FILE}")
    print(f"Salida:  {OUT_CLEAN}")
    
    save_clean_lyrics(
        aligned_csv_path=INPUT_FILE,
        output_path=OUT_CLEAN,
        format='parquet'  # Importante: Usar Parquet
    )

    # -------------------------------------------------
    # 2. Generación de features 1D (Base para Chi²)
    # -------------------------------------------------
    print(f"\n--- Paso 2/2: Generación de Vector Base (TF-IDF) ---")
    print("Generando vocabulario amplio para posterior selección Chi²...")
    
    generate_text_1d_features(
        clean_file_path=OUT_CLEAN,
        output_path=OUT_FEATURES_1D,
        # Usamos un vocabulario amplio (3000-5000) para darle opciones 
        # al Chi² de elegir las mejores después.
        max_vocab_size=3000, 
        format='parquet'
    )

    # Reporte final
    print("\n✅ Pipeline de texto completado con éxito.")
    print("Archivos generados:")
    print(f"1. Texto Limpio:   {OUT_CLEAN}")
    print(f"2. Features 1D:    {OUT_FEATURES_1D}")
    print("\nSiguiente paso sugerido: Generar Embeddings 2D (BERT).")


if __name__ == "__main__":
    run_pipeline()