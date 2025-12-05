"""
Orquestador del Pipeline de Extracción V2.1 (Kaggle -> Genius -> YouTube)
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# === CONFIGURACIÓN DE RUTAS ===
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_V2_DIR = DATA_DIR / "raw_v2"
AUDIO_DIR = RAW_V2_DIR / "audio"

# INPUT: Dataset de Kaggle
KAGGLE_DATASET = RAW_V2_DIR / "archive" / "dataset.csv"

# OUTPUTS: Checkpoints
CSV_STEP1 = RAW_V2_DIR / "metadata_step1_spotify.csv"
#CSV_STEP2 = RAW_V2_DIR / "metadata_step2_lyrics.csv" # Archivo de Metadatos + letras originales
#CSV_STEP2 = RAW_V2_DIR / "metadata_step2_lyrics_clean.csv" # Archivo de Metadatos + letras limpio
CSV_STEP2 = RAW_V2_DIR / "metadata_part_brenda.csv" # Archivo de Metadatos + letras limpio (Uzi only) cambia el nombre segun convenga


load_dotenv(PROJECT_ROOT / ".env")

try:
    from pipeline.step01_spotify_seeds import extract_spotify_metadata
    from pipeline.step02_genius_filter import filter_and_fetch_lyrics
    from pipeline.step03_youtube_dl import download_audio_batch
except ImportError as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

def main():
    print(f"Iniciando Pipeline de Extracción V2.1 (Kaggle Strategy)")
    
    RAW_V2_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # --- PASO 1: SELECCIÓN DE SEMILLAS (KAGGLE) ---
    print("\n" + "="*50)
    print("[PASO 1] Selección de semillas desde Dataset Kaggle")
    print("="*50)
    
    if CSV_STEP1.exists():
        print(f"Checkpoint encontrado: {CSV_STEP1.name}")
    else:
        # Cambio: Aumentamos de 500 a 2000 para maximizar la captura
        # Esto tomará todo lo disponible de Q4 (1258) y 2000 de los demás.
        extract_spotify_metadata(kaggle_csv_path=KAGGLE_DATASET, 
                                 output_csv=CSV_STEP1, 
                                 samples_per_quadrant=6000) # Aumentado a 6000 para capturar todo lo posible

    # --- PASO 2: GENIUS (LETRAS + FILTRO) ---
    print("\n" + "="*50)
    print("[PASO 2] Extracción de Letras y Filtrado (Genius)")
    print("="*50)
    
    if CSV_STEP2.exists():
        print(f"Checkpoint encontrado: {CSV_STEP2.name}")
    else:
        # Si el paso 1 falló o no generó archivo, salimos
        if not CSV_STEP1.exists():
            print("Error: Falta el archivo del Paso 1.")
            sys.exit(1)
        filter_and_fetch_lyrics(input_csv=CSV_STEP1, output_csv=CSV_STEP2)
        
    '''
    # LÓGICA NUEVA (descomentar solo si se quiere forzar siempre el paso 2)
    print(f"   Iniciando/Reanudando proceso sobre {CSV_STEP1.name}...")
    
    # Verificación de seguridad básica
    if not CSV_STEP1.exists():
        print("❌ Error: No existe el archivo de semillas (Paso 1).")
        sys.exit(1)

    # El módulo step02 se encargará de leer CSV_STEP2 y saltar lo que ya existe.
    filter_and_fetch_lyrics(input_csv=CSV_STEP1, output_csv=CSV_STEP2)
    '''
    # --- PASO 3: YOUTUBE (DESCARGA) ---
    print("\n" + "="*50)
    print("[PASO 3] Descarga de Audio (YouTube DL)")
    print("="*50)
    
    if CSV_STEP2.exists():
        download_audio_batch(input_csv=CSV_STEP2, audio_output_dir=AUDIO_DIR)
    else:
        print("Error: No se puede proceder sin el archivo del Paso 2.")
        sys.exit(1)

    print("\nPipeline finalizado exitosamente.")

if __name__ == "__main__":
    main()