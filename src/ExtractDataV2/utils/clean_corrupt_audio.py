import os
import glob
import librosa
from pathlib import Path
from tqdm import tqdm

# Configura tu ruta correcta
BASE_DIR = Path(__file__).resolve().parents[3]
AUDIO_DIR = BASE_DIR / "data" / "raw_v2" / "audio"

def delete_bad_files():
    print(f"🕵️‍♂️ Buscando archivos corruptos en: {AUDIO_DIR}")
    files = list(AUDIO_DIR.glob("*.mp3"))
    
    bad_files = 0
    
    for file_path in tqdm(files):
        try:
            # Intentamos leer solo la duración (es rápido y detecta corrupción)
            librosa.get_duration(path=file_path)
            
            # Opcional: Verificar tamaño mínimo (ej. < 50KB es sospechoso para 30s)
            if file_path.stat().st_size < 50000:
                raise ValueError("Archivo demasiado pequeño")
                
        except Exception as e:
            print(f"   🗑️ Eliminando corrupto: {file_path.name} ({e})")
            os.remove(file_path)
            bad_files += 1

    print(f"\n✨ Limpieza terminada. Se eliminaron {bad_files} archivos corruptos.")
    print("👉 Ahora puedes volver a correr el scraper (ExtractDataV2/main.py) y descargará estos faltantes.")

if __name__ == "__main__":
    delete_bad_files()