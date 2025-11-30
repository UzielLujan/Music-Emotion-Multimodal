"""
@author: Brenda Tránsito and Uziel Luján
"""
import os
import zipfile
from pathlib import Path

def extract_fma_medium_and_metadata(raw_dir):
    raw_dir = Path(raw_dir)

    zip_medium = raw_dir / "fma_medium.zip"
    zip_meta = raw_dir / "fma_metadata.zip"
    extract_medium = raw_dir / "fma_medium"
    extract_meta = raw_dir / "fma_metadata"

    if not zip_medium.exists():
        raise FileNotFoundError("No existe fma_medium.zip")

# carpeta existe pero está vacía → hay que extraer
    if not extract_medium.exists() or len(list(extract_medium.iterdir())) == 0:
        print("Extrayendo fma_medium.zip...")
        with zipfile.ZipFile(str(zip_medium), "r") as z:
            z.extractall(str(raw_dir))

    if not zip_meta.exists():
        raise FileNotFoundError("No existe fma_metadata.zip")

# carpeta existe pero está vacía → hay que extraer
    if not extract_meta.exists() or len(list(extract_meta.iterdir())) == 0:
        print("Extrayendo fma_metadata.zip...")
        with zipfile.ZipFile(str(zip_meta), "r") as z:
            z.extractall(str(raw_dir))

    return extract_medium, extract_meta


