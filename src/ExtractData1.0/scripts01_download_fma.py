# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 16:01:19 2025

@author: Brenda Tránsito
"""

import os
import urllib.request
import zipfile
import shutil


def download_file(url, dest_path):
    """Descargar archivo con manejo de errores y reporte de progreso."""
    print(f"Descargando:\n{url}")
    print(f"Guardando en: {dest_path}")

    def progress(blocks, block_size, total_size):
        downloaded = blocks * block_size
        percent = downloaded * 100 / total_size if total_size > 0 else 0
        print(f"\r   {percent:.2f}% ({downloaded/1e6:.1f} MB)", end="")

    urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    print("\n Descarga completa.")


def safe_extract(zip_path, extract_to):
    """Extraer ZIP sin errores ZIP64 y asegurando compatibilidad con Windows."""
    print(f"Extrayendo: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        print(f"Archivos extraídos en: {extract_to}")

    except zipfile.BadZipFile:
        print("ERROR: Archivo ZIP corrupto o incompleto.")
        print("   Intenta borrar el archivo y volver a descargar.")
    except Exception as e:
        print("Error al descomprimir:", e)


from pathlib import Path

def download_fma_medium():
    # Ruta absoluta a data/raw basada en la ubicación de este script
    base_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'raw'
    base_dir.mkdir(parents=True, exist_ok=True)

    # URLs oficiales del dataset FMA (cambiado a medium)
    URL_MEDIUM = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    URL_META = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

    # Paths de destino (cambiado a medium)
    PATH_MEDIUM = base_dir / "fma_medium.zip"
    PATH_META = base_dir / "fma_metadata.zip"
    EXTRACT_MEDIUM = base_dir / "fma_medium"
    EXTRACT_META = base_dir / "fma_metadata"

    # Descargar fma_medium
    if not PATH_MEDIUM.exists():
        download_file(URL_MEDIUM, str(PATH_MEDIUM))
    else:
        print("fma_medium.zip ya existe, no se descarga.")

    # Descargar fma_metadata
    if not PATH_META.exists():
        download_file(URL_META, str(PATH_META))
    else:
        print("fma_metadata.zip ya existe, no se descarga.")

    # Extraer fma_medium
    if not EXTRACT_MEDIUM.exists():
        safe_extract(str(PATH_MEDIUM), str(base_dir))
    else:
        print("fma_medium ya extraído.")

    # Extraer fma_metadata
    if not EXTRACT_META.exists():
        safe_extract(str(PATH_META), str(base_dir))
    else:
        print("fma_metadata ya extraído.")

    print("\n FMA Medium + Metadata listos para usar.")


if __name__ == "__main__":
    download_fma_medium()



