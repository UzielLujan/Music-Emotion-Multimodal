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

def download_fma_medium(raw_dir):
    """
    Descarga fma_medium.zip y fma_metadata.zip en raw_dir,
    y los extrae en ese mismo directorio.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    URL_MEDIUM = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    URL_META = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

    path_medium = raw_dir / "fma_medium.zip"
    path_meta = raw_dir / "fma_metadata.zip"
    extract_medium = raw_dir / "fma_medium"
    extract_meta = raw_dir / "fma_metadata"

    # Descargar
    if not path_medium.exists():
        download_file(URL_MEDIUM, str(path_medium))
    else:
        print("[INFO] fma_medium.zip ya existe.")

    if not path_meta.exists():
        download_file(URL_META, str(path_meta))
    else:
        print("[INFO] fma_metadata.zip ya existe.")

    # Extraer
    if not extract_medium.exists():
        safe_extract(str(path_medium), str(raw_dir))
    else:
        print("[INFO] Carpeta fma_medium ya existente.")

    if not extract_meta.exists():
        safe_extract(str(path_meta), str(raw_dir))
    else:
        print("[INFO] Carpeta fma_metadata ya existente.")

    print("[✔] FMA Medium + Metadata listos.")




