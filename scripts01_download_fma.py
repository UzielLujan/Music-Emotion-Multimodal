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
    print(f"📥 Descargando:\n{url}")
    print(f"➡ Guardando en: {dest_path}")

    def progress(blocks, block_size, total_size):
        downloaded = blocks * block_size
        percent = downloaded * 100 / total_size if total_size > 0 else 0
        print(f"\r   {percent:.2f}% ({downloaded/1e6:.1f} MB)", end="")

    urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    print("\n✅ Descarga completa.")


def safe_extract(zip_path, extract_to):
    """Extraer ZIP sin errores ZIP64 y asegurando compatibilidad con Windows."""
    print(f"📦 Extrayendo: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_to)
        print(f"✅ Archivos extraídos en: {extract_to}")

    except zipfile.BadZipFile:
        print("❌ ERROR: Archivo ZIP corrupto o incompleto.")
        print("   Intenta borrar el archivo y volver a descargar.")
    except Exception as e:
        print("❌ Error al descomprimir:", e)


def download_fma_full(base_dir="data/raw"):
    os.makedirs(base_dir, exist_ok=True)

    # ------------------------------
    # URLs oficiales del dataset FMA
    # ------------------------------
    URL_SMALL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    URL_META = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

    # ------------------------------
    # Paths de destino
    # ------------------------------
    PATH_SMALL = os.path.join(base_dir, "C:/Users/Brenda Tránsito/Documents/Maestría/Tercer Semestre/MIR/Proyecto/Proyecto_MIR/data/raw/fma_small.zip")
    PATH_META = os.path.join(base_dir, "C:/Users/Brenda Tránsito/Documents/Maestría/Tercer Semestre/MIR/Proyecto/Proyecto_MIR/data/raw/fma_metadata.zip")

    EXTRACT_SMALL = os.path.join(base_dir, "C:/Users/Brenda Tránsito/Documents/Maestría/Tercer Semestre/MIR/Proyecto/Proyecto_MIR/data/raw/fma_small")
    EXTRACT_META = os.path.join(base_dir, "C:/Users/Brenda Tránsito/Documents/Maestría/Tercer Semestre/MIR/Proyecto/Proyecto_MIR/data/raw/fma_metadata")

    # ------------------------------
    # Descargar fma_small
    # ------------------------------
    if not os.path.exists(PATH_SMALL):
        download_file(URL_SMALL, PATH_SMALL)
    else:
        print("✔ fma_small.zip ya existe, no se descarga.")

    # ------------------------------
    # Descargar fma_metadata
    # ------------------------------
    if not os.path.exists(PATH_META):
        download_file(URL_META, PATH_META)
    else:
        print("✔ fma_metadata.zip ya existe, no se descarga.")

    # ------------------------------
    # Extraer fma_small
    # ------------------------------
    if not os.path.exists(EXTRACT_SMALL):
        safe_extract(PATH_SMALL, base_dir)
    else:
        print("✔ fma_small ya extraído.")

    # ------------------------------
    # Extraer fma_metadata
    # ------------------------------
    if not os.path.exists(EXTRACT_META):
        safe_extract(PATH_META, base_dir)
    else:
        print("✔ fma_metadata ya extraído.")

    print("\n🎉 FMA + Metadata listos para usar.")


if __name__ == "__main__":
    download_fma_full()



