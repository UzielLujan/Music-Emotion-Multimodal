# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 09:55:12 2025

@author: Brenda Tránsito
"""

import os
import zipfile


from pathlib import Path


def extract_fma_medium_and_metadata():
    # Ruta absoluta a data/raw basada en la ubicación de este script
    base_dir = Path(__file__).resolve().parent.parent.parent  # .../Music-Emotion-Multimodal/
    raw_dir = base_dir / 'data' / 'raw'
    zip_medium = raw_dir / 'fma_medium.zip'
    zip_meta = raw_dir / 'fma_metadata.zip'
    extract_medium = raw_dir / 'fma_medium'
    extract_meta = raw_dir / 'fma_metadata'

    # Extraer fma_medium.zip
    if not zip_medium.exists():
        raise FileNotFoundError(f"No existe {zip_medium}. Descarga fma_medium.zip primero.")
    if not extract_medium.exists():
        print("Extrayendo fma_medium.zip...")
        with zipfile.ZipFile(str(zip_medium), "r") as z:
            z.extractall(str(raw_dir))
        print("Extracción de fma_medium completada.")
    else:
        print("fma_medium ya está extraída.")

    # Extraer fma_metadata.zip
    if not zip_meta.exists():
        raise FileNotFoundError(f"No existe {zip_meta}. Descarga fma_metadata.zip primero.")
    if not extract_meta.exists():
        print("Extrayendo fma_metadata.zip...")
        with zipfile.ZipFile(str(zip_meta), "r") as z:
            z.extractall(str(raw_dir))
        print("Extracción de fma_metadata completada.")
    else:
        print("fma_metadata ya está extraída.")

    return str(extract_medium), str(extract_meta)

