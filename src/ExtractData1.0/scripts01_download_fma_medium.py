# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 09:55:12 2025

@author: Brenda Tránsito
"""

import os
import zipfile


from pathlib import Path

def extract_fma_medium():
    # Ruta absoluta a data/raw basada en la ubicación de este script
    base_dir = Path(__file__).resolve().parent.parent.parent  # .../Music-Emotion-Multimodal/
    raw_dir = base_dir / 'src' / 'data' / 'raw'
    zip_path = raw_dir / 'fma_medium.zip'
    extract_path = raw_dir / 'fma_medium'

    if not zip_path.exists():
        raise FileNotFoundError(f"No existe {zip_path}. Descarga fma_medium.zip primero.")

    if extract_path.exists():
        print("fma_medium ya está extraída.")
        return str(extract_path)

    print("Extrayendo fma_medium.zip...")
    with zipfile.ZipFile(str(zip_path), "r") as z:
        z.extractall(str(raw_dir))

    print("Extracción completada.")
    return str(extract_path)

