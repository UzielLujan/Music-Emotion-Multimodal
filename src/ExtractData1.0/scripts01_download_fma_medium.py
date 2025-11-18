# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 09:55:12 2025

@author: Brenda Tránsito
"""

import os
import zipfile

def extract_fma_medium():
    raw_dir = "C:/Users/Brenda Tránsito/Documents/Maestría/Tercer Semestre/MIR/Proyecto/Proyecto_MIR/data/raw/"
    zip_path = os.path.join(raw_dir, "fma_medium.zip")
    extract_path = os.path.join(raw_dir, "fma_medium")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"No existe {zip_path}. Descarga fma_medium.zip primero.")

    if os.path.exists(extract_path):
        print("fma_medium ya está extraída.")
        return extract_path

    print("Extrayendo fma_medium.zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)

    print("Extracción completada.")
    return extract_path

