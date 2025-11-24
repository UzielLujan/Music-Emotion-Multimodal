# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 16:03:30 2025

@author: Brenda Tránsito
"""

import os
import pandas as pd

def merge_valence_arousal(df_feats, metadata_dir):
    echonest_path = os.path.join(metadata_dir, "echonest.csv")

    # Detectar MultiIndex de 3 niveles
    echo = pd.read_csv(echonest_path, index_col=0, header=[0,1,2])

    # Confirmar columnas disponibles
    print("Primeras columnas de echonest:")
    print(echo.columns[:10])

    # Verificar que valence y energy existan
    if ("echonest", "audio_features", "valence") not in echo.columns:
        raise KeyError("No existe valence en echonest.csv")

    if ("echonest", "audio_features", "energy") not in echo.columns:
        raise KeyError("No existe energy en echonest.csv")

    # Extraer columnas
    df_val = echo[[("echonest", "audio_features", "valence"),
                   ("echonest", "audio_features", "energy")]]

    # Renombrar columnas
    df_val.columns = ["valence", "arousal"]

    # Alinearlos con df_feats
    df_merged = df_feats.join(df_val, how="inner")

    print("Tamaño antes:", df_feats.shape)
    print("Tamaño después:", df_merged.shape)

    # Retorna con valence y arousal incluidas
    return df_merged
