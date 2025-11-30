# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:47:22 2025

@author: Brenda Tránsito
"""

# scripts/scripts05_add_valence_arousal.py
import os
import pandas as pd

def add_valence_arousal(df_feats, metadata_dir):

    echo_path = os.path.join(metadata_dir, "echonest.csv")

    echo = pd.read_csv(
        echo_path,
        index_col=0,
        header=[0,1,2]  # MultiIndex de 3 niveles
    )

    # columnas reales de valence y energy en echonest
    col_val = ("echonest", "audio_features", "valence")
    col_ar = ("echonest", "audio_features", "energy")

    echo_va = echo[[col_val, col_ar]].copy()
    echo_va.columns = ["valence", "arousal"]

    merged = df_feats.merge(
        echo_va,
        left_on="track_id",
        right_index=True,
        how="inner"
    )

    print(f"Merged VA size: {len(merged)}")

    return merged
