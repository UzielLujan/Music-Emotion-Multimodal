# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 22:38:53 2025

@author: Brenda Tránsito
"""

def map_emotions(valence, arousal):
    if valence >= 0.5 and arousal >= 0.5:
        return "happy"
    elif valence < 0.5 and arousal >= 0.5:
        return "angry"
    elif valence < 0.5 and arousal < 0.5:
        return "sad"
    elif valence >= 0.5 and arousal < 0.5:
        return "relaxed"


from pathlib import Path

def add_emotion_labels(df):
    df["emotion"] = df.apply(
        lambda r: map_emotions(r["valence"], r["arousal"]), axis=1
    )
    # Ruta absoluta a data/features basada en la ubicación de este script
    base_dir = Path(__file__).resolve().parent.parent.parent / 'src' / 'data' / 'features'
    base_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_dir / 'fma_labeled.csv'
    df.to_csv(str(output_path), index=False)
    return df
