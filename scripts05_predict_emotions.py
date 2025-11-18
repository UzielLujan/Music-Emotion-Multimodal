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

def add_emotion_labels(df):
    df["emotion"] = df.apply(
        lambda r: map_emotions(r["valence"], r["arousal"]), axis=1
    )
    df.to_csv("C:/Users/Brenda Tránsito/Documents/Maestría/Tercer Semestre/MIR/Proyecto/Proyecto_MIR/data/features/fma_labeled.csv", index=False)
    return df
