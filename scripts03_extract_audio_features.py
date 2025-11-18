# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 16:03:15 2025

@author: Brenda Tránsito
"""

# scripts/scripts03_extract_audio_features_medium.py
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_audio_features(track_ids, audio_dir, sampling_rate=22050):

    records = []

    for tid in tqdm(track_ids):
        folder = f"{tid:06d}"[:3]
        file_path = os.path.join(audio_dir, folder, f"{tid:06d}.mp3")

        if not os.path.exists(file_path):
            continue

        try:
            y, sr = librosa.load(file_path, sr=sampling_rate)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)

            feats = {
                "track_id": tid,
                **{f"mfcc_{i}": mfcc[i] for i in range(20)}
            }

            records.append(feats)

        except Exception as e:
            print(f"Error en track {tid}: {e}")
            continue

    df = pd.DataFrame(records)
    return df
