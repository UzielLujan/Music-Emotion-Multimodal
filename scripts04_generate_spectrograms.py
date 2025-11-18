# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 16:03:22 2025

@author: Brenda Tránsito
"""

# scripts/scripts04_generate_spectrograms_medium.py
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def generate_spectrograms(track_ids, audio_dir, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    for tid in tqdm(track_ids):
        folder = f"{tid:06d}"[:3]
        file_path = os.path.join(audio_dir, folder, f"{tid:06d}.mp3")

        if not os.path.exists(file_path):
            continue

        try:
            y, sr = librosa.load(file_path, sr=22050)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_db = librosa.power_to_db(S, ref=np.max)

            plt.figure(figsize=(3,3))
            librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None)
            plt.axis("off")

            plt.savefig(os.path.join(out_dir, f"{tid}.png"),
                        bbox_inches="tight", pad_inches=0)
            plt.close()

        except Exception as e:
            print(f"Error espectrograma {tid}: {e}")
            continue
