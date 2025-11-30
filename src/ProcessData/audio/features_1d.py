import numpy as np
import pandas as pd
import librosa

def extract_hsfs(y, sr):
    """
    Calcula High-Level Statistical Functions (HSFs) a partir de LLDs.
    Devuelve un Diccionario (listo para ser fila de Pandas).
    """
    features = {}
    
    # --- 1. EXTRAER LLDs (Low Level Descriptors) ---
    
    # MFCCs (Timbre) - Extraemos 13 coeficientes
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Chroma (Armonía/Tonos)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # Spectral Contrast (Textura/Picos vs Valles)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Zero Crossing Rate (Ruidisidad/Percusión)
    zcr = librosa.feature.zero_crossing_rate(y)
    
    # Spectral Centroid (Brillo)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    # --- 2. CALCULAR HSFs (Media y Std) ---
    
    # Función auxiliar para guardar stats
    def add_stats(name, matrix):
        features[f'{name}_mean'] = np.mean(matrix)
        features[f'{name}_std'] = np.std(matrix)

    # MFCCs (Iteramos por cada coeficiente para no perder detalle)
    for i in range(mfcc.shape[0]):
        add_stats(f'mfcc_{i+1}', mfcc[i, :])
        
    # Para los demás, hacemos promedio global de la matriz o vector
    add_stats('chroma', chroma)
    add_stats('contrast', contrast)
    add_stats('zcr', zcr)
    add_stats('centroid', centroid)
    
    return features