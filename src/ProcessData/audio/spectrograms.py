import numpy as np
import librosa

def extract_melspectrogram(y, sr, n_mels=128, target_shape=(128, 646)):
    """
    Genera un Mel-Spectrogram en escala Logarítmica (dB).
    Devuelve: Matriz NumPy (n_mels, time_steps).
    """
    # 1. Calcular Mel-Spectrogram
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    
    # 2. Convertir a dB (Escala Logarítmica es mejor para percepción humana/redes)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    
    # 3. Normalización/Fix Shape (Opcional pero recomendado para CNNs)
    # Si el audio dura exactamente 15s a 22050Hz con hop 512, el ancho es ~646.
    # Si queremos asegurar un tamaño fijo para la CNN, recortamos o hacemos padding.
    
    current_width = melspec_db.shape[1]
    target_width = target_shape[1]
    
    if current_width < target_width:
        # Padding con el valor mínimo (silencio)
        pad_width = target_width - current_width
        melspec_db = np.pad(melspec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=melspec_db.min())
    else:
        # Recorte
        melspec_db = melspec_db[:, :target_width]
        
    return melspec_db