import numpy as np
import librosa

def smart_trim(y, sr, duration=15):
    """
    Encuentra y recorta el segmento de 'duration' segundos con mayor energía.
    Input:
        y: Señal de audio completa.
        sr: Sample rate.
        duration: Duración deseada en segundos.
    Output:
        y_trimmed: Segmento recortado.
    """
    target_samples = int(duration * sr)
    
    # Si el audio es más corto que el objetivo, lo devolvemos tal cual (o con padding)
    if len(y) <= target_samples:
        # Opcional: Podríamos hacer padding aquí si fuera necesario
        return librosa.util.fix_length(y, size=target_samples)

    # Calcular la energía RMS en ventanas (frame_length)
    # Hop length determina el paso de la ventana deslizante
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # El RMS nos da la energía por frames. Necesitamos mapear frames a muestras.
    # Queremos encontrar la ventana de frames que cubra 15 segundos.
    frames_per_sec = sr / hop_length
    target_frames = int(duration * frames_per_sec)
    
    # Buscamos el índice donde la suma de energía de los siguientes 'target_frames' es máxima
    # (Convolución simple para suma móvil)
    if len(rms) < target_frames:
        start_frame = 0
    else:
        # Truco rápido: Suma móvil usando convolución
        window = np.ones(target_frames)
        energy_profile = np.convolve(rms, window, mode='valid')
        max_energy_index = np.argmax(energy_profile)
        start_frame = max_energy_index

    # Convertir frame inicial a muestra inicial
    start_sample = start_frame * hop_length
    end_sample = start_sample + target_samples
    
    # Recorte seguro
    y_trimmed = y[start_sample:end_sample]
    
    return y_trimmed