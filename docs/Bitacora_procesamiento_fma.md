Bitácora de Procesamiento — FMA Medium (Pipeline v1 · 2025)

Autor: Uziel Luján (Uzi)

# Bitácora de Procesamiento — FMA Medium (Pipeline v1 · 2025)

**Autor:** Uziel Luján (Uzi)  
**Proyecto:** Music Emotion Multimodal

---

## 1. Objetivo de la bitácora

Documentar los pasos realizados para reconstruir, depurar y ejecutar el pipeline de procesamiento del dataset FMA Medium, originalmente desarrollado por Brenda, con el fin de obtener un conjunto de datos multimodal (audio + letras) y etiquetado por valence/arousal -> (angry,sad,happy,relaxed) para modelado posterior.

Esta bitácora resume lo logrado, lo corregido y lo pendiente, sin saturar detalles técnicos.

---

## 2. Reconstrucción del entorno y estructura del proyecto

- Se corrigió el manejo de rutas usando **Pathlib** y centralizando la definición de rutas en `main.py`.
- Se creó un environment Conda funcional.

### Estructura del proyecto (simplificada)

```text
Music-Emotion-Multimodal/
│
├── data/
│   ├── raw/ (fma_medium, fma_metadata, versiones .zip)
│   ├── processed/ (lyrics.csv)
│   ├── spectrograms_medium/
│   └── features/ (fma_labeled.csv)
│
├── src/
│   └── ExtractData1.0/
│   ... Resto de scripts con la estructura modular definida
```
### Rutas definidas en `main.py`:
```text
PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
META = RAW / "fma_metadata"
AUDIO = RAW / "fma_medium"
SPEC_DIR = DATA_DIR / "spectrograms_medium"
FEATURES_DIR = DATA_DIR / "features"
LYRICS_PATH = DATA_DIR / "processed" / "lyrics.csv"
```

Esto reemplaza rutas absolutas y elimina dependencias locales del entorno original de Brenda.

---

## 3. Exploración del dataset FMA (metadata + audio)

- **Tracks totales en metadata:** 106,574
- **MP3 reales detectados (subset medium):** 25,000

Se generó un script exploratorio para:

1. Cargar `tracks.csv` correctamente
2. Analizar distribución de `language_code`
3. Validar estructura del dataset
4. Verificar disponibilidad real de archivos MP3
5. Detectar posibles corruptos

---

## 4. Idioma inglés — Reconstrucción completa

Brenda usaba:

```python
language_code == "en"
```

Esto daba un subset teórico de:

- 14,255 tracks etiquetados como inglés
- pero solo 3,887 con MP3 real

**Mejora:** heurística por título

- Se detectaron 2,769 títulos con patrones de inglés en los títulos de las canciones.

**Resultado final:**

| Etapa                  | Tracks |
|------------------------|--------|
| Inglés oficial + MP3   | 3,887  |
| Rescatados desde NaN   | 2,283  |
| **TOTAL inglés real con mp3**  | 6,170  |

Se consigió pasar de 3,887 respecto al pipeline original a 6,170.

---

## 5. Valence & Arousal (Echonest)

Corrección crítica:

Brenda usaba energy como sustituto de arousal:

```python
arousal = ("echonest", "audio_features", "energy")
```

Este fue aplicado correctamente.

- **Tracks con VA:** 13,129
- **Intersección con inglés_real:**

```python
english_real ∩ VA = 2,677 tracks
```

Este número supera ampliamente los ~1200 originales del pipeline de Brenda.

---

## 6. Extracción de características acústicas (MFCC)

- Se procesaron 6,170 candidatos iniciales.
- Algunas pistas no pudieron cargarse (MP3 corruptos).

**Resultados:**

- MFCC extraídos correctamente
- Features unidas con VA/labels
- 1,889 tracks finales con features completas (vs. ~1,267 de Brenda)

Este es el número base para modelado audio-only.

---

## 7. Espectrogramas

- Generación parcial: 2,864 espectrogramas generados
- El proceso consumió RAM intensivamente (`matplotlib` + `librosa`)
- Se recomienda batch-processing en pipeline v2
- No son necesarios para modelado inmediato mañana.

---

## 8. Predicción de emociones (labels finales)

Se utilizó:

```python
emotion = map(valence, arousal)
```

Se generó:

```text
data/features/fma_labeled.csv   (1,889 tracks)
```

Este archivo se convierte en el dataset base hasta ahora.

---

## 9. Descarga de letras (Genius API)

- Se implementó `fetch_lyrics_v2`
- Cache incremental
- Protección contra caídas
- Manejo de errores
- Descarga exclusiva para `track_ids` finales del modelo

**Pendiente:** corregir el mapeo `artist/title → track_id` (aparentemente debido a la estructura multiindex de `tracks.csv`).

Esta es la única etapa NO finalizada, pero ya está perfectamente diagnosticada, según jaja.

---

## 10. Merge final (features + lyrics)

- El merge funcionó estructuralmente
- Actualmente produce 0 letras unidas
- **Causa:** mismatch entre `track_ids` descargados y metadatos reales
- **Solución planificada:** reconstrucción correcta del mapeo `metadata → track_id` (pipeline v2)

---

## 11. Estado final del pipeline v1 (HOY)

- FMA Medium descargado y extraído
- Rutas corregidas
- English_real generado: 6,170 tracks
- VA intersección: 2,677 tracks
- Audio features finales: 1,889 tracks
- Espectrogramas generados: 2,864 imágenes
- Lyrics pipeline implementado (pendiente mapeo)
- Dataset final para modelado: `fma_labeled.csv`

El pipeline funciona hasta la etapa de dataset para modelado audio-only, es decir, extracción de features y etiquetas emocionales, lo suficiente para comenzar fase 2.

---

## 12. Próximos pasos (pipeline v2)
0. Planificar pipeline v2 modular como una mejora estructurada del actual
1. Reconstruir extracción de artista/título de `tracks.csv` correctamente
2. Corregir `fetch_lyrics` usando el mapping real
3. Fusionar letras + features
4. Implementar fallback textual (usar título si no hay letra)
5. Procesamiento por lotes de espectrogramas
6. Documentar pipeline v2
7. Iniciar modelado multimodal
8. Evaluar arquitectura CNN-LSTM + transformadores para texto

---

## 13. Conclusión

El trabajo realizado hoy:

- desbloquea el pipeline completo,
- supera ampliamente las limitaciones del original,
- deja un dataset robusto para comenzar modelado,
- y sienta las bases del pipeline v2 mejorado.

Esta bitácora documenta claramente los avances del proyecto hasta ahora.