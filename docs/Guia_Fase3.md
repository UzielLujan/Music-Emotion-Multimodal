
# Guía de Ejecución: Fase 3 (Modelado y Entrenamiento)
**Fecha**: Diciembre 2025 
**Contexto**: Construcción del "Stacking Ensemble" Multimodal (Audio + Texto). 
**Entradas**: Datos procesados en Fase 2 (.npy Embeddings, .parquet Features, .npy Espectrogramas).

---

## 1. Objetivo General
Entrenar una arquitectura de Stacking Ensemble compuesta por 4 modelos base (Expertos) y 1 meta-modelo (Juez), logrando superar el baseline aleatorio (25% accuracy) y el baseline unimodal.
La Arquitectura "4 Expertos + 1 Juez"
1. Experto Texto 2D (Tú): BERT Embeddings $\rightarrow$ CNN-LSTM.Experto Texto 1D (Tú): TF-IDF $\rightarrow$ 
2. Selección Chi² $\rightarrow$ DNN.Experto Audio 2D (Uziel): Espectrogramas Mel $\rightarrow$ CNN (VGG-ish).
3. Experto Audio 1D (Uziel): Features Estadísticos (OpenSMILE/Librosa) $\rightarrow$ DNN.
4. Meta-Learner (Fusión): Predicciones de los 4 expertos $\rightarrow$ Regresión Logística / MLP.
---

## 2. Estructura de Trabajo (src/Models)
Para mantener el orden, crearemos scripts de entrenamiento dedicados. No haremos un solo script gigante.

```bash
src/
├── Models/
│   ├── __init__.py
│   ├── utils.py               <-- Funciones compartidas (Carga de datos, Splitter)
│   ├── train_text_2d.py       <-- Script Principal Brenda (Deep Learning)
│   ├── train_text_1d.py       <-- Script Secundario Brenda (Machine Learning + DNN)
│   ├── train_audio_2d.py      <-- Script Uziel
│   ├── train_audio_1d.py      <-- Script Uziel
│   └── train_fusion.py        <-- Script Final (Stacking)
└── ...
```
---

## 3. Plan Detallado por Módulos
**Paso 0: Preparación del Terreno (utils.py)**

Antes de entrenar, necesitamos una forma estándar de cargar los datos que alinee los IDs.

- [ ] Dataset Loader: Crear una clase que cargue el aligned_metadata.csv y busque los archivos .npy o .parquet correspondientes usando el spotify_id.

- [ ] Stratified Split: Implementar una función que divida en Train/Validation/Test (ej. 70/15/15) asegurando que los 4 cuadrantes estén balanceados en todos los sets.
---
### Paso 1: Rama de Texto (Responsable: Brenda)
- A. Modelo 2D: Semántica Profunda (train_text_2d.py)
    - Input: embeddings_2d.npy (Tensor: $N \times 512 \times 768$).
    - Arquitectura:Entrada (Input Layer).
        - Capa Convolucional (Conv1D): Para detectar n-gramas locales (frases clave).
        - Max Pooling: Para reducir dimensión.
        - LSTM (Opcional/Paper): Para capturar dependencias a largo plazo en la narrativa.
        - Dense Layer + Dropout: Clasificación.
    * Reto Técnico: Manejar la memoria de la GPU con tensores grandes. Usar DataLoaders de PyTorch.

- B. Modelo 1D: Palabras Clave (train_text_1d.py)
    - Input: features_text_1d.parquet (TF-IDF amplio, ~3000 dims).
    - Pre-procesamiento (IN-LOOP):
        - Aquí entra el Chi-Cuadrado ($\chi^2$): Dentro del script, aplicar SelectKBest(chi2, k=500) usando solo el set de entrenamiento
    - Arquitectura: DNN (Red Densa Simple).
        - Capas: Input(500) $\rightarrow$ Dense(256) $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Dense(4).
    - Objetivo: Capturar palabras específicas ("party", "pain") que discriminan fuertemente.

### Paso 2: Rama de Audio (Responsable: Uziel)

Debe seguir la misma lógica que texto.

- Modelo 2D (Espectrogramas): Usar una CNN 2D estándar (como una ResNet18 simplificada o una VGG-16 pequeña) para analizar la imagen del sonido.

- Modelo 1D (Features): Una DNN simple para procesar vectores numéricos de timbre y ritmo.

### Paso 3: Fusión y Stacking (train_fusion.py)
Una vez que los 4 modelos anteriores estén entrenados y guardados (.pth o .h5):
1. Congelar: Poner los 4 modelos en modo eval().
2. Extracción de Probabilidades: Pasar todo el dataset por los modelos.
    - Output: Un vector de 16 dimensiones por canción (4 emociones $\times$ 4 modelos).

3. Entrenamiento del Meta-Model:
    - Usar esas 16 dimensiones como entrada (X_meta).
    - Etiquetas reales como objetivo (y_true).
    - Entrenar una Regresión Logística o una red muy pequeña para encontrar el peso ideal de cada experto.
--- 

### 4. Estándares Técnicos y Métricas
Hiperparámetros Sugeridos
- Optimizador: AdamW (suele funcionar mejor que Adam estándar).
- Learning Rate:

    - BERT/CNN: Bajo (ej. 1e-4 o 5e-5).

    - DNN: Estándar (ej. 1e-3).

- Batch Size: 16 o 32 (dependiendo de la VRAM).

- Early Stopping: Obligatorio. Si la val_loss no mejora en 5 épocas, detener entrenamiento y guardar el mejor modelo.

**Métricas de Evaluación** 

Para cada modelo (y el final), debemos reportar:
- Accuracy Global.
- Matriz de Confusión: Para ver si confunde "Angry" con "Happy" (High Arousal) o "Sad" con "Relaxed" (Low Arousal).
- F1-Score por Clase: Importante si hay desbalance leve.

### 5. Checklist de Entregables (Fase 3)

Al finalizar esta fase, debemos tener en la carpeta models/saved/:

- [ ] text_2d_cnn.pth (Modelo pesado de Texto)

- [ ] text_1d_dnn.pth (Modelo ligero de Texto)

- [ ] audio_2d_cnn.pth (Modelo pesado de Audio)

- [ ] audio_1d_dnn.pth (Modelo ligero de Audio)

- [ ] fusion_meta_learner.pkl (El cerebro final)

- [ ] reporte_resultados.md (Tabla comparativa de accuracies).