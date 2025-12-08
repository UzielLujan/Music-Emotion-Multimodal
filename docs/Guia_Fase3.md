
# Guía de Ejecución: Fase 3 (Modelado y Entrenamiento)
**Fecha**: Diciembre 2025 
**Contexto**: Construcción del "Stacking Ensemble" Multimodal (Audio + Texto). 
**Entradas**: Datos procesados en Fase 2 (.npy Embeddings, .parquet Features, .npy Espectrogramas).

---

## 1. Arquitectura y Filosofía de Trabajo
No entrenaremos una red gigante "End-to-End". Usaremos la estrategia de Stacking Ensemble.

El Flujo de Trabajo en 2 Etapas:

Etapa de Expertos (Nivel 0): Entrenamos 4 modelos independientemente. Cada uno se vuelve especialista en su dominio. Al terminar, guardamos sus "cerebros" (pesos .pth) y los congelamos.

Etapa de Fusión (Nivel 1): Usamos los 4 modelos congelados para generar predicciones sobre el set de validación. Entrenamos un "Juez" (Meta-Learner) que aprende a ponderar esas predicciones para dar el veredicto final.

## 2. Objetivo General
Entrenar una arquitectura de Stacking Ensemble compuesta por 4 modelos base (Expertos) y 1 meta-modelo (Juez), logrando superar el baseline aleatorio (25% accuracy) y el baseline unimodal.
La Arquitectura "4 Expertos + 1 Juez"
1. Experto Texto 2D (Tú): BERT Embeddings $\rightarrow$ CNN-LSTM.Experto Texto 1D (Tú): TF-IDF $\rightarrow$ 
2. Selección Chi² $\rightarrow$ DNN.Experto Audio 2D (Uziel): Espectrogramas Mel $\rightarrow$ CNN (VGG-ish).
3. Experto Audio 1D (Uziel): Features Estadísticos (OpenSMILE/Librosa) $\rightarrow$ DNN.
4. Meta-Learner (Fusión): Predicciones de los 4 expertos $\rightarrow$ Regresión Logística / MLP.
---

## 3. Estructura de Trabajo (src/Models)
Para mantener el orden, crearemos scripts de entrenamiento dedicados. No haremos un solo script gigante.

```bash
src/
├── Models/
│   ├── __init__.py
│   ├── utils.py               # <--- Carga de datos y Splitter (CRÍTICO)
│   ├── definitions.py         # <--- Clases de las Redes (TextCNN, AudioCNN, etc.)
│   ├── train_text_2d.py       # Script Brenda
│   ├── train_text_1d.py       # Script Brenda
│   ├── train_audio_2d.py      # Script Uziel
│   ├── train_audio_1d.py      # Script Uziel
│   ├── generate_meta_features.py # <--- Script intermedio (Inferencia de expertos)
│   └── train_fusion.py        # Script Final (El Juez)
└── saved_models/              # <--- AQUÍ SE GUARDAN LOS .PTH
    ├── text_2d_best.pth
    ├── text_1d_best.pth
    ├── audio_2d_best.pth
    └── audio_1d_best.pth
```
---

## 📂 Rutas de Datos (Entradas y Salidas)

| Tipo de Dato | Ruta de Archivo | Descripción |
|--------------|------------------|-------------|
| Master Split | `data/processed/splits/master_split.csv` | La Biblia. Dice qué ID es Train, Val o Test. |
| Texto 2D | `data/processed/features_2d/embeddings/embeddings_2d.npy` | Tensores BERT (Input Texto Profundo) |
| Texto 1D | `data/processed/features_1d/features_text_1d.parquet` | TF-IDF (Input Texto Estadístico) |
| Audio 2D | `data/processed/features_2d/spectrograms/specs.npy` | Espectrogramas (Input Audio Profundo) |
| Audio 1D | `data/processed/features_1d/features_audio.csv` | OpenSMILE (Input Audio Estadístico) |
---

## 4. El "Master Split" (Paso 0 - Obligatorio)
Antes de entrenar nada, debemos garantizar que todos los modelos vean las mismas canciones en los mismos grupos.

Crearemos un script src/ProcessData/make_splits.py que genere el archivo master_split.csv.

- Train (70%): Usado para entrenar los 4 expertos.

- Validation (15%): Usado para entrenar al Meta-Learner (Fusion).

- Test (15%): Usado SOLO para la evaluación final y reporte de tesis.

Regla de Oro: Ningún script de entrenamiento hace train_test_split aleatorio. Todos leen el master_split.csv y filtran por ID.

---

## 5. Detalle de los Expertos (Fase 3-A)
En esta fase, los modelos NO se conocen entre sí. Se entrenan y guardan por separado.

**🧠 Rama de Texto (Responsable: Brenda)**
- A. Experto Texto 2D (Deep Learning)
    - Script: train_text_2d.py

    - Entrada: Tensor (Batch, 256, 768) (donde 256 es tiempo, 768 features).

    - Arquitectura (CNN-LSTM):

    - Transposición: Cambiar a (Batch, 768, 256) para que la conv. sea temporal.

    - Conv1D Block: Filtros de tamaño 3, 4 y 5 (detectan n-gramas) + ReLU + MaxPool.

    - LSTM: Procesa la secuencia de características extraídas por la CNN.

    - Clasificador: Dense -> Softmax (4 emociones).

    - Guardado: Al tener la mejor Validation Accuracy, guardar en src/saved_models/text_2d_best.pth.

- B. Experto Texto 1D (Machine Learning)
Script: train_text_1d.py

    - Entrada: Vector TF-IDF (3000 dimensiones).

    - Lógica Interna:

        - Leer master_split.csv.

        - Separar X_train y X_val.

        - Fit Chi²: Aprender las 500 mejores palabras usando solo X_train.

        - Transform: Reducir X_train y X_val a 500 dims.

        - DNN: Entrenar red densa (500 -> 256 -> 64 -> 4).

- Guardado: src/saved_models/text_1d_best.pth.

**Rama de Audio (Responsable: Uziel)**
- Audio 2D: CNN clásica (tipo VGG) sobre imágenes de espectrogramas. Guarda audio_2d_best.pth.

- Audio 1D: DNN sobre features numéricos. Guarda audio_1d_best.pth.


---

## 6. La Fusión / Stacking (Fase 3-B)
Aquí es donde ocurre la magia del "Model Freezing".

Paso Intermedio: Generación de Meta-Features
Script: generate_meta_features.py

1. Cargar los 4 modelos (.pth).

2. Ponerlos en modo evaluación:

```Python
model_text_2d.eval()
for param in model_text_2d.parameters():
    param.requires_grad = False  # <--- CONGELADO ❄️
```

3. Pasar todo el dataset (Train, Val, Test) por los 4 modelos.

4. No queremos la clase final, queremos las probabilidades.

    - Ejemplo salida Texto 2D: [0.1, 0.8, 0.05, 0.05] (Probabilidad de cada emoción).

5. Concatenar los 4 vectores.

    - Total features: 4 modelos x 4 clases = 16 features.

6. Guardar este nuevo dataset pequeño como meta_dataset.csv.
---

**Entrenamiento del Juez** 
Script: train_fusion.py
1. Cargar meta_dataset.csv.
2. Usar las particiones del master_split.csv.
3. Entrenar una Regresión Logística o una Red Neuronal muy simple (Perceptrón).

    - Input: 16 probabilidades.

    - Target: Emoción real.

4. Este modelo aprenderá a decir: "El modelo de Audio miente mucho en canciones tristes, mejor le hago caso al de Texto".


**Métricas de Evaluación** 

Para cada modelo (y el final), debemos reportar:
- Accuracy Global.
- Matriz de Confusión: Para ver si confunde "Angry" con "Happy" (High Arousal) o "Sad" con "Relaxed" (Low Arousal).
- F1-Score por Clase: Importante si hay desbalance leve.

6. Entregables Finales y MétricasPara la tesis, presentaremos una tabla de resultados basada en el Test Set (15%):

| Modelo | Accuracy | F1-Score | Observación |
|--------|----------|----------|-------------|
| Aleatorio | 25.0% | 0.25 | Línea base mínima. |
| Texto 1D (DNN) | ~35–40% | ... | Detecta palabras clave. |
| Texto 2D (CNN) | ~45–50% | ... | Entiende contexto. |
| Audio 1D (DNN) | ... | ... | En proceso de evaluación. |
| Audio 2D (CNN) | ... | ... | En proceso de evaluación. |
| FUSIÓN (Stacking) | ~55–65% | ... | El objetivo final. |


** Checklist de Inicio para Fase 3** 
- [ ] Generar src/ProcessData/make_splits.py y correrlo para tener el master_split.csv.
- [ ] Crear el archivo src/Models/utils.py (Clase Dataset que lee el split maestro).
- [ ] Programar train_text_1d.py y los pesos entrenados (Brenda).
- [ ] Programar train_text_2d.py y los pesos entrenados(Brenda).
- [ ] Programar train_audio_1d.py y los pesos entrenados(Uzi).
- [ ] Programar train_audio_2d.py y los pesos entrenados(Uzi).