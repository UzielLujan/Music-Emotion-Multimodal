# Anexo Técnico: Definición de Arquitectura de Modelos (Alineación con Paper)


## 1. Detalle de los Expertos (Fase 3-A)
Siguiendo la arquitectura del paper *A Multimodal Music Emotion Classification Method Based on Multifeature Combined Network Classifier*, entrenaremos **2 Redes Combinadas** (Combined Networks) en lugar de 4 modelos independientes. La fusión de características (1D + 2D) ocurre **dentro** de la red neuronal.

**🧠 A. Red Combinada de Texto (Text Network): Modelo Unimodal de Texto**
* **Responsable:** Brenda
* **Script:** `train_text_network.py`
* **Entradas:**
    * `Texto 2D`: Tensor de Embeddings (BERT/DistilBERT).
    * `Texto 1D`: Vector TF-IDF.
* **Arquitectura Interna:**
    1.  Rama 2D: Procesada por CNN o BiGRU para extraer patrones secuenciales.
    2.  Rama 1D: Procesada por DNN (Capas densas) para extraer patrones estadísticos.
    3.  **Concatenación:** Se unen los vectores latentes de ambas ramas.
    4.  **Clasificador:** Capas densas finales $\rightarrow$ Softmax (4 emociones).
* **Guardado:** `src/saved_models/text_network_best.pth`.

```bash
graph TD
    Emb[Embeddings 2D] --> CNN2[CNN + BiLSTM]
    TF[TF-IDF 1D] --> DNN2[DNN]
    CNN2 --> |Vector C| Concat2((Concatenación))
    DNN2 --> |Vector D| Concat2
    Concat2 --> Softmax2[Capa Softmax]
    Softmax2 --> ProbT[4 Probabilidades Texto]

```


**🔊 B. Red Combinada de Audio (Audio Network): Modelo Unimodal de Audio**
* **Responsable:** Uziel
* **Script:** `train_audio_network.py`
* **Entradas:**
    * `Audio 2D`: Tensor de Espectrogramas (Imagen).
    * `Audio 1D`: Vector de HSFs (Estadísticos).
* **Arquitectura Interna:**
    1.  Rama 2D: Procesada por CNN (tipo VGG/ResNet) para patrones espectrales.
    2.  Rama 1D: Procesada por DNN.
    3.  **Concatenación:** Fusión de características auditivas.
    4.  **Clasificador:** Capas densas finales $\rightarrow$ Softmax (4 emociones).
* **Guardado:** `src/saved_models/audio_network_best.pth`.
```bash
graph TD
    Spec[Espectrograma 2D] --> CNN[CNN + BiLSTM]
    HSF[Features 1D] --> DNN[DNN]
    CNN --> |Vector A| Concat((Concatenación))
    DNN --> |Vector B| Concat
    Concat --> Softmax[Capa Softmax]
    Softmax --> ProbA[4 Probabilidades Audio]
```

---

## 2. La Fusión / Stacking (Fase 3-B)
Una vez entrenados los dos expertos, congelamos sus pesos y entrenamos al "Juez".

**Paso Intermedio: Generación de Meta-Features**
* **Script:** `generate_meta_features.py`
* **Proceso:**
    1.  Cargar los 2 modelos (.pth) en modo `eval()`.
    2.  Pasar todo el dataset (Train, Val, Test).
    3.  Obtener las **Probabilidades** (Softmax) de cada red.
        * Salida Audio: 4 probabilidades.
        * Salida Texto: 4 probabilidades.
    4.  **Concatenar:** Generar un vector de **8 dimensiones** (4+4) por canción.
    5.  Guardar como `meta_dataset.csv`.


**Entrenamiento del Juez (Meta-Learner)**
* **Script:** `train_fusion.py`
* **Input:** 8 meta-features (probabilidades).
* **Modelo:** Regresión Logística o MLP ligero.
* **Lógica:** Aprende a ponderar qué modalidad es más confiable para cada tipo de emoción (ej. "Para enojo, confía más en el audio; para relajación, confía más en el texto").

```bash
graph TD
    ProbA[4 Probs Audio] --> Meta[Meta-Learner]
    ProbT[4 Probs Texto] --> Meta
    Meta --> Final[Predicción Final]
```