# Bitácora técnica: Clasificación multimodal de emociones en música
**Artículo base:** *Multi-Modality in Music: Predicting Emotion in Music from High-Level Audio Features and Lyrics*

**Proyecto MIR:** *Análisis de emoción y sentimiento en música desde una perspectiva multimodal*

## Resumen

### Objetivo Principal
El objetivo de este artículo es comparar el rendimiento de un enfoque multimodal (audio + letras) frente a uno unimodal (solo audio) para el reconocimiento de emociones en la música (MER).

### Metodología
Utilizaron características de audio (API Spotify) y letras (Vader, TFIDF) para predecir valencia y activación en el dataset DMDD con 4 modelos de regresión.

### Descubrimientos Clave
* El enfoque multimodal (combinando audio y letras) superó al enfoque de solo audio.
* Esta mejora fue específicamente notoria al predecir la valencia.
* Solo 5 de las 11 características de audio de Spotify resultaron ser influyentes.

---

## 1. Motivación y Objetivo del Estudio

### ¿Qué problema se aborda?
Determinar qué características, como las de la API de Spotify, pueden predecir eficazmente la emoción en una canción.

### ¿Por qué es relevante para MIR?
El estudio se enmarca en el Reconocimiento de Emociones Musicales (MER), un área de creciente interés para aplicaciones como los sistemas de recomendación musical.

### ¿Qué se propone como solución?
Probar la hipótesis de que un modelo multimodal (audio + letras) supera a uno unimodal. Se usaron 4 modelos de regresión:
* Linear Regression
* Random Forest Regression
* Support Vector Regression (SVR)
* Multilayer Perceptron (MLP) Regressor

---

## 2. Trabajos Relacionados

Se detalla la evolución de los modelos de regresión para predecir valencia (V) y activación (A).

* **2008: Y.-H. Yang et al.:** Primeros en regresión VA. Usaron SVR con audio de bajo nivel, alcanzando $R^2$ de 0.282 (V) y 0.538 (A).
* **2014: Soleymani et al.:** Usaron SVR y MLP con *features* supra-segmentales, alcanzando $R^2$ de 0.42 (V) y 0.52 (A).
* **2016: Bai et al.:** Usaron RFR con 548 *features* de audio, alcanzando $R^2$ de 0.293 (V) y 0.625 (A).
* **2017: Jeon et al.:** Presentaron un modelo bi-modal (audio+letras) con Deep Learning (DL) y fusión media. Superó a los unimodales y encontró que las letras predicen mejor la emoción que el audio.
* **2018: Delbouys et al.:** Compararon *feature engineering* vs. DL en DMDD. La fusión multimodal a nivel medio superó a ambos modelos unimodales.
* **2018-2019:** Se menciona el uso de MLR (X. Yang et al., Vatolkin & Nagathil) y MLP (Bhattarai & Lee).

### 2.1 Letras (Lyrics) como Métrica de Predicción
* Pioneros (Yang & Lee, 2004) usaron Bag-of-Words (BoW).
* Trabajos posteriores (Hu et al.) expandieron a n-gramas y ANEW.
* Hallazgo clave: Las letras suelen superar al audio, pero la combinación de ambas modalidades logra los mejores resultados.
* BoW, TF-IDF y *word embeddings* son los métodos de representación de texto más usados.
### 2.2 Características de Alto Nivel (Higher-Level Features)
* Se describe la relación entre *features* y emoción: *Arousal* (tempo, pitch, loudness) y *Valencia* (modo, armonía, ritmo).
* Panda et al. (2021) evaluaron 12 *features* de Spotify y encontraron que solo 3 eran relevantes: energía, valencia y acousticness.

### 2.3 Spotify
* Es el servicio de *streaming* más grande, con una API que da acceso a canciones y *features* anotadas.
* El proceso de anotación es automatizado (desarrollado por The Echo Nest) pero no es público.
---

## 3. Datos y Metodología

### 3.1 Dataset y "Ground Truth" (Valor Real)
* **Dataset Principal:** Deezer Mood Detection Dataset (DMDD), basado en el Million Song Dataset y tags de Last.fm.
* **Anotaciones (Ground Truth):** Puntuaciones VA obtenidas aplicando el léxico XANEW a los tags de Last.fm (no son de Spotify).
### 3.2 Modalidad Acústica (Audio)
* **Fuente:** API de Spotify (11 *features* originales, 23 con *dummies*).
* **Reducción:** 13,445 canciones del DMDD encontradas en Spotify.

### 3.3 Modalidad Textual (Letras)
* **Fuente:** Letras extraídas de genius.com.
* **Dataset Final:** 12,471 canciones con audio y letras disponibles.
* **División:** 60% Entrenamiento, 20% Validación, 20% Prueba.

### 3.4 Extracción y Selección de Características

### 3.4 Extracción y Selección de Características
El objetivo de esta fase fue optimizar el conjunto de predictores, reteniendo únicamente las características más informativas de cada modalidad antes de la fusión. Este proceso de selección se aplicó de forma independiente a cada modalidad, utilizando en ambos casos las puntuaciones de **Valencia y Activación (VA)** como las variables respuesta.

#### Selección de Características de Audio

* **Método:** Análisis de significancia estadística mediante Regresión Lineal Múltiple (MLR).
* **Proceso:** Se entrenó un modelo MLR utilizando las 23 características de audio (incluyendo las *dummy variables* para `key`) para predecir las puntuaciones VA. Se examinaron los coeficientes de regresión y sus valores p.
* **Resultado:** Se retuvieron únicamente las 5 características que demostraron tener un **impacto estadísticamente significativo** (p < 0.05) en la predicción: `danceability`, `energy`, `instrumentalness`, `valence` (de Spotify) y `mode`.

#### Selección de Características de Letras

* **Método:** Evaluación de rendimiento en el conjunto de validación.
* **Proceso:** Inicialmente se generaron tres conjuntos de características textuales (Vader Sentiment, TF-IDF y XANEW). Se entrenaron y evaluaron modelos con diferentes combinaciones de estos conjuntos.
* **Resultado:** La combinación que obtuvo el **mejor rendimiento predictivo para las puntuaciones VA** en el conjunto de validación fue la compuesta por **Vader Sentiment (4 *features*) + TF-IDF (100 PCs)**. Las características derivadas de XANEW fueron descartadas al no mejorar el rendimiento del modelo.

> **Nota:** Vader Sentiment no es un modelo de *Deep Learning*, sino que utiliza un diccionario (léxico) donde cada palabra tiene una puntuación de sentimiento preasignada.
---

## 4. Configuración Experimental y Análisis de Características

### 4.1 Modelos y Optimización
Se usaron 4 modelos de regresión (Linear, Random Forest, SVR, MLP) de scikit-learn. Los hiperparámetros se optimizaron con GridSearchCV.

> Nota: ¿Qué es Support Vector Regression (SVR)?

>SVR (Regresión de Vectores de Soporte) es la versión de regresión del modelo de clasificación *Support Vector Machines (SVM)*.

### 4.2 Modelo Base (Baseline)
Se creó un *baseline* MLR para audio-solo, letras-solo y multimodal para comparación.

### 4.3 Análisis de Características (Post-Hoc)
Se usó Recursive Feature Elimination (RFE) para investigar las características más significativas.

---

## 5. Resultados y Hallazgos

La métrica utilizada para evaluar el poder predictivo fue el $R^2$ (R-cuadrado).

### 5.1 Hallazgos Clave
* **Valencia:** La multimodalidad obtiene el R² más alto y supera significativamente a los enfoques unimodales.
* **Activación (Arousal):** Los resultados son complicados. Solo audio (unimodal) funcionó mejor en 3 de los 4 modelos. Las letras por sí solas son inútiles (R² casi 0) para predecir *arousal*.
* **Inferencia:** Añadir las *features* de letras (inútiles para *arousal*) puede interferir e introducir error al modelo multimodal de *arousal*.
* **Rendimiento de Modelos:** Ningún algoritmo fue el mejor; SVR fue consistentemente el peor. Los modelos complejos (MLP, RFR) a menudo no superaron al *baseline* simple (MLR).

### 5.2 Análisis de Características
* RFE mostró que `valence` (Spotify) y *features* de letras (Vader, TF-IDF) eran importantes para ambas tareas.
* `danceability` fue específica para valencia.
* `energy` y `negative score` (Vader) fueron específicas para *arousal*.

### 5.3 Rendimiento de Características Seleccionadas
* **Hallazgo Clave:** Usar el subconjunto de *features* seleccionadas (5 de audio, 104 de letras) dio un impulso pequeño pero consistente al R² en todas las modalidades.
* **Conclusión:** Los modelos finales del estudio se entrenaron usando solo este conjunto seleccionado de características.


## 5. Discusión de Resultados

Los resultados obtenidos muestran dos conclusiones principales:

1. **Las características unimodales** (tanto de letras como de audio) **predicen razonablemente la valencia**, pero el **modelo multimodal supera a cualquiera de las modalidades individuales**.  
2. **La predicción del arousal** es más compleja cuando se utilizan letras, ya que los **atributos acústicos por sí solos** logran un rendimiento similar al modelo combinado.

### 5.1 Efecto de las características

**Valencia:**
- Las variables **danceability**, **energy**, **valence (Spotify)**, **mode**, **acousticness**, **liveness** y el **sentimiento compuesto** contribuyen positivamente.  
- **Danceability**, **energy** y **valence** son las más influyentes.  
- La relación positiva entre **modo mayor** y **valencia** concuerda con estudios previos sobre la asociación entre tonalidad y emociones positivas.  
- **Energy** también está fuertemente relacionada con **arousal**, apoyando su vínculo con la percepción de intensidad sonora.

**Arousal:**
- Las características **energy**, **speechiness**, **mode** y **valence** tienen efecto positivo.  
- **Danceability** e **instrumentalness** presentan coeficientes negativos.  
- **Energy** es el predictor dominante de arousal.  
- Las características **XANEW** fueron descartadas por baja capacidad predictiva, mientras que **TF-IDF** resultó más efectiva para capturar el vocabulario emocional en letras.

### 5.2 Modelos Multimodales vs. Unimodales

**Valencia:**
- El enfoque **multimodal** supera claramente a los modelos unimodales.  
- Las características de **audio y texto son complementarias** en la predicción de valencia.  
- El predominio del audio coincide con investigaciones previas que señalan que los oyentes se basan más en la música que en las letras para percibir emociones.

**Arousal:**
- No se observa una mejora consistente al combinar modalidades.  
- En la mayoría de los casos, los **modelos basados solo en audio** ofrecen mejor rendimiento.  
- Esto sugiere que el **arousal depende principalmente de propiedades rítmicas y de tempo**, mientras que las letras aportan poca información útil o incluso ruido.

---
