# Plan del Proyecto — Sistema Multimodal para Clasificación de Emociones en Música

## 1. Introducción

Este proyecto desarrolla un sistema de clasificación de emociones en música desde una perspectiva **multimodal**, integrando información **acústica** y **textual**.  
La metodología se basa en arquitecturas neurales profundas **por modalidad** y en una **fusión final** mediante *stacking*, un enfoque que ha mostrado mejoras frente a fusiones tempranaso combinaciones simples.

Este documento resume las decisiones de diseño acordadas y las ideas esenciales extraídas de los artículos base que guían el desarrollo del sistema.


## 2. Motivación del enfoque multimodal

La emoción musical depende tanto del contenido sonoro (timbre, energía, ritmo) como del contenido semántico en las letras.  
Cada modalidad aporta información complementaria y, al combinarlas mediante modelos profundos y fusión supervisada, se obtiene una representación emocional más robusta.

Por ello, el proyecto adopta un pipeline que:
- extrae representaciones 2D y 1D del audio y texto,
- entrena modelos unimodales independientes,
- y combina sus salidas mediante *stacking ensemble learning*.


## 3. Relación con los artículos base

El diseño metodológico se inspira en tres trabajos clave:

1. **“A Multimodal Music Emotion Classification Method Based on Multifeature Combined 
   Network Classifier”**  
   → Modelo base: características 2D/1D, CNN–BiLSTM/DNN por modalidad y fusión por stacking.

2. **“Music Emotion Classification Method Based on Deep Learning and Explicit Sparse 
   Attention Network”**  
   → Ideas adicionales:  
     - evaluación comparativa de audio original / voz / fondo,  
     - uso potencial de **atención dispersa**,  
     - y la noción de **votación por bloques** como mejora opcional.

3. **“Multi-Modality in Music: Predicting Emotion in Music from High-Level Audio 
   Features and Lyrics”**  
   → Relación entre valence/arousal, letras y rasgos altos; fundamento para incorporar
     modelos modernos como **BERT**.

Estas fuentes definen los elementos esenciales del pipeline y las mejoras que se
proponen integrar en este proyecto.


## 4. Preprocesamiento del Dataset

El preprocesamiento organiza y normaliza los datos provenientes de las modalidades
acústica y textual antes de la extracción de características. Esta etapa garantiza que
tanto el audio como las letras puedan integrarse coherentemente dentro del pipeline
multimodal.

### 4.1 Segmentación del audio
Cada pista se divide en segmentos de **15 segundos**, siguiendo la evidencia del
artículo base que muestra que intervalos cortos capturan mejor patrones emocionales
locales. Esto permite generar múltiples muestras por canción y facilita el análisis
por bloques.

### 4.2 Separación de fuentes (voz y fondo)
Con el fin de evaluar qué componentes del audio aportan más a la clasificación,
se consideran tres variantes por segmento:

- **Audio original**  
- **Pista instrumental** (solo fondo musical)  
- **Pista vocal**

Los artículos analizados reportan que la pista instrumental tiende a ofrecer un
desempeño más estable, mientras que la pista vocal puede introducir mayor ruido
emocional. Esta comparación se conserva como parte del diseño experimental.

### 4.3 Limpieza y normalización de letras
Las letras se depuran para eliminar ruido estructural (HTML, anotaciones innecesarias,
coros repetidos) y se normalizan para facilitar su tokenización y posterior
transformación en embeddings o vectores estadísticos.

### 4.4 Emparejamiento audio–texto
Cada segmento de audio se vincula con su letra correspondiente usando metadatos como
título, artista y track ID. Este alineamiento garantiza consistencia entre modalidades
durante el entrenamiento unimodal y la fusión multimodal.

### 4.5 Etiquetado emocional con el modelo Arousal–Valence
El etiquetado se organiza bajo el modelo bidimensional de **Arousal–Valence (Russell)**.  
Este esquema permite mapear cada canción (o segmento) a cuatro categorías emocionales:

- **Q1:** valence +, arousal + (Happy / Energetic)  
- **Q2:** valence –, arousal + (Angry / Tense)  
- **Q3:** valence –, arousal – (Sad)  
- **Q4:** valence +, arousal – (Calm / Relaxed)

Este modelo sirve como base para la clasificación supervisada y para organizar los
experimentos posteriores.


## 5. Extracción de Características

La extracción de características transforma las señales acústicas y textuales en
representaciones numéricas aptas para el entrenamiento de los modelos profundos.
Este proyecto adopta un enfoque dual por modalidad, considerando tanto
representaciones **2D** como **1D**.

---

### 5.1 Características acústicas (Audio)

#### 5.1.1 Representaciones 2D: espectrogramas
Los segmentos de audio se convierten en espectrogramas mediante la 
**Transformada de Fourier de corto tiempo (STFT)**.  
Esta representación tiempo–frecuencia actúa como una “imagen acústica” y permite:

- capturar patrones espectrales asociados al timbre y textura,
- preservar variaciones temporales relevantes para la emoción,
- utilizar arquitecturas **CNN–BiLSTM** para modelar simultáneamente espacio y tiempo.

Los espectrogramas son consistentes con el enfoque del Artículo 1, que propone
esquemas híbridos para modelar música en el dominio 2D.

#### 5.1.2 Representaciones 1D: LLDs y HSFs
Además de los espectrogramas, se extraen **Low-Level Descriptors (LLDs)** como:

- MFCCs  
- Zero-Crossing Rate  
- Spectral centroid, roll-off y bandwidth  
- Chroma  

De estos descriptores se generan estadísticas agregadas
(**High-Level Statistical Features, HSFs**) como media, varianza y máximo,
obteniendo vectores 1D compactos.

Estas características permiten capturar información global y estable del segmento
acústico, y sirven como entrada para un **modelo DNN**, complementando la información
2D del espectrograma.

---

### 5.2 Características textuales (Letras)

#### 5.2.1 Representaciones 2D: embeddings distribucionales
El contenido semántico de las letras se modela como una matriz 2D donde:

- cada fila corresponde a una palabra,
- y cada columna a un vector de embedding.

El Artículo 1 utiliza **Word2Vec**, mientras que este proyecto considera además
la incorporación de **BERT**, un modelo contextualizado que captura relaciones
profundas entre palabras.  
BERT mejora la expresividad semántica y puede aportar mayor discriminación emocional.

Estas matrices se procesan mediante **CNN–BiLSTM**, análogo a las características
acústicas 2D.

#### 5.2.2 Representaciones 1D: TF–IDF y Chi-cuadrado
Como complemento de los embeddings, se consideran representaciones estadísticas:

- **TF–IDF** para capturar la relevancia de términos frecuentes o distintivos,
- **Chi-cuadrado (CHI)** para seleccionar palabras asociadas a categorías emocionales.

Estas representaciones se utilizan como entrada para una **DNN**, aportando señales
discriminativas adicionales que enriquecen el modelo textual.

---

### 5.3 Resumen de la extracción multimodal
En ambas modalidades (audio y texto), se utilizan dos tipos de representaciones:

- **2D** → procesadas por una red **CNN–BiLSTM**  
- **1D** → procesadas por una **DNN**

Esta dualidad permite capturar tanto patrones locales como información global,
proporcionando dos vistas complementarias por modalidad que se integrarán en las
etapas posteriores del pipeline.

## 6. Modelado Unimodal

Cada modalidad (audio y texto) cuenta con su propio modelo especializado.  
El objetivo es capturar patrones particulares de cada dominio antes de integrarlos
mediante fusión multimodal.  
Ambos modelos comparten una estructura similar basada en dos flujos paralelos:

- un flujo **2D**, procesado con **CNN–BiLSTM**,  
- y un flujo **1D**, procesado con una **DNN**.

Las salidas de ambos flujos se combinan internamente para producir una
**predicción unimodal** por emoción.

---

### 6.1 Modelo de Audio

El modelo acústico recibe:

- un **espectrograma 2D** (imagen tiempo–frecuencia), y  
- un vector **1D** con HSFs derivados de LLDs.

Su funcionamiento se basa en:

1. **CNN–BiLSTM (rama 2D)**  
   - La CNN captura patrones espectrales como timbre, brillo o textura sonora.  
   - La BiLSTM modela variaciones temporales del segmento.  

2. **DNN (rama 1D)**  
   - Procesa HSFs que resumen información global del segmento.  

3. **Fusión interna**  
   - Las salidas de ambas ramas se combinan mediante capas densas.  
   - Se obtiene un vector softmax que representa la **predicción unimodal de audio**.

Este diseño está basado directamente en la arquitectura del Artículo 1.

*(En la presentación, esta sección se acompaña con la figura `Audio input`.)*

---

### 6.2 Modelo de Texto

El modelo textual opera de forma análoga con:

- una representación **2D**: matriz de embeddings (Word2Vec o BERT),  
- una representación **1D**: vectores TF–IDF o Chi-cuadrado.

Su estructura es:

1. **CNN–BiLSTM (rama 2D)**  
   - Detecta patrones sintácticos y semánticos en la secuencia de palabras.  
   - Si se usa BERT, los embeddings son contextualizados, mejorando la discriminación emocional.

2. **DNN (rama 1D)**  
   - Procesa la relevancia o discriminación estadística de las palabras.

3. **Fusión interna**  
   - Se combinan ambas salidas para obtener la **predicción unimodal de letras**.

Esta arquitectura también proviene del Artículo 1, con la mejora opcional del uso de BERT.

*(En la presentación, esta sección se acompaña con la figura `Lyric input`.)*

---

### 6.3 Rol del modelado unimodal en el sistema final

Las predicciones unimodales (audio-only y lyrics-only) son esenciales para:

- comparar el desempeño por modalidad,  
- analizar cómo contribuye cada dominio emocionalmente,  
- y servir como **entrada** para la etapa de *stacking*, donde se obtiene la
  predicción multimodal final.

De esta manera, los modelos unimodales forman la base estructural del sistema.

## 7. Fusión Multimodal por *Stacking*

La combinación de las modalidades acústica y textual se realiza mediante un esquema de
**stacking ensemble learning**, una forma avanzada de fusión tardía que ha demostrado
mejor desempeño que técnicas tradicionales de *early fusion* o *late fusion* simples.

El enfoque consiste en entrenar primero dos modelos unimodales (audio y texto) y luego
utilizar sus salidas probabilísticas para entrenar un tercer modelo encargado de
integrar ambas fuentes de información.

---

### 7.1 Funcionamiento general

1. **Predicciones unimodales**  
   - Cada modelo produce un vector softmax con la probabilidad de cada clase emocional.  
   - Estas salidas representan la “opinión” de cada modalidad sobre el estado emocional.

2. **Concatenación de salidas**  
   - Los vectores softmax se combinan en un único vector conjunto.  
   - Este vector contiene información complementaria derivada del análisis acústico
     y textual.

3. **Subclasificador**  
   - El vector concatenado alimenta a un modelo final (p. ej. una DNN ligera o un 
     clasificador de *scikit-learn*).  
   - Este subclasificador aprende patrones intermodales que los unimodales no capturan
     por separado.

4. **Predicción final**  
   - La salida del subclasificador constituye la **predicción multimodal definitiva**.

---

### 7.2 Tipo de fusión en el pipeline

El esquema implementado corresponde a **late fusion mejorada**:

- Las modalidades no se mezclan a nivel de características crudas (evitando problemas
  de escala y dimensionalidad).
- Tampoco se hace una simple votación, sino un proceso supervisado que **aprende**
  cómo combinar las salidas unimodales.
- El stacking permite capturar relaciones complementarias entre audio y texto de manera
  flexible y optimizada.

Este tipo de fusión es recomendado en el Artículo 1, al mostrar mejoras significativas
frente a esquemas tradicionales.

---

### 7.3 Extensiones opcionales inspiradas en la literatura

El Artículo 2 introduce dos ideas adicionales que pueden explorarse como mejoras:

- **Atención dispersa (explicit sparse attention)**  
  Un mecanismo que prioriza características relevantes y reduce ruido en el modelo.

- **Votación por bloques**  
  Si se utiliza segmentación múltiple, las predicciones por segmento pueden agregarse
  mediante votación mayoritaria para obtener la emoción final de la canción completa.

Ambas extensiones pueden incorporarse si el tiempo lo permite y podrían mejorar la
interpretabilidad o estabilidad del sistema multimodal.

## 8. Predicción Final de Emoción

La salida final del sistema multimodal proviene del **subclasificador** entrenado mediante
*stacking*, el cual combina las predicciones unimodales de audio y texto para generar un
vector softmax con la probabilidad de cada estado emocional.

---

### 8.1 Flujo de predicción

1. **Entrada del segmento**  
   Cada segmento de audio y su letra asociada se procesan por los modelos unimodales.

2. **Obtención de predicciones unimodales**  
   Ambos modelos producen un vector softmax que refleja su predicción emocional basada
   exclusivamente en su modalidad.

3. **Fusión supervisada**  
   Las salidas unimodales se concatenan y se introducen en el subclasificador, que aprende
   a ponderar y combinar la información de ambas fuentes.

4. **Predicción multimodal**  
   La salida final del subclasificador se interpreta como la emoción detectada para el
   segmento.

---

### 8.2 Voto por bloques (opcional)

Si una canción se procesa en varios segmentos (p. ej. múltiples ventanas de 15 segundos),
el sistema puede generar varias predicciones multimodales por canción.  
En ese caso, se puede aplicar un esquema de **votación mayoritaria**:

- Cada segmento aporta una predicción multimodal.
- La emoción final de la canción se determina por la categoría más votada.

Este mecanismo fue mencionado en la literatura como una herramienta potencial para
capturar estabilidad emocional a nivel global de la obra.

---

### 8.3 Resultado del pipeline

La predicción final integra:
- las características acústicas locales (del espectrograma),
- la información global del audio (HSFs),
- el contenido semántico de la letra (embeddings),
- y su relevancia estadística (TF–IDF / CHI),

logrando un sistema multimodal completo, consistente y basado en evidencia académica.
