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


