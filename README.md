# TomateCNN
**Diagnóstico en tiempo real de 10 enfermedades en hojas de tomate**  
MobileNetV2 + TensorFlow Lite · 2025  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)](https://www.tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green)](https://opencv.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.1%25-brightgreen)](https://github.com/Cristian-Acevedo-C/TomateCNN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live Demo**  
<a href="https://tomatecnn-ganknc3pg2vlhaznlbt8ob.streamlit.app" target="_blank">
  <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Abrir en Streamlit" width="300"/>
</a>

**Autor:** Cristian Acevedo Cifuentes  

## Características clave

- **10 clases**: 9 enfermedades de la hoja de tomate + hoja sana.
- **Modelo ligero**: MobileNetV2 cuantizado a TFLite (~6 MB), ideal para tiempo real.
- **Modo webcam**: Diagnóstico en vivo usando la cámara del PC.
- **Dashboard web (Streamlit)**: Panel para cargar imágenes, revisar métricas y ver ejemplos.
- **Reporte de sesión**: Registro de hojas analizadas, clases predichas y niveles de confianza.
- **Entrenado con datos reales**: Dataset de tomate con más de 11.000 imágenes anotadas.

---

## Resultados reales (11 000 imágenes)

| Métrica                    | Valor      |
|----------------------------|------------|
| **Accuracy global**        | **93.1 %** |
| **F1-score macro**         | **0.93**   |
| Mejor clase                | Virus Mosaico → **99.9 %** |
| Clase más difícil          | Tizón Temprano → **84.5 %** |

---

## Componentes principales

| Archivo                    | Descripción                                           |
|----------------------------|-------------------------------------------------------|
| `detector_tomates.py`      | Detector en tiempo real con webcam (OpenCV)           |
| `panel_gestion.py`         | Dashboard web profesional (Streamlit)                 |
| `procesador_lotes.py`      | Evaluación masiva sobre carpetas                      |
| `generar_matriz_full.py`   | Métricas completas + matriz de confusión              |
| `test_model_info.py`       | Diagnóstico rápido del modelo                         |
| `model.tflite`             | Modelo cuantizado (~6 MB)                             |

---
## Estructura del proyecto

```text
TomateCNN/
├─ panel_gestion.py         # Dashboard web en Streamlit
├─ detector_tomates.py      # Detección en tiempo real con webcam
├─ procesador_lotes.py      # Procesamiento por lotes (carpetas de imágenes)
├─ generar_matriz_full.py   # Cálculo de métricas + matriz de confusión
├─ test_model_info.py       # Info rápida del modelo TFLite
├─ model.tflite             # Modelo entrenado y cuantizado
├─ labels.json              # Mapeo índice → nombre de enfermedad
├─ requirements.txt         # Dependencias para la app
└─ assets/
   └─ tablero_streamlit.jpg # Captura del dashboard web
```
## Requisitos

- **Python**: 3.10 o 3.11 (recomendado)
- **Sistema**: Windows / Linux con webcam (para el detector en vivo)
- **Dependencias**: se instalan con `requirements.txt`

>  Nota  
> El modelo que usa la app está en formato **TFLite (`model.tflite`)**, por lo que **no es necesario instalar TensorFlow completo** para correr el dashboard o el detector en tiempo real.  
> TensorFlow solo se utiliza en la etapa de **entrenamiento** del modelo lo pueden buscar en el repositorio como Entrenamiento_CNN_Tomates.ipynb

## Instalación

```bash
git clone https://github.com/Cristian-Acevedo-C/TomateCNN.git
cd TomateCNN

# (Opcional) Crear entorno virtual
python -m venv .venv
# En Windows:
# .venv\Scripts\activate
# En Linux/Mac:
# source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso rápido

```bash
# 1) Dashboard web (recomendado)
streamlit run panel_gestion.py

# 2) Detector con webcam
python detector_tomates.py

# 3) Procesar una carpeta de imágenes
python procesador_lotes.py
```


## Capturas

## Dashboard web (Streamlit)

[![Dashboard web](assets/assetstablero_streamlit.jpg)](https://tomatecnn-ganknc3pg2vlhaznlbt8ob.streamlit.app/)

**Live Demo**  
<a href="https://tomatecnn-ganknc3pg2vlhaznlbt8ob.streamlit.app" target="_blank">
  <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Abrir en Streamlit" width="360"/>
</a>



### Matriz de confusión (11 000 imágenes)
![Matriz de confusión](matriz_confusion_full.png)

### Detector en tiempo real
*Captura en proceso…*

---

## Instalación

```bash
pip install -r requirements.txt
```

Uso rápido
Bash# Dashboard web (recomendado)
```bash
streamlit run panel_gestion.py
```

# Detector con webcam
```bash
python detector_tomates.py
```

Fuente del dataset
Subconjunto “tomato” del dataset PlantVillage
Hughes & Salathé (2015) · arXiv:1511.08060
Licencia: CC0 Public Domain
Versión usada en Kaggle: https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf

TomateCNN — Probando DL.
Cristian Acevedo Cifuentes · 2025



