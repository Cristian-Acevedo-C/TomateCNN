# TomateCNN

**Diagnóstico en tiempo real de 9 enfermedades en hojas de tomate + hoja sana (10 clases)**  
MobileNetV2 + TensorFlow Lite · 2025

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)](https://www.tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green)](https://opencv.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.1%25-brightgreen)](https://github.com/Cristian-Acevedo-C/TomateCNN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Autor: **Cristian Acevedo Cifuentes**

---

## Resultados reales (11 000 imágenes)

| Métrica                    | Valor      |
|----------------------------|------------|
| **Accuracy global**        | **93.1 %** |
| **F1-score macro**         | **0.93**   |
| Mejor clase                | *Virus Mosaico* → **99.9 %** |
| Clase más difícil          | *Tizón Temprano* → **84.5 %** |

---

## Componentes principales

| Archivo                    | Descripción                                           |
|----------------------------|-------------------------------------------------------|
| `detector_tomates.py`      | Detector en tiempo real con webcam (OpenCV)           |
| `panel_gestion.py`         | Dashboard web profesional (Streamlit)                 |
| `procesador_lotes.py`      | Evaluación masiva sobre carpetas                      |
| `generar_matriz_full.py`   | Métricas completas (train + val) y matriz de confusión|
| `generar_matriz_train.py`  | Métricas solo TRAIN                                   |
| `generar_matriz_val.py`    | Métricas solo VAL                                     |
| `test_model_info.py`       | Diagnóstico rápido del modelo                         |
| `model.tflite`             | Modelo cuantizado (~6 MB)                             |
| `labels.json`              | Mapeo de clases del modelo                            |
| `requirements.txt`         | Dependencias del proyecto                             |

---

## Capturas

### Dashboard web (Streamlit)
![Dashboard web](capturas_tomate/dashboard_streamlit.jpg)

### Matriz de confusión (11 000 imágenes)
![Matriz de confusión](matriz_confusion_full.png)

### Detector en tiempo real
*Captura en proceso…*

---

## Instalación

```bash
pip install -r requirements.txt
```
# Dashboard web (recomendado para demos)
```bash
streamlit run panel_gestion.py
```

# Detector con webcam (modo consola)
```bash
python detector_tomates.py
```
Fuente del dataset

Subconjunto “tomato” del dataset PlantVillage
Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060
Licencia: CC0 Public Domain

Versión usada en Kaggle:
https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf

TomateCNN-Probando la DL

Cristian Acevedo · 2025



