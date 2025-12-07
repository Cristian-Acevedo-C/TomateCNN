"""
Proyecto: Deteccion de enfermedades en hojas de tomate
Script: detector_tomates.py
Autores: Cristian Acevedo, Paola Quinde Baythiare Carriz

Descripcion:
  - Captura video desde la camara.
  - Analiza un ROI central usando un modelo TFLite (MobileNetV2).
  - Clasifica la hoja como SANA, ENFERMA o DUDOSA segun confianza.
  - Guarda cada captura en una carpeta y registra los datos en un CSV.
  - Interfaz pensada para demostracion en laboratorio / aula.
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import datetime
import time
import sys
from pathlib import Path
import csv

# ==========================================
# CONFIGURACION GENERAL
# ==========================================

# Rutas del modelo y labels (se asume que este script esta en la carpeta del modelo)
MODEL_PATH = 'model.tflite'
LABELS_PATH = 'labels.json'

# Carpeta donde se guardan las capturas de la sesion
CARPETA_SALIDA = Path('capturas_tomate')

# Umbral de confianza para marcar un caso como "dudoso"
UMBRAL_DUDOSO = 60.0  # en porcentaje

# Configuracion de ventana y texto
WINDOW_NAME = "TomateCNN Vision"
APP_TITLE = "TomateCNN - Sistema de Diagnostico de Hojas"
APP_VERSION = "v1.0"

# Crear carpeta de salida si no existe
CARPETA_SALIDA.mkdir(exist_ok=True)

# Archivo CSV para registrar resultados de la sesion
timestamp_sesion = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = CARPETA_SALIDA / f"registro_{timestamp_sesion}.csv"

with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['fecha_hora', 'archivo', 'estado', 'enfermedad', 'confianza_%'])

# ==========================================
# PALETA DE COLORES (SIN ACENTOS)
# ==========================================

C_FONDO_BARRA = (30, 30, 30)
C_TEXTO_CLARO = (235, 235, 235)
C_TEXTO_SUAVE = (170, 170, 170)
C_ACENTO = (255, 200, 0)
C_VERDE = (60, 200, 60)
C_ROJO = (70, 70, 220)
C_AMARILLO = (0, 210, 210)

# Traducciones: nombres en ingles (labels) -> texto en espanol sin acentos (para overlay)
TRADUCCIONES = {
    'Bacterial_spot': 'MANCHA BACTERIANA',
    'Early_blight': 'TIZON TEMPRANO',
    'Late_blight': 'TIZON TARDIO',
    'Leaf_Mold': 'MOHO FOLIAR',
    'Septoria_leaf_spot': 'MANCHA SEPTORIA',
    'Spider_mites Two-spotted_spider_mite': 'ACARO ROJO',
    'Target_Spot': 'MANCHA DIANA',
    'Tomato_Yellow_Leaf_Curl_Virus': 'VIRUS RIZADO AMARILLO',
    'Tomato_mosaic_virus': 'VIRUS DEL MOSAICO',
    'healthy': 'HOJA SANA'
}

# ==========================================
# CARGA DEL MODELO TFLITE Y LABELS
# ==========================================

def cargar_sistema():
    """
    Carga el modelo TFLite y el archivo labels.json.
    Devuelve:
      - labels: lista de nombres de clase en ingles (sin Tomato___).
      - interpreter: interprete TFLite ya inicializado.
      - input_det, output_det: detalles de entrada y salida.
      - h, w: alto y ancho de entrada esperados por el modelo.
      - dtype: tipo de dato esperado por el modelo (normalmente float32).
    """
    print("Cargando modelo y etiquetas...")

    # Carga del modelo TFLite
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_det = interpreter.get_input_details()[0]
        output_det = interpreter.get_output_details()[0]
        h = input_det['shape'][1]
        w = input_det['shape'][2]
        dtype = input_det['dtype']
    except Exception as e:
        sys.exit(f"Error cargando modelo TFLite: {e}")

    # Carga de labels
    try:
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        labels = [''] * len(data)

        # Formato: {"Tomato___Bacterial_spot": 0, ...}
        if any("Tomato___" in k for k in data.keys()):
            for full, idx in data.items():
                base_name = full.replace("Tomato___", "")
                labels[int(idx)] = base_name
        # Formato: {"0": "Tomato___Bacterial_spot", ...}
        elif any(k.isdigit() for k in data.keys()):
            for idx, name in data.items():
                base_name = name.replace("Tomato___", "")
                labels[int(idx)] = base_name
        else:
            # Formato generico
            for idx, name in data.items():
                labels[int(idx)] = name

        print(f"  Clases cargadas: {len(labels)}")
        return labels, interpreter, input_det, output_det, h, w, dtype

    except Exception as e:
        print(f"Error cargando labels.json, usando nombres genericos: {e}")
        labels = [f"CLASE_{i}" for i in range(10)]
        return labels, interpreter, input_det, output_det, h, w, dtype


labels, interpreter, input_det, output_det, height, width, input_dtype = cargar_sistema()

# ==========================================
# CONFIGURACION DE CAMARA
# ==========================================

cap = cv2.VideoCapture(0)  # cambiar a 1 si se usa camara USB externa
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    sys.exit("Error: no se detecto camara")

# "Calentar" la camara (algunos drivers ajustan brillo/foco en los primeros frames)
print("Ajustando brillo y enfoque automatico...")
for _ in range(20):
    cap.read()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Contadores de diagnosticos
total_fotos = 0
total_sanas = 0
total_enfermas = 0
total_dudosas = 0

font_main = cv2.FONT_HERSHEY_SIMPLEX
font_bold = cv2.FONT_HERSHEY_SIMPLEX

print("\n" + "═" * 60)
print(f"{APP_TITLE}")
print("═" * 60)
print(f"  Sesion:   {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print(f"  Carpeta:  {CARPETA_SALIDA}/")
print(f"  CSV:      {CSV_PATH.name}")
print("  Controles: 'C' = Analizar ROI central  |  'Q' = Salir")
print("═" * 60 + "\n")

# ==========================================
# UTILIDADES GRAFICAS PARA LA UI
# ==========================================

def dibujar_panel_superior(frame, fps):
    """
    Dibuja la barra superior con el titulo de la aplicacion y los FPS.
    """
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), C_FONDO_BARRA, -1)
    cv2.putText(frame, APP_TITLE, (20, 32), font_bold, 0.7, C_TEXTO_CLARO, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Sesion {timestamp_sesion}", (20, 48),
                font_main, 0.45, C_TEXTO_SUAVE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 30),
                font_main, 0.6, C_ACENTO, 1, cv2.LINE_AA)


def dibujar_panel_inferior(frame):
    """
    Dibuja la barra inferior con los contadores y los atajos de teclado.
    """
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 45), (w, h), C_FONDO_BARRA, -1)

    texto_izq = (
        f"Capturas: {total_fotos}  |  Sanas: {total_sanas}  |  Enf: {total_enfermas}  |  Dud: {total_dudosas}"
    )
    texto_der = "[C] Analizar ROI  |  [Q] Salir"

    cv2.putText(frame, texto_izq, (20, h - 18),
                font_main, 0.5, C_TEXTO_CLARO, 1, cv2.LINE_AA)
    cv2.putText(frame, texto_der, (w - 360, h - 18),
                font_main, 0.5, C_TEXTO_SUAVE, 1, cv2.LINE_AA)


def dibujar_roi(frame):
    """
    Dibuja un ROI cuadrado en el centro de la imagen con esquinas tipo visor.
    Devuelve las coordenadas (x1, y1, x2, y2) del ROI.
    """
    h, w = frame.shape[:2]
    lado = min(h, w) // 2
    x1 = w // 2 - lado // 2
    y1 = h // 2 - lado // 2
    x2 = x1 + lado
    y2 = y1 + lado

    color = (220, 220, 220)
    grosor = 2
    l = 25  # longitud de esquina

    # Esquinas tipo visor
    cv2.line(frame, (x1, y1), (x1 + l, y1), color, grosor)
    cv2.line(frame, (x1, y1), (x1, y1 + l), color, grosor)

    cv2.line(frame, (x2, y1), (x2 - l, y1), color, grosor)
    cv2.line(frame, (x2, y1), (x2, y1 + l), color, grosor)

    cv2.line(frame, (x1, y2), (x1 + l, y2), color, grosor)
    cv2.line(frame, (x1, y2), (x1, y2 - l), color, grosor)

    cv2.line(frame, (x2, y2), (x2 - l, y2), color, grosor)
    cv2.line(frame, (x2, y2), (x2, y2 - l), color, grosor)

    # Cruz central
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 8, cy), (cx + 8, cy), C_ACENTO, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 8), (cx, cy + 8), C_ACENTO, 1, cv2.LINE_AA)

    # Texto guia
    cv2.putText(frame, "Centre la hoja en el recuadro y presione C",
                (w // 2 - 260, y1 - 10), font_main, 0.6, C_TEXTO_SUAVE, 1, cv2.LINE_AA)

    return x1, y1, x2, y2

# ==========================================
# ANALISIS Y GUARDADO (USANDO ROI)
# ==========================================

def analizar_y_guardar(frame, roi_coords):
    """
    Recorta el ROI central, lo preprocesa igual que en el entrenamiento (MobileNetV2),
    ejecuta la inferencia TFLite y guarda:
      - La imagen completa en CARPETA_SALIDA.
      - Un registro en el CSV de la sesion.
    Devuelve: (nombre_enfermedad_es, estado, confianza, color_estado)
    """
    global total_fotos, total_sanas, total_enfermas, total_dudosas

    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]

    # Redimension a input del modelo
    img = cv2.resize(roi, (width, height))

    # Convertir de BGR (OpenCV) a RGB (modelo)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocesamiento MobileNetV2: rango [-1, 1]
    data = img.astype(input_dtype)
    if input_dtype == np.float32:
        data = data / 127.5 - 1.0

    data = np.expand_dims(data, axis=0)

    # Inferencia
    interpreter.set_tensor(input_det['index'], data)
    interpreter.invoke()
    out = interpreter.get_tensor(output_det['index'])[0]

    idx = int(np.argmax(out))
    conf = float(out[idx] * 100.0)

    nombre_ing = labels[idx]
    nombre_es = TRADUCCIONES.get(
        nombre_ing, nombre_ing.replace("_", " ").upper()
    )

    # Clasificacion segun confianza y tipo de clase
    if conf < UMBRAL_DUDOSO:
        estado = "DUDOSO"
        color = C_AMARILLO
        total_dudosas += 1
    elif nombre_ing == "healthy":
        estado = "SANA"
        color = C_VERDE
        total_sanas += 1
    else:
        estado = "ENFERMA"
        color = C_ROJO
        total_enfermas += 1

    ahora = datetime.datetime.now()
    ts = ahora.strftime("%Y%m%d_%H%M%S")
    nombre_file = f"{estado}_{nombre_es.replace(' ', '_')}_{ts}.jpg"

    # Guardar frame completo (no solo ROI) para tener contexto
    cv2.imwrite(str(CARPETA_SALIDA / nombre_file), frame)

    # Registrar en CSV
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([ahora.isoformat(), nombre_file, estado, nombre_es, f"{conf:.2f}"])

    total_fotos += 1

    print(f"[{total_fotos:03d}] {estado:7} | {nombre_es:25} | {conf:5.1f}% -> {nombre_file}")

    return nombre_es, estado, conf, color

# ==========================================
# BUCLE PRINCIPAL (INTERFAZ EN VIVO)
# ==========================================

prev_time = time.time()
salir = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camara desconectada")
            break

        h_img, w_img = frame.shape[:2]

        # Calculo de FPS
        curr_time = time.time()
        if prev_time > 0:
            delta = curr_time - prev_time
            fps = 1.0 / delta if delta > 0 else 0.0
        else:
            fps = 0.0
        prev_time = curr_time

        # Dibujar interfaz
        dibujar_panel_superior(frame, fps)
        roi_coords = dibujar_roi(frame)
        dibujar_panel_inferior(frame)

        cv2.imshow(WINDOW_NAME, frame)

        k = cv2.waitKey(1) & 0xFF

        # Salir con ESC o Q
        if k in [27, ord('q'), ord('Q')]:
            break

        # Analizar ROI con tecla C
        elif k in [ord('c'), ord('C')]:
            frame_analisis = frame.copy()
            nombre, estado, conf, color = analizar_y_guardar(frame_analisis, roi_coords)

            # Pantalla de resultado
            resultado = frame_analisis.copy()
            overlay = resultado.copy()
            cv2.rectangle(overlay, (0, 0), (w_img, h_img), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, resultado, 0.45, 0, resultado)

            # Tarjeta central
            card_w, card_h = 540, 260
            card_x = (w_img - card_w) // 2
            card_y = (h_img - card_h) // 2

            cv2.rectangle(resultado, (card_x, card_y),
                          (card_x + card_w, card_y + card_h), (25, 25, 25), -1)
            cv2.rectangle(resultado, (card_x, card_y),
                          (card_x + card_w, card_y + card_h), color, 2)

            # Texto de resultado
            cv2.putText(resultado, "Resultado de analisis",
                        (card_x + 20, card_y + 40), font_bold, 0.7, C_TEXTO_CLARO, 1, cv2.LINE_AA)

            cv2.putText(resultado, "Diagnostico:",
                        (card_x + 20, card_y + 90), font_main, 0.55, C_TEXTO_SUAVE, 1, cv2.LINE_AA)
            cv2.putText(resultado, nombre,
                        (card_x + 20, card_y + 120), font_bold, 0.9, C_TEXTO_CLARO, 2, cv2.LINE_AA)

            cv2.putText(resultado, "Confianza:",
                        (card_x + 20, card_y + 160), font_main, 0.55, C_TEXTO_SUAVE, 1, cv2.LINE_AA)
            cv2.putText(resultado, f"{conf:.1f}%",
                        (card_x + 20, card_y + 190), font_main, 0.9, C_TEXTO_CLARO, 1, cv2.LINE_AA)

            # Barra de confianza
            bar_w = 260
            fill_w = int(bar_w * (conf / 100.0))
            cv2.rectangle(resultado, (card_x + 180, card_y + 176),
                          (card_x + 180 + bar_w, card_y + 188), (60, 60, 60), -1)
            cv2.rectangle(resultado, (card_x + 180, card_y + 176),
                          (card_x + 180 + fill_w, card_y + 188), color, -1)

            # Footer
            cv2.putText(resultado, "Datos guardados en CSV y carpeta de capturas.",
                        (card_x + 20, card_y + 230), font_main, 0.45, C_TEXTO_SUAVE, 1, cv2.LINE_AA)

            # Mostrar resultado unos segundos
            t0 = time.time()
            while time.time() - t0 < 4.0:
                cv2.imshow(WINDOW_NAME, resultado)
                ch = cv2.waitKey(10) & 0xFF
                if ch in [27, ord('q'), ord('Q')]:
                    salir = True
                    break

            if salir:
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "═" * 60)
    print("          RESUMEN FINAL DE LA SESION")
    print("═" * 60)
    print(f"  Hojas SANAS       : {total_sanas}")
    print(f"  Hojas ENFERMAS    : {total_enfermas}")
    print(f"  Clasif. DUDOSAS   : {total_dudosas}")
    print("  ------------------------------")
    print(f"  TOTAL FOTOS       : {total_fotos}")
    print("═" * 60)
    print(f"  CSV generado      : {CSV_PATH.name}")
    print("  Carpeta imagenes  : capturas_tomate/")
    print("═" * 60)
