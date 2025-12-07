"""
Proyecto: Deteccion de enfermedades en hojas de tomate
Script: procesador_lotes.py
Autor: Cristian Acevedo

Descripcion:
  - Recorre una carpeta de imagenes organizada por clases (subcarpetas).
  - Clasifica cada imagen usando un modelo TFLite basado en MobileNetV2.
  - Compara la prediccion con la clase real (segun la carpeta de origen).
  - Guarda copias de las imagenes con overlay en carpetas de aciertos y errores.
  - Imprime un resumen con accuracy global y accuracy por clase.
  - Genera un archivo resumen_evaluacion.txt con los resultados principales.

Uso esperado:
  - Carpeta "capturas/" con subcarpetas tipo:
        capturas/
            Tomato___Bacterial_spot/
            Tomato___Early_blight/
            ...
  - Ejecutar:
        python procesador_lotes.py
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
import time
import datetime
from collections import Counter

# ==========================================
# GESTION DE LIBRERIAS TFLITE
# ==========================================
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
        from tensorflow.lite.python.interpreter import Interpreter
        tflite.Interpreter = Interpreter
    except ImportError:
        print("ERROR: No tienes tensorflow ni tflite_runtime instalado.")
        sys.exit()

# ==========================================
# CONFIGURACION GENERAL
# ==========================================

class Config:
    """
    Parametros centrales del script:
      - rutas de modelo, labels e imagenes,
      - carpetas de salida y mapeos de clases.
    """
    BASE_DIR = Path(__file__).parent

    # Modelo y labels
    MODEL_PATH = BASE_DIR / "model.tflite"
    LABEL_PATH = BASE_DIR / "labels.json"

    # Carpeta raiz donde estan las fotos a evaluar
    # Estructura esperada: IMAGES_DIR / "Tomato___Clase" / imagenes
    IMAGES_DIR = BASE_DIR / "capturas"

    # Carpeta donde se guardan resultados de la clasificacion
    RESULTS_DIR = BASE_DIR / "resultados_clasificacion"

    # Umbral de confianza (no se usa para filtrar, pero puede servir en mejoras futuras)
    CONF_THRESHOLD = 0.50

    # Mapeo de nombres originales a etiquetas en espanol (sin tildes para overlays)
    SPANISH_LABELS = {
        "Tomato___Bacterial_spot": "MANCHA BACTERIANA",
        "Tomato___Early_blight": "TIZON TEMPRANO",
        "Tomato___Late_blight": "TIZON TARDIO",
        "Tomato___Leaf_Mold": "MOHO FOLIAR",
        "Tomato___Septoria_leaf_spot": "MANCHA SEPTORIA",
        "Tomato___Spider_mites Two-spotted_spider_mite": "ACARO ROJO",
        "Tomato___Target_Spot": "MANCHA DIANA",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "VIRUS RIZADO AMARILLO",
        "Tomato___Tomato_mosaic_virus": "VIRUS DEL MOSAICO",
        "Tomato___healthy": "HOJA SANA",
    }

    # Mapa inverso: espanol -> ingles completo (por si se necesita)
    SPANISH_TO_ENGLISH = {v: k for k, v in SPANISH_LABELS.items()}


# ==========================================
# CLASE PRINCIPAL PARA MANEJAR EL MODELO
# ==========================================

class TomatoBatch:
    """
    Encapsula la carga del modelo TFLite y la logica de inferencia
    sobre una imagen individual.
    """

    def __init__(self):
        self._load_model()
        self._load_labels()

    def _load_model(self):
        """
        Carga el modelo TFLite y determina el tamano de entrada.
        """
        if not Config.MODEL_PATH.exists():
            print(f"ERROR: No se encontro el modelo en {Config.MODEL_PATH}")
            sys.exit()

        self.interpreter = tflite.Interpreter(model_path=str(Config.MODEL_PATH))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        self.input_dtype = self.input_details[0]["dtype"]

        print(f"Modelo cargado. Input shape: {self.input_shape}, dtype={self.input_dtype}")

    def _load_labels(self):
        """
        Carga el archivo labels.json y construye:
          - self.idx_to_english: indice -> nombre completo (Tomato___Clase).
          - self.idx_to_spanish: indice -> nombre en espanol (sin tildes).
        """
        if not Config.LABEL_PATH.exists():
            print(f"ADVERTENCIA: No se encontro labels.json en {Config.LABEL_PATH}")
            self.idx_to_english = {}
            self.idx_to_spanish = {}
            return

        with open(Config.LABEL_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)

        idx_to_english = {}

        # Formato 1: {"Tomato___Bacterial_spot": 0, ...}
        if any("Tomato___" in k for k in raw.keys()):
            for full_name, idx in raw.items():
                idx = int(idx)
                idx_to_english[idx] = full_name

        # Formato 2: {"0": "Tomato___Bacterial_spot", ...}
        elif any(k.isdigit() for k in raw.keys()):
            for idx_str, full_name in raw.items():
                idx = int(idx_str)
                idx_to_english[idx] = full_name

        else:
            # Formato generico
            for idx_str, name in raw.items():
                idx = int(idx_str)
                idx_to_english[idx] = name

        self.idx_to_english = idx_to_english

        # Construimos un mapeo en espanol seguro
        self.idx_to_spanish = {}
        for idx, full_name in self.idx_to_english.items():
            self.idx_to_spanish[idx] = Config.SPANISH_LABELS.get(
                full_name, full_name.replace("Tomato___", "").upper()
            )

        print("Clases detectadas en labels.json:")
        for idx in sorted(self.idx_to_english.keys()):
            print(f"  {idx}: {self.idx_to_english[idx]} -> {self.idx_to_spanish[idx]}")

    def predict(self, image_path):
        """
        Ejecuta la prediccion sobre una imagen individual.

        Devuelve:
          - original_img: imagen original leida.
          - label_spanish: etiqueta en espanol (para overlay).
          - conf: confianza en [0, 1].
          - label_english_full: nombre completo de la clase (Tomato___Clase).
        """
        original_img = cv2.imread(str(image_path))
        if original_img is None:
            print(f"[SKIP] No se pudo leer: {image_path}")
            return None, None, None, None

        # Redimensionamos a la entrada del modelo
        img = cv2.resize(
            original_img, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA
        )

        # MobileNetV2 se entrena normalmente en RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocesamiento MobileNetV2: rango [-1, 1]
        data = img.astype(self.input_dtype)
        if self.input_dtype == np.float32:
            data = data / 127.5 - 1.0

        data = np.expand_dims(data, axis=0)

        # Inferencia TFLite
        self.interpreter.set_tensor(self.input_details[0]["index"], data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        idx = int(np.argmax(output))
        conf = float(output[idx])

        label_english_full = self.idx_to_english.get(idx, "Desconocido")
        label_spanish = self.idx_to_spanish.get(idx, "DESCONOCIDO")

        return original_img, label_spanish, conf, label_english_full


# ==========================================
# FUNCION PRINCIPAL PARA PROCESAR POR LOTES
# ==========================================

def main():
    """
    Recorre todas las imagenes dentro de IMAGES_DIR (recursivo),
    aplica el modelo y separa aciertos y errores en carpetas.
    Tambien calcula accuracy global y por clase.
    """
    if not Config.IMAGES_DIR.exists():
        print(f"ERROR: No existe la carpeta de imagenes: {Config.IMAGES_DIR}")
        return

    # Crear carpetas de resultados
    Config.RESULTS_DIR.mkdir(exist_ok=True)
    aciertos_dir = Config.RESULTS_DIR / "aciertos"
    errores_dir = Config.RESULTS_DIR / "errores"
    aciertos_dir.mkdir(exist_ok=True)
    errores_dir.mkdir(exist_ok=True)

    # Buscar imagenes de forma recursiva (rglob)
    print(f">> Buscando imagenes dentro de: {Config.IMAGES_DIR}")
    extensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"}
    files = []
    for ext in extensions:
        files.extend(list(Config.IMAGES_DIR.rglob(ext)))

    if not files:
        print("No se encontraron imagenes. Revisar extensiones y ruta IMAGES_DIR.")
        return

    # Orden determinista: facilita reproducir experimentos
    files.sort()

    print(f">> Encontradas {len(files)} imagenes. Iniciando evaluacion...\n")
    brain = TomatoBatch()

    total = len(files)
    correctos = 0
    incorrectos = 0

    # Contadores por clase real (solo imagenes realmente procesadas)
    aciertos_por_clase = Counter()
    total_por_clase = Counter()

    print("Controles durante la revision:")
    print("  - ESC: salir inmediatamente")
    print("  - Espacio: pausar/reanudar visualizacion\n")

    paused = False

    for i, path in enumerate(files):
        original_img, label_spanish, conf, label_english_full = brain.predict(path)
        if original_img is None:
            # Imagen ilegible: no cuenta para total_por_clase ni accuracy
            continue

        carpeta_origen = path.parent.name  # clase real (ej: Tomato___Early_blight)
        total_por_clase[carpeta_origen] += 1

        # Comparar clase real vs prediccion
        es_correcto = (label_english_full == carpeta_origen)

        if es_correcto:
            status = "OK"
            dest_dir = aciertos_dir
            correctos += 1
            color = (0, 255, 0)
            aciertos_por_clase[carpeta_origen] += 1
        else:
            status = "FAIL"
            dest_dir = errores_dir
            incorrectos += 1
            color = (0, 0, 255)

        # Preparo overlay para guardar la imagen con texto
        display_img = cv2.resize(original_img, (800, 600))
        cv2.rectangle(display_img, (0, 0), (800, 50), (0, 0, 0), -1)

        texto_pred = f"{label_spanish} ({conf*100:.1f}%)"
        cv2.putText(display_img, texto_pred, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        info_real_pred = (
            f"Real: {carpeta_origen} | Pred (EN): {label_english_full} | Archivo: {path.name}"
        )
        cv2.putText(display_img, info_real_pred, (20, 580),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Nombre de salida informativo
        out_name = f"{status}__real-{carpeta_origen}__pred-{label_english_full}{path.suffix}"
        out_path = dest_dir / out_name
        cv2.imwrite(str(out_path), display_img)

        # Mostrar en pantalla (opcional, util para auditoria visual)
        cv2.imshow("Prueba Masiva TomateCNN", display_img)

        print(f"[{i+1:04d}/{total:04d}] {status} | Pred: {label_spanish:25s} "
              f"| Real: {carpeta_origen:30s} | conf={conf*100:.1f}%")

        # Teclas de control
        delay = 0 if paused else 1
        key = cv2.waitKey(delay) & 0xFF

        if key == 27:  # ESC
            print("Salida anticipada por ESC.")
            break
        elif key == 32:  # Espacio
            paused = not paused
            if paused:
                print("\n--- PAUSADO (presionar espacio para continuar) ---\n")
            else:
                print("--- REANUDANDO ---")

    cv2.destroyAllWindows()

    # Calculamos totales reales (solo imagenes procesadas con exito)
    total_procesadas = sum(total_por_clase.values())

    # ==========================================
    # RESUMEN FINAL EN CONSOLA
    # ==========================================

    print("\n================ RESUMEN ================")
    print(f"Total imagenes encontradas:   {total}")
    print(f"Total imagenes procesadas:    {total_procesadas}")
    print(f"Aciertos (OK):                {correctos}")
    print(f"Errores (FAIL):               {incorrectos}")

    if total_procesadas > 0:
        acc = correctos / total_procesadas
        print(f"Accuracy global aprox.:       {acc:.1%}")
    else:
        acc = 0.0
        print("Accuracy global aprox.:       N/A (0 procesadas)")

    print(f"Resultados guardados en:      {Config.RESULTS_DIR}")

    print("\nAccuracy por clase (solo sobre imagenes procesadas):")
    for clase in sorted(total_por_clase.keys()):
        total_clase = total_por_clase[clase]
        aciertos_clase = aciertos_por_clase.get(clase, 0)
        acc_clase = aciertos_clase / total_clase if total_clase > 0 else 0.0
        print(f"  {clase:40} -> {acc_clase:6.1%} ({aciertos_clase}/{total_clase})")

    print("=========================================")

    # ==========================================
    # RESUMEN EN ARCHIVO DE TEXTO
    # ==========================================

    resumen_path = Config.RESULTS_DIR / "resumen_evaluacion.txt"
    with open(resumen_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluacion realizada el {datetime.datetime.now().isoformat()}\n")
        f.write(f"Carpeta evaluada: {Config.IMAGES_DIR}\n\n")
        f.write(f"Total imagenes encontradas:   {total}\n")
        f.write(f"Total imagenes procesadas:    {total_procesadas}\n")
        f.write(f"Aciertos (OK):                {correctos}\n")
        f.write(f"Errores (FAIL):               {incorrectos}\n")
        f.write(f"Accuracy global aprox.:       {acc:.1%}\n\n")
        f.write("Accuracy por clase (solo sobre imagenes procesadas):\n")
        for clase in sorted(total_por_clase.keys()):
            total_clase = total_por_clase[clase]
            aciertos_clase = aciertos_por_clase.get(clase, 0)
            acc_clase = aciertos_clase / total_clase if total_clase > 0 else 0.0
            f.write(f"  {clase:40} -> {acc_clase:6.1%} ({aciertos_clase}/{total_clase})\n")

    print(f"\nResumen tambien guardado en: {resumen_path}")


if __name__ == "__main__":
    main()
