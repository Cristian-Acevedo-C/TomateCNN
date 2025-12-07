"""
Proyecto: Deteccion de enfermedades en hojas de tomate
Script:   generar_matriz_full.py
Autor:    Cristian Acevedo

Descripcion:
    Evalua el rendimiento del modelo TFLite sobre todo el dataset
    (conjuntos TRAIN + VAL) y genera:

    - Matriz de confusion (matriz_confusion_full.png)
    - Matriz de confusion en CSV (matriz_confusion_full.csv)
    - Reporte de clasificacion por clase (precision, recall, f1-score)
    - Resumen en consola con accuracy global y por clase
    - Archivo de log: resumen_evaluacion_full.txt
"""

import cv2
import numpy as np
import json
import sys
import datetime
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------------------------------------------------
#  GESTION DE LIBRERIAS TFLITE
# ----------------------------------------------------------------------
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
        from tensorflow.lite.python.interpreter import Interpreter
        tflite.Interpreter = Interpreter
    except ImportError:
        print("ERROR: No se encontro tensorflow ni tflite_runtime instalado.")
        sys.exit()

# ----------------------------------------------------------------------
#  CONFIGURACION
# ----------------------------------------------------------------------
class Config:
    # Carpeta base del proyecto (Tomate_CNN)
    BASE_DIR = Path(__file__).parent

    # Rutas a modelo y labels
    MODEL_PATH = BASE_DIR / "model.tflite"
    LABEL_PATH = BASE_DIR / "labels.json"

    # Ruta a la carpeta "tomato" (dataset completo)
    # Debe contener:
    #   tomato/train/Tomato___...
    #   tomato/val/Tomato___...
    DATASET_ROOT = Path(r"C:\Users\crist\Downloads\tomato")

    # Subconjuntos a evaluar en este script (train + val)
    SUBDIRS = ["train", "val"]

    # Nombres abreviados en espanol para graficos/informes
    TRADUCCIONES = {
        "Bacterial_spot": "Mancha Bact.",
        "Early_blight": "Tizon Temp.",
        "Late_blight": "Tizon Tardio",
        "Leaf_Mold": "Moho Foliar",
        "Septoria_leaf_spot": "Septoria",
        "Spider_mites Two-spotted_spider_mite": "Acaro Rojo",
        "Target_Spot": "Mancha Diana",
        "Tomato_Yellow_Leaf_Curl_Virus": "Virus Rizado",
        "Tomato_mosaic_virus": "Virus Mosaico",
        "healthy": "Sana",
    }

    # Salidas
    CONFUSION_PNG = BASE_DIR / "matriz_confusion_full.png"
    CONFUSION_CSV = BASE_DIR / "matriz_confusion_full.csv"
    LOG_PATH = BASE_DIR / "resumen_evaluacion_full.txt"


# ----------------------------------------------------------------------
#  CARGA DEL MODELO TFLITE
# ----------------------------------------------------------------------
def cargar_interpreter():
    """
    Carga el modelo TFLite y devuelve el interprete junto con
    los detalles de entrada/salida.
    """
    if not Config.MODEL_PATH.exists():
        print(f"ERROR: No se encontro el modelo en {Config.MODEL_PATH}")
        sys.exit()

    interpreter = tflite.Interpreter(model_path=str(Config.MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    h = input_details["shape"][1]
    w = input_details["shape"][2]
    dtype = input_details["dtype"]

    print(f"Modelo cargado -> input shape: {input_details['shape']}, dtype={dtype}")
    return interpreter, input_details, output_details, h, w, dtype


# ----------------------------------------------------------------------
#  CARGA DE LABELS
# ----------------------------------------------------------------------
def cargar_labels():
    """
    Carga labels.json y devuelve:

        labels_map:     dict {indice -> nombre_base_en_ingles}
        indices:        lista de indices ordenados
        nombres_en:     lista de nombres base en ingles
        nombres_es:     lista de nombres abreviados en espanol
    """
    if not Config.LABEL_PATH.exists():
        print(f"ERROR: No se encontro labels.json en {Config.LABEL_PATH}")
        sys.exit()

    with open(Config.LABEL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels_map = {}

    # Caso 1: {"Tomato___Bacterial_spot": 0, ...}
    if any("Tomato___" in k for k in data.keys()):
        for full, idx in data.items():
            idx = int(idx)
            labels_map[idx] = full.replace("Tomato___", "")
    else:
        # Caso 2: {"0": "Tomato___Bacterial_spot", ...}
        for idx, name in data.items():
            idx = int(idx)
            labels_map[idx] = name.replace("Tomato___", "")

    indices = sorted(labels_map.keys())
    nombres_ingles = [labels_map[i] for i in indices]
    nombres_espanol = [Config.TRADUCCIONES.get(n, n) for n in nombres_ingles]

    print("\nClases detectadas (por indice):")
    for i, (en, es) in enumerate(zip(nombres_ingles, nombres_espanol)):
        print(f"  {i}: {en} -> {es}")

    return labels_map, indices, nombres_ingles, nombres_espanol


# ----------------------------------------------------------------------
#  SCRIPT PRINCIPAL
# ----------------------------------------------------------------------
def main():
    # Verificar estructura del dataset
    if not Config.DATASET_ROOT.exists():
        print(f"ERROR: No existe la carpeta del dataset: {Config.DATASET_ROOT}")
        sys.exit()

    for sub in Config.SUBDIRS:
        if not (Config.DATASET_ROOT / sub).exists():
            print(f"ERROR: No se encontro subcarpeta '{sub}' en {Config.DATASET_ROOT}")
            sys.exit()

    print(f"\nUsando dataset ROOT en: {Config.DATASET_ROOT}")
    print(f"Subcarpetas evaluadas: {Config.SUBDIRS}")

    interpreter, input_det, output_det, h, w, dtype = cargar_interpreter()
    labels_map, indices, nombres_ingles, nombres_espanol = cargar_labels()

    # Listas globales para etiquetas reales y predichas
    y_true = []
    y_pred = []

    # Contadores por clase (para accuracy por clase)
    total_por_clase = Counter()
    aciertos_por_clase = Counter()

    total_imgs = 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"}

    # Recorremos train y val
    for sub in Config.SUBDIRS:
        base = Config.DATASET_ROOT / sub
        print(f"\n--- Recorriendo subcarpeta: {base}")

        # Orden determinista de archivos (facilita reproducibilidad)
        all_imgs = sorted(p for p in base.rglob("*") if p.suffix in exts)
        print(f"Evaluando {len(all_imgs)} imagenes de {sub}...")

        if not all_imgs:
            print(f"  [AVISO] No se encontraron imagenes en {base}")
            continue

        for img_path in all_imgs:
            # Clase real = primer directorio bajo train/ o val/
            # Ejemplo: tomato/train/Tomato___Early_blight/xxx.jpg
            rel = img_path.relative_to(base)
            if len(rel.parts) == 0:
                print(f"  [SKIP] No se pudo determinar clase para: {img_path}")
                continue

            clase_folder = rel.parts[0]  # "Tomato___Early_blight"

            # Normalizacion de nombre base
            nombre_base = clase_folder.replace("Tomato___", "").strip()

            # Buscar indice de clase en labels_map
            clase_id = None
            for idx, nombre in labels_map.items():
                if nombre == nombre_base:
                    clase_id = int(idx)
                    break

            if clase_id is None:
                print(f"  [SKIP] Carpeta '{clase_folder}' no coincide con ninguna clase en labels.json")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [SKIP] No se pudo leer: {img_path}")
                continue

            # ------------------------------------------------------------------
            #  PREPROCESAMIENTO (ALINEADO CON TRAIN/VAL)
            # ------------------------------------------------------------------
            # Redimensionar a tamaño esperado por el modelo
            img = cv2.resize(img, (w, h))

            # Asegurar 3 canales
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # BGR -> RGB (igual que en generar_matriz_train/val)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalizacion MobileNetV2: rango [-1, 1]
            data = img.astype(dtype)
            if dtype == np.float32:
                data = data / 127.5 - 1.0

            data = np.expand_dims(data, axis=0)

            # Inferencia
            interpreter.set_tensor(input_det["index"], data)
            interpreter.invoke()
            out = interpreter.get_tensor(output_det["index"])[0]
            pred_id = int(np.argmax(out))

            # Acumular etiquetas
            y_true.append(clase_id)
            y_pred.append(pred_id)
            total_imgs += 1

            # Contadores por clase (usando nombre_base)
            total_por_clase[nombre_base] += 1
            if pred_id == clase_id:
                aciertos_por_clase[nombre_base] += 1

            if total_imgs % 500 == 0:
                print(f"  {total_imgs} imagenes procesadas...")

    if total_imgs == 0:
        print("ERROR: No se proceso ninguna imagen valida.")
        sys.exit()

    print(f"\nTotal de imagenes procesadas (TRAIN + VAL): {total_imgs}")

    # ------------------------------------------------------------------
    #  MATRIZ DE CONFUSION
    # ------------------------------------------------------------------
    print("\nGenerando matriz de confusion (TRAIN + VAL)...")
    cm = confusion_matrix(y_true, y_pred, labels=indices)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=nombres_espanol,
        yticklabels=nombres_espanol,
    )
    plt.ylabel("Clase real (TRAIN + VAL)")
    plt.xlabel("Prediccion del modelo")
    plt.title(f"Matriz de confusion (TRAIN + VAL, total: {total_imgs} imagenes)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(Config.CONFUSION_PNG)
    print(f"Imagen de matriz de confusion guardada como: {Config.CONFUSION_PNG}")

    # Guardar tambien la matriz como CSV (para analisis externo)
    np.savetxt(Config.CONFUSION_CSV, cm, delimiter=",", fmt="%d")
    print("Matriz de confusion guardada tambien como CSV.")

    # ------------------------------------------------------------------
    #  REPORTE DETALLADO (CLASSIFICATION REPORT)
    # ------------------------------------------------------------------
    print("\n--- REPORTE DETALLADO (TRAIN + VAL) ---")
    report = classification_report(
        y_true,
        y_pred,
        labels=indices,
        target_names=nombres_espanol,
    )
    print(report)

    # Accuracy global
    aciertos_globales = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc_global = aciertos_globales / total_imgs
    print(f"\nAccuracy global: {acc_global:.3f} ({aciertos_globales}/{total_imgs})")

    # ------------------------------------------------------------------
    #  ACCURACY POR CLASE
    # ------------------------------------------------------------------
    print("\nAccuracy por clase:")
    for nombre_base in sorted(total_por_clase.keys()):
        total_c = total_por_clase[nombre_base]
        ok_c = aciertos_por_clase.get(nombre_base, 0)
        acc_c = ok_c / total_c if total_c > 0 else 0.0

        nombre_es = Config.TRADUCCIONES.get(nombre_base, nombre_base)
        print(f"  {nombre_es:20s} ({nombre_base:30s}) -> {acc_c:6.1%} ({ok_c}/{total_c})")

    # ------------------------------------------------------------------
    #  LOG EN ARCHIVO DE TEXTO
    # ------------------------------------------------------------------
    try:
        with open(Config.LOG_PATH, "w", encoding="utf-8") as f:
            f.write("Resumen de evaluacion del modelo (TRAIN + VAL)\n")
            f.write(f"Fecha y hora:   {datetime.datetime.now().isoformat()}\n")
            f.write(f"Dataset root:   {Config.DATASET_ROOT}\n")
            f.write(f"Subconjuntos:   {', '.join(Config.SUBDIRS)}\n")
            f.write(f"Total imagenes: {total_imgs}\n")
            f.write(f"Accuracy global: {acc_global:.3%} ({aciertos_globales}/{total_imgs})\n")
            f.write(f"Imagen matriz de confusion: {Config.CONFUSION_PNG.name}\n")
            f.write(f"CSV matriz de confusion:    {Config.CONFUSION_CSV.name}\n\n")

            f.write("=== Accuracy por clase ===\n")
            for nombre_base in sorted(total_por_clase.keys()):
                total_c = total_por_clase[nombre_base]
                ok_c = aciertos_por_clase.get(nombre_base, 0)
                acc_c = ok_c / total_c if total_c > 0 else 0.0
                nombre_es = Config.TRADUCCIONES.get(nombre_base, nombre_base)
                f.write(
                    f"- {nombre_es:20s} ({nombre_base:30s}) -> "
                    f"{acc_c:6.1%} ({ok_c}/{total_c})\n"
                )

            f.write("\n=== Classification report (sklearn) ===\n")
            f.write(report)
        print(f"\nResumen detallado escrito en: {Config.LOG_PATH}")
    except Exception as e:
        print(f"\n[AVISO] No se pudo escribir el log de resumen: {e}")


if __name__ == "__main__":
    main()
