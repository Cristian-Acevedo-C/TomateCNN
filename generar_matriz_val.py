"""
Proyecto: Deteccion de enfermedades en hojas de tomate
Script:   generar_matriz_val.py
Autor:    Cristian Acevedo

Descripcion:
    - Evalua el conjunto de validacion (val) del dataset local.
    - Genera:
        * Matriz de confusion (PNG + CSV)
        * Reporte de clasificacion
        * Accuracy global sobre VAL
        * Log de resumen en TXT

Estructura esperada del dataset:

    dataset/
        tomato/
            train/
            val/
                Tomato___Bacterial_spot/
                Tomato___Early_blight/
                ...

Archivos requeridos en la misma carpeta del script:

    model.tflite
    labels.json
"""

import cv2
import numpy as np
import json
import sys
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

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
        print("ERROR: No se encontro ni tflite_runtime ni tensorflow.lite.")
        sys.exit()


# ==========================================
# CONFIGURACION
# ==========================================
class Config:
    BASE_DIR = Path(__file__).parent

    MODEL_PATH = BASE_DIR / "model.tflite"
    LABEL_PATH = BASE_DIR / "labels.json"

    # Dataset local: dataset/tomato/val/...
    DATASET_ROOT = BASE_DIR / "dataset" / "tomato"
    SUBDIR = "val"

    OUTPUT_PNG = BASE_DIR / "matriz_confusion_val.png"
    OUTPUT_CSV = BASE_DIR / "matriz_confusion_val.csv"
    LOG_PATH = BASE_DIR / "resumen_evaluacion_val.txt"

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


def cargar_interpreter():
    """
    Carga el modelo TFLite y devuelve:
      - interpreter
      - input_details
      - output_details
      - height, width, dtype
    """
    if not Config.MODEL_PATH.exists():
        print(f"ERROR: No se encontro el modelo en {Config.MODEL_PATH}")
        sys.exit()

    interpreter = tflite.Interpreter(model_path=str(Config.MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    height = input_details["shape"][1]
    width = input_details["shape"][2]
    dtype = input_details["dtype"]

    print(
        f"Modelo cargado correctamente. "
        f"Input shape: {input_details['shape']} | dtype={dtype}"
    )
    return interpreter, input_details, output_details, height, width, dtype


def cargar_labels():
    """
    Carga labels.json y devuelve un diccionario:
        idx -> nombre_base_en_ingles (ej: 'Bacterial_spot', 'healthy', etc.).
    """
    if not Config.LABEL_PATH.exists():
        print(f"ERROR: No se encontro labels.json en {Config.LABEL_PATH}")
        sys.exit()

    with open(Config.LABEL_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    idx_to_name = {}

    if any("Tomato___" in k for k in raw.keys()):
        for full_name, idx in raw.items():
            idx_to_name[int(idx)] = full_name.replace("Tomato___", "")
    else:
        for idx, name in raw.items():
            idx_to_name[int(idx)] = name.replace("Tomato___", "")

    print("\nClases detectadas en labels.json:")
    for i in sorted(idx_to_name.keys()):
        print(f"  {i}: {idx_to_name[i]}")

    return idx_to_name


def main():
    # Verificar estructura del dataset
    if not Config.DATASET_ROOT.exists():
        print(f"ERROR: No existe la carpeta del dataset: {Config.DATASET_ROOT}")
        sys.exit()

    val_dir = Config.DATASET_ROOT / Config.SUBDIR
    if not val_dir.exists():
        print(
            f"ERROR: No existe la subcarpeta '{Config.SUBDIR}' dentro de "
            f"{Config.DATASET_ROOT}"
        )
        sys.exit()

    interpreter, input_details, output_details, H, W, DTYPE = cargar_interpreter()
    idx_to_name = cargar_labels()

    y_true = []
    y_pred = []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"}
    total = 0

    print(f"\nLeyendo imagenes desde: {val_dir}")
    images = [p for p in val_dir.rglob("*") if p.suffix in exts]
    images.sort()
    print(f"  Imagenes encontradas: {len(images)}")

    for img_path in images:
        rel = img_path.relative_to(val_dir)
        if len(rel.parts) == 0:
            continue

        folder_name = rel.parts[0]
        if folder_name.startswith("Tomato___"):
            real_name = folder_name.replace("Tomato___", "").strip()
        else:
            real_name = folder_name.strip()

        # Buscar indice real
        true_id = None
        for idx, name in idx_to_name.items():
            if name == real_name:
                true_id = int(idx)
                break

        if true_id is None:
            print(f"[SKIP] Carpeta '{folder_name}' no coincide con labels.json")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[SKIP] No se pudo leer: {img_path}")
            continue

        img = cv2.resize(img, (W, H))

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = img.astype(DTYPE)
        if DTYPE == np.float32:
            # MobileNetV2: rango [-1, 1]
            x = x / 127.5 - 1.0

        x = np.expand_dims(x, axis=0)

        interpreter.set_tensor(input_details["index"], x)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details["index"])[0]
        pred_id = int(np.argmax(logits))

        y_true.append(true_id)
        y_pred.append(pred_id)
        total += 1

        if total % 100 == 0:
            print(f"  Imagenes procesadas: {total}...")

    if total == 0:
        print("No se proceso ninguna imagen del conjunto de validacion.")
        sys.exit()

    print(f"\nTotal de imagenes evaluadas (VAL): {total}")

    indices = sorted(idx_to_name.keys())
    nombres_ingles = [idx_to_name[i] for i in indices]
    nombres_es = [Config.TRADUCCIONES.get(n, n) for n in nombres_ingles]

    cm = confusion_matrix(y_true, y_pred, labels=indices)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=nombres_es,
        yticklabels=nombres_es,
        ax=ax,
    )
    plt.ylabel("Clase real (validacion)")
    plt.xlabel("Prediccion del modelo")
    plt.title(f"Matriz de confusion - VAL (n={total})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(Config.OUTPUT_PNG)
    print(f"\nMatriz de confusion guardada en: {Config.OUTPUT_PNG}")

    # Guardar matriz como CSV
    np.savetxt(Config.OUTPUT_CSV, cm, delimiter=",", fmt="%d")
    print(f"Matriz de confusion guardada como CSV en: {Config.OUTPUT_CSV}")

    print("\n--- REPORTE DETALLADO (solo VALIDACION) ---")
    report = classification_report(
        y_true,
        y_pred,
        labels=indices,
        target_names=nombres_es,
    )
    print(report)

    aciertos = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = aciertos / total
    print(f"Accuracy global VAL: {acc:.3f} ({aciertos}/{total})")

    # Log en TXT
    try:
        with open(Config.LOG_PATH, "w", encoding="utf-8") as f:
            f.write("Resumen de evaluacion del modelo (VAL)\n")
            f.write(f"Fecha y hora:   {datetime.datetime.now().isoformat()}\n")
            f.write(f"Dataset root:   {Config.DATASET_ROOT}\n")
            f.write(f"Subconjunto:    {Config.SUBDIR}\n")
            f.write(f"Total imagenes: {total}\n")
            f.write(
                f"Accuracy global: {acc:.3%} "
                f"({aciertos}/{total})\n"
            )
            f.write(
                f"PNG matriz de confusion: {Config.OUTPUT_PNG.name}\n"
                f"CSV matriz de confusion: {Config.OUTPUT_CSV.name}\n\n"
            )
            f.write("=== Classification report (sklearn) ===\n")
            f.write(report)
        print(f"Resumen escrito en: {Config.LOG_PATH}")
    except Exception as e:
        print(f"[AVISO] No se pudo escribir el log de resumen: {e}")


if __name__ == "__main__":
    main()
