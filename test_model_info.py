i"""
Proyecto: Deteccion de enfermedades en hojas de tomate
Script:   test_model_info.py
Autor:    Cristian Acevedo

Descripcion:
    Utilidad rapida para inspeccionar el modelo TFLite:
    - Ver forma y tipo de entrada/salida.
    - Probar una inferencia con ruido aleatorio.
    - Dar pistas sobre el preprocesamiento esperado.
"""

import os
from pathlib import Path

import numpy as np

# ============================================================
# CARGA ROBUSTA DEL INTERPRETE TFLITE
# ============================================================
try:
    # Opcion ligera: tflite_runtime (Raspberry Pi / Linux)
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    print(">> Usando libreria ligera: tflite_runtime")
except ImportError:
    try:
        # Opcion PC: TensorFlow completo
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print(">> Usando TensorFlow (tf.lite.Interpreter)")
    except ImportError:
        print("ERROR: No se encontro ni tflite_runtime ni TensorFlow.")
        print("Instala al menos uno, por ejemplo:")
        print("    pip install tensorflow")
        raise SystemExit(1)

# ============================================================
# CONFIGURACION BASICA
# ============================================================
BASE_DIR = Path(__file__).parent
MODEL_NAME = "model.tflite"
MODEL_PATH = BASE_DIR / MODEL_NAME

# ============================================================
# FUNCION PRINCIPAL
# ============================================================
def main() -> None:
    """
    Carga el modelo TFLite y muestra informacion basica sobre:
      - Forma de entrada (input_shape)
      - Tipo de dato (dtype)
      - Forma de salida
    Ademas ejecuta una inferencia con ruido aleatorio para asegurar
    que el modelo responde correctamente.
    """
    if not MODEL_PATH.exists():
        print(f"\nERROR: No se encontro el archivo de modelo: {MODEL_PATH}")
        print("Asegurate de copiar 'model.tflite' a la carpeta del proyecto.")
        return

    # Cargar interprete
    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details["shape"]   # ej: [1, 160, 160, 3]
    input_dtype = input_details["dtype"]  # normalmente float32
    output_shape = output_details["shape"]

    print("\n" + "=" * 46)
    print("        DIAGNOSTICO RAPIDO DEL MODELO       ")
    print("=" * 46)
    print(f"Archivo modelo    : {MODEL_NAME}")
    print(f"1) Input shape    : {input_shape}")
    print(f"2) Input dtype    : {input_dtype}")
    print(f"3) Output shape   : {output_shape}")
    print("-" * 46)

    # ========================================================
    # PRUEBA DE INFERENCIA CON RUIDO
    # ========================================================
    # Generamos un "batch" de prueba con ruido aleatorio
    if input_dtype == np.uint8:
        dummy = np.random.randint(0, 256, size=input_shape, dtype=input_dtype)
        print(
            "NOTA: El modelo espera datos uint8 (0–255).\n"
            "      Normalmente NO se divide por 255.\n"
        )
    else:
        dummy = np.random.rand(*input_shape).astype(input_dtype)
        print(
            "NOTA: El modelo espera datos float.\n"
            "      Para modelos tipo MobileNetV2 (como el de este proyecto)\n"
            "      se usa normalmente:\n"
            "          x = imagen / 127.5 - 1.0   # rango [-1, 1]\n"
        )

    # Enviamos el tensor al modelo
    interpreter.set_tensor(input_details["index"], dummy)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details["index"])[0]

    print("-" * 46)
    print("4) Vector de prediccion con ruido (primer forward):")
    print(output_data)

    top_idx = int(np.argmax(output_data))
    top_conf = float(np.max(output_data))

    print(f"\n5) Clase con mayor probabilidad (con ruido): {top_idx}")
    print(f"   Confianza maxima: {top_conf:.4f}")
    print("=" * 46)
    print("DIAGNOSTICO: Modelo cargado correctamente y responde a inferencia.")
    print("Listo para usar en detector_tomates.py o procesador_lotes.py\n")


if __name__ == "__main__":
    main()
