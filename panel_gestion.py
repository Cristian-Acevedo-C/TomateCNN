"""
Proyecto: TomateCNN Vision
Script:   panel_gestion.py
Autor:    Cristian Acevedo

Descripcion general
-------------------
Panel web construido con Streamlit para operar el modelo TFLite de
deteccion de enfermedades en hojas de tomate. Permite:

- Capturar imagen desde la camara (Streamlit camera_input).
- Aplicar inferencia sobre la region central (ROI) de la imagen.
- Clasificar la hoja como SANA / ENFERMA / DUDOSA.
- Llevar contadores de la sesion (sanas, enfermas, dudosas, total).
- Registrar cada analisis en un archivo CSV.
- Guardar la imagen original en disco.
- Mostrar historial de escaneos en una tabla interactiva.
"""

import datetime
import json
import time
import csv
from pathlib import Path
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


# ----------------------------------------------------------------------
# 1. CONFIGURACION DE PAGINA (DEBE IR ANTES DE CUALQUIER STREAMLIT)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="TomateCNN Vision",
    page_icon="🍅",
    layout="wide",
)


# ----------------------------------------------------------------------
# 2. CONFIGURACION GENERAL DEL PROYECTO
# ----------------------------------------------------------------------
class Config:
    """Parametros basicos utilizados por el panel."""

    MODEL_PATH: Path = Path("model.tflite")
    LABELS_PATH: Path = Path("labels.json")

    # Carpeta donde se guardan las capturas de la camara
    OUTPUT_DIR: Path = Path("capturas_tomate")

    # Umbral para marcar una clasificacion como "dudosa"
    UMBRAL_DUDOSO: float = 60.0

    # Traductor de nombres tecnicos (ingles) a descripciones en espanol
    TRADUCCIONES: Dict[str, str] = {
        "Bacterial_spot": "MANCHA BACTERIANA",
        "Early_blight": "TIZON TEMPRANO",
        "Late_blight": "TIZON TARDIO",
        "Leaf_Mold": "MOHO FOLIAR",
        "Septoria_leaf_spot": "MANCHA SEPTORIA",
        "Spider_mites Two-spotted_spider_mite": "ACARO ROJO",
        "Target_Spot": "MANCHA DIANA",
        "Tomato_Yellow_Leaf_Curl_Virus": "VIRUS RIZADO AMARILLO",
        "Tomato_mosaic_virus": "VIRUS DEL MOSAICO",
        "healthy": "HOJA SANA",
    }


# Crear carpeta de salida si no existe
Config.OUTPUT_DIR.mkdir(exist_ok=True)

# Nombre de archivo CSV por sesion
SESSION_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = Config.OUTPUT_DIR / f"registro_{SESSION_ID}.csv"

# Crear CSV inicial si no existe
if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fecha_hora", "archivo", "estado", "enfermedad", "confianza_%"])


# ----------------------------------------------------------------------
# 3. MOTOR DE INFERENCIA (CACHEADO EN MEMORIA)
# ----------------------------------------------------------------------
@st.cache_resource
def cargar_sistema_inferencia():
    """
    Carga el modelo TFLite y el archivo de etiquetas.

    Devuelve
    -------
    interpreter : tf.lite.Interpreter
        Motor TFLite listo para usarse.
    input_det : dict
        Detalles del tensor de entrada.
    output_det : dict
        Detalles del tensor de salida.
    labels : list[str]
        Lista de nombres de clases (en ingles).
    height, width : int
        Altura y ancho esperados por el modelo.
    input_dtype : np.dtype
        Tipo de dato requerido por el tensor de entrada.
    """
    if not Config.MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontro el modelo: {Config.MODEL_PATH}")

    interpreter = tf.lite.Interpreter(model_path=str(Config.MODEL_PATH))
    interpreter.allocate_tensors()

    input_det = interpreter.get_input_details()[0]
    output_det = interpreter.get_output_details()[0]

    height = int(input_det["shape"][1])
    width = int(input_det["shape"][2])
    input_dtype = input_det["dtype"]

    # Carga robusta de labels.json (soporta distintos formatos)
    labels: List[str]
    try:
        if not Config.LABELS_PATH.exists():
            raise FileNotFoundError(f"No se encontro labels.json en {Config.LABELS_PATH}")

        with open(Config.LABELS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels = [""] * len(data)

        # Formato 1: {"Tomato___Bacterial_spot": 0, ...}
        if any("Tomato___" in k for k in data.keys()):
            for full, idx in data.items():
                labels[int(idx)] = full.replace("Tomato___", "")
        # Formato 2: {"0": "Tomato___Bacterial_spot", ...}
        elif any(k.isdigit() for k in data.keys()):
            for k, v in data.items():
                labels[int(k)] = v.replace("Tomato___", "")
        # Formato generico
        else:
            for k, v in data.items():
                labels[int(k)] = v
    except Exception:
        # En caso de error, se usan etiquetas genericas
        labels = [f"CLASE_{i}" for i in range(10)]

    return interpreter, input_det, output_det, labels, height, width, input_dtype


interpreter, input_det, output_det, labels, HEIGHT, WIDTH, INPUT_DTYPE = cargar_sistema_inferencia()


# ----------------------------------------------------------------------
# 4. ESTADO DE SESION (CONTADORES E HISTORIAL)
# ----------------------------------------------------------------------
if "historial" not in st.session_state:
    st.session_state.update(
        {
            "total_sanas": 0,
            "total_enfermas": 0,
            "total_dudosas": 0,
            "total_fotos": 0,
            "historial": [],
        }
    )


# ----------------------------------------------------------------------
# 5. LOGICA PRINCIPAL DE DIAGNOSTICO
# ----------------------------------------------------------------------
def procesar_muestra(img_pil: Image.Image) -> Tuple[str, str, float, Path, np.ndarray]:
    """
    Ejecuta el pipeline completo de inferencia para una imagen capturada.

    Pasos:
    - Convierte de PIL a OpenCV.
    - Extrae una region central (ROI) cuadrada.
    - Redimensiona al tamano esperado por el modelo.
    - Aplica el preprocesamiento segun el tipo de dato.
    - Ejecuta la inferencia TFLite.
    - Determina el diagnostico y actualiza contadores.
    - Guarda la imagen y el registro en el CSV.

    Parametros
    ----------
    img_pil : PIL.Image.Image
        Imagen capturada desde la camara de Streamlit.

    Devuelve
    --------
    nombre_es : str
        Nombre de la enfermedad (o sano) en espanol.
    estado : str
        Categoria final: "SANA", "ENFERMA" o "DUDOSO".
    conf : float
        Confianza del modelo en porcentaje (0 a 100).
    path : Path
        Ruta del archivo de imagen guardado.
    roi : np.ndarray
        Region de interes utilizada para la inferencia (formato BGR).
    """
    # PIL -> OpenCV (RGB -> BGR)
    img_np = np.array(img_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ROI central cuadrada
    h_img, w_img = img_cv.shape[:2]
    roi_size = min(h_img, w_img) // 2
    center_x, center_y = w_img // 2, h_img // 2

    y1 = max(center_y - roi_size // 2, 0)
    y2 = min(center_y + roi_size // 2, h_img)
    x1 = max(center_x - roi_size // 2, 0)
    x2 = min(center_x + roi_size // 2, w_img)

    roi = img_cv[y1:y2, x1:x2]

    # Redimension y preprocesamiento
    resized = cv2.resize(roi, (WIDTH, HEIGHT))
    data = np.expand_dims(resized, axis=0).astype(INPUT_DTYPE)

    if INPUT_DTYPE == np.float32:
        # Modelo entrenado con imagenes normalizadas a [0, 1]
        data = data / 255.0

    # Inferencia TFLite
    interpreter.set_tensor(input_det["index"], data)
    interpreter.invoke()
    out = interpreter.get_tensor(output_det["index"])[0]

    idx = int(np.argmax(out))
    conf = float(out[idx] * 100.0)

    nombre_ing = labels[idx]
    nombre_es = Config.TRADUCCIONES.get(
        nombre_ing, nombre_ing.replace("_", " ").upper()
    )

    # Clasificacion segun umbral y tipo de clase
    if conf < Config.UMBRAL_DUDOSO:
        estado = "DUDOSO"
        st.session_state.total_dudosas += 1
    elif nombre_ing == "healthy":
        estado = "SANA"
        st.session_state.total_sanas += 1
    else:
        estado = "ENFERMA"
        st.session_state.total_enfermas += 1

    st.session_state.total_fotos += 1

    # Guardar imagen completa en disco
    ahora = datetime.datetime.now()
    ts = ahora.strftime("%Y%m%d_%H%M%S")
    filename = f"{estado}_{nombre_es.replace(' ', '_')}_{ts}.jpg"
    path = Config.OUTPUT_DIR / filename
    cv2.imwrite(str(path), img_cv)

    # Registrar en CSV
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [ahora.isoformat(), filename, estado, nombre_es, f"{conf:.2f}"]
        )

    # Registrar en historial en memoria (tabla de la derecha)
    st.session_state.historial.append(
        {
            "Fecha/Hora": ahora.strftime("%d-%m-%Y %H:%M:%S"),
            "Diagnostico": nombre_es,
            "Estado": estado,
            "Confianza": f"{conf:.1f}%",
            "Archivo": filename,
        }
    )

    return nombre_es, estado, conf, path, roi


# ----------------------------------------------------------------------
# 6. ESTILOS CSS PERSONALIZADOS
# ----------------------------------------------------------------------
st.markdown(
    """
<style>
    .main {background-color: #0e1117;}
    h1 {color: #00e676 !important; font-family: 'Roboto', sans-serif;}
    h2, h3 {color: #e0e0e0 !important;}

    div[data-testid="stMetric"] {
        background-color: #1a1c24;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #00e676;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    .status-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .status-sana {
        background-color: #064e3b;
        border: 2px solid #059669;
        color: #a7f3d0;
    }
    .status-enferma {
        background-color: #7f1d1d;
        border: 2px solid #dc2626;
        color: #fecaca;
    }
    .status-dudoso {
        background-color: #78350f;
        border: 2px solid #d97706;
        color: #fde68a;
    }

    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# 7. INTERFAZ DE USUARIO (STREAMLIT)
# ----------------------------------------------------------------------
st.title("TomateCNN Vision")
st.markdown("Sistema de diagnostico de enfermedades en hojas de tomate")
st.markdown("---")

col_izq, col_der = st.columns([1.5, 1])


# --- PANEL IZQUIERDO: CAMARA + RESULTADOS ---
with col_izq:
    st.subheader("📡 Modulo de analisis en vivo")
    img_file = st.camera_input(
        "Capture una imagen de la hoja (ROI central)", key="camera"
    )

    if img_file is not None:
        img_pil = Image.open(img_file)

        if st.button("▶ Ejecutar diagnostico", type="primary"):
            with st.spinner("Procesando inferencia..."):
                # Pausa minima para que el spinner sea visible
                time.sleep(0.3)
                nombre, estado, conf, path, roi_img = procesar_muestra(img_pil)

            st.markdown("### Resultado del analisis")

            # Bloques de estado
            if estado == "SANA":
                st.markdown(
                    f"""
                    <div class="status-box status-sana">
                        <h2>✅ MUESTRA SANA</h2>
                        <p style="font-size: 18px;">
                            Diagnostico: <b>{nombre}</b> | Confianza: <b>{conf:.1f}%</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif estado == "ENFERMA":
                st.markdown(
                    f"""
                    <div class="status-box status-enferma">
                        <h2>🚨 PATOLOGIA DETECTADA</h2>
                        <p style="font-size: 18px;">
                            Diagnostico: <b>{nombre}</b> | Confianza: <b>{conf:.1f}%</b>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="status-box status-dudoso">
                        <h2>⚠️ CLASIFICACION DUDOSA</h2>
                        <p style="font-size: 18px;">
                            Posible: <b>{nombre}</b> | Confianza: <b>{conf:.1f}%</b>
                        </p>
                        <small>Se recomienda repetir la captura con mejor iluminacion y enfoque.</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Evidencia visual: imagen completa + ROI usado
            st.markdown("#### Evidencia visual")
            c1, c2 = st.columns(2)
            with c1:
                st.image(
                    img_pil,
                    caption="Imagen completa (campo amplio)",
                    use_column_width=True,
                )
            with c2:
                roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
                st.image(
                    roi_rgb,
                    caption="Region de interes procesada por la IA",
                    use_column_width=True,
                )

            st.success(f"Registro guardado como: `{path.name}`")


# --- PANEL DERECHO: METRICAS E HISTORIAL ---
with col_der:
    st.subheader("📊 Resumen de sesion")

    k1, k2 = st.columns(2)
    k1.metric("Hojas sanas", st.session_state.total_sanas)
    k2.metric("Hojas enfermas", st.session_state.total_enfermas)

    k3, k4 = st.columns(2)
    k3.metric("Clasif. dudosas", st.session_state.total_dudosas)
    k4.metric("Total analizadas", st.session_state.total_fotos)

    st.markdown("---")
    st.subheader("📝 Historial de escaneos")

    if st.session_state.historial:
        df = pd.DataFrame(st.session_state.historial)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=350,
        )

        # Boton para descargar el CSV de la sesion
        if CSV_PATH.exists():
            with open(CSV_PATH, "rb") as f:
                st.download_button(
                    label="⬇️ Exportar reporte CSV",
                    data=f,
                    file_name=CSV_PATH.name,
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.info("Aun no se han registrado analisis en esta sesion.")


# Footer simple con ID de sesion
st.markdown("---")
st.markdown(
    f"<center style='color: #666; font-size: 12px;'>TomateCNN · Sesion ID: {SESSION_ID}</center>",
    unsafe_allow_html=True,
)
