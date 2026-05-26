"""
7_pipeline_camara.py
====================
Fase 7  Pipeline de Camara en Vivo para Reconocimiento Facial

Captura video de la camara, detecta todos los rostros en cada frame,
identifica cada persona usando el modelo entrenado en la Fase 5, y
muestra los resultados en tiempo real con bounding boxes coloreados.

Funcionalidades:
  - Deteccion multi-rostro en tiempo real (MTCNN  Haar Cascade)
  - Reconocimiento facial con umbral de confianza
  - Optimizacion con skip_frames y resize_factor para mantener fluidez
  - Grabacion opcional del video procesado
  - Screenshots con tecla 'S'
  - Reporte de sesion al finalizar

Uso:
    python Scripts/7_pipeline_camara.py
    python Scripts/7_pipeline_camara.py --fuente 1
    python Scripts/7_pipeline_camara.py --fuente 0 --skip_frames 3 --resize_factor 0.4
    python Scripts/7_pipeline_camara.py --fuente 0 --grabar
    python Scripts/7_pipeline_camara.py --modelo models/otro_run/best_model.pth

Autor: Generado automaticamente para el proyecto Face-Recognition Proyect
"""


import sys
import time
import argparse
import warnings
import importlib.util
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import cv2

warnings.filterwarnings("ignore", category=UserWarning)

# Forzar UTF-8 en stdout/stderr para evitar errores de encoding en Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# 
#  1) IMPORTACION DINAMICA DE LA FASE 6
# 
# El archivo 6_inferencia.py comienza con un numero, por lo que no se puede
# importar con `import` directo. Usamos importlib para cargarlo dinamicamente.

def _importar_fase6():
    """Importa el modulo 6_inferencia.py dinamicamente."""
    scripts_dir = Path(__file__).resolve().parent
    fase6_path  = scripts_dir / "6_inferencia.py"

    if not fase6_path.exists():
        print(f"\n   ERROR: No se encontro 6_inferencia.py en: {scripts_dir}")
        print(f"     Este script depende de la Fase 6 para cargar el modelo y hacer predicciones.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("inferencia", fase6_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["inferencia"] = mod
    spec.loader.exec_module(mod)
    return mod


_fase6 = _importar_fase6()

# Funciones importadas de la Fase 6
load_model                 = _fase6.load_model
predict                    = _fase6.predict
detectar_todos_los_rostros = _fase6.detectar_todos_los_rostros
preprocess_image           = _fase6.preprocess_image


#  Rutas del proyecto 

PROJECT_ROOT    = Path(__file__).resolve().parent.parent          # Face-Recognition Proyect/
MODELS_DIR      = PROJECT_ROOT / "models"
DEFAULT_MODEL   = MODELS_DIR / "best_model.pth"
RESULTADOS_DIR  = PROJECT_ROOT / "resultados"
SCREENSHOTS_DIR = RESULTADOS_DIR / "screenshots"
GRABACIONES_DIR = RESULTADOS_DIR / "grabaciones"


# 
#  2) FUNCIONES AUXILIARES DE DIBUJO
# 

def _crop_con_margen(frame, bbox, margen=0.15):
    """
    Recorta un rostro del frame con un margen adicional alrededor del bbox.
    El margen proporciona contexto extra al modelo para mejor reconocimiento.

    Args:
        frame:  Frame BGR de OpenCV (array NumPy)
        bbox:   [x1, y1, x2, y2] del rostro
        margen: Proporcion de margen a anadir (0.15 = 15%)

    Retorna:
        Crop del frame como array NumPy BGR, o array vacio si bbox invalido.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Calcular margen proporcional al tamano del rostro
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margen)
    my = int(bh * margen)

    # Aplicar margen sin salir de los limites del frame
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    return frame[y1:y2, x1:x2]


def _dibujar_rostro(frame, bbox, result):
    """
    Dibuja el bounding box y la etiqueta de un rostro sobre el frame.

      - Verde (0, 200, 0) si conocido=True
      - Rojo  (0, 0, 220) si conocido=False
      - Texto con fondo semitransparente para legibilidad
    """
    clase     = result["clase"]
    confianza = result["confianza"]
    conocido  = result["conocido"]

    x1, y1, x2, y2 = bbox

    # Color segun si es conocido o no
    color = (0, 200, 0) if conocido else (0, 0, 220)

    #  Bounding box 
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    #  Etiqueta: nombre + porcentaje 
    if conocido:
        texto = f"{clase} ({confianza*100:.1f}%)"
    else:
        texto = f"Desconocido ({confianza*100:.1f}%)"

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness  = 2
    (text_w, text_h), baseline = cv2.getTextSize(texto, font, font_scale, thickness)

    # Fondo semitransparente detras del texto
    text_y = max(y1 - 6, text_h + 6)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x1, text_y - text_h - 6),
        (x1 + text_w + 8, text_y + 4),
        color, -1,
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Texto blanco sobre el fondo
    cv2.putText(
        frame, texto,
        (x1 + 4, text_y - 2),
        font, font_scale, (255, 255, 255), thickness,
    )


def _dibujar_hud(frame, fps, n_rostros, model_name, grabando=False):
    """
    Dibuja el HUD (heads-up display) en la esquina superior izquierda:
      - FPS actual (media movil)
      - Numero de rostros detectados
      - Nombre del modelo
      - Indicador de grabacion si esta activa
    """
    lineas = [
        f"FPS: {fps:.1f}",
        f"Rostros: {n_rostros}",
        f"Modelo: {model_name}",
    ]
    if grabando:
        lineas.append("REC")

    #  Fondo semitransparente 
    line_height = 24
    panel_h = line_height * len(lineas) + 16
    panel_w = 260

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    #  Texto del HUD 
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1

    for i, linea in enumerate(lineas):
        y = 26 + i * line_height

        if linea == "REC":
            # Indicador de grabacion: circulo rojo + texto
            cv2.circle(frame, (20, y - 5), 6, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (32, y), font, font_scale, (0, 0, 255), thickness + 1)
        else:
            cv2.putText(frame, linea, (12, y), font, font_scale, (0, 255, 0), thickness)


def _guardar_screenshot(frame, screenshots_dir):
    """
    Guarda el frame actual como screenshot con timestamp en el nombre.
    Retorna la ruta del archivo guardado.
    """
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"screenshot_{timestamp}.jpg"
    filepath  = screenshots_dir / filename

    # Usar cv2.imencode para manejar rutas Unicode en Windows
    success, img_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if success:
        img_encoded.tofile(str(filepath))
    else:
        cv2.imwrite(str(filepath), frame)

    print(f"   Screenshot guardado: {filepath}")
    return filepath


def _formatear_duracion(segundos):
    """Convierte segundos a formato HH:MM:SS."""
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# 
#  3) REPORTE DE SESION
# 

def _imprimir_reporte(duracion, fps_promedio, frames, conocidos, desconocidos,
                      screenshots, grabacion_path):
    """Imprime el resumen de la sesion de camara al cerrar."""
    print(f"\n{'-'*50}")
    print(f"  RESUMEN DE SESION")
    print(f"{'-'*50}")
    print(f"  Duracion:              {_formatear_duracion(duracion)}")
    print(f"  FPS promedio:          {fps_promedio:.1f}")
    print(f"  Frames procesados:     {frames}")
    print(f"  Rostros conocidos:     {conocidos}")
    print(f"  Rostros desconocidos:  {desconocidos}")
    print(f"  Screenshots guardados: {screenshots}")
    if grabacion_path:
        print(f"  Video guardado en:     {grabacion_path}")
    else:
        print(f"  Grabacion:             desactivada")
    print(f"{'-'*50}\n")


# 
#  4) PIPELINE PRINCIPAL DE CAMARA EN VIVO
# 

def run_pipeline(args, model, class_names, img_size, confidence_threshold, device,
                 model_display_name):
    """
    Loop principal de reconocimiento facial en vivo.

    Captura frames de la camara, detecta rostros, ejecuta inferencia
    y muestra resultados en tiempo real con bounding boxes.

    Optimizaciones:
      - skip_frames: ejecuta deteccion + prediccion solo cada N frames
      - resize_factor: reduce el tamano del frame para deteccion mas rapida
    """

    #  Abrir camara 
    cap = cv2.VideoCapture(args.fuente)
    if not cap.isOpened():
        print(f"\n   ERROR: No se pudo abrir la camara con indice {args.fuente}")
        print(f"     Verifica que la camara esta conectada y no la usa otra aplicacion.")
        sys.exit(1)

    # Leer un frame de prueba para obtener las dimensiones
    ret, test_frame = cap.read()
    if not ret:
        print(f"\n   ERROR: La camara {args.fuente} se abrio pero no devuelve frames.")
        cap.release()
        sys.exit(1)

    frame_h, frame_w = test_frame.shape[:2]

    #  Configurar grabacion si esta activada 
    writer         = None
    grabacion_path = None

    if args.grabar:
        GRABACIONES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        grabacion_path = GRABACIONES_DIR / f"grabacion_{timestamp}.avi"

        # Usar FPS reportado por la camara, o 30 como fallback
        cam_fps = cap.get(cv2.CAP_PROP_FPS)
        if cam_fps <= 0:
            cam_fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(grabacion_path), fourcc, cam_fps, (frame_w, frame_h))
        print(f"   Grabando en: {grabacion_path}")

    #  Crear directorio de screenshots 
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    #  Variables de estado 
    frame_count      = 0
    known_count      = 0
    unknown_count    = 0
    screenshot_count = 0

    # FPS con media movil de los ultimos 30 frames
    fps_buffer = deque(maxlen=30)
    last_time  = time.time()
    fps        = 0.0

    # Resultados en cache para redibujar en frames intermedios (skip_frames)
    cached_results = []  # lista de (bbox, result_dict)

    start_time  = time.time()
    window_name = "Fase 7 - Reconocimiento Facial en Vivo"

    # Ventana redimensionable
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"\n   Camara abierta: indice {args.fuente} ({frame_w}{frame_h})")
    print(f"    Skip frames: {args.skip_frames} | Resize factor: {args.resize_factor}")
    print(f"  Presiona 'Q' para salir, 'S' para screenshot\n")

    #  Procesar el frame de prueba como frame #1 
    # (para no perder el frame que ya leimos)
    frames_pendientes = [test_frame]

    # 
    #  BUCLE PRINCIPAL
    # 

    while True:
        # Obtener el siguiente frame (primero el frame de prueba, luego la camara)
        if frames_pendientes:
            frame = frames_pendientes.pop(0)
            ret   = True
        else:
            ret, frame = cap.read()

        if not ret:
            print("   No se pudo leer frame de la camara. Saliendo...")
            break

        frame_count += 1

        #  Calcular FPS (media movil) 
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        if dt > 0:
            fps_buffer.append(dt)
        if len(fps_buffer) > 0:
            fps = 1.0 / (sum(fps_buffer) / len(fps_buffer))

        #  Deteccion + prediccion (cada skip_frames frames) 
        # Frame 1, 1+skip, 1+2*skip, ... ejecutan deteccion completa
        if (frame_count - 1) % args.skip_frames == 0:

            # Redimensionar frame para acelerar la deteccion
            if args.resize_factor < 1.0:
                new_w = int(frame_w * args.resize_factor)
                new_h = int(frame_h * args.resize_factor)
                small_frame = cv2.resize(frame, (new_w, new_h))
            else:
                small_frame = frame

            # Detectar todos los rostros en el frame (redimensionado)
            bboxes = detectar_todos_los_rostros(small_frame, device)

            # Escalar bboxes de vuelta al tamano original del frame
            if args.resize_factor < 1.0:
                scale  = 1.0 / args.resize_factor
                bboxes = [
                    [int(b[0] * scale), int(b[1] * scale),
                     int(b[2] * scale), int(b[3] * scale)]
                    for b in bboxes
                ]

            # Ejecutar prediccion para cada rostro detectado
            cached_results = []

            for bbox in bboxes:
                # Recortar el rostro del frame original (con margen)
                crop = _crop_con_margen(frame, bbox)
                if crop.size == 0:
                    continue

                # Llamar a predict() con el crop como array NumPy BGR
                result = predict(
                    crop, model, class_names,
                    img_size, confidence_threshold, device,
                )
                cached_results.append((bbox, result))

                # Actualizar contadores de sesion
                if result["conocido"]:
                    known_count += 1
                else:
                    unknown_count += 1

        #  Dibujar resultados sobre el frame 
        # En frames intermedios (skip), redibujar los resultados en cache
        for bbox, result in cached_results:
            _dibujar_rostro(frame, bbox, result)

        #  Escribir en el video si se esta grabando 
        if writer is not None:
            writer.write(frame)

        #  Mostrar frame 
        cv2.imshow(window_name, frame)

        #  Manejar teclado 
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

        elif key == ord("s") or key == ord("S"):
            _guardar_screenshot(frame, SCREENSHOTS_DIR)
            screenshot_count += 1

    # 
    #  CIERRE Y LIMPIEZA
    # 

    total_time = time.time() - start_time

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    #  Reporte de sesion 
    fps_promedio = frame_count / total_time if total_time > 0 else 0

    _imprimir_reporte(
        duracion=total_time,
        fps_promedio=fps_promedio,
        frames=frame_count,
        conocidos=known_count,
        desconocidos=unknown_count,
        screenshots=screenshot_count,
        grabacion_path=str(grabacion_path) if grabacion_path else None,
    )


# 
#  5) FUNCION PRINCIPAL
# 

def main(args):
    print("\n" + "" * 70)
    print("  FASE 7  PIPELINE DE CAMARA EN VIVO")
    print("" * 70)

    #  Determinar ruta del modelo 
    modelo_path = Path(args.modelo) if args.modelo else DEFAULT_MODEL

    if not modelo_path.exists():
        print(f"\n   ERROR: Modelo no encontrado: {modelo_path}")
        print(f"     Entrena primero un modelo con la Fase 5 (5_entrenar_cnn.py).")
        sys.exit(1)

    #  Cargar modelo 
    print(f"\n  Cargando modelo...")
    model, class_names, img_size, confidence_threshold, device = load_model(modelo_path)

    # Obtener nombre legible del modelo para el HUD
    model_display_name = type(model).__name__  # "ResNet" o "EfficientNet"

    print(f"   Clases: {len(class_names)} ({', '.join(class_names[:5])}{'...' if len(class_names) > 5 else ''})")
    print()

    #  Ejecutar pipeline de camara 
    run_pipeline(
        args, model, class_names,
        img_size, confidence_threshold, device,
        model_display_name,
    )


# 
#  ENTRY POINT
# 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 7  Pipeline de Camara en Vivo para Reconocimiento Facial",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fuente",        type=int,   default=0,
                        help="Indice de la camara a usar (0 = camara por defecto).")
    parser.add_argument("--skip_frames",   type=int,   default=2,
                        help="Ejecutar deteccion + prediccion cada N frames. "
                             "En frames intermedios se redibujan los ultimos resultados.")
    parser.add_argument("--resize_factor", type=float, default=0.5,
                        help="Factor de escala del frame para la deteccion de rostros "
                             "(0.5 = mitad de tamano). El crop para predict() usa el "
                             "frame original.")
    parser.add_argument("--grabar",        action="store_true",
                        help="Graba el video procesado en resultados/grabaciones/ "
                             "con codec XVID.")
    parser.add_argument("--modelo",        type=str,   default=None,
                        help="Ruta al checkpoint .pth del modelo. "
                             "Por defecto: models/best_model.pth")

    args = parser.parse_args()
    main(args)


