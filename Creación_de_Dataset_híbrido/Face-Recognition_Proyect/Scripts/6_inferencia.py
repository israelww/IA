"""
6_inferencia.py
===============
Fase 6  Inferencia con CNN para Reconocimiento Facial

Pipeline de inferencia que utiliza el modelo entrenado en la Fase 5 para
reconocer rostros. Dos modos de uso: imagen subida y camara en vivo (Fase 7).

Funcionalidades:
  - Carga del checkpoint con reconstruccion automatica de la arquitectura
  - Deteccion de rostros: MTCNN (facenet_pytorch)  Haar Cascade  imagen completa
  - Inferencia con umbral de confianza para clasificar como "Desconocido"
  - Modo imagen subida: analiza, anota y guarda la imagen con bounding box
  - Deteccion multi-rostro para camara en vivo (exportable a Fase 7)
  - Visualizacion con bounding box coloreado segun confianza

Uso:
    python Scripts/6_inferencia.py --imagen ruta/foto.jpg
    python Scripts/6_inferencia.py --imagen ruta/foto.jpg --mostrar
    python Scripts/6_inferencia.py --imagen foto.jpg --modelo models/otro_run/best_model.pth

Autor: Generado automaticamente para el proyecto Face-Recognition Proyect
"""


import os
import sys
import json
import argparse
import warnings
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

#  Intentar importar MTCNN de facenet_pytorch 
try:
    from facenet_pytorch import MTCNN
    MTCNN_DISPONIBLE = True
except ImportError:
    MTCNN_DISPONIBLE = False

warnings.filterwarnings("ignore", category=UserWarning)

# Forzar UTF-8 en stdout/stderr para evitar errores de encoding en Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

#  Rutas del proyecto 

PROJECT_ROOT    = Path(__file__).resolve().parent.parent          # Face-Recognition Proyect/
MODELS_DIR      = PROJECT_ROOT / "models"
DEFAULT_MODEL   = MODELS_DIR / "best_model.pth"
RESULTADOS_DIR  = PROJECT_ROOT / "resultados"

#  Extensiones de imagen soportadas 

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Cache global para Haar Cascade (evita recargar en cada frame)
_haar_cascade = None
_haar_cascade_warned = False


def _get_haar_cascade():
    """
    Carga Haar Cascade de forma robusta.
    En Windows, algunas instalaciones de OpenCV fallan al abrir rutas con
    caracteres no ASCII; por eso se intenta copiar a %TEMP% y cargar desde ahi.
    """
    global _haar_cascade, _haar_cascade_warned

    if _haar_cascade is not None:
        return _haar_cascade

    cascade_name = "haarcascade_frontalface_default.xml"
    candidatos = []

    try:
        candidatos.append(Path(cv2.data.haarcascades) / cascade_name)
    except Exception:
        pass

    try:
        candidatos.append(Path(cv2.__file__).resolve().parent / "data" / cascade_name)
    except Exception:
        pass

    for ruta in candidatos:
        if not ruta.exists():
            continue

        # Intento directo
        cascade = cv2.CascadeClassifier(str(ruta))
        if not cascade.empty():
            _haar_cascade = cascade
            return _haar_cascade

        # Fallback: copiar a una ruta ASCII (temp) y volver a cargar
        try:
            temp_dir = Path(tempfile.gettempdir()) / "opencv_haar_cache"
            temp_dir.mkdir(parents=True, exist_ok=True)
            ruta_temp = temp_dir / cascade_name
            shutil.copyfile(str(ruta), str(ruta_temp))
            cascade = cv2.CascadeClassifier(str(ruta_temp))
            if not cascade.empty():
                _haar_cascade = cascade
                return _haar_cascade
        except Exception:
            continue

    if not _haar_cascade_warned:
        print("   No se pudo cargar Haar Cascade. Se omitira fallback Haar.")
        _haar_cascade_warned = True
    return None


# 
#  1) MODELO CNN (MISMA ARQUITECTURA QUE LA FASE 5)
# 

def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Construye el modelo de transfer learning.
    Opciones: resnet18, efficientnet_b0
    """
    model_name = model_name.lower().strip()

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    else:
        raise ValueError(
            f"Modelo '{model_name}' no soportado. Usa 'resnet18' o 'efficientnet_b0'."
        )

    return model


# 
#  2) CARGA DEL MODELO DESDE CHECKPOINT
# 

def load_model(checkpoint_path: str):
    """
    Carga un checkpoint .pth generado por la Fase 5.

    Reconstruye la arquitectura con build_model usando los metadatos embebidos
    en el checkpoint (model_name, num_classes), carga los pesos y pone el
    modelo en modo eval().

    Retorna: (model, class_names, img_size, confidence_threshold, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"\n   ERROR: No se encontro el checkpoint: {checkpoint_path}")
        print(f"     Verifica que el modelo fue entrenado en la Fase 5.")
        sys.exit(1)

    # Cargar checkpoint completo.
    # weights_only=False porque el checkpoint contiene objetos Python complejos
    # (listas, strings, dicts del optimizer) ademas de los tensores del modelo.
    # El checkpoint proviene de nuestro propio pipeline (fuente confiable).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extraer metadatos embebidos en el checkpoint
    model_name           = checkpoint["model_name"]
    num_classes          = checkpoint["num_classes"]
    class_names          = checkpoint["class_names"]
    img_size             = checkpoint["img_size"]
    confidence_threshold = checkpoint.get("confidence_threshold", 0.70)

    # Reconstruir arquitectura SIN pesos pretrained (los cargamos del checkpoint)
    model = build_model(model_name, num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    print(f"   Modelo cargado: {model_name} ({num_classes} clases)")
    print(f"   Checkpoint: {checkpoint_path.name} (epoca {checkpoint.get('epoch', '?')})")
    print(f"   Umbral de confianza: {confidence_threshold:.2f}")
    print(f"   Dispositivo: {device}")

    return model, class_names, img_size, confidence_threshold, device


# 
#  3) TRANSFORM DE EVALUACION (IDENTICO A LA FASE 5)
# 

def get_eval_transform(img_size: int):
    """
    Retorna el mismo transform de evaluacion usado en la Fase 5:
    Resize  ToTensor  Normalize con media/std de ImageNet.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# 
#  4) DETECCION DE ROSTROS (MTCNN  HAAR CASCADE  IMAGEN COMPLETA)
# 

# Instancias globales de MTCNN (se inicializan una sola vez, lazy)
# _mtcnn_single: detecta un solo rostro (para inferencia en imagen individual)
# _mtcnn_multi:  detecta todos los rostros (para camara en vivo, Fase 7)
_mtcnn_single = None
_mtcnn_multi  = None


def _get_mtcnn(device, keep_all: bool = False):
    """
    Inicializa MTCNN de forma lazy (solo la primera vez que se necesita).

    Args:
        device:   Dispositivo (cuda/cpu)
        keep_all: Si True, detecta todos los rostros del frame (modo camara).
                  Si False, detecta solo el rostro principal (modo imagen).
    """
    global _mtcnn_single, _mtcnn_multi

    if not MTCNN_DISPONIBLE:
        return None

    if keep_all:
        if _mtcnn_multi is None:
            _mtcnn_multi = MTCNN(
                keep_all=True,
                device=device,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
            )
        return _mtcnn_multi
    else:
        if _mtcnn_single is None:
            _mtcnn_single = MTCNN(
                keep_all=False,
                device=device,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
            )
        return _mtcnn_single


def _detectar_rostro_mtcnn(pil_image, device):
    """
    Detecta el rostro principal usando MTCNN (facenet_pytorch).
    Retorna: (bbox [x1, y1, x2, y2], True) si se detecta, o (None, False).
    """
    mtcnn = _get_mtcnn(device)
    if mtcnn is None:
        return None, False

    try:
        boxes, probs = mtcnn.detect(pil_image)
        if boxes is not None and len(boxes) > 0:
            # Tomar el rostro con mayor probabilidad
            best_idx = int(np.argmax(probs))
            bbox = boxes[best_idx].astype(int).tolist()
            return bbox, True
    except Exception:
        pass

    return None, False


def _detectar_rostro_haar(cv2_bgr):
    """
    Detecta el rostro principal usando Haar Cascade (fallback cuando
    MTCNN no esta disponible o no detecta nada).
    Retorna: (bbox [x1, y1, x2, y2], True) si se detecta, o (None, False).
    """
    cascade = _get_haar_cascade()
    if cascade is None:
        return None, False

    gray = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2GRAY)
    try:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
    except cv2.error:
        return None, False

    if len(faces) > 0:
        # Tomar el rostro mas grande (mayor area)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return [int(x), int(y), int(x + w), int(y + h)], True

    return None, False


def _recortar_rostro(pil_image, bbox, margen=0.15):
    """
    Recorta el rostro de la imagen con un margen adicional alrededor del
    bounding box. El margen evita perder informacion contextual util
    (frente, menton, orejas) que ayuda al reconocimiento.
    """
    ancho, alto = pil_image.size
    x1, y1, x2, y2 = bbox

    # Calcular margen proporcional al tamano del rostro
    face_w = x2 - x1
    face_h = y2 - y1
    margin_x = int(face_w * margen)
    margin_y = int(face_h * margen)

    # Aplicar margen sin salir de los limites de la imagen
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(ancho, x2 + margin_x)
    y2 = min(alto, y2 + margin_y)

    return pil_image.crop((x1, y1, x2, y2))


# 
#  5) PREPROCESAMIENTO DE IMAGEN
# 

def _cargar_imagen(source):
    """
    Carga una imagen desde multiples formatos de entrada.

    Acepta:
      - str o Path:    ruta a archivo de imagen
      - np.ndarray:    array BGR de OpenCV
      - PIL.Image:     imagen PIL

    Retorna: (PIL Image RGB, array NumPy BGR para OpenCV)
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {path}")
        pil_img = Image.open(str(path)).convert("RGB")
        # cv2.imread puede fallar con rutas Unicode en Windows,
        # usar np.fromfile + imdecode como alternativa robusta
        try:
            img_bytes = np.fromfile(str(path), dtype=np.uint8)
            cv2_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        except Exception:
            cv2_img = None
        if cv2_img is None:
            # Fallback: convertir PIL  NumPy  BGR
            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    elif isinstance(source, np.ndarray):
        # Array NumPy  se asume BGR (convencion de OpenCV)
        cv2_img = source.copy()
        pil_img = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))

    elif isinstance(source, Image.Image):
        pil_img = source.convert("RGB")
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    else:
        raise TypeError(
            f"Tipo de imagen no soportado: {type(source)}. "
            f"Usa str, Path, np.ndarray (BGR) o PIL.Image."
        )

    return pil_img, cv2_img


def preprocess_image(source, img_size: int, device):
    """
    Preprocesa una imagen para inferencia:
      1. Carga la imagen desde cualquier formato aceptado
      2. Detecta y recorta el rostro (MTCNN  Haar Cascade  imagen completa)
      3. Aplica el transform de evaluacion (Resize + ToTensor + Normalize)

    Args:
        source:   Ruta a archivo (str/Path), array NumPy BGR, o PIL Image
        img_size: Tamano de imagen esperado por el modelo (ej. 160)
        device:   Dispositivo (cuda/cpu)

    Retorna:
        tensor:        Tensor [1, 3, img_size, img_size] listo para el modelo
        face_detected: True si se detecto un rostro, False si se uso la imagen completa
        bbox:          [x1, y1, x2, y2] del rostro detectado, o None
    """
    pil_img, cv2_img = _cargar_imagen(source)

    #  Cascada de deteccion de rostros 
    # Intento 1: MTCNN (mas preciso, basado en deep learning)
    bbox, face_detected = _detectar_rostro_mtcnn(pil_img, device)

    # Intento 2: Haar Cascade (clasico, mas rapido pero menos preciso)
    if not face_detected:
        bbox, face_detected = _detectar_rostro_haar(cv2_img)

    # Intento 3: Usar la imagen completa redimensionada como ultimo recurso
    if face_detected and bbox is not None:
        face_crop = _recortar_rostro(pil_img, bbox)
    else:
        face_crop = pil_img  # Imagen completa
        bbox = None

    #  Aplicar transform de evaluacion (identico a Fase 5) 
    eval_transform = get_eval_transform(img_size)
    tensor = eval_transform(face_crop)
    tensor = tensor.unsqueeze(0).to(device)  # Agregar dimension de batch: [1, 3, H, W]

    return tensor, face_detected, bbox


# 
#  6) INFERENCIA
# 

@torch.no_grad()
def predict(source, model, class_names, img_size, confidence_threshold, device):
    """
    Ejecuta inferencia sobre una imagen.

    Args:
        source:               Imagen (ruta, array NumPy BGR, o PIL Image)
        model:                Modelo CNN cargado en modo eval()
        class_names:          Lista de nombres de clase
        img_size:             Tamano de imagen del modelo
        confidence_threshold: Umbral minimo para considerar la prediccion como "conocida"
        device:               Dispositivo (cuda/cpu)

    Retorna:
        dict con:
          - clase:         nombre de la clase predicha (o "Desconocido" si confianza < umbral)
          - confianza:     probabilidad softmax maxima (float 0-1)
          - conocido:      True si confianza >= confidence_threshold
          - top3:          lista de dicts {clase, confianza} con las 3 mejores predicciones
          - face_detected: True si se detecto un rostro en la imagen
          - bbox:          [x1, y1, x2, y2] del rostro detectado, o None
    """
    # Preprocesar imagen (deteccion de rostro + transform)
    tensor, face_detected, bbox = preprocess_image(source, img_size, device)

    # Forward pass
    outputs = model(tensor)
    probabilities = torch.softmax(outputs, dim=1).squeeze(0)  # [num_classes]

    # Obtener prediccion principal
    max_prob, max_idx = torch.max(probabilities, dim=0)
    confianza = round(max_prob.item(), 4)
    clase_idx = max_idx.item()

    # Determinar si es "conocido" segun el umbral de confianza
    conocido = confianza >= confidence_threshold
    clase = class_names[clase_idx] if conocido else "Desconocido"

    # Top-3 predicciones (o menos si hay pocas clases)
    k = min(3, len(class_names))
    top3_probs, top3_indices = torch.topk(probabilities, k=k)
    top3 = [
        {
            "clase":     class_names[idx.item()],
            "confianza": round(prob.item(), 4),
        }
        for prob, idx in zip(top3_probs, top3_indices)
    ]

    return {
        "clase":         clase,
        "confianza":     confianza,
        "conocido":      conocido,
        "top3":          top3,
        "face_detected": face_detected,
        "bbox":          bbox,
    }


# 
#  7) DETECCION MULTI-ROSTRO (PARA CAMARA EN VIVO  FASE 7)
# 

def detectar_todos_los_rostros(frame_bgr, device):
    """
    Detecta TODOS los rostros en un frame BGR de OpenCV.
    Disenada para ser usada en el loop de camara en vivo (Fase 7),
    donde puede haber varias personas en el encuadre.

    Cascada de deteccion:
      1. MTCNN con keep_all=True (detecta multiples rostros)
      2. Haar Cascade como fallback (detectMultiScale ya retorna multiples)

    Args:
        frame_bgr: Frame BGR de OpenCV (array NumPy)
        device:    Dispositivo (cuda/cpu)

    Retorna:
        Lista de bboxes [[x1, y1, x2, y2], ...]
        Lista vacia si no se detectan rostros.
    """
    bboxes = []

    #  Intento 1: MTCNN con keep_all=True 
    mtcnn = _get_mtcnn(device, keep_all=True)
    if mtcnn is not None:
        try:
            # MTCNN espera PIL Image en RGB
            pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            boxes, probs = mtcnn.detect(pil_img)
            if boxes is not None and len(boxes) > 0:
                for box, prob in zip(boxes, probs):
                    if prob is not None and prob > 0.5:
                        bboxes.append(box.astype(int).tolist())
                if bboxes:
                    return bboxes
        except Exception:
            pass

    #  Intento 2: Haar Cascade (fallback, tambien detecta multiples) 
    cascade = _get_haar_cascade()
    if cascade is None:
        return bboxes
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    try:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
    except cv2.error:
        return bboxes

    for (x, y, w, h) in faces:
        bboxes.append([int(x), int(y), int(x + w), int(y + h)])

    return bboxes


# 
#  8) IMAGEN SUBIDA  INFERENCIA + ANOTACION + GUARDADO
# 

def _anotar_imagen(cv2_img, result):
    """
    Dibuja bounding box y texto sobre una copia de la imagen original.
    Logica de anotacion reutilizada por predict_imagen_subida() y visualizar().

    Retorna: imagen cv2 anotada (copia, no modifica el original).
    """
    img_vis = cv2_img.copy()

    clase         = result["clase"]
    confianza     = result["confianza"]
    conocido      = result["conocido"]
    bbox          = result.get("bbox")
    face_detected = result["face_detected"]

    # Colores: Verde para conocido, Rojo para desconocido (formato BGR)
    color = (0, 200, 0) if conocido else (0, 0, 220)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness  = 2

    if face_detected and bbox is not None:
        x1, y1, x2, y2 = bbox

        #  Dibujar bounding box 
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

        #  Texto: nombre + porcentaje de confianza 
        texto = f"{clase} ({confianza*100:.1f}%)"
        (text_w, text_h), baseline = cv2.getTextSize(texto, font, font_scale, thickness)

        # Fondo solido detras del texto para legibilidad
        text_y = max(y1 - 8, text_h + 8)
        cv2.rectangle(
            img_vis,
            (x1, text_y - text_h - 8),
            (x1 + text_w + 8, text_y + 4),
            color, -1,
        )
        cv2.putText(
            img_vis, texto,
            (x1 + 4, text_y - 4),
            font, font_scale, (255, 255, 255), thickness,
        )

    else:
        #  Sin rostro detectado 
        cv2.putText(
            img_vis, "Sin rostro detectado",
            (15, 35),
            font, 0.9, (0, 0, 220), 2,
        )
        # Igualmente mostrar la prediccion (sobre imagen completa)
        pred_texto = f"Prediccion: {clase} ({confianza*100:.1f}%)"
        cv2.putText(
            img_vis, pred_texto,
            (15, 70),
            font, 0.7, (200, 200, 200), 2,
        )

    return img_vis


def predict_imagen_subida(imagen_path, model, class_names, img_size,
                          confidence_threshold, device):
    """
    Modo "imagen subida": analiza una sola imagen, la anota con el resultado
    y guarda la imagen anotada en disco.

    Pipeline:
      1. Llama a predict() para obtener la prediccion
      2. Dibuja bounding box y nombre sobre la imagen original
      3. Guarda la imagen anotada en resultados/imagen_anotada_<timestamp>.jpg
      4. Imprime resultado detallado en consola

    Args:
        imagen_path:          Ruta a la imagen
        model:                Modelo CNN cargado en modo eval()
        class_names:          Lista de nombres de clase
        img_size:             Tamano de imagen del modelo
        confidence_threshold: Umbral de confianza
        device:               Dispositivo (cuda/cpu)

    Retorna:
        dict con el resultado de predict() mas:
          - imagen_anotada: ruta absoluta (str) del archivo guardado
    """
    img_path = Path(imagen_path)
    if not img_path.exists():
        print(f"   ERROR: Imagen no encontrada: {img_path}")
        sys.exit(1)

    #  Ejecutar inferencia 
    result = predict(
        img_path, model, class_names,
        img_size, confidence_threshold, device,
    )

    #  Anotar la imagen original 
    _, cv2_img = _cargar_imagen(img_path)
    img_anotada = _anotar_imagen(cv2_img, result)

    #  Guardar imagen anotada 
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"imagen_anotada_{timestamp}.jpg"
    output_path = RESULTADOS_DIR / output_filename

    # Usar cv2.imencode para manejar rutas Unicode en Windows
    success, img_encoded = cv2.imencode(".jpg", img_anotada, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if success:
        img_encoded.tofile(str(output_path))
    else:
        cv2.imwrite(str(output_path), img_anotada)

    #  Imprimir resultado en consola 
    print(f"\n  {'-'*55}")
    print(f"  Resultado de la inferencia:")
    print(f"  {'-'*55}")
    print(f"  Clase:            {result['clase']}")
    print(f"  Confianza:        {result['confianza']*100:.1f}%")
    print(f"  Conocido:         {'Si' if result['conocido'] else 'No'}")
    print(f"  Rostro detectado: {'Si' if result['face_detected'] else 'No'}")
    print(f"\n  Top-3 predicciones:")
    for i, pred in enumerate(result["top3"], 1):
        print(f"    {i}. {pred['clase']:<25} {pred['confianza']*100:.1f}%")
    print(f"\n   Imagen anotada guardada en: {output_path}")
    print(f"  {'-'*55}")

    #  Agregar ruta de la imagen anotada al resultado 
    result["imagen_anotada"] = str(output_path)

    return result


# 
#  9) VISUALIZACION (--mostrar)
# 

def visualizar(source, result):
    """
    Muestra la imagen original con bounding box y nombre de la persona
    usando cv2.imshow. Reutiliza _anotar_imagen() para el dibujo.

      - Bounding box verde si conocido=True, rojo si conocido=False
      - Texto con nombre y porcentaje de confianza encima del bounding box
      - Si no se detecto rostro, muestra texto informativo en la esquina
    """
    # Cargar imagen original y anotarla
    _, cv2_img = _cargar_imagen(source)
    img_vis = _anotar_imagen(cv2_img, result)

    #  Mostrar imagen con cv2.imshow 
    clase = result["clase"]
    window_name = f"Inferencia - {clase}"
    cv2.imshow(window_name, img_vis)
    print(f"\n  Presiona cualquier tecla para cerrar la ventana...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 
#  10) FUNCION PRINCIPAL
# 

def main(args):
    print("\n" + "" * 70)
    print("  FASE 6  INFERENCIA CNN PARA RECONOCIMIENTO FACIAL")
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

    if not MTCNN_DISPONIBLE:
        print(f"   facenet_pytorch no instalado. Se usara solo Haar Cascade para deteccion.")

    print()

    #  Modo: imagen subida 
    if args.imagen:
        print(f"   Imagen: {args.imagen}")

        result = predict_imagen_subida(
            args.imagen, model, class_names,
            img_size, confidence_threshold, device,
        )

        # Visualizacion opcional con cv2.imshow si se pidio --mostrar
        if args.mostrar:
            visualizar(args.imagen, result)

    else:
        print("   Debes especificar --imagen con la ruta a la imagen.")
        print("  Usa --help para ver las opciones disponibles.")
        sys.exit(1)

    print(f"\n{'-'*70}\n")


# 
#  ENTRY POINT
# 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 6  Inferencia CNN para Reconocimiento Facial",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--imagen",  type=str, default=None,
                        help="Ruta a una imagen para analizar. Guarda la imagen "
                             "anotada automaticamente en resultados/.")
    parser.add_argument("--modelo",  type=str, default=None,
                        help="Ruta al checkpoint .pth del modelo. "
                             "Por defecto: models/best_model.pth")
    parser.add_argument("--mostrar", action="store_true",
                        help="Ademas de guardar, abre la imagen anotada con "
                             "cv2.imshow para visualizacion interactiva.")

    args = parser.parse_args()
    main(args)


