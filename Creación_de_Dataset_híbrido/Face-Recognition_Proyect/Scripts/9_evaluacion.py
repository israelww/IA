# -*- coding: utf-8 -*-
"""
9_evaluacion.py
===============
Fase 9 - Reporte final y evaluacion integral del sistema.

Combina resultados de:
  - Fase 5 (metrics_test.json, history.json)
  - Fase 8 (resultado_umbral.json)
Y permite:
  - Imprimir reporte final en consola
  - Generar curvas de entrenamiento
  - Generar matriz de confusion visual
  - Medir rendimiento del pipeline en camara
  - Guardar reporte consolidado en JSON
"""

import sys
import json
import time
import argparse
import warnings
import importlib.util
from pathlib import Path
from datetime import datetime

import cv2
import torch

warnings.filterwarnings("ignore", category=UserWarning)

# Forzar UTF-8 en stdout/stderr para compatibilidad en Windows/Linux
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Rutas del proyecto (mismo estandar de fases anteriores)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTADOS_EVAL_DIR = PROJECT_ROOT / "resultados" / "evaluacion"
DEFAULT_MODEL = MODELS_DIR / "best_model.pth"
DEFAULT_METRICS = MODELS_DIR / "metrics_test.json"
DEFAULT_HISTORY = MODELS_DIR / "history.json"
DEFAULT_UMBRAL = PROJECT_ROOT / "resultados" / "umbral" / "resultado_umbral.json"


def _importar_fase6():
    """Importa dinamicamente 6_inferencia.py."""
    scripts_dir = Path(__file__).resolve().parent
    fase6_path = scripts_dir / "6_inferencia.py"

    if not fase6_path.exists():
        print(f"\n  ERROR: No se encontro 6_inferencia.py en: {scripts_dir}")
        print("  Este script depende de la Fase 6 para inferencia y deteccion.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("inferencia", fase6_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inferencia"] = mod
    spec.loader.exec_module(mod)
    return mod


_fase6 = _importar_fase6()
load_model = _fase6.load_model
predict = _fase6.predict
detectar_todos_los_rostros = _fase6.detectar_todos_los_rostros


def _leer_json(path_obj):
    """Lee un JSON de forma segura. Retorna dict o None."""
    path_obj = Path(path_obj)
    if not path_obj.exists():
        return None
    try:
        with path_obj.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as ex:
        print(f"  Aviso: no se pudo leer {path_obj.name}: {ex}")
        return None


def _crop_con_margen(frame, bbox, margen=0.15):
    """
    Recorta un rostro desde el frame con un margen adicional para conservar
    contexto facial util para la inferencia.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margen)
    my = int(bh * margen)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return frame[y1:y2, x1:x2]


def _extraer_peores_clases(metrics_data, top_n=5):
    """
    Extrae las peores clases por F1-score a partir de metrics_test.json.
    Omite filas agregadas como accuracy/macro avg/weighted avg.
    """
    if not metrics_data:
        return []

    per_class = metrics_data.get("per_class", {})
    if not isinstance(per_class, dict):
        return []

    filas = []
    for nombre, met in per_class.items():
        if not isinstance(met, dict):
            continue
        if "f1-score" not in met:
            continue
        f1 = float(met.get("f1-score", 0.0))
        support = int(float(met.get("support", 0)))
        filas.append(
            {
                "clase": str(nombre),
                "f1": f1,
                "support": support,
            }
        )

    filas.sort(key=lambda x: (x["f1"], x["support"]))
    return filas[:top_n]


def construir_resumen(modelo_path, metrics_data, history_data, umbral_data):
    """Arma un resumen consolidado para reporte y exportacion."""
    ckpt = {}
    if Path(modelo_path).exists():
        try:
            ckpt = torch.load(modelo_path, map_location="cpu", weights_only=False)
        except Exception as ex:
            print(f"  Aviso: no se pudo leer checkpoint para metadatos: {ex}")

    arquitectura = (
        ckpt.get("model_name")
        or (metrics_data or {}).get("model_name")
        or ((history_data or {}).get("config", {}) if history_data else {}).get("model_name")
        or "desconocido"
    )
    num_clases = (
        ckpt.get("num_classes")
        or ((history_data or {}).get("config", {}) if history_data else {}).get("num_classes")
        or len((metrics_data or {}).get("class_names", []))
        or "N/D"
    )

    best_epoch = (
        ckpt.get("epoch")
        or (metrics_data or {}).get("best_epoch")
        or ((history_data or {}).get("config", {}) if history_data else {}).get("best_epoch")
    )
    total_epochs = ((history_data or {}).get("config", {}) if history_data else {}).get("epochs_run")
    if total_epochs is None and history_data and isinstance(history_data.get("train_loss"), list):
        total_epochs = len(history_data["train_loss"])

    device_train = (
        (metrics_data or {}).get("config", {}).get("device")
        or ((history_data or {}).get("config", {}) if history_data else {}).get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    m_test = None
    if metrics_data:
        m_test = {
            "accuracy": float((metrics_data.get("accuracy", {}) or {}).get("test", 0.0)),
            "precision": float((metrics_data.get("test_metrics", {}) or {}).get("precision_weighted", 0.0)),
            "recall": float((metrics_data.get("test_metrics", {}) or {}).get("recall_weighted", 0.0)),
            "f1": float((metrics_data.get("test_metrics", {}) or {}).get("f1_weighted", 0.0)),
        }

    m_umbral = None
    if umbral_data:
        met = umbral_data.get("metricas_optimo", {})
        m_umbral = {
            "threshold": float(umbral_data.get("threshold_optimo", 0.0)),
            "precision": float(met.get("precision", 0.0)),
            "recall": float(met.get("recall", 0.0)),
            "f1": float(met.get("f1", 0.0)),
            "cobertura": float(met.get("cobertura", 0.0)),
        }

    return {
        "arquitectura": str(arquitectura),
        "num_clases": num_clases,
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "device": str(device_train),
        "metricas_test": m_test,
        "metricas_umbral": m_umbral,
        "peores_clases": _extraer_peores_clases(metrics_data, top_n=5),
    }


def imprimir_reporte(resumen, metrics_data_disponible, umbral_disponible):
    """Imprime el reporte final en consola."""
    print("\n" + "" * 70)
    print("  REPORTE FINAL  SISTEMA DE RECONOCIMIENTO FACIAL")
    print("" * 70)

    print("\n  MODELO")
    print("  " + "" * 66)
    print(f"  Arquitectura:     {resumen['arquitectura']}")
    print(f"  Clases:           {resumen['num_clases']}")
    if resumen["best_epoch"] is not None and resumen["total_epochs"] is not None:
        print(f"  Mejor epoca:      {resumen['best_epoch']} / {resumen['total_epochs']}")
    elif resumen["best_epoch"] is not None:
        print(f"  Mejor epoca:      {resumen['best_epoch']}")
    else:
        print("  Mejor epoca:      N/D")
    print(f"  Dispositivo:      {resumen['device']}")

    print("\n  METRICAS EN TEST (Fase 5  sin umbral de confianza)")
    print("  " + "" * 66)
    if metrics_data_disponible and resumen["metricas_test"]:
        mt = resumen["metricas_test"]
        print(f"  Accuracy:         {mt['accuracy']:.4f}")
        print(f"  Precision:        {mt['precision']:.4f}")
        print(f"  Recall:           {mt['recall']:.4f}")
        print(f"  F1-score:         {mt['f1']:.4f}")
    else:
        print("  No disponible. Ejecuta la Fase 5 para generar metrics_test.json.")

    print("\n  UMBRAL DE CONFIANZA (Fase 8  calibrado)")
    print("  " + "" * 66)
    if umbral_disponible and resumen["metricas_umbral"]:
        mu = resumen["metricas_umbral"]
        print(f"  Threshold:        {mu['threshold']:.2f}")
        print(f"  Precision:        {mu['precision']:.4f}")
        print(f"  Recall:           {mu['recall']:.4f}")
        print(f"  F1:               {mu['f1']:.4f}")
        print(f"  Cobertura:        {mu['cobertura']*100:.1f}%")
    else:
        print("  No disponible. La Fase 8 no se ha ejecutado (o falta resultado_umbral.json).")

    print("\n  PEORES 5 CLASES (por F1 en test)")
    print("  " + "" * 66)
    peores = resumen["peores_clases"]
    if peores:
        for i, fila in enumerate(peores, start=1):
            print(
                f"  {i}. {fila['clase']:<18} "
                f"F1: {fila['f1']:.4f}   support: {fila['support']}"
            )
    else:
        print("  No disponible (faltan metricas por clase de la Fase 5).")

    print("" * 70 + "\n")


def guardar_curvas_entrenamiento(history_data, out_path):
    """Genera curvas de loss/accuracy de entrenamiento y validacion."""
    if not history_data:
        print("  Aviso: no hay history.json; se omite la curva de entrenamiento.")
        return False

    train_loss = history_data.get("train_loss", [])
    val_loss = history_data.get("val_loss", [])
    train_acc = history_data.get("train_acc", [])
    val_acc = history_data.get("val_acc", [])
    if not train_loss or not val_loss or not train_acc or not val_acc:
        print("  Aviso: history.json incompleto; se omite la curva de entrenamiento.")
        return False

    best_epoch = (history_data.get("config", {}) or {}).get("best_epoch", None)
    epocas = list(range(1, len(train_loss) + 1))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Aviso: matplotlib no esta instalado; no se puede generar curvas.")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: loss
    axes[0].plot(epocas, train_loss, label="Train loss", linewidth=2)
    axes[0].plot(epocas, val_loss, label="Val loss", linewidth=2)
    if best_epoch is not None:
        axes[0].axvline(best_epoch, color="black", linestyle="--", label=f"Mejor epoca: {best_epoch}")
    axes[0].set_title("Perdida por epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    # Subplot 2: accuracy
    axes[1].plot(epocas, train_acc, label="Train acc", linewidth=2)
    axes[1].plot(epocas, val_acc, label="Val acc", linewidth=2)
    if best_epoch is not None:
        axes[1].axvline(best_epoch, color="black", linestyle="--", label=f"Mejor epoca: {best_epoch}")
    axes[1].set_title("Exactitud por epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("Curvas de entrenamiento (Fase 5)")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"  Curvas guardadas en: {out_path}")
    return True


def guardar_matriz_confusion(metrics_data, out_path):
    """Genera heatmap de matriz de confusion a partir de metrics_test.json."""
    if not metrics_data:
        print("  Aviso: no hay metrics_test.json; se omite matriz de confusion.")
        return False

    cm = metrics_data.get("confusion_matrix")
    class_names = metrics_data.get("class_names", [])
    if not cm:
        print("  Aviso: metrics_test.json no contiene confusion_matrix.")
        return False

    n_clases = len(class_names) if class_names else len(cm)
    annot = n_clases <= 20

    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Aviso: falta matplotlib/numpy; no se puede generar matriz.")
        return False

    cm_arr = np.array(cm)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Intentar seaborn; fallback a matplotlib puro.
    try:
        import seaborn as sns

        fig_w = max(8, n_clases * 0.45)
        fig_h = max(6, n_clases * 0.40)
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(
            cm_arr,
            cmap="Blues",
            annot=annot,
            fmt="d",
            cbar=True,
            xticklabels=class_names if class_names else "auto",
            yticklabels=class_names if class_names else "auto",
        )
        plt.title("Matriz de confusion  split test")
        plt.xlabel("Prediccion")
        plt.ylabel("Etiqueta real")
        plt.xticks(rotation=60, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150)
        plt.close()
        print(f"  Matriz de confusion guardada en: {out_path}")
        return True

    except Exception:
        fig_w = max(8, n_clases * 0.45)
        fig_h = max(6, n_clases * 0.40)
        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(cm_arr, cmap="Blues")
        plt.colorbar()
        ticks = range(n_clases)
        if class_names:
            plt.xticks(ticks, class_names, rotation=60, ha="right")
            plt.yticks(ticks, class_names)
        else:
            plt.xticks(ticks)
            plt.yticks(ticks)

        if annot:
            for i in range(cm_arr.shape[0]):
                for j in range(cm_arr.shape[1]):
                    plt.text(j, i, int(cm_arr[i, j]), ha="center", va="center", color="black", fontsize=8)

        plt.title("Matriz de confusion  split test")
        plt.xlabel("Prediccion")
        plt.ylabel("Etiqueta real")
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150)
        plt.close()
        print(f"  Matriz de confusion guardada en: {out_path}")
        return True


def evaluar_camara(model, class_names, img_size, confidence_threshold, device, fuente=0, duracion_seg=30):
    """
    Ejecuta evaluacion de rendimiento en camara por 30 segundos (o hasta Q).
    Reusa detectar_todos_los_rostros + predict de la Fase 6.
    """
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"  Aviso: no se pudo abrir la camara (fuente={fuente}).")
        return None

    ret, frame_prueba = cap.read()
    if not ret:
        cap.release()
        print(f"  Aviso: la camara se abrio pero no entrega frames (fuente={fuente}).")
        return None

    frame_h, frame_w = frame_prueba.shape[:2]
    print(f"\n  Camara iniciada: {fuente} ({frame_w}x{frame_h})")
    print(f"  Medicion activa por {duracion_seg} segundos. Presiona 'Q' para salir antes.\n")

    latencias_ms = []
    fps_inst = []
    total_frames = 0

    window_name = "Fase 9 - Evaluacion de rendimiento (Q para salir)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    start_global = time.time()
    prev_end = start_global
    pendientes = [frame_prueba]

    while True:
        if pendientes:
            frame = pendientes.pop(0)
            ok = True
        else:
            ok, frame = cap.read()

        if not ok:
            break

        t0 = time.time()

        # Deteccion + inferencia (reutilizando funciones existentes)
        bboxes = detectar_todos_los_rostros(frame, device)
        for bbox in bboxes:
            crop = _crop_con_margen(frame, bbox)
            if crop.size == 0:
                continue
            _ = predict(crop, model, class_names, img_size, confidence_threshold, device)

        t1 = time.time()
        lat_ms = (t1 - t0) * 1000.0
        latencias_ms.append(lat_ms)
        total_frames += 1

        dt_total = t1 - prev_end
        prev_end = t1
        if dt_total > 0:
            fps_inst.append(1.0 / dt_total)

        # Overlay minimo para referencia visual
        elapsed = t1 - start_global
        fps_show = fps_inst[-1] if fps_inst else 0.0
        cv2.putText(
            frame,
            f"Tiempo: {elapsed:5.1f}/{duracion_seg}s  FPS: {fps_show:4.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if elapsed >= duracion_seg:
            break

    cap.release()
    cv2.destroyAllWindows()

    if total_frames == 0:
        print("  Aviso: no se procesaron frames de camara.")
        return None

    fps_prom = sum(fps_inst) / len(fps_inst) if fps_inst else 0.0
    fps_min = min(fps_inst) if fps_inst else 0.0
    fps_max = max(fps_inst) if fps_inst else 0.0
    lat_prom = sum(latencias_ms) / len(latencias_ms) if latencias_ms else 0.0

    resultado = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duracion_objetivo_s": int(duracion_seg),
        "frames_procesados": int(total_frames),
        "fps_promedio": round(float(fps_prom), 3),
        "fps_minimo": round(float(fps_min), 3),
        "fps_maximo": round(float(fps_max), 3),
        "latencia_promedio_ms": round(float(lat_prom), 3),
    }

    print("  RENDIMIENTO EN CAMARA (30 seg)")
    print("  " + "" * 66)
    print(f"  FPS promedio:     {resultado['fps_promedio']:.1f}")
    print(f"  FPS minimo:       {resultado['fps_minimo']:.1f}")
    print(f"  FPS maximo:       {resultado['fps_maximo']:.1f}")
    print(f"  Latencia / frame: {resultado['latencia_promedio_ms']:.1f} ms")

    return resultado


def guardar_json(path_obj, data):
    """Guarda JSON en UTF-8 con indentacion."""
    path_obj = Path(path_obj)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fase 9 - Reporte final del sistema de reconocimiento facial."
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=DEFAULT_MODEL,
        help="Ruta al checkpoint .pth (default: models/best_model.pth)",
    )
    parser.add_argument(
        "--curvas",
        action="store_true",
        help="Genera curvas de entrenamiento en resultados/evaluacion/curvas_entrenamiento.png",
    )
    parser.add_argument(
        "--confusion",
        action="store_true",
        help="Genera matriz de confusion en resultados/evaluacion/matriz_confusion.png",
    )
    parser.add_argument(
        "--camara",
        action="store_true",
        help="Mide rendimiento en vivo del pipeline con la camara",
    )
    parser.add_argument(
        "--guardar",
        action="store_true",
        help="Guarda reporte consolidado en resultados/evaluacion/reporte_final.json",
    )
    parser.add_argument(
        "--fuente",
        type=int,
        default=0,
        help="Indice de camara para --camara (default: 0)",
    )
    parser.add_argument(
        "--duracion",
        type=int,
        default=30,
        help="Duracion en segundos para --camara (default: 30)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    RESULTADOS_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = args.modelo.parent / "metrics_test.json"
    history_path = args.modelo.parent / "history.json"
    umbral_path = DEFAULT_UMBRAL

    metrics_data = _leer_json(metrics_path)
    history_data = _leer_json(history_path)
    umbral_data = _leer_json(umbral_path)

    if metrics_data is None:
        print("  Aviso: no se encontro metrics_test.json. Ejecuta Fase 5 para metricas completas.")
    if umbral_data is None:
        print("  Aviso: no se encontro resultado_umbral.json. La seccion de umbral se omitira.")

    resumen = construir_resumen(
        modelo_path=args.modelo,
        metrics_data=metrics_data,
        history_data=history_data,
        umbral_data=umbral_data,
    )
    imprimir_reporte(
        resumen,
        metrics_data_disponible=metrics_data is not None,
        umbral_disponible=umbral_data is not None,
    )

    if args.curvas:
        out_curvas = RESULTADOS_EVAL_DIR / "curvas_entrenamiento.png"
        guardar_curvas_entrenamiento(history_data, out_curvas)

    if args.confusion:
        out_conf = RESULTADOS_EVAL_DIR / "matriz_confusion.png"
        guardar_matriz_confusion(metrics_data, out_conf)

    rendimiento_camara = None
    if args.camara:
        if not Path(args.modelo).exists():
            print(f"  Aviso: modelo no encontrado en {args.modelo}. Se omite evaluacion de camara.")
        else:
            model, class_names, img_size, confidence_threshold, device = load_model(str(args.modelo))
            rendimiento_camara = evaluar_camara(
                model=model,
                class_names=class_names,
                img_size=img_size,
                confidence_threshold=confidence_threshold,
                device=device,
                fuente=args.fuente,
                duracion_seg=args.duracion,
            )
            if rendimiento_camara is not None:
                out_cam = RESULTADOS_EVAL_DIR / "rendimiento_camara.json"
                guardar_json(out_cam, rendimiento_camara)
                print(f"  Rendimiento de camara guardado en: {out_cam}")

    if args.guardar:
        if rendimiento_camara is None:
            rendimiento_camara = _leer_json(RESULTADOS_EVAL_DIR / "rendimiento_camara.json")

        reporte = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "modelo": {
                "arquitectura": resumen["arquitectura"],
                "num_clases": resumen["num_clases"],
                "mejor_epoca": resumen["best_epoch"],
                "dispositivo": resumen["device"],
            },
            "metricas_test_fase5": resumen["metricas_test"] if resumen["metricas_test"] else {},
            "umbral_fase8": resumen["metricas_umbral"] if resumen["metricas_umbral"] else {},
            "peores_clases": resumen["peores_clases"],
            "rendimiento_camara": rendimiento_camara if rendimiento_camara else {},
        }

        out_rep = RESULTADOS_EVAL_DIR / "reporte_final.json"
        guardar_json(out_rep, reporte)
        print(f"  Reporte final guardado en: {out_rep}")


if __name__ == "__main__":
    main()



