# -*- coding: utf-8 -*-
"""
8_umbral.py
===========
Fase 8 - Calibracion de umbral de confianza para reconocimiento facial.

Este script evalua el modelo sobre el split de test para encontrar el
threshold optimo de confianza (0.10 a 0.99), maximizando F1 con la
restriccion de cobertura minima del 70%.
"""

import sys
import csv
import json
import argparse
import warnings
import shutil
import importlib.util
from pathlib import Path
from datetime import datetime

import torch

warnings.filterwarnings("ignore", category=UserWarning)

# Forzar UTF-8 en stdout/stderr para evitar problemas en Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Rutas del proyecto (mismo patron que fases previas)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_DIR = PROJECT_ROOT / "splits"
RESULTADOS_UMBRAL_DIR = PROJECT_ROOT / "resultados" / "umbral"
DEFAULT_MODEL = MODELS_DIR / "best_model.pth"


def _importar_fase6():
    """Importa dinamicamente 6_inferencia.py para reutilizar load_model/predict."""
    scripts_dir = Path(__file__).resolve().parent
    fase6_path = scripts_dir / "6_inferencia.py"

    if not fase6_path.exists():
        print(f"\n  ERROR: No se encontro 6_inferencia.py en: {scripts_dir}")
        print("  Este script depende de la Fase 6 para cargar el modelo y predecir.")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("inferencia", fase6_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inferencia"] = mod
    spec.loader.exec_module(mod)
    return mod


_fase6 = _importar_fase6()
load_model = _fase6.load_model
predict = _fase6.predict


def _barra_progreso(actual, total, ancho=20):
    """Renderiza una barra simple de progreso en una sola linea."""
    if total <= 0:
        return
    proporcion = actual / total
    llenos = int(proporcion * ancho)
    if llenos >= ancho:
        barra = "=" * ancho
    else:
        barra = ("=" * llenos) + ">" + (" " * (ancho - llenos - 1))
    print(f"\r  [{barra}] {actual}/{total}", end="", flush=True)
    if actual == total:
        print()


def _resolver_ruta_imagen(ruta_raw):
    """
    Intenta resolver rutas absolutas o relativas del CSV incluso si el proyecto
    fue movido de carpeta.
    """
    ruta_txt = str(ruta_raw).strip().strip('"').strip("'")
    path = Path(ruta_txt)

    if path.exists():
        return path

    # Caso ruta relativa al proyecto actual
    candidato = PROJECT_ROOT / ruta_txt
    if candidato.exists():
        return candidato

    ruta_norm = ruta_txt.replace("\\", "/")

    # Caso CSV con ruta absoluta de otra maquina/carpeta:
    # recorta desde "Face-Recognition Proyect/" y reconstruye localmente.
    marca = "Face-Recognition Proyect/"
    if marca in ruta_norm:
        sub = ruta_norm.split(marca, 1)[1]
        candidato = PROJECT_ROOT / Path(sub)
        if candidato.exists():
            return candidato

    # Fallback por carpetas conocidas del proyecto
    carpetas = [
        "Dataset_aumentado/",
        "Dataset_procesado/",
        "Dataset/",
    ]
    for c in carpetas:
        if c in ruta_norm:
            sub = ruta_norm.split(c, 1)[1]
            candidato = PROJECT_ROOT / c.rstrip("/") / Path(sub)
            if candidato.exists():
                return candidato

    return path


def _leer_test(test_csv_path):
    """Lee test.csv completo (estructura por persona directa, sin grupos)."""
    if not test_csv_path.exists():
        print("\n  ERROR: No existe splits/test.csv")
        print("  Debes ejecutar primero la Fase 4 (4_preparar_clasificacion.py).")
        sys.exit(1)

    filas = []

    with test_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columnas_esperadas = {"ruta", "clase", "clase_idx"}
        if not columnas_esperadas.issubset(set(reader.fieldnames or [])):
            print("\n  ERROR: test.csv no tiene las columnas esperadas.")
            print("  Se requieren: ruta, clase, clase_idx")
            sys.exit(1)

        for row in reader:
            filas.append(row)

    return filas


def recolectar_predicciones(test_csv_path, model, class_names, img_size, device):
    """
    Recorre el split de test y recolecta predicciones 'raw' sin umbral
    (confidence_threshold=0.0).
    """
    filas = _leer_test(test_csv_path)
    total = len(filas)

    if total == 0:
        print("\n  ERROR: No hay imagenes en test.csv para evaluar.")
        return []

    print(f"\n  Recolectando predicciones en test ({total} imagenes)...")
    resultados = []

    for i, row in enumerate(filas, start=1):
        ruta_csv = row["ruta"]
        clase_real = str(row["clase"]).strip()
        ruta_img = _resolver_ruta_imagen(ruta_csv)

        try:
            # Sin umbral para obtener clase/confianza crudas en TODO el test
            pred = predict(
                source=str(ruta_img),
                model=model,
                class_names=class_names,
                img_size=img_size,
                confidence_threshold=0.0,
                device=device,
            )

            clase_predicha = pred["clase"]
            confianza = float(pred["confianza"])
            correcta = clase_predicha == clase_real

            resultados.append(
                {
                    "ruta": str(ruta_img),
                    "clase_real": clase_real,
                    "clase_predicha": clase_predicha,
                    "confianza": confianza,
                    "correcta": bool(correcta),
                }
            )
        except Exception as ex:
            # Se registra el error pero se continua con el resto del dataset.
            resultados.append(
                {
                    "ruta": str(ruta_img),
                    "clase_real": clase_real,
                    "clase_predicha": "ERROR",
                    "confianza": 0.0,
                    "correcta": False,
                    "error": str(ex),
                }
            )

        _barra_progreso(i, total)

    errores = sum(1 for r in resultados if r["clase_predicha"] == "ERROR")
    if errores > 0:
        print(f"  Aviso: {errores} imagenes no pudieron procesarse y quedaron como ERROR.")

    return resultados


def _metricas_clases_conocidas(y_true, y_pred, class_names):
    """
    Calcula precision/recall/f1 ponderados por soporte para clasificacion
    multiclase, excluyendo cualquier etiqueta fuera de class_names.
    """
    if not y_true:
        return 0.0, 0.0, 0.0

    clases = list(class_names)

    total_support = 0
    precision_w = 0.0
    recall_w = 0.0
    f1_w = 0.0

    for c in clases:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        support = sum(1 for yt in y_true if yt == c)

        prec_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_c = (2 * prec_c * rec_c / (prec_c + rec_c)) if (prec_c + rec_c) > 0 else 0.0

        precision_w += prec_c * support
        recall_w += rec_c * support
        f1_w += f1_c * support
        total_support += support

    if total_support == 0:
        return 0.0, 0.0, 0.0

    return (
        precision_w / total_support,
        recall_w / total_support,
        f1_w / total_support,
    )


def buscar_threshold_optimo(predicciones, class_names, cobertura_minima=0.70):
    """
    Evalua thresholds de 0.10 a 0.99 y selecciona el optimo:
    - Maximizar F1
    - Restriccion: cobertura >= cobertura_minima
    """
    if not predicciones:
        return None, None, []

    total = len(predicciones)
    tabla = []

    for t_int in range(10, 100):
        threshold = round(t_int / 100.0, 2)

        # Cobertura: porcentaje de ejemplos que NO caen en "Desconocido"
        aceptadas = [r for r in predicciones if r["confianza"] >= threshold]
        cobertura = len(aceptadas) / total if total > 0 else 0.0

        y_true = [r["clase_real"] for r in aceptadas if r["clase_predicha"] != "ERROR"]
        y_pred = [r["clase_predicha"] for r in aceptadas if r["clase_predicha"] != "ERROR"]

        precision, recall, f1 = _metricas_clases_conocidas(y_true, y_pred, class_names)

        fila = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cobertura": cobertura,
        }
        tabla.append(fila)

    # Filtrar candidatos validos por cobertura minima.
    candidatos = [r for r in tabla if r["cobertura"] >= cobertura_minima]

    # Si no hay candidatos validos, usar fallback al mejor F1 global.
    if not candidatos:
        mejor = max(
            tabla,
            key=lambda r: (r["f1"], r["precision"], r["recall"], r["threshold"]),
        )
        return mejor["threshold"], mejor, tabla

    # Criterio principal: mayor F1 manteniendo cobertura minima.
    # Desempate: mayor precision, mayor recall y luego threshold mas alto
    # (mas estricto, sin sacrificar desempeno).
    mejor = max(
        candidatos,
        key=lambda r: (r["f1"], r["precision"], r["recall"], r["threshold"]),
    )
    return mejor["threshold"], mejor, tabla


def _buscar_fila_por_threshold(tabla, threshold, tol=1e-9):
    for fila in tabla:
        if abs(fila["threshold"] - threshold) <= tol:
            return fila
    return None


def imprimir_reporte_consola(tabla_completa, threshold_optimo):
    """Imprime tabla resumen para 0.50..0.99 paso 0.05 y marca el optimo."""
    thresholds_mostrar = {round(x / 100.0, 2) for x in range(50, 100, 5)}
    thresholds_mostrar.add(0.99)
    thresholds_mostrar.add(round(float(threshold_optimo), 2))

    filas = []
    for t in sorted(thresholds_mostrar):
        fila = _buscar_fila_por_threshold(tabla_completa, t)
        if fila is not None:
            filas.append(fila)

    print("\n  threshold  precision  recall     f1     cobertura")
    print("  ")
    for fila in filas:
        mark = "" if abs(fila["threshold"] - threshold_optimo) < 1e-9 else " "
        print(
            f" {mark}  {fila['threshold']:>5.2f}   "
            f"   {fila['precision']:.4f}  "
            f"  {fila['recall']:.4f} "
            f"  {fila['f1']:.4f} "
            f"  {fila['cobertura']*100:5.1f}%"
        )


def guardar_grafica_pr(tabla_completa, threshold_optimo, salida_png):
    """Genera curva de precision/recall/f1 vs threshold."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  Aviso: matplotlib no esta instalado; se omite la grafica.")
        return

    thresholds = [r["threshold"] for r in tabla_completa]
    precision = [r["precision"] for r in tabla_completa]
    recall = [r["recall"] for r in tabla_completa]
    f1 = [r["f1"] for r in tabla_completa]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision, label="Precision", linewidth=2)
    plt.plot(thresholds, recall, label="Recall", linewidth=2)
    plt.plot(thresholds, f1, label="F1", linewidth=2)
    plt.axvline(
        x=threshold_optimo,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Optimo: {threshold_optimo:.2f}",
    )

    plt.title("Curva Precision-Recall-F1 por Threshold")
    plt.xlabel("Threshold de confianza")
    plt.ylabel("Metrica")
    plt.ylim(0.0, 1.0)
    plt.xlim(0.10, 0.99)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    salida_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(salida_png), dpi=150)
    plt.close()
    print(f"\n  Grafica guardada en: {salida_png}")


def aplicar_threshold_checkpoint(modelo_path, threshold_nuevo):
    """
    Actualiza solo `confidence_threshold` en el checkpoint y crea backup previo.
    """
    modelo_path = Path(modelo_path)
    backup_path = modelo_path.with_name("best_model_backup.pth")

    checkpoint = torch.load(modelo_path, map_location="cpu", weights_only=False)
    threshold_anterior = float(checkpoint.get("confidence_threshold", 0.70))

    shutil.copy2(modelo_path, backup_path)
    checkpoint["confidence_threshold"] = float(round(threshold_nuevo, 2))
    torch.save(checkpoint, modelo_path)

    print("\n  Checkpoint actualizado correctamente:")
    print(f"    Backup: {backup_path}")
    print(f"    Umbral: {threshold_anterior:.2f} -> {threshold_nuevo:.2f}")

    return threshold_anterior, float(round(threshold_nuevo, 2))


def guardar_json_resultados(
    output_json,
    modelo_nombre,
    threshold_anterior,
    threshold_optimo,
    metrica_optima,
    tabla_completa,
):
    """Guarda resultados de calibracion en JSON."""
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modelo": modelo_nombre,
        "threshold_anterior": float(round(threshold_anterior, 2)),
        "threshold_optimo": float(round(threshold_optimo, 2)),
        "metricas_optimo": {
            "precision": round(float(metrica_optima["precision"]), 4),
            "recall": round(float(metrica_optima["recall"]), 4),
            "f1": round(float(metrica_optima["f1"]), 4),
            "cobertura": round(float(metrica_optima["cobertura"]), 4),
        },
        "tabla_completa": [
            {
                "threshold": round(float(r["threshold"]), 2),
                "precision": round(float(r["precision"]), 4),
                "recall": round(float(r["recall"]), 4),
                "f1": round(float(r["f1"]), 4),
                "cobertura": round(float(r["cobertura"]), 4),
            }
            for r in tabla_completa
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n  JSON guardado en: {output_json}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fase 8 - Calibracion de umbral de confianza (threshold)."
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=DEFAULT_MODEL,
        help="Ruta al checkpoint .pth (default: models/best_model.pth)",
    )
    parser.add_argument(
        "--grafica",
        action="store_true",
        help="Genera curva precision/recall/f1 en resultados/umbral/curva_pr.png",
    )
    parser.add_argument(
        "--aplicar",
        action="store_true",
        help="Actualiza confidence_threshold del checkpoint con el valor optimo",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    test_csv = SPLITS_DIR / "test.csv"
    out_json = RESULTADOS_UMBRAL_DIR / "resultado_umbral.json"
    out_png = RESULTADOS_UMBRAL_DIR / "curva_pr.png"

    if not test_csv.exists():
        print("\n  ERROR: No se encontro splits/test.csv")
        print("  Ejecuta primero la Fase 4 para generar los splits.")
        sys.exit(1)

    print("\n" + "=" * 58)
    print("  FASE 8 - BUSQUEDA DE UMBRAL OPTIMO")
    print("=" * 58)
    print(f"  Modelo: {args.modelo}")
    print("  Estructura: Persona directa (sin grupos)")

    # Cargar modelo y metadatos del checkpoint via Fase 6 (requisito del proyecto).
    model, class_names, img_size, threshold_anterior, device = load_model(str(args.modelo))

    # Leer nombre de modelo desde checkpoint para el JSON.
    ckpt_meta = torch.load(args.modelo, map_location="cpu", weights_only=False)
    modelo_nombre = ckpt_meta.get("model_name", args.modelo.name)

    predicciones = recolectar_predicciones(
        test_csv_path=test_csv,
        model=model,
        class_names=class_names,
        img_size=img_size,
        device=device,
    )
    if not predicciones:
        print("\n  No se recolectaron predicciones. Fin.")
        sys.exit(1)

    threshold_optimo, metrica_optima, tabla_completa = buscar_threshold_optimo(
        predicciones=predicciones,
        class_names=class_names,
        cobertura_minima=0.70,
    )
    if threshold_optimo is None:
        print("\n  ERROR: No fue posible calcular el threshold optimo.")
        sys.exit(1)

    print("\n  Resultado optimo:")
    print(f"    Threshold optimo: {threshold_optimo:.2f}")
    print(f"    Precision:        {metrica_optima['precision']:.4f}")
    print(f"    Recall:           {metrica_optima['recall']:.4f}")
    print(f"    F1:               {metrica_optima['f1']:.4f}")
    print(f"    Cobertura:        {metrica_optima['cobertura']*100:.1f}%")

    imprimir_reporte_consola(tabla_completa, threshold_optimo)

    if args.grafica:
        guardar_grafica_pr(tabla_completa, threshold_optimo, out_png)

    if args.aplicar:
        threshold_anterior, _ = aplicar_threshold_checkpoint(args.modelo, threshold_optimo)

    guardar_json_resultados(
        output_json=out_json,
        modelo_nombre=modelo_nombre,
        threshold_anterior=threshold_anterior,
        threshold_optimo=threshold_optimo,
        metrica_optima=metrica_optima,
        tabla_completa=tabla_completa,
    )

    print("\n  Proceso completado.")


if __name__ == "__main__":
    main()


