# CreaciÃ³n de Dataset HÃ­brido - Reconocimiento Facial con CNN

Proyecto integral de reconocimiento facial por fases, desde la captura de imÃ¡genes hasta la demo en vivo con cÃ¡mara.

Incluye:
- ConstrucciÃ³n de dataset por persona.
- Preprocesamiento y aumento de datos.
- Entrenamiento CNN (ResNet18 / EfficientNet-B0).
- Inferencia en imagen y video en tiempo real.
- CalibraciÃ³n de umbral de confianza.
- EvaluaciÃ³n final y reporte consolidado.

---

## 1. Objetivo del proyecto

Construir un sistema de reconocimiento facial capaz de:
- Identificar personas conocidas.
- Rechazar predicciones de baja confianza como `Desconocido`.
- Funcionar en imagen estÃ¡tica y en cÃ¡mara en tiempo real.

---

## 2. Estructura del repositorio

```text
CreaciÃ³n_de_Dataset_hÃ­brido/
â”œâ”€â”€ Face-Recognition Proyect/
â”‚   â”œâ”€â”€ Dataset/                     # Fase 1 (capturas crudas por persona)
â”‚   â”œâ”€â”€ Dataset_procesado/           # Fase 2
â”‚   â”œâ”€â”€ Dataset_aumentado/           # Fase 3
â”‚   â”œâ”€â”€ splits/                      # Fase 4
â”‚   â”œâ”€â”€ models/                      # Fase 5 (checkpoints y mÃ©tricas)
â”‚   â”œâ”€â”€ resultados/                  # Fases 6â€“9 y demo
â”‚   â”œâ”€â”€ logs/                        # logs auxiliares
â”‚   â””â”€â”€ Scripts/
â”‚       â”œâ”€â”€ 1_Captura.py
â”‚       â”œâ”€â”€ 2_Preprocesar.py
â”‚       â”œâ”€â”€ 3_Aumentar.py
â”‚       â”œâ”€â”€ 4_preparar_clasificacion.py
â”‚       â”œâ”€â”€ 5_entrenar_cnn.py
â”‚       â”œâ”€â”€ 6_inferencia.py
â”‚       â”œâ”€â”€ 7_pipeline_camara.py
â”‚       â”œâ”€â”€ 8_umbral.py
â”‚       â”œâ”€â”€ 9_evaluacion.py
â”‚       â””â”€â”€ demo.py
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

---

## 3. Requisitos e instalaciÃ³n

## 3.1 Requisitos mÃ­nimos sugeridos

- Python 3.10 o 3.11.
- Webcam para fases de cÃ¡mara.
- GPU CUDA opcional (acelera entrenamiento/inferencia).

## 3.2 Crear entorno virtual

### Windows (PowerShell)

```powershell
cd "C:\Users\joeyk\OneDrive\Desktop\CNN\CreaciÃ³n_de_Dataset_hÃ­brido"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 3.3 Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencias principales del proyecto:
- `opencv-python`
- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`
- `Pillow`
- `torch`, `torchvision`
- `matplotlib`, `seaborn`
- `mtcnn`

Nota:
- Para detecciÃ³n facial avanzada en inferencia/cÃ¡mara tambiÃ©n se usa `facenet_pytorch` cuando estÃ¡ disponible. Si no estÃ¡, el sistema usa Haar Cascade como fallback.

---

## 4. Formato de dataset soportado

Estructura actual soportada por las fases:

```text
Dataset/
â”œâ”€â”€ Persona_1/
â”‚   â”œâ”€â”€ img1.jpg
â”‚   â”œâ”€â”€ img2.jpg
â”‚   â””â”€â”€ ...
â”œâ”€â”€ Persona_2/
â””â”€â”€ ...
```

---

## 5. Flujo completo (Fases 1 a 9)

Primero entra al proyecto:

```powershell
cd "C:\Users\joeyk\OneDrive\Desktop\CNN\CreaciÃ³n_de_Dataset_hÃ­brido\Face-Recognition Proyect"
```

## Fase 1 - Captura de dataset (`1_Captura.py`)

QuÃ© hace:
- Captura rostros desde cÃ¡mara y guarda imÃ¡genes por persona en `Dataset/`.

ParÃ¡metros:

| ParÃ¡metro | Tipo | Default | DescripciÃ³n |
|---|---|---:|---|
| `--persona` | str | `Joey` | Nombre de la persona (carpeta destino). |
| `--meta` | int | `220` | Cantidad de fotos nuevas a capturar. |
| `--cooldown` | float | `0.65` | Segundos mÃ­nimos entre fotos. |
| `--fuente` | int | `0` | Ãndice de cÃ¡mara. |

Comando ejemplo:

```powershell
python Scripts/1_Captura.py --persona "Joey" --meta 300 --cooldown 0.8
```

## Fase 2 - Preprocesamiento (`2_Preprocesar.py`)

QuÃ© hace:
- Detecta rostro, alinea/recorta y normaliza tamaÃ±o a 160x160.
- Guarda en `Dataset_procesado/`.

ParÃ¡metros:

| ParÃ¡metro | Tipo | Default | DescripciÃ³n |
|---|---|---:|---|
| `--confianza` | float | `0.90` | Umbral de detecciÃ³n MTCNN. |

Comando:

```powershell
python Scripts/2_Preprocesar.py --confianza 0.90
```

## Fase 3 - Aumento de datos (`3_Aumentar.py`)

QuÃ© hace:
- Genera variaciones de cada imagen (rotaciÃ³n, brillo, espejo, ruido, etc.).
- Guarda en `Dataset_aumentado/`.

ParÃ¡metros:

| ParÃ¡metro | Tipo | Default | DescripciÃ³n |
|---|---|---:|---|
| `--factor` | int | `6` | Multiplicador total por imagen. |
| `--solo_basicas` | flag | `False` | Solo rotaciÃ³n/brillo/espejo. |

Comando:

```powershell
python Scripts/3_Aumentar.py --factor 4
```

## Fase 4 - Splits train/val/test (`4_preparar_clasificacion.py`)

QuÃ© hace:
- Genera `train.csv`, `val.csv`, `test.csv` en `splits/`.
- Split estratificado por clase.
- Incluye protecciÃ³n contra fuga de datos agrupando por imagen base (`_augXX` queda en el mismo split que su original).

ParÃ¡metros:

| ParÃ¡metro | Tipo | Default | DescripciÃ³n |
|---|---|---:|---|
| `--train` | float | `0.70` | ProporciÃ³n train. |
| `--val` | float | `0.15` | ProporciÃ³n validaciÃ³n. |
| `--test` | float | `0.15` | ProporciÃ³n test. |

Comando:

```powershell
python Scripts/4_preparar_clasificacion.py --train 0.70 --val 0.15 --test 0.15
```

## Fase 5 - Entrenamiento CNN (`5_entrenar_cnn.py`)

QuÃ© hace:
- Entrena modelo de clasificaciÃ³n facial.
- Guarda `best_model.pth`, `metrics_test.json`, `history.json`.

ParÃ¡metros:

| ParÃ¡metro | Tipo | Default | DescripciÃ³n |
|---|---|---:|---|
| `--epochs` | int | `50` | MÃ¡ximo de Ã©pocas. |
| `--batch_size` | int | `32` | TamaÃ±o de batch. |
| `--lr` | float | `0.0003` | Learning rate inicial. |
| `--img_size` | int | `160` | TamaÃ±o de entrada de imagen. |
| `--num_workers` | int | `2` | Workers de DataLoader. |
| `--model_name` | str | `resnet18` | `resnet18` o `efficientnet_b0`. |
| `--grupos` | lista str | `None` | Filtro opcional de grupos si aplica. |

Comando recomendado:

```powershell
python Scripts/5_entrenar_cnn.py --epochs 50 --batch_size 32 --model_name resnet18
```

## Fase 6 - Inferencia en imagen (`6_inferencia.py`)

QuÃ© hace:
- Carga modelo entrenado y predice sobre una imagen.
- Guarda imagen anotada en `resultados/`.

ParÃ¡metros:

| ParÃ¡metro | Tipo | Default | DescripciÃ³n |
|---|---|---:|---|
| `--imagen` | str | `None` | Ruta de imagen a analizar. |
| `--modelo` | str | `models/best_model.pth` | Checkpoint a usar. |
| `--mostrar` | flag | `False` | Abre ventana con la imagen anotada. |

Comando:

```powershell
python Scripts/6_inferencia.py --imagen "C:\Users\joeyk\Downloads\foto.jpg" --mostrar
```

## Fase 7 - Pipeline cÃ¡mara en vivo (`7_pipeline_camara.py`)

QuÃ© hace:
- Ejecuta detecciÃ³n + clasificaciÃ³n en tiempo real.
- Puede grabar sesiÃ³n en `resultados/grabaciones/`.

ParÃ¡metros:

| ParÃ¡metro | Tipo | Default | DescripciÃ³n |
|---|---|---:|---|
| `--fuente` | int | `0` | Ãndice de cÃ¡mara. |
| `--skip_frames` | int | `2` | Recalcular cada N frames. |
| `--resize_factor` | float | `0.5` | Escalado para detecciÃ³n. |
| `--grabar` | flag | `False` | Graba video procesado. |
| `--modelo` | str | `models/best_model.pth` | Checkpoint. |

Comando:

```powershell
python Scripts/7_pipeline_camara.py --fuente 0 --skip_frames 2 --resize_factor 0.5 --grabar
```

## Fase 8 - CalibraciÃ³n de umbral (`8_umbral.py`)

QuÃ© hace:
- Busca threshold Ã³ptimo de confianza.
- Opcionalmente actualiza el checkpoint.

ParÃ¡metros:

| ParÃ¡metro | Tipo | Default | DescripciÃ³n |
|---|---|---:|---|
| `--modelo` | str | `models/best_model.pth` | Checkpoint. |
| `--grafica` | flag | `False` | Genera curva PR/F1. |
| `--aplicar` | flag | `False` | Escribe threshold Ã³ptimo al checkpoint. |

Comando recomendado:

```powershell
python Scripts/8_umbral.py --grafica --aplicar
```

## Fase 9 - EvaluaciÃ³n final (`9_evaluacion.py`)

QuÃ© hace:
- Consolida mÃ©tricas del entrenamiento, umbral y rendimiento.
- Genera grÃ¡ficas y reporte final en JSON.

ParÃ¡metros:

| ParÃ¡metro | Tipo | Default | DescripciÃ³n |
|---|---|---:|---|
| `--modelo` | str | `models/best_model.pth` | Checkpoint. |
| `--curvas` | flag | `False` | Guarda curvas de entrenamiento. |
| `--confusion` | flag | `False` | Guarda matriz de confusiÃ³n. |
| `--camara` | flag | `False` | Mide rendimiento en vivo. |
| `--guardar` | flag | `False` | Guarda `reporte_final.json`. |
| `--fuente` | int | `0` | CÃ¡mara para `--camara`. |
| `--duracion` | int | `30` | Segundos para mediciÃ³n de cÃ¡mara. |

Comando completo:

```powershell
python Scripts/9_evaluacion.py --curvas --confusion --camara --guardar
```

## Demo unificada (`demo.py`)

QuÃ© hace:
- Punto de entrada final para usuario.
- Usa lÃ³gica ya implementada en fases 6 y 7.

ParÃ¡metros:

| ParÃ¡metro | Tipo | Default | DescripciÃ³n |
|---|---|---:|---|
| `--camara` | flag | `False` | Modo cÃ¡mara en vivo. |
| `--imagen` | str | `None` | Modo imagen estÃ¡tica. |
| `--grabar` | flag | `False` | Grabar sesiÃ³n de cÃ¡mara. |
| `--mostrar` | flag | `False` | Mostrar imagen anotada. |
| `--modelo` | str | ruta por defecto del checkpoint | Modelo alternativo. |
| `--fuente` | int | `0` | Ãndice de cÃ¡mara. |

Comandos:

```powershell
python Scripts/demo.py --camara
python Scripts/demo.py --camara --grabar
python Scripts/demo.py --imagen "C:\Users\joeyk\Downloads\foto.jpg" --mostrar
```

---

## 6. Comandos rÃ¡pidos por escenario

## 6.1 Pipeline desde cero

```powershell
python Scripts/1_Captura.py --persona "Joey" --meta 300 --cooldown 0.8
python Scripts/2_Preprocesar.py --confianza 0.90
python Scripts/3_Aumentar.py --factor 4
python Scripts/4_preparar_clasificacion.py --train 0.70 --val 0.15 --test 0.15
python Scripts/5_entrenar_cnn.py --epochs 50 --batch_size 32 --model_name resnet18
python Scripts/8_umbral.py --grafica --aplicar
python Scripts/9_evaluacion.py --curvas --confusion --guardar
python Scripts/demo.py --camara
```

## 6.2 Rehacer solo splits (Fase 4)

```powershell
Remove-Item ".\splits" -Recurse -Force -ErrorAction SilentlyContinue
python Scripts/4_preparar_clasificacion.py --train 0.70 --val 0.15 --test 0.15
```

## 6.3 Reentrenar desde Fase 5

```powershell
Remove-Item ".\models\best_model.pth" -Force -ErrorAction SilentlyContinue
Remove-Item ".\models\last_model.pth" -Force -ErrorAction SilentlyContinue
Remove-Item ".\models\history.json" -Force -ErrorAction SilentlyContinue
Remove-Item ".\models\metrics_test.json" -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force ".\models" | Out-Null
python Scripts/5_entrenar_cnn.py --epochs 50 --batch_size 32 --model_name resnet18
python Scripts/8_umbral.py --grafica --aplicar
```

---

## 7. Integrantes

- Rodriguez Rojo Israel Josue - 22170799
- Samano Machado Kevin Jasiel - 22170815
- Quevedo Castellon Joey Kelvin - 22170777



