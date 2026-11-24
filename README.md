# Heritage Guardian — Mural vs. Grafiti

Proyecto aplicado de Aprendizaje Automático II · Pontificia Universidad Católica de Valparaíso · 2026.

**Integrantes:** Hugo González, Rodrigo Venegas y Víctor Córdova (según la presentación de avance; confirmar nómina final).

Prototipo de clasificación visual de intervenciones urbanas. Compara ResNet18, ViT-Small y Random Forest con embeddings de ImageNet; estudia balanceo, fine-tuning, ensemble, interpretabilidad y generalización externa. Las etiquetas son proxies visuales, no determinaciones de vandalismo ni de valor patrimonial.

## Resultados principales

| Experimento | Accuracy test | Macro F1 test | Recall mural test | Macro F1 local | AUC local |
|---|---:|---:|---:|---:|---:|
| ResNet baseline | .7355 | .6454 | .3590 | .5217 | .5747 |
| ResNet improved | .7810 | .7502 | .6667 | .4763 | .6374 |
| ResNet balanced | .8264 | .8027 | .7436 | .5055 | .7212 |
| ViT baseline | .8182 | .7789 | .6154 | .4763 | .6212 |
| ViT improved | .8347 | .8177 | .8205 | .5055 | .6606 |
| ViT balanced | .8306 | .8104 | .7821 | .4457 | .5424 |
| Random Forest | .7273 | .6879 | .5769 | .5055 | .8141 |

ViT Improved fue seleccionado por Macro F1 en validación (.8432). Weighted soft voting exploró 21 pesos y seleccionó 0% ResNet / 100% ViT: no aportó mejora. En Valparaíso, ViT reconoció solo 5/30 murales y los tres modelos principales clasificaron los cinco casos ambiguos como grafiti. El sistema no está listo para uso operativo.

## Estructura

```text
heritage-guardian/
  app.py
  requirements.txt
  src/                         # preparación, entrenamiento, evaluación, interpretabilidad
  notebooks/00_avance_original.ipynb
  docs/DATA_CARD.md
  docs/informe.tex
  data/valparaiso_metadata.csv
  data/raw/                    # archivos de trabajo, no versionar imágenes
  data/dataset_estructurado/    # train/val/test, grafiti/mural
  data/valparaiso_raw/
  data/test_valparaiso/
  data/valparaiso_ambiguous/
  checkpoints/                 # baseline, final, classical (local)
  results/                     # histories, metrics, predictions, figures, embeddings
```

## Instalación (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Python 3.12.6; entorno local comprobado con PyTorch 2.7.1 y torchvision 0.22.1 CUDA 12.6, GTX 1660 Ti Max-Q. ViT se entrenó en Colab Tesla T4 con AMP. Para reproducir el entorno CUDA local, instalar las ruedas correspondientes en vez de las CPU:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

La inferencia admite CPU si CUDA no está disponible. El entrenamiento completo es más costoso en CPU. Las versiones restantes deben registrarse con `python -m pip freeze` en el entorno utilizado; no se garantiza identidad numérica entre plataformas.

## Datos y reproducción

Consultar [Data Card](docs/DATA_CARD.md). Los archivos de imágenes y pesos no se incluyen por defecto en el repositorio público: su distribución está condicionada a la revisión de derechos. Si se dispone de acceso autorizado, `src.prepare_data` contiene las referencias a los archivos de trabajo de Google Drive. No se garantiza acceso público a dichos archivos. Conservar los splits existentes y el manifiesto; no regenerarlos durante la comparación.

```powershell
python -m src.prepare_data
python -m src.prepare_data --audit-only
python -m src.analyze_data
```

Se esperan 1127/242/242 imágenes en train/val/test (1611 en total). Semilla 42.

### Entrenamiento desde cero (solo si no existen resultados)

```powershell
python -m src.train --model resnet18 --mode baseline --no-amp
python -m src.train --model resnet18 --mode improved --no-amp
python -m src.train --model resnet18 --mode balanced --no-amp
python -m src.train --model vit_small --mode baseline --batch-size 16 --workers 2
python -m src.train --model vit_small --mode improved --batch-size 16 --workers 2
python -m src.train --model vit_small --mode balanced --batch-size 16 --workers 2
```

Las ejecuciones ViT originales se realizaron en Colab con CUDA/AMP. Los scripts protegen resultados existentes; usar una copia de trabajo limpia si se desea reproducir. Baseline: 5 épocas, Adam y selección por accuracy de validación. Improved/balanced: hasta 25 épocas, AdamW, cosine annealing, label smoothing, early stopping (paciencia 5), selección por Macro F1. Balanced agrega oversampling ponderado.

### Evaluación original

```powershell
python -m src.evaluate --split val
python -m src.ensemble --split val
python -m src.evaluate --split test
python -m src.ensemble --split test
python -m src.random_forest
python -m src.explainability --split val
```

La optimización del ensemble se hace únicamente sobre validación; test reutiliza los pesos guardados. Random Forest aprende solo de train con 300 árboles y embeddings de 512 dimensiones.

### Evaluación externa

Con acceso autorizado a las 68 imágenes, guardarlas en `data/valparaiso_raw` usando los nombres `valparaiso_001` a `valparaiso_068` y las extensiones originales; el CSV incluye sus fuentes. Las carpetas de destino deben estar vacías (incluidos archivos `.gitkeep`) para la primera preparación.

```powershell
python -m src.label_local
python -m src.evaluate --split local
python -m src.ensemble --split local
python -m src.explainability --split local
python -m src.evaluate_rf_local
python -m src.evaluate_ambiguous
```

No ajustar hiperparámetros usando el test externo. Los cinco casos dudosos se evalúan sin etiquetas binarias de referencia.

## Demo Gradio

Se requieren estos tres archivos obtenidos por entrenamiento o por acceso autorizado a los checkpoints del equipo:

- `checkpoints/final/resnet18_balanced_seed42_best.pth`
- `checkpoints/final/vit_small_improved_seed42_best.pth`
- `checkpoints/classical/random_forest_resnet18_seed42.joblib`

El extractor congelado necesita los pesos ImageNet de torchvision (descarga inicial si no están en caché). Cargar únicamente pesos y archivos joblib de confianza.

```powershell
python app.py
```

Abrir `http://127.0.0.1:7860` (o la URL indicada). La interfaz permite cargar imágenes, comparar puntuaciones, activar Grad-CAM y Attention Rollout y revisar resultados, metodología y limitaciones. Se ejecuta localmente, sin enlace público. El tiempo de inferencia depende del dispositivo.

## Evidencia experimental

Las tablas de comparación están en `results/metrics/{val,test,local}/model_comparison_all.csv`. Los historiales están en `results/histories`; las predicciones individuales en `results/predictions`; las matrices, ROC y mapas en `results/figures`. `results/metrics/ensemble_config.json` registra los pesos fijados en validación. Los resultados agregados deben mantenerse junto al código, evitando publicar material visual sin permisos.

## Uso responsable y derechos

El sistema distingue categorías visuales aproximadas. No determina autorización, vandalismo, patrimonio ni decisiones de conservación. La muestra local es pequeña, de conveniencia y con sesgo de procedencia; los modelos no están calibrados para Valparaíso. No usar como única base de decisiones. La licencia del código debe acordarse entre los autores; las licencias de imágenes y pesos se revisan por separado. Ver [Data Card](docs/DATA_CARD.md).
