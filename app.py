
import time
import threading

import gradio as gr
import joblib
import numpy as np
import torch

from PIL import Image

from src.config import (
    CHECKPOINTS,
    CLASSES,
    IMAGE_SIZE
)

from src.dataset import get_transforms
from src.models import get_device
from src.evaluate import load_checkpoint

from src.random_forest import (
    EXPERIMENT,
    create_feature_extractor
)

from src.explainability import (
    grad_cam,
    attention_rollout,
    overlay_heatmap
)


# ==========================================
# 1. CARGAR MODELOS
# ==========================================

DEVICE = get_device()

print("\n=== HERITAGE GUARDIAN APP ===")
print("Dispositivo:", DEVICE)

print("Cargando ResNet18 Balanced...")

RESNET, _ = load_checkpoint(
    "resnet18",
    "balanced",
    DEVICE
)

print("Cargando ViT-Small Improved...")

VIT, _ = load_checkpoint(
    "vit_small",
    "improved",
    DEVICE
)

print("Cargando Random Forest...")

RF_PATH = (
    CHECKPOINTS /
    "classical" /
    f"{EXPERIMENT}.joblib"
)

RF = joblib.load(RF_PATH)

EXTRACTOR = create_feature_extractor(DEVICE)

if not np.array_equal(RF.classes_, [0, 1]):
    raise ValueError(
        "El orden de clases de Random Forest no coincide."
    )

_, TRANSFORM = get_transforms("baseline")

# Evitar ejecuciones simultáneas que interfieran
# con los hooks de interpretabilidad.

MODEL_LOCK = threading.Lock()

print("Modelos cargados correctamente.")


# ==========================================
# 2. FUNCIONES AUXILIARES
# ==========================================

def probabilities_to_dict(probabilities):

    return {
        "Grafiti": float(probabilities[0]),
        "Mural": float(probabilities[1])
    }


def to_display_image(array):

    array = np.clip(array, 0, 1)

    return Image.fromarray(
        (array * 255).astype(np.uint8)
    )


def prepare_input(image):

    image = image.convert("RGB")

    tensor = TRANSFORM(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    display = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    display = np.asarray(
        display,
        dtype=np.float32
    ) / 255.0

    return tensor, display


# ==========================================
# 3. INFERENCIA SIN MAPAS
# ==========================================

def predict_neural(model, tensor):

    model.eval()

    with torch.inference_mode():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0].cpu().numpy()

    prediction = int(
        probabilities.argmax()
    )

    return prediction, probabilities


def predict_random_forest(tensor):

    with torch.inference_mode():

        embedding = EXTRACTOR(
            tensor
        ).cpu().numpy()

    probabilities = RF.predict_proba(
        embedding
    )[0]

    prediction = int(
        probabilities.argmax()
    )

    return prediction, probabilities


# ==========================================
# 4. GENERAR RESUMEN
# ==========================================

def build_summary(
    resnet_pred,
    resnet_probs,
    vit_pred,
    vit_probs,
    rf_pred,
    rf_probs,
    elapsed
):

    main_class = CLASSES[vit_pred].capitalize()

    main_score = float(
        vit_probs[vit_pred]
    )

    agreement = (
        resnet_pred == vit_pred == rf_pred
    )

    summary = f"""
## Resultado principal: {main_class}

**Modelo:** ViT-Small Improved

**Puntuación Softmax:** {main_score:.2%}

### Comparación de modelos

| Modelo | Predicción | Puntuación |
|---|---|---:|
| ViT-Small Improved | {main_class} | {main_score:.2%} |
| ResNet18 Balanced | {CLASSES[resnet_pred].capitalize()} | {resnet_probs[resnet_pred]:.2%} |
| Random Forest | {CLASSES[rf_pred].capitalize()} | {rf_probs[rf_pred]:.2%} |

**Tiempo de procesamiento:** {elapsed:.2f} segundos

**Dispositivo:** {DEVICE}
"""

    warnings = []

    # Indicador heurístico, no calibrado.

    if main_score < 0.70:

        warnings.append(
            "La predicción principal presenta "
            "un margen de decisión reducido."
        )

    if not agreement:

        warnings.append(
            "Existe desacuerdo entre los modelos."
        )

    else:

        warnings.append(
            "Los tres modelos coinciden. "
            "Esto no garantiza que la predicción "
            "sea correcta."
        )

    warning_text = "\n".join(
        f"- {warning}" for warning in warnings
    )

    summary += f"""

### Indicadores de revisión

{warning_text}

**Importante:** las puntuaciones no están calibradas
para fotografías de Valparaíso. Incluso una puntuación
elevada puede corresponder a una clasificación incorrecta.
"""

    return summary


# ==========================================
# 5. FUNCIÓN PRINCIPAL DE CLASIFICACIÓN
# ==========================================

def classify_image(image, generate_maps):

    if image is None:

        raise gr.Error(
            "Primero debes cargar una imagen."
        )

    with MODEL_LOCK:

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        tensor, display = prepare_input(image)

        resnet_overlay = None
        vit_overlay = None

        # ==================================
        # INFERENCIA CON INTERPRETABILIDAD
        # ==================================

        if generate_maps:

            resnet_map, resnet_pred, resnet_probs = (
                grad_cam(
                    RESNET,
                    tensor
                )
            )

            vit_map, vit_pred, vit_probs = (
                attention_rollout(
                    VIT,
                    tensor
                )
            )

            resnet_overlay = to_display_image(
                overlay_heatmap(
                    display,
                    resnet_map
                )
            )

            vit_overlay = to_display_image(
                overlay_heatmap(
                    display,
                    vit_map
                )
            )

        # ==================================
        # INFERENCIA SIN INTERPRETABILIDAD
        # ==================================

        else:

            resnet_pred, resnet_probs = (
                predict_neural(
                    RESNET,
                    tensor
                )
            )

            vit_pred, vit_probs = (
                predict_neural(
                    VIT,
                    tensor
                )
            )

        # ==================================
        # RANDOM FOREST
        # ==================================

        rf_pred, rf_probs = (
            predict_random_forest(tensor)
        )

        for probabilities in [
            resnet_probs,
            vit_probs,
            rf_probs
        ]:

            if not np.isfinite(probabilities).all():

                raise gr.Error(
                    "El modelo produjo valores inválidos."
                )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        summary = build_summary(
            resnet_pred,
            resnet_probs,
            vit_pred,
            vit_probs,
            rf_pred,
            rf_probs,
            elapsed
        )

        return (
            summary,
            probabilities_to_dict(vit_probs),
            probabilities_to_dict(resnet_probs),
            probabilities_to_dict(rf_probs),
            resnet_overlay,
            vit_overlay
        )


# ==========================================
# 6. LIMPIAR INTERFAZ
# ==========================================

def clear_interface():

    return (
        None,
        "Carga una imagen para comenzar.",
        None,
        None,
        None,
        None,
        None
    )


# ==========================================
# 7. INTERFAZ GRADIO
# ==========================================

with gr.Blocks(
    title="Heritage Guardian"
) as demo:

    gr.Markdown(
        """
# Heritage Guardian

### Clasificación visual de intervenciones urbanas

Prototipo experimental de Machine Learning para el
análisis de grafitis y murales, mediante redes
convolucionales, Vision Transformers y Random Forest.

**Sube una fotografía para comparar las predicciones
de las tres arquitecturas.**
"""
    )

    with gr.Tabs():

        # ==================================
        # PESTAÑA 1: CLASIFICACIÓN
        # ==================================

        with gr.Tab("Clasificador"):

            with gr.Row():

                with gr.Column():

                    input_image = gr.Image(
                        label="Imagen de entrada",
                        type="pil",
                        height=380
                    )

                    generate_maps = gr.Checkbox(
                        label=(
                            "Generar mapas de interpretabilidad"
                        ),
                        value=True
                    )

                    with gr.Row():

                        predict_button = gr.Button(
                            "Clasificar imagen",
                            variant="primary"
                        )

                        clear_button = gr.Button(
                            "Limpiar"
                        )

                    gr.Markdown(
                        """
Formatos compatibles: JPG, JPEG, PNG y WEBP.

La imagen se transforma internamente a
224 × 224 píxeles.
"""
                    )

                with gr.Column():

                    summary_output = gr.Markdown(
                        "Carga una imagen para comenzar."
                    )

            gr.Markdown(
                "## Puntuaciones de clasificación"
            )

            with gr.Row():

                vit_output = gr.Label(
                    label="ViT-Small Improved",
                    num_top_classes=2
                )

                resnet_output = gr.Label(
                    label="ResNet18 Balanced",
                    num_top_classes=2
                )

                rf_output = gr.Label(
                    label="Random Forest",
                    num_top_classes=2
                )

            gr.Markdown(
                """
## Interpretabilidad

Los mapas destacan regiones relacionadas con las
activaciones o la atención de las redes neuronales.
"""
            )

            with gr.Row():

                gradcam_output = gr.Image(
                    label="ResNet18 — Grad-CAM",
                    height=320,
                    interactive=False
                )

                attention_output = gr.Image(
                    label="ViT — Attention Rollout",
                    height=320,
                    interactive=False
                )

            gr.Markdown(
                """
Los colores cálidos representan mayor intensidad
relativa dentro de cada mapa.

Grad-CAM es específico de la clase predicha.
Attention Rollout representa atención acumulada y
no constituye una explicación causal de la decisión.
"""
            )

        # ==================================
        # PESTAÑA 2: RESULTADOS
        # ==================================

        with gr.Tab("Resultados experimentales"):

            gr.Markdown(
                """
# Evaluación de los modelos

Los modelos fueron entrenados utilizando un dataset
de 1.611 imágenes, dividido en:

- Entrenamiento: 1.127 imágenes.
- Validación: 242 imágenes.
- Test: 242 imágenes.

Posteriormente se realizó una evaluación externa
sobre 63 fotografías de Valparaíso, junto a cinco
casos ambiguos analizados por separado.

## Test original

| Modelo | Accuracy | Macro F1 | Recall Mural | AUC-ROC |
|---|---:|---:|---:|---:|
| ResNet Baseline | 73,55% | 64,54% | 35,90% | 0,7539 |
| ResNet Improved | 78,10% | 75,02% | 66,67% | 0,8547 |
| ResNet Balanced | 82,64% | 80,27% | 74,36% | 0,8949 |
| ViT Baseline | 81,82% | 77,89% | 61,54% | 0,8986 |
| ViT Improved | 83,47% | 81,77% | 82,05% | 0,8945 |
| ViT Balanced | 83,06% | 81,04% | 78,21% | 0,8975 |
| Random Forest | 72,73% | 68,79% | 57,69% | 0,8011 |

## Test independiente de Valparaíso

| Modelo | Accuracy | Macro F1 | Recall Mural | AUC-ROC |
|---|---:|---:|---:|---:|
| ResNet Baseline | 60,32% | 52,17% | 20,00% | 0,5747 |
| ResNet Improved | 58,73% | 47,63% | 13,33% | 0,6374 |
| ResNet Balanced | 60,32% | 50,55% | 16,67% | 0,7212 |
| ViT Baseline | 58,73% | 47,63% | 13,33% | 0,6212 |
| ViT Improved | 60,32% | 50,55% | 16,67% | 0,6606 |
| ViT Balanced | 57,14% | 44,57% | 10,00% | 0,5424 |
| Random Forest | 60,32% | 50,55% | 16,67% | 0,8141 |

## Conclusiones

ViT-Small Improved fue seleccionado utilizando
el conjunto de validación.

El ensemble mediante Weighted Soft Voting no
superó al mejor modelo individual.

La evaluación externa evidenció problemas de
generalización al dominio de Valparaíso,
especialmente en el reconocimiento de murales.

Random Forest obtuvo el mayor AUC-ROC local,
aunque sus clasificaciones con el umbral habitual
también presentaron limitaciones.

Los tres modelos clasificaron los cinco casos
ambiguos como grafiti.
"""
            )

        # ==================================
        # PESTAÑA 3: METODOLOGÍA
        # ==================================

        with gr.Tab("Metodología"):

            gr.Markdown(
                """
# Arquitecturas utilizadas

## ResNet18 Balanced

- Red neuronal convolucional residual.
- Transfer Learning desde ImageNet.
- Fine-tuning completo.
- Data Augmentation.
- Weighted Random Sampling.
- AdamW, Label Smoothing y Early Stopping.
- Interpretabilidad mediante Grad-CAM.

## ViT-Small Improved

- Vision Transformer.
- Imágenes de 224 × 224 píxeles.
- Parches de 16 × 16 píxeles.
- 196 parches y un token CLS.
- Fine-tuning completo.
- Data Augmentation.
- AdamW y Cosine Annealing.
- Label Smoothing y Early Stopping.
- Interpretabilidad mediante Attention Rollout.

## Random Forest

- ResNet18 preentrenada como extractor congelado.
- Embeddings de 512 dimensiones.
- Random Forest de 300 árboles.
- Balanceo mediante class weights.

## Ensemble

Se evaluaron 21 combinaciones de pesos
mediante Weighted Soft Voting.

La configuración seleccionada en validación
correspondió a 0% ResNet y 100% ViT.

Por ello, la aplicación utiliza ViT-Small Improved
como modelo principal y presenta los demás modelos
como herramientas de comparación.
"""
            )

        # ==================================
        # PESTAÑA 4: LIMITACIONES
        # ==================================

        with gr.Tab("Limitaciones"):

            gr.Markdown(
                """
# Limitaciones y uso responsable

Heritage Guardian es un prototipo académico
de clasificación visual.

## Definición de las clases

- Grafiti: predominio de firmas, tags, rayados
  o letras estilizadas.
- Mural: predominio de composiciones pictóricas,
  figurativas o abstractas.

Estas categorías pueden superponerse y dependen
del criterio de etiquetado.

## Limitaciones del dataset

- Desbalance de clases en el entrenamiento.
- Presencia de collages y marcas de agua.
- Posible ruido en las etiquetas originales.
- Diferencias entre las fuentes de las fotografías.
- Posible dependencia del contexto visual.
- Tamaño reducido del conjunto de evaluación local.

## Generalización

El modelo principal obtuvo:

- Macro F1 original: 81,77%.
- Macro F1 Valparaíso: 50,55%.
- Recall mural Valparaíso: 16,67%.

Estos resultados indican que el sistema requiere
mejoras antes de ser utilizado en escenarios reales.

## Interpretación de probabilidades

Las puntuaciones entregadas por los modelos
no deben interpretarse como garantías de acierto.

Los modelos pueden equivocarse incluso cuando
sus puntuaciones son elevadas o cuando todos
coinciden en una misma clase.

## Uso responsable

El sistema no determina:

- Si una intervención está autorizada.
- Si constituye vandalismo.
- Si posee valor artístico o patrimonial.
- Si corresponde eliminar o conservar una obra.

Las predicciones no deben utilizarse como base
única para decisiones legales, patrimoniales
o de intervención urbana.

## Trabajo futuro

- Incorporar datos representativos de Valparaíso.
- Mejorar la curación y consistencia del etiquetado.
- Evaluar la calibración de las probabilidades.
- Explorar técnicas de adaptación de dominio.
- Incorporar mecanismos de revisión humana.
"""
            )

    # ======================================
    # 8. EVENTOS DE LA INTERFAZ
    # ======================================

    predict_button.click(

        fn=classify_image,

        inputs=[
            input_image,
            generate_maps
        ],

        outputs=[
            summary_output,
            vit_output,
            resnet_output,
            rf_output,
            gradcam_output,
            attention_output
        ]
    )

    clear_button.click(

        fn=clear_interface,

        inputs=None,

        outputs=[
            input_image,
            summary_output,
            vit_output,
            resnet_output,
            rf_output,
            gradcam_output,
            attention_output
        ]
    )

    gr.Markdown(
        """
---

**Heritage Guardian | Proyecto académico de Machine Learning**

Clasificación visual, interpretabilidad y evaluación
de generalización al dominio objetivo.
"""
    )


# ==========================================
# 9. EJECUTAR APLICACIÓN
# ==========================================

if __name__ == "__main__":

    demo.queue().launch(
        server_name="127.0.0.1",
        inbrowser=True,
        share=False
    )
