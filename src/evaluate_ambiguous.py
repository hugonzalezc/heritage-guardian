
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch

from src.config import (
    ROOT,
    RESULTS,
    CHECKPOINTS,
    CLASSES
)

from src.models import get_device

from src.evaluate import load_checkpoint

from src.random_forest import (
    EXPERIMENT,
    create_feature_extractor
)

from src.explainability import (
    explain_image,
    prepare_image,
    save_explanation
)


# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

AMBIGUOUS_DIR = (
    ROOT / "data" / "valparaiso_ambiguous"
)

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff"
}


# ==========================================
# 2. PREDICCIÓN RANDOM FOREST
# ==========================================

def predict_rf(
    image_path,
    extractor,
    classifier,
    device
):

    tensor, _ = prepare_image(
        image_path,
        device
    )

    with torch.inference_mode():

        embedding = extractor(
            tensor
        ).cpu().numpy()

    if not np.array_equal(
        classifier.classes_,
        [0, 1]
    ):

        raise ValueError(
            "Orden de clases incorrecto."
        )

    probabilities = classifier.predict_proba(
        embedding
    )[0]

    prediction = int(
        probabilities.argmax()
    )

    return prediction, probabilities


# ==========================================
# 3. GRÁFICO DE PROBABILIDADES
# ==========================================

def plot_probabilities(
    predictions,
    output_path
):

    names = predictions["image"].tolist()

    x = np.arange(len(names))

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        x - width,
        predictions["resnet_prob_mural"],
        width,
        label="ResNet18 Balanced",
        color="#3b82f6"
    )

    ax.bar(
        x,
        predictions["vit_prob_mural"],
        width,
        label="ViT-Small Improved",
        color="#8b5cf6"
    )

    ax.bar(
        x + width,
        predictions["rf_prob_mural"],
        width,
        label="Random Forest",
        color="#d97706"
    )

    ax.axhline(
        0.5,
        color="black",
        linestyle="--",
        label="Umbral de decisión"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        names,
        rotation=20,
        ha="right"
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel(
        "Probabilidad asignada a mural"
    )

    ax.set_title(
        "Predicciones sobre casos ambiguos"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close(fig)


# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================

def main():

    print(
        "\n=== HERITAGE GUARDIAN AMBIGUOUS CASES ==="
    )

    device = get_device()

    print("Dispositivo:", device)

    if not AMBIGUOUS_DIR.exists():

        raise FileNotFoundError(
            f"No existe: {AMBIGUOUS_DIR}"
        )

    paths = sorted([

        path

        for path in AMBIGUOUS_DIR.iterdir()

        if path.is_file()

        and path.suffix.lower()
        in VALID_EXTENSIONS
    ])

    if not paths:

        raise RuntimeError(
            "No se encontraron imágenes dudosas."
        )

    print("Casos ambiguos:", len(paths))

    # Crear carpetas de salida

    figures_dir = (
        RESULTS / "figures" / "ambiguous"
    )

    predictions_dir = (
        RESULTS / "predictions" / "ambiguous"
    )

    for folder in [
        figures_dir,
        predictions_dir
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    # ======================================
    # CARGAR MODELOS
    # ======================================

    print("\nCargando ResNet18 Balanced...")

    resnet, _ = load_checkpoint(
        "resnet18",
        "balanced",
        device
    )

    print("Cargando ViT-Small Improved...")

    vit, _ = load_checkpoint(
        "vit_small",
        "improved",
        device
    )

    print("Cargando Random Forest...")

    rf_path = (
        CHECKPOINTS /
        "classical" /
        f"{EXPERIMENT}.joblib"
    )

    classifier = joblib.load(rf_path)

    extractor = create_feature_extractor(
        device
    )

    records = []

    # ======================================
    # ANALIZAR CADA IMAGEN
    # ======================================

    for index, path in enumerate(paths, start=1):

        print(
            f"\n[{index}/{len(paths)}] {path.name}"
        )

        explanation = explain_image(
            path,
            resnet,
            vit,
            device
        )

        resnet_pred = explanation["resnet_pred"]

        vit_pred = explanation["vit_pred"]

        resnet_probs = explanation["resnet_probs"]

        vit_probs = explanation["vit_probs"]

        rf_pred, rf_probs = predict_rf(
            path,
            extractor,
            classifier,
            device
        )

        # Guardar figura de interpretabilidad

        output_path = (
            figures_dir /
            f"{path.stem}_explanation.png"
        )

        save_explanation(
            explanation,
            output_path,
            "dudoso",
            "caso_ambiguo"
        )

        # Guardar mapas numéricos

        np.savez_compressed(

            figures_dir /
            f"{path.stem}_maps.npz",

            grad_cam=explanation["resnet_map"],

            attention_rollout=explanation["vit_map"]
        )

        agreement = (
            resnet_pred == vit_pred == rf_pred
        )

        records.append({

            "image": path.name,

            "human_label": "dudoso",

            "resnet_prediction":
                CLASSES[resnet_pred],

            "resnet_prob_grafiti":
                float(resnet_probs[0]),

            "resnet_prob_mural":
                float(resnet_probs[1]),

            "resnet_confidence":
                float(resnet_probs.max()),

            "vit_prediction":
                CLASSES[vit_pred],

            "vit_prob_grafiti":
                float(vit_probs[0]),

            "vit_prob_mural":
                float(vit_probs[1]),

            "vit_confidence":
                float(vit_probs.max()),

            "rf_prediction":
                CLASSES[rf_pred],

            "rf_prob_grafiti":
                float(rf_probs[0]),

            "rf_prob_mural":
                float(rf_probs[1]),

            "rf_confidence":
                float(rf_probs.max()),

            "resnet_vit_agreement":
                resnet_pred == vit_pred,

            "all_models_agreement":
                agreement
        })

        print(
            f"ResNet: {CLASSES[resnet_pred]} "
            f"({resnet_probs.max():.1%})"
        )

        print(
            f"ViT: {CLASSES[vit_pred]} "
            f"({vit_probs.max():.1%})"
        )

        print(
            f"Random Forest: {CLASSES[rf_pred]} "
            f"({rf_probs.max():.1%})"
        )

        print(
            "Coincidencia de los tres:",
            "Sí" if agreement else "No"
        )

    # ======================================
    # GUARDAR RESULTADOS
    # ======================================

    predictions = pd.DataFrame(records)

    csv_path = (
        predictions_dir /
        "ambiguous_predictions.csv"
    )

    predictions.to_csv(
        csv_path,
        index=False
    )

    plot_probabilities(
        predictions,
        figures_dir /
        "ambiguous_probabilities.png"
    )

    total = len(predictions)

    agreement_all = int(
        predictions[
            "all_models_agreement"
        ].sum()
    )

    agreement_neural = int(
        predictions[
            "resnet_vit_agreement"
        ].sum()
    )

    print("\n=== RESUMEN DE CASOS AMBIGUOS ===")

    print(

        predictions[[

            "image",

            "resnet_prediction",

            "resnet_confidence",

            "vit_prediction",

            "vit_confidence",

            "rf_prediction",

            "rf_confidence",

            "all_models_agreement"

        ]].round(4).to_string(index=False)
    )

    print("\nTotal de imágenes:", total)

    print(
        "Coincidencias ResNet y ViT:",
        f"{agreement_neural}/{total}"
    )

    print(
        "Coincidencias de los tres modelos:",
        f"{agreement_all}/{total}"
    )

    print(
        "Desacuerdos entre los tres modelos:",
        total - agreement_all
    )

    print(
        "\nNo se calculan métricas de acierto "
        "porque no existen etiquetas binarias "
        "de referencia."
    )

    print("\nAnálisis completado.")

    print("Predicciones:", csv_path)

    print("Figuras:", figures_dir)


if __name__ == "__main__":

    main()
