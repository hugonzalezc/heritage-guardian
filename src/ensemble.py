
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from src.config import RESULTS, CLASSES

from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_comparison
)


# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RESNET_NAME = "resnet18_balanced_seed42"

VIT_NAME = "vit_small_improved_seed42"

ENSEMBLE_NAME = "ensemble_resnet_vit_seed42"

CONFIG_PATH = (
    RESULTS / "metrics" / "ensemble_config.json"
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Heritage Guardian Ensemble"
    )

    parser.add_argument(
        "--split",
        choices=["val", "test", "local"],
        default="val"
    )

    return parser.parse_args()


# ==========================================
# 2. CARGAR PREDICCIONES
# ==========================================

def load_predictions(split, experiment):

    path = (
        RESULTS / "predictions" / split /
        f"{experiment}_predictions.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"No existen las predicciones: {path}"
        )

    predictions = pd.read_csv(path)

    required_columns = [

        "image",
        "y_true",
        "y_pred",
        "prob_grafiti",
        "prob_mural"
    ]

    for column in required_columns:

        if column not in predictions.columns:

            raise ValueError(
                f"Columna faltante: {column}"
            )

    if predictions["image"].duplicated().any():

        raise ValueError(
            "Existen imágenes duplicadas."
        )

    probs = predictions[
        ["prob_grafiti", "prob_mural"]
    ].to_numpy()

    if not np.isfinite(probs).all():

        raise ValueError(
            "Existen probabilidades no finitas."
        )

    if (probs < 0).any() or (probs > 1).any():

        raise ValueError(
            "Probabilidades fuera del rango válido."
        )

    if not np.allclose(
        probs.sum(axis=1),
        1.0,
        atol=1e-5
    ):

        raise ValueError(
            "Las probabilidades no suman uno."
        )

    return predictions


# ==========================================
# 3. ALINEAR LAS PREDICCIONES
# ==========================================

def align_predictions(resnet, vit):

    resnet = resnet.sort_values(
        "image"
    ).reset_index(drop=True)

    vit = vit.sort_values(
        "image"
    ).reset_index(drop=True)

    if len(resnet) != len(vit):

        raise ValueError(
            "Los modelos tienen distintas "
            "cantidades de predicciones."
        )

    if not resnet["image"].equals(
        vit["image"]
    ):

        raise ValueError(
            "Los nombres de las imágenes "
            "no coinciden."
        )

    if not resnet["y_true"].equals(
        vit["y_true"]
    ):

        raise ValueError(
            "Las etiquetas reales no coinciden."
        )

    return resnet, vit


# ==========================================
# 4. CONSTRUIR EL ENSEMBLE
# ==========================================

def combine_predictions(
    resnet,
    vit,
    alpha
):

    if not 0 <= alpha <= 1:

        raise ValueError(
            "Alpha debe estar entre 0 y 1."
        )

    resnet_probs = resnet[

        ["prob_grafiti", "prob_mural"]

    ].to_numpy()

    vit_probs = vit[

        ["prob_grafiti", "prob_mural"]

    ].to_numpy()

    ensemble_probs = (

        alpha * resnet_probs

        + (1 - alpha) * vit_probs
    )

    y_true = resnet["y_true"].to_numpy()

    y_pred = ensemble_probs.argmax(
        axis=1
    )

    predictions = pd.DataFrame({

        "image": resnet["image"],

        "y_true": y_true,

        "y_pred": y_pred,

        "true_class": [

            CLASSES[i] for i in y_true
        ],

        "predicted_class": [

            CLASSES[i] for i in y_pred
        ],

        "prob_grafiti": ensemble_probs[:, 0],

        "prob_mural": ensemble_probs[:, 1],

        "confidence": ensemble_probs.max(
            axis=1
        ),

        "correct": y_true == y_pred
    })

    return predictions


# ==========================================
# 5. BUSCAR EL MEJOR PESO
# ==========================================

def optimize_alpha(resnet, vit):

    print("\n=== OPTIMIZACIÓN DEL ENSEMBLE ===")

    results = []

    # 21 configuraciones: 0.00, 0.05, ..., 1.00

    for alpha in np.linspace(0, 1, 21):

        alpha = round(float(alpha), 2)

        predictions = combine_predictions(

            resnet,

            vit,

            alpha
        )

        metrics = calculate_metrics(
            predictions
        )

        metrics["alpha_resnet"] = alpha

        metrics["alpha_vit"] = round(
            1 - alpha, 2
        )

        results.append(metrics)

        print(

            f"ResNet={alpha:.2f} | "

            f"ViT={1-alpha:.2f} | "

            f"F1={metrics['macro_f1']:.4f} | "

            f"Recall mural="
            f"{metrics['recall_mural']:.4f}"
        )

    results_df = pd.DataFrame(results)

    # Selección principal: Macro F1
    # Desempate: Recall Mural
    # Segundo desempate: Accuracy
    # Si persiste, favorecer el menor peso ResNet

    ranked = results_df.sort_values(

        by=[

            "macro_f1",

            "recall_mural",

            "accuracy",

            "alpha_resnet"
        ],

        ascending=[

            False,

            False,

            False,

            True
        ]
    )

    best = ranked.iloc[0]

    best_alpha = float(
        best["alpha_resnet"]
    )

    return best_alpha, results_df


# ==========================================
# 6. ANALIZAR COMPLEMENTARIEDAD
# ==========================================

def analyze_complementarity(resnet, vit):

    resnet_correct = (

        resnet["y_true"]

        == resnet["y_pred"]
    ).to_numpy()

    vit_correct = (

        vit["y_true"]

        == vit["y_pred"]
    ).to_numpy()

    both_correct = int(

        (resnet_correct & vit_correct).sum()
    )

    only_resnet = int(

        (resnet_correct & ~vit_correct).sum()
    )

    only_vit = int(

        (~resnet_correct & vit_correct).sum()
    )

    both_wrong = int(

        (~resnet_correct & ~vit_correct).sum()
    )

    disagreement = int(

        (
            resnet["y_pred"]

            != vit["y_pred"]
        ).sum()
    )

    oracle_accuracy = float(

        (resnet_correct | vit_correct).mean()
    )

    result = {

        "total_images": len(resnet),

        "both_correct": both_correct,

        "only_resnet_correct": only_resnet,

        "only_vit_correct": only_vit,

        "both_wrong": both_wrong,

        "disagreement": disagreement,

        "oracle_accuracy": oracle_accuracy
    }

    print("\n=== COMPLEMENTARIEDAD ===")

    print("Ambos correctos:", both_correct)

    print("Solo ResNet correcto:", only_resnet)

    print("Solo ViT correcto:", only_vit)

    print("Ambos incorrectos:", both_wrong)

    print("Desacuerdos:", disagreement)

    print(
        f"Oracle Accuracy: {oracle_accuracy:.4f}"
    )

    return result


# ==========================================
# 7. GRÁFICOS DE OPTIMIZACIÓN
# ==========================================

def plot_alpha_search(
    search_df,
    best_alpha,
    output_dir
):

    search_df = search_df.sort_values(
        "alpha_resnet"
    )

    plt.figure(figsize=(10, 6))

    plt.plot(

        search_df["alpha_resnet"],

        search_df["macro_f1"],

        marker="o",

        label="Macro F1"
    )

    plt.plot(

        search_df["alpha_resnet"],

        search_df["accuracy"],

        marker="s",

        label="Accuracy"
    )

    plt.plot(

        search_df["alpha_resnet"],

        search_df["recall_mural"],

        marker="^",

        label="Recall Mural"
    )

    plt.axvline(

        best_alpha,

        color="black",

        linestyle="--",

        label=f"Mejor alpha={best_alpha:.2f}"
    )

    plt.xlabel("Peso de ResNet18")

    plt.ylabel("Score")

    plt.title(
        "Optimización del Ensemble - Validación"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        output_dir / "ensemble_alpha_search.png",

        dpi=200
    )

    plt.close()


def plot_complementarity(
    complementarity,
    output_dir
):

    labels = [

        "Ambos correctos",

        "Solo ResNet",

        "Solo ViT",

        "Ambos incorrectos"
    ]

    values = [

        complementarity["both_correct"],

        complementarity["only_resnet_correct"],

        complementarity["only_vit_correct"],

        complementarity["both_wrong"]
    ]

    colors = [

        "#16a34a",

        "#3b82f6",

        "#8b5cf6",

        "#dc2626"
    ]

    plt.figure(figsize=(9, 5))

    bars = plt.bar(

        labels,

        values,

        color=colors
    )

    plt.bar_label(bars)

    plt.ylabel("Cantidad de imágenes")

    plt.title(
        "Complementariedad entre ResNet y ViT"
    )

    plt.xticks(rotation=10)

    plt.tight_layout()

    plt.savefig(

        output_dir /
        "ensemble_complementarity.png",

        dpi=200
    )

    plt.close()


# ==========================================
# 8. EJECUCIÓN PRINCIPAL
# ==========================================

def run_ensemble(split):

    print(
        "\n=== HERITAGE GUARDIAN ENSEMBLE ==="
    )

    print("Split:", split)

    figures_dir = (

        RESULTS / "figures" / split
    )

    metrics_dir = (

        RESULTS / "metrics" / split
    )

    predictions_dir = (

        RESULTS / "predictions" / split
    )

    for folder in [

        figures_dir,

        metrics_dir,

        predictions_dir
    ]:

        folder.mkdir(

            parents=True,

            exist_ok=True
        )

    # Cargar las predicciones individuales

    resnet = load_predictions(

        split,

        RESNET_NAME
    )

    vit = load_predictions(

        split,

        VIT_NAME
    )

    resnet, vit = align_predictions(

        resnet,

        vit
    )

    print("Imágenes alineadas:", len(resnet))

    # Solo se optimiza utilizando validación

    if split == "val":

        alpha, search_df = optimize_alpha(

            resnet,

            vit
        )

        search_df.to_csv(

            metrics_dir /
            "ensemble_alpha_search.csv",

            index=False
        )

        plot_alpha_search(

            search_df,

            alpha,

            figures_dir
        )

        ensemble_config = {

            "resnet_experiment": RESNET_NAME,

            "vit_experiment": VIT_NAME,

            "alpha_resnet": alpha,

            "alpha_vit": round(1-alpha, 2),

            "selection_metric": "macro_f1",

            "selection_split": "val",

            "tie_breakers": [

                "recall_mural",

                "accuracy",

                "lower_alpha_resnet"
            ],

            "weight_step": 0.05
        }

        with open(

            CONFIG_PATH,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                ensemble_config,

                file,

                indent=4
            )

    else:

        # Test y local utilizan el peso
        # previamente seleccionado en validación

        if not CONFIG_PATH.exists():

            raise FileNotFoundError(

                "Primero debes optimizar "
                "el ensemble en validación."
            )

        with open(

            CONFIG_PATH,

            "r",

            encoding="utf-8"

        ) as file:

            ensemble_config = json.load(file)

        if (

            ensemble_config["resnet_experiment"]

            != RESNET_NAME

            or ensemble_config["vit_experiment"]

            != VIT_NAME

            or ensemble_config["selection_split"]

            != "val"
        ):

            raise ValueError(
                "Configuración incompatible."
            )

        alpha = float(

            ensemble_config["alpha_resnet"]
        )

    print("\n=== CONFIGURACIÓN SELECCIONADA ===")

    print(f"Peso ResNet: {alpha:.2f}")

    print(f"Peso ViT: {1-alpha:.2f}")

    # Predicciones finales del ensemble

    predictions = combine_predictions(

        resnet,

        vit,

        alpha
    )

    metrics = calculate_metrics(
        predictions
    )

    metrics.update({

        "experiment": ENSEMBLE_NAME,

        "model": "ensemble",

        "mode": "weighted_soft_voting",

        "split": split,

        "alpha_resnet": alpha,

        "alpha_vit": round(1-alpha, 2)
    })

    predictions.to_csv(

        predictions_dir /
        f"{ENSEMBLE_NAME}_predictions.csv",

        index=False
    )

    predictions[
        ~predictions["correct"]
    ].to_csv(

        predictions_dir /
        f"{ENSEMBLE_NAME}_errors.csv",

        index=False
    )

    with open(

        metrics_dir /
        f"{ENSEMBLE_NAME}_metrics.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            metrics,

            file,

            indent=4
        )

    report = classification_report(

        predictions["y_true"],

        predictions["y_pred"],

        labels=[0, 1],

        target_names=CLASSES,

        zero_division=0
    )

    with open(

        metrics_dir /
        f"{ENSEMBLE_NAME}_report.txt",

        "w",

        encoding="utf-8"

    ) as file:

        file.write(report)

    # Complementariedad de los modelos

    complementarity = analyze_complementarity(

        resnet,

        vit
    )

    with open(

        metrics_dir /
        "ensemble_complementarity.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            complementarity,

            file,

            indent=4
        )

    plot_complementarity(

        complementarity,

        figures_dir
    )

    # Matriz de confusión

    plot_confusion_matrix(

        predictions,

        ENSEMBLE_NAME,

        split,

        figures_dir
    )

    # Comparar ensemble con los seis modelos

    comparison_path = (

        metrics_dir / "model_comparison.csv"
    )

    if not comparison_path.exists():

        raise FileNotFoundError(
            f"No existe: {comparison_path}"
        )

    comparison = pd.read_csv(
        comparison_path
    )

    comparison = comparison[

        comparison["experiment"]

        != ENSEMBLE_NAME
    ]

    comparison = pd.concat(

        [

            comparison,

            pd.DataFrame([metrics])
        ],

        ignore_index=True
    )

    comparison = comparison.sort_values(

        "macro_f1",

        ascending=False
    )

    comparison.to_csv(

        metrics_dir /
        "model_comparison_with_ensemble.csv",

        index=False
    )

    plot_comparison(

        comparison,

        figures_dir,

        f"{split} + ensemble"
    )

    # Curvas ROC incluyendo el ensemble

    all_predictions = {}

    for experiment in comparison["experiment"]:

        all_predictions[experiment] = (

            load_predictions(

                split,

                experiment
            )
        )

    plot_roc_curves(

        all_predictions,

        figures_dir,

        f"{split} + ensemble"
    )

    print("\n=== ENSEMBLE CLASSIFICATION REPORT ===")

    print(report)

    print(
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )

    print(
        f"Recall mural: "
        f"{metrics['recall_mural']:.4f}"
    )

    print(
        f"AUC-ROC: {metrics['auc_roc']:.4f}"
    )

    print(
        f"MCC: {metrics['mcc']:.4f}"
    )

    print("\n=== COMPARACIÓN CON ENSEMBLE ===")

    print(

        comparison[[

            "experiment",

            "accuracy",

            "macro_f1",

            "recall_mural",

            "auc_roc",

            "mcc"

        ]].round(4).to_string(index=False)
    )

    print("\nEnsemble completado.")

    print("Configuración:", CONFIG_PATH)


if __name__ == "__main__":

    args = parse_args()

    run_ensemble(args.split)
