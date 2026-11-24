
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.config import (
    RESULTS,
    CHECKPOINTS,
    CLASSES,
    SEED
)

from src.models import (
    build_model,
    get_device
)

from src.evaluate import (
    create_eval_loader,
    calculate_metrics,
    plot_confusion_matrix,
    plot_comparison,
    plot_roc_curves
)


EXPERIMENT = "random_forest_resnet18_seed42"

RF_CONFIG = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1
}


# ==========================================
# 1. EXTRACTOR DE CARACTERÍSTICAS
# ==========================================

def create_feature_extractor(device):

    model = build_model(
        "resnet18",
        pretrained=True,
        freeze_backbone=True
    )

    # Eliminar el clasificador final.
    # La salida ahora tiene 512 características.

    model.fc = nn.Identity()

    for parameter in model.parameters():

        parameter.requires_grad = False

    model = model.to(device)

    model.eval()

    return model


# ==========================================
# 2. EXTRAER EMBEDDINGS
# ==========================================

def extract_embeddings(model, split, device):

    loader, dataset, root = create_eval_loader(
        split=split,
        batch_size=32,
        workers=0
    )

    features = []
    labels = []

    print(f"\nExtrayendo embeddings: {split}")

    with torch.inference_mode():

        for images, targets in tqdm(
            loader,
            desc=f"Embeddings {split}",
            unit="batch"
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            embeddings = model(images)

            features.append(
                embeddings.cpu().numpy()
            )

            labels.extend(
                targets.numpy().tolist()
            )

    X = np.concatenate(
        features,
        axis=0
    )

    y = np.asarray(labels)

    if not np.isfinite(X).all():

        raise RuntimeError(
            "Embeddings NaN o infinitos."
        )

    paths = [
        __import__("pathlib").Path(path)
        .relative_to(root)
        .as_posix()
        for path, _ in dataset.samples
    ]

    print("Características:", X.shape)
    print("Etiquetas:", y.shape)

    return X, y, paths


# ==========================================
# 3. ENTRENAR RANDOM FOREST
# ==========================================

def train_random_forest(X_train, y_train):

    print("\n=== ENTRENANDO RANDOM FOREST ===")

    classifier = RandomForestClassifier(
        **RF_CONFIG
    )

    start = time.perf_counter()

    classifier.fit(
        X_train,
        y_train
    )

    elapsed = time.perf_counter() - start

    print(
        f"Tiempo entrenamiento RF: "
        f"{elapsed:.2f} segundos"
    )

    return classifier, elapsed


# ==========================================
# 4. GENERAR PREDICCIONES
# ==========================================

def predict_random_forest(
    classifier,
    X,
    y,
    paths
):

    probabilities = classifier.predict_proba(X)

    if not np.array_equal(
        classifier.classes_,
        [0, 1]
    ):

        raise ValueError(
            "El orden de clases no coincide."
        )

    predictions = probabilities.argmax(axis=1)

    return pd.DataFrame({

        "image": paths,

        "y_true": y,

        "y_pred": predictions,

        "true_class": [
            CLASSES[i] for i in y
        ],

        "predicted_class": [
            CLASSES[i] for i in predictions
        ],

        "prob_grafiti": probabilities[:, 0],

        "prob_mural": probabilities[:, 1],

        "confidence": probabilities.max(axis=1),

        "correct": y == predictions
    })


# ==========================================
# 5. GUARDAR RESULTADOS
# ==========================================

def save_results(predictions, split):

    metrics_dir = RESULTS / "metrics" / split

    predictions_dir = (
        RESULTS / "predictions" / split
    )

    figures_dir = RESULTS / "figures" / split

    for folder in [
        metrics_dir,
        predictions_dir,
        figures_dir
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    metrics = calculate_metrics(
        predictions
    )

    metrics.update({

        "experiment": EXPERIMENT,

        "model": "random_forest",

        "mode": "frozen_resnet18_embeddings",

        "split": split
    })

    predictions.to_csv(

        predictions_dir /
        f"{EXPERIMENT}_predictions.csv",

        index=False
    )

    predictions[
        ~predictions["correct"]
    ].to_csv(

        predictions_dir /
        f"{EXPERIMENT}_errors.csv",

        index=False
    )

    with open(

        metrics_dir /
        f"{EXPERIMENT}_metrics.json",

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
        f"{EXPERIMENT}_report.txt",

        "w",

        encoding="utf-8"

    ) as file:

        file.write(report)

    plot_confusion_matrix(

        predictions,

        EXPERIMENT,

        split,

        figures_dir
    )

    print(f"\n=== RANDOM FOREST: {split} ===")

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

    return metrics


# ==========================================
# 6. COMPARACIÓN CON LOS OTROS MODELOS
# ==========================================

def update_comparison(metrics, split):

    metrics_dir = RESULTS / "metrics" / split

    figures_dir = RESULTS / "figures" / split

    comparison_path = (

        metrics_dir /
        "model_comparison_with_ensemble.csv"
    )

    comparison = pd.read_csv(
        comparison_path
    )

    comparison = comparison[
        comparison["experiment"] != EXPERIMENT
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
        "model_comparison_all.csv",

        index=False
    )

    # Carpeta separada para conservar
    # los gráficos anteriores.

    comparison_figures = (

        figures_dir /
        "all_models"
    )

    comparison_figures.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_comparison(

        comparison,

        comparison_figures,

        split
    )

    all_predictions = {}

    for experiment in comparison["experiment"]:

        path = (

            RESULTS /
            "predictions" /
            split /
            f"{experiment}_predictions.csv"
        )

        all_predictions[experiment] = (
            pd.read_csv(path)
        )

    plot_roc_curves(

        all_predictions,

        comparison_figures,

        split
    )

    print(f"\n=== COMPARACIÓN FINAL: {split} ===")

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


# ==========================================
# 7. EJECUCIÓN PRINCIPAL
# ==========================================

def main():

    print(
        "\n=== HERITAGE GUARDIAN RANDOM FOREST ==="
    )

    device = get_device()

    print("Dispositivo:", device)

    checkpoint_dir = (
        CHECKPOINTS / "classical"
    )

    embedding_dir = (
        RESULTS / "embeddings"
    )

    for folder in [
        checkpoint_dir,
        embedding_dir
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    print("\nCargando ResNet18 preentrenada...")

    extractor = create_feature_extractor(
        device
    )

    # Extraer características de train
    # sin augmentation aleatorio.

    start = time.perf_counter()

    X_train, y_train, train_paths = (
        extract_embeddings(
            extractor,
            "train",
            device
        )
    )

    extraction_seconds = (
        time.perf_counter() - start
    )

    np.savez_compressed(

        embedding_dir / "train_embeddings.npz",

        X=X_train,

        y=y_train,

        paths=np.asarray(train_paths)
    )

    # Random Forest solamente aprende
    # utilizando el conjunto de entrenamiento.

    classifier, training_seconds = (
        train_random_forest(
            X_train,
            y_train
        )
    )

    checkpoint_path = (

        checkpoint_dir /
        f"{EXPERIMENT}.joblib"
    )

    joblib.dump(

        classifier,

        checkpoint_path
    )

    config = {

        "experiment": EXPERIMENT,

        "feature_extractor": "resnet18",

        "pretrained": True,

        "fine_tuning": False,

        "embedding_dimension": 512,

        "class_to_idx": {
            name: i
            for i, name in enumerate(CLASSES)
        },

        "random_forest": RF_CONFIG,

        "train_feature_extraction_seconds":
            extraction_seconds,

        "rf_training_seconds":
            training_seconds
    }

    with open(

        RESULTS / "metrics" /
        f"{EXPERIMENT}_config.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            config,

            file,

            indent=4
        )

    # Evaluar sin actualizar los parámetros.

    for split in ["val", "test"]:

        X, y, paths = extract_embeddings(

            extractor,

            split,

            device
        )

        np.savez_compressed(

            embedding_dir /
            f"{split}_embeddings.npz",

            X=X,

            y=y,

            paths=np.asarray(paths)
        )

        predictions = predict_random_forest(

            classifier,

            X,

            y,

            paths
        )

        metrics = save_results(

            predictions,

            split
        )

        update_comparison(

            metrics,

            split
        )

    print("\nRandom Forest completado.")

    print("Clasificador:", checkpoint_path)

    print("Embeddings:", embedding_dir)


if __name__ == "__main__":

    main()
