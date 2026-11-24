
import argparse
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from tqdm import tqdm

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from src.config import (
    DATA,
    RESULTS,
    CHECKPOINTS,
    LOCAL_TEST,
    CLASSES
)

from src.dataset import get_transforms

from src.models import (
    build_model,
    get_device
)


# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

EXPERIMENTS = [

    ("resnet18", "baseline"),
    ("resnet18", "improved"),
    ("resnet18", "balanced"),

    ("vit_small", "baseline"),
    ("vit_small", "improved"),
    ("vit_small", "balanced")
]

SEED = 42

sns.set_theme(style="whitegrid")


def parse_args():

    parser = argparse.ArgumentParser(
        description="Heritage Guardian Evaluation"
    )

    parser.add_argument(
        "--split",
        choices=["val", "test", "local"],
        default="val"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0
    )

    return parser.parse_args()


# ==========================================
# 2. CARGAR CHECKPOINT
# ==========================================

def load_checkpoint(model_name, mode, device):

    experiment = (
        f"{model_name}_{mode}_seed{SEED}"
    )

    group = (
        "baseline"
        if mode == "baseline"
        else "final"
    )

    checkpoint_path = (
        CHECKPOINTS / group /
        f"{experiment}_best.pth"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"No existe: {checkpoint_path}"
        )

    checkpoint = torch.load(

        checkpoint_path,

        map_location="cpu",

        weights_only=True
    )

    expected_mapping = {
        name: index
        for index, name in enumerate(CLASSES)
    }

    if checkpoint["class_to_idx"] != expected_mapping:

        raise ValueError(
            "El orden de las clases no coincide."
        )

    if checkpoint["model_name"] != model_name:

        raise ValueError(
            "La arquitectura del checkpoint no coincide."
        )

    if checkpoint["mode"] != mode:

        raise ValueError(
            "La configuración del checkpoint no coincide."
        )

    model = build_model(

        model_name,

        pretrained=False,

        freeze_backbone=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, checkpoint


# ==========================================
# 3. PREPARAR DATASET DE EVALUACIÓN
# ==========================================

def create_eval_loader(
    split,
    batch_size,
    workers
):

    root = (
        LOCAL_TEST
        if split == "local"
        else DATA / split
    )

    _, eval_transform = get_transforms(
        "baseline"
    )

    dataset = ImageFolder(

        root,

        transform=eval_transform
    )

    expected_mapping = {
        name: index
        for index, name in enumerate(CLASSES)
    }

    if dataset.class_to_idx != expected_mapping:

        raise ValueError(
            "Las clases del dataset no coinciden."
        )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=workers,

        pin_memory=torch.cuda.is_available()
    )

    return loader, dataset, root


# ==========================================
# 4. GENERAR PREDICCIONES
# ==========================================

def predict(
    model,
    loader,
    dataset,
    root,
    device
):

    probabilities = []
    labels = []

    if device.type == "cuda":

        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        for images, targets in tqdm(

            loader,

            desc="Evaluando",

            unit="batch"
        ):

            images = images.to(

                device,

                non_blocking=True
            )

            # Inferencia FP32 para estabilidad
            logits = model(images)

            probs = torch.softmax(

                logits,

                dim=1
            )

            if not torch.isfinite(probs).all():

                raise RuntimeError(
                    "Predicciones NaN o infinitas."
                )

            probabilities.append(
                probs.cpu().numpy()
            )

            labels.extend(
                targets.numpy().tolist()
            )

    if device.type == "cuda":

        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    probabilities = np.concatenate(

        probabilities,

        axis=0
    )

    y_true = np.array(labels)

    y_pred = probabilities.argmax(
        axis=1
    )

    image_paths = [

        str(
            __import__("pathlib").Path(path)
            .relative_to(root)
            .as_posix()
        )

        for path, _ in dataset.samples
    ]

    predictions = pd.DataFrame({

        "image": image_paths,

        "y_true": y_true,

        "y_pred": y_pred,

        "true_class": [
            CLASSES[i] for i in y_true
        ],

        "predicted_class": [
            CLASSES[i] for i in y_pred
        ],

        "prob_grafiti": probabilities[:, 0],

        "prob_mural": probabilities[:, 1],

        "confidence": probabilities.max(axis=1),

        "correct": y_true == y_pred
    })

    return predictions, elapsed


# ==========================================
# 5. CALCULAR MÉTRICAS
# ==========================================

def calculate_metrics(predictions):

    y_true = predictions["y_true"]

    y_pred = predictions["y_pred"]

    prob_mural = predictions["prob_mural"]

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {

        "accuracy": accuracy_score(
            y_true, y_pred
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),

        "macro_f1": f1_score(

            y_true,

            y_pred,

            labels=[0, 1],

            average="macro",

            zero_division=0
        ),

        "f1_grafiti": f1_score(

            y_true,

            y_pred,

            pos_label=0,

            zero_division=0
        ),

        "f1_mural": f1_score(

            y_true,

            y_pred,

            pos_label=1,

            zero_division=0
        ),

        "precision_mural": precision_score(

            y_true,

            y_pred,

            pos_label=1,

            zero_division=0
        ),

        "recall_mural": recall_score(

            y_true,

            y_pred,

            pos_label=1,

            zero_division=0
        ),

        "auc_roc": roc_auc_score(

            y_true,

            prob_mural
        ),

        "mcc": matthews_corrcoef(

            y_true,

            y_pred
        ),

        "specificity": (

            float(tn / (tn + fp))

            if tn + fp > 0

            else 0.0
        ),

        "grafitis_correctos": int(tn),

        "grafitis_como_mural": int(fp),

        "murales_como_grafiti": int(fn),

        "murales_correctos": int(tp)
    }

    return metrics


# ==========================================
# 6. MATRIZ DE CONFUSIÓN
# ==========================================

def plot_confusion_matrix(
    predictions,
    experiment,
    split,
    output_dir
):

    cm = confusion_matrix(

        predictions["y_true"],

        predictions["y_pred"],

        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        cbar=False,

        xticklabels=["Grafiti", "Mural"],

        yticklabels=["Grafiti", "Mural"]
    )

    plt.title(
        f"{experiment}\n{split} set"
    )

    plt.xlabel("Predicción")

    plt.ylabel("Clase real")

    plt.tight_layout()

    plt.savefig(

        output_dir /
        f"{experiment}_confusion_matrix.png",

        dpi=200
    )

    plt.close()


# ==========================================
# 7. CURVA ROC
# ==========================================

def plot_roc_curves(
    all_predictions,
    output_dir,
    split
):

    plt.figure(figsize=(9, 7))

    for experiment, predictions in (
        all_predictions.items()
    ):

        y_true = predictions["y_true"]

        probs = predictions["prob_mural"]

        fpr, tpr, _ = roc_curve(

            y_true,

            probs,

            pos_label=1
        )

        auc = roc_auc_score(

            y_true,

            probs
        )

        name = experiment.replace(
            "_seed42", ""
        )

        plt.plot(

            fpr,

            tpr,

            linewidth=2,

            label=f"{name} (AUC={auc:.3f})"
        )

    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--",

        color="gray",

        label="Clasificador aleatorio"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(f"Curvas ROC - {split}")

    plt.legend(loc="lower right")

    plt.tight_layout()

    plt.savefig(

        output_dir / "roc_comparison.png",

        dpi=200
    )

    plt.close()


# ==========================================
# 8. COMPARACIÓN GENERAL
# ==========================================

def plot_comparison(
    metrics_df,
    output_dir,
    split
):

    selected_metrics = [

        "accuracy",

        "macro_f1",

        "recall_mural",

        "auc_roc"
    ]

    plot_df = metrics_df.melt(

        id_vars=["experiment"],

        value_vars=selected_metrics,

        var_name="metric",

        value_name="score"
    )

    plt.figure(figsize=(12, 8))

    sns.barplot(

        data=plot_df,

        y="experiment",

        x="score",

        hue="metric"
    )

    plt.xlim(0, 1)

    plt.title(
        f"Comparación de modelos - {split}"
    )

    plt.xlabel("Score")

    plt.ylabel("Modelo")

    plt.tight_layout()

    plt.savefig(

        output_dir /
        "model_comparison.png",

        dpi=200
    )

    plt.close()


# ==========================================
# 9. EVALUACIÓN COMPLETA
# ==========================================

def evaluate(args):

    device = get_device()

    print("\n=== HERITAGE GUARDIAN EVALUATION ===")

    print("Dispositivo:", device)

    print("Split:", args.split)

    loader, dataset, root = create_eval_loader(

        split=args.split,

        batch_size=args.batch_size,

        workers=args.workers
    )

    print("Cantidad de imágenes:", len(dataset))

    print("Clases:", dataset.class_to_idx)

    figures_dir = (
        RESULTS / "figures" / args.split
    )

    predictions_dir = (
        RESULTS / "predictions" / args.split
    )

    metrics_dir = (
        RESULTS / "metrics" / args.split
    )

    for folder in [

        figures_dir,

        predictions_dir,

        metrics_dir
    ]:

        folder.mkdir(

            parents=True,

            exist_ok=True
        )

    all_metrics = []

    all_predictions = {}

    for model_name, mode in EXPERIMENTS:

        experiment = (
            f"{model_name}_{mode}_seed{SEED}"
        )

        print(f"\n=== {experiment.upper()} ===")

        model, checkpoint = load_checkpoint(

            model_name,

            mode,

            device
        )

        predictions, elapsed = predict(

            model,

            loader,

            dataset,

            root,

            device
        )

        metrics = calculate_metrics(
            predictions
        )

        metrics.update({

            "experiment": experiment,

            "model": model_name,

            "mode": mode,

            "split": args.split,

            "checkpoint_epoch": checkpoint["epoch"],

            "inference_seconds": elapsed
        })

        all_metrics.append(metrics)

        all_predictions[experiment] = predictions

        predictions.to_csv(

            predictions_dir /
            f"{experiment}_predictions.csv",

            index=False
        )

        errors = predictions[
            ~predictions["correct"]
        ]

        errors.to_csv(

            predictions_dir /
            f"{experiment}_errors.csv",

            index=False
        )

        with open(

            metrics_dir /
            f"{experiment}_metrics.json",

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
            f"{experiment}_report.txt",

            "w",

            encoding="utf-8"
        ) as file:

            file.write(report)

        plot_confusion_matrix(

            predictions,

            experiment,

            args.split,

            figures_dir
        )

        print(report)

        print(
            f"Accuracy: {metrics['accuracy']:.4f}"
        )

        print(
            f"Macro F1: {metrics['macro_f1']:.4f}"
        )

        print(
            f"Recall mural: {metrics['recall_mural']:.4f}"
        )

        print(
            f"AUC-ROC: {metrics['auc_roc']:.4f}"
        )

        print(
            f"MCC: {metrics['mcc']:.4f}"
        )

        print(
            f"Murales como grafiti: "
            f"{metrics['murales_como_grafiti']}"
        )

        del model, checkpoint

        if device.type == "cuda":

            torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(
        all_metrics
    )

    metrics_df = metrics_df.sort_values(

        "macro_f1",

        ascending=False
    )

    metrics_df.to_csv(

        metrics_dir /
        "model_comparison.csv",

        index=False
    )

    plot_roc_curves(

        all_predictions,

        figures_dir,

        args.split
    )

    plot_comparison(

        metrics_df,

        figures_dir,

        args.split
    )

    print("\n=== COMPARACIÓN FINAL ===")

    print(

        metrics_df[[

            "experiment",

            "accuracy",

            "macro_f1",

            "recall_mural",

            "auc_roc",

            "mcc"

        ]].round(4).to_string(index=False)
    )

    print("\nEvaluación completada.")

    print("Métricas:", metrics_dir)

    print("Predicciones:", predictions_dir)

    print("Gráficos:", figures_dir)


if __name__ == "__main__":

    args = parse_args()

    evaluate(args)
