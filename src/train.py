
import argparse
import json
import time

import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)

from src.config import (
    CHECKPOINTS,
    RESULTS,
    CLASSES,
    SEED
)

from src.dataset import create_dataloaders

from src.models import (
    build_model,
    get_device,
    count_parameters
)


# ==========================================
# 1. CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Heritage Guardian Training"
    )

    parser.add_argument(
        "--model",
        choices=["resnet18", "vit_small"],
        required=True
    )

    parser.add_argument(
        "--mode",
        choices=["baseline", "improved", "balanced"],
        default="improved"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    parser.add_argument(
        "--no-amp",
        action="store_true"
    )

    return parser.parse_args()


def get_experiment_config(args):

    baseline = args.mode == "baseline"
    is_vit = args.model == "vit_small"

    config = {

        "model": args.model,

        "mode": args.mode,

        "epochs": 5 if baseline else 25,

        "batch_size": 16 if is_vit else 32,

        "lr": (
            (1e-4 if is_vit else 1e-3)
            if baseline
            else
            (3e-5 if is_vit else 1e-4)
        ),

        "weight_decay": (
            0.0 if baseline
            else (0.01 if is_vit else 1e-4)
        ),

        "label_smoothing": (
            0.0 if baseline
            else (0.1 if is_vit else 0.05)
        ),

        "optimizer": "Adam" if baseline else "AdamW",

        "scheduler": not baseline,

        "early_stopping": not baseline,

        "selection_metric": (
            "accuracy" if baseline else "macro_f1"
        ),

        "patience": args.patience,

        "workers": args.workers,

        "seed": args.seed,

        "amp": (
            torch.cuda.is_available()
            and not args.no_amp
            and (is_vit or not baseline)
        )
    }

    if args.epochs is not None:
        config["epochs"] = args.epochs

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    if args.lr is not None:
        config["lr"] = args.lr

    return config


# ==========================================
# 2. MÉTRICAS
# ==========================================

def calculate_metrics(
    y_true,
    y_pred,
    total_loss,
    total_samples
):

    return {

        "loss": total_loss / total_samples,

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0
        ),

        "f1_mural": f1_score(
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
        )
    }


# ==========================================
# 3. EJECUCIÓN DE UNA ÉPOCA
# ==========================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    use_amp=False
):

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_samples = 0

    y_true = []
    y_pred = []

    progress_bar = tqdm(
    loader,
    desc="Train" if training else "Validation",
    leave=False,
    unit="batch"
)

    for images, labels in progress_bar:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    labels
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Se detectó una pérdida NaN o infinita."
                )

            if training:

                scaler.scale(loss).backward()

                scaler.step(optimizer)

                scaler.update()

        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{total_loss / total_samples:.4f}"
        )

        y_true.extend(
            labels.detach().cpu().tolist()
        )

        y_pred.extend(
            predictions.detach().cpu().tolist()
        )

    return calculate_metrics(
        y_true,
        y_pred,
        total_loss,
        total_samples
    )


# ==========================================
# 4. ENTRENAMIENTO COMPLETO
# ==========================================

def train(config):

    device = get_device()

    experiment_name = (
        f"{config['model']}_"
        f"{config['mode']}_"
        f"seed{config['seed']}"
    )

    checkpoint_group = (
        "baseline"
        if config["mode"] == "baseline"
        else "final"
    )

    checkpoint_dir = (
        CHECKPOINTS / checkpoint_group
    )

    history_dir = RESULTS / "histories"

    metrics_dir = RESULTS / "metrics"

    for folder in [
        checkpoint_dir,
        history_dir,
        metrics_dir
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    checkpoint_path = (
        checkpoint_dir /
        f"{experiment_name}_best.pth"
    )

    history_path = (
        history_dir /
        f"{experiment_name}.csv"
    )

    config_path = (
        metrics_dir /
        f"{experiment_name}_config.json"
    )

    summary_path = (
        metrics_dir /
        f"{experiment_name}_summary.json"
    )

    existing_files = [
        path for path in [
            checkpoint_path,
            history_path,
            config_path,
            summary_path
        ]
        if path.exists()
    ]

    if existing_files:

        raise FileExistsError(
            "El experimento ya tiene archivos. "
            "Muévelos antes de repetirlo:\n"
            + "\n".join(map(str, existing_files))
        )

    print("\n=== HERITAGE GUARDIAN TRAINING ===")

    print(
        json.dumps(
            config,
            indent=4
        )
    )

    print("\nDispositivo:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        torch.cuda.reset_peak_memory_stats()

    print("\nPreparando DataLoaders...")

    loaders, datasets = create_dataloaders(

        batch_size=config["batch_size"],

        num_workers=config["workers"],

        mode=config["mode"],

        seed=config["seed"]
    )

    print("\nCargando modelo preentrenado...")

    model = build_model(

        config["model"],

        pretrained=True,

        freeze_backbone=False

    ).to(device)

    total, trainable = count_parameters(model)

    print(f"Parámetros totales: {total:,}")

    print(
        f"Parámetros entrenables: {trainable:,}"
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config["label_smoothing"]
    )

    optimizer_class = (
        torch.optim.Adam
        if config["optimizer"] == "Adam"
        else torch.optim.AdamW
    )

    optimizer = optimizer_class(

        model.parameters(),

        lr=config["lr"],

        weight_decay=config["weight_decay"]
    )

    scheduler = None

    if config["scheduler"]:

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

            optimizer,

            T_max=config["epochs"],

            eta_min=config["lr"] * 0.01
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config["amp"]
    )

    config["classes"] = CLASSES

    config["class_to_idx"] = (
        datasets["train"].class_to_idx
    )

    config["device"] = str(device)

    config["torch_version"] = str(
        torch.__version__
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    best_score = float("-inf")

    best_epoch = 0

    best_metrics = {}

    epochs_without_improvement = 0

    history = []

    start_time = time.perf_counter()

    # ======================================
    # BUCLE PRINCIPAL DE ENTRENAMIENTO
    # ======================================

    for epoch in range(
        1,
        config["epochs"] + 1
    ):

        epoch_start = time.perf_counter()

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        train_metrics = run_epoch(

            model=model,

            loader=loaders["train"],

            criterion=criterion,

            device=device,

            optimizer=optimizer,

            scaler=scaler,

            use_amp=config["amp"]
        )

        val_metrics = run_epoch(

            model=model,

            loader=loaders["val"],

            criterion=criterion,

            device=device,

            use_amp=config["amp"]
        )

        epoch_time = (
            time.perf_counter() - epoch_start
        )

        row = {

            "epoch": epoch,

            "lr": current_lr,

            "seconds": epoch_time
        }

        row.update({

            f"train_{key}": value

            for key, value
            in train_metrics.items()
        })

        row.update({

            f"val_{key}": value

            for key, value
            in val_metrics.items()
        })

        history.append(row)

        pd.DataFrame(history).to_csv(

            history_path,

            index=False
        )

        print(
            f"\nEpoch {epoch}/{config['epochs']} "
            f"| {epoch_time:.1f}s"
        )

        print(

            f"Train Loss: {train_metrics['loss']:.4f} "

            f"| Acc: {train_metrics['accuracy']:.4f} "

            f"| F1: {train_metrics['macro_f1']:.4f}"
        )

        print(

            f"Val Loss: {val_metrics['loss']:.4f} "

            f"| Acc: {val_metrics['accuracy']:.4f} "

            f"| F1: {val_metrics['macro_f1']:.4f}"
        )

        print(

            f"Recall mural: "
            f"{val_metrics['recall_mural']:.4f} "

            f"| LR: {current_lr:.7f}"
        )

        score = val_metrics[
            config["selection_metric"]
        ]

        if score > best_score:

            best_score = score

            best_epoch = epoch

            best_metrics = val_metrics.copy()

            epochs_without_improvement = 0

            checkpoint = {

                "model_name": config["model"],

                "mode": config["mode"],

                "epoch": epoch,

                "model_state_dict": (
                    model.state_dict()
                ),

                "class_to_idx": (
                    datasets["train"].class_to_idx
                ),

                "config": config,

                "val_metrics": val_metrics
            }

            temp_path = checkpoint_path.with_suffix(
                ".tmp"
            )

            torch.save(
                checkpoint,
                temp_path
            )

            temp_path.replace(checkpoint_path)

            print(
                ">> Nuevo mejor modelo guardado."
            )

        else:

            epochs_without_improvement += 1

            print(

                "Épocas sin mejora: "

                f"{epochs_without_improvement}"
            )

        if scheduler is not None:

            scheduler.step()

        if (

            config["early_stopping"]

            and

            epochs_without_improvement
            >= config["patience"]
        ):

            print("\nEarly stopping activado.")

            break

    elapsed = time.perf_counter() - start_time

    peak_vram = None

    if device.type == "cuda":

        peak_vram = (

            torch.cuda.max_memory_allocated()

            / 1024**3
        )

    summary = {

        "experiment": experiment_name,

        "model": config["model"],

        "mode": config["mode"],

        "best_epoch": best_epoch,

        "best_score": best_score,

        "selection_metric": (
            config["selection_metric"]
        ),

        "best_val_metrics": best_metrics,

        "epochs_completed": len(history),

        "training_seconds": elapsed,

        "peak_vram_gb": peak_vram,

        "checkpoint": str(checkpoint_path)
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n=== ENTRENAMIENTO COMPLETADO ===")

    print(
        f"Mejor época: {best_epoch}"
    )

    print(
        f"Mejor score: {best_score:.4f}"
    )

    print(
        f"Tiempo total: {elapsed / 60:.2f} minutos"
    )

    if peak_vram is not None:

        print(
            f"Pico de memoria asignada: "
            f"{peak_vram:.2f} GB"
        )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        f"Historial: {history_path}"
    )

    print(
        f"Resumen: {summary_path}"
    )

    return summary


# ==========================================
# 5. PUNTO DE ENTRADA
# ==========================================

def main():

    args = parse_args()

    config = get_experiment_config(args)

    train(config)


if __name__ == "__main__":

    main()
