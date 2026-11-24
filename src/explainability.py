
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn.functional as F

from src.config import (
    DATA,
    LOCAL_TEST,
    RESULTS,
    CLASSES,
    SEED,
    IMAGE_SIZE
)

from src.dataset import get_transforms
from src.models import get_device
from src.evaluate import load_checkpoint


RESNET_NAME = "resnet18_balanced_seed42"
VIT_NAME = "vit_small_improved_seed42"


# ==========================================
# 1. NORMALIZACIÓN DE MAPAS
# ==========================================

def normalize_map(heatmap):

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    if not np.isfinite(heatmap).all():
        raise RuntimeError(
            "Mapa con valores no finitos."
        )

    heatmap = heatmap - heatmap.min()

    maximum = heatmap.max()

    if maximum > 1e-8:
        heatmap = heatmap / maximum

    return heatmap


# ==========================================
# 2. GRAD-CAM PARA RESNET18
# ==========================================

def grad_cam(model, image_tensor):

    activations = {}

    target_layer = model.layer4[-1]

    def capture_activation(
        module,
        inputs,
        output
    ):

        activations["value"] = output

    handle = target_layer.register_forward_hook(
        capture_activation
    )

    try:

        model.eval()

        with torch.enable_grad():

            model.zero_grad(set_to_none=True)

            x = image_tensor.detach().requires_grad_(True)

            logits = model(x)

            probabilities = torch.softmax(
                logits.detach(),
                dim=1
            )

            predicted_class = int(
                probabilities.argmax(dim=1).item()
            )

            target_score = logits[
                0,
                predicted_class
            ]

            feature_maps = activations["value"]

            gradients = torch.autograd.grad(
                target_score,
                feature_maps
            )[0]

            # Promedio espacial de los gradientes

            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            # Combinación ponderada de activaciones

            cam = (
                weights * feature_maps
            ).sum(dim=1, keepdim=True)

            cam = F.relu(cam)

            cam = F.interpolate(
                cam,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )

            cam = cam.squeeze().detach().cpu().numpy()

        cam = normalize_map(cam)

        return (
            cam,
            predicted_class,
            probabilities[0].cpu().numpy()
        )

    finally:

        handle.remove()


# ==========================================
# 3. ATTENTION ROLLOUT PARA VIT
# ==========================================

def attention_rollout(model, image_tensor):

    attentions = []
    handles = []
    previous_settings = []

    def capture_attention(
        module,
        inputs,
        output
    ):

        attentions.append(
            output.detach().float().cpu()
        )

    try:

        model.eval()

        # Capturar atención explícita
        # en cada bloque Transformer.

        for block in model.blocks:

            attention = block.attn

            if hasattr(attention, "fused_attn"):

                previous_settings.append(
                    (attention, attention.fused_attn)
                )

                attention.fused_attn = False

            handle = (
                attention.attn_drop
                .register_forward_hook(
                    capture_attention
                )
            )

            handles.append(handle)

        with torch.no_grad():

            logits = model(image_tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

        predicted_class = int(
            probabilities.argmax(dim=1).item()
        )

        if len(attentions) != len(model.blocks):

            raise RuntimeError(
                "No se capturaron todas "
                "las matrices de atención."
            )

        num_tokens = attentions[0].shape[-1]

        identity = torch.eye(num_tokens)

        rollout = identity.clone()

        for attention in attentions:

            # attention:
            # batch, heads, tokens, tokens

            attention = attention[0].mean(dim=0)

            # Incorporar conexión residual

            attention = attention + identity

            # Normalizar las filas

            attention = attention / (
                attention.sum(
                    dim=-1,
                    keepdim=True
                )
            )

            # Propagación entre capas

            rollout = attention @ rollout

        # Atención desde CLS hacia los parches

        prefix_tokens = model.num_prefix_tokens

        mask = rollout[0, prefix_tokens:]

        grid_h, grid_w = model.patch_embed.grid_size

        if mask.numel() != grid_h * grid_w:

            raise RuntimeError(
                "La cantidad de parches no coincide."
            )

        mask = mask.reshape(
            1, 1, grid_h, grid_w
        )

        mask = F.interpolate(
            mask,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        mask = mask.squeeze().cpu().numpy()

        mask = normalize_map(mask)

        return (
            mask,
            predicted_class,
            probabilities[0].cpu().numpy()
        )

    finally:

        # Retirar los hooks.

        for handle in handles:
            handle.remove()

        # Restaurar configuración original.

        for attention, setting in previous_settings:
            attention.fused_attn = setting


# ==========================================
# 4. PREPARAR IMAGEN
# ==========================================

def prepare_image(image_path, device):

    image = Image.open(
        image_path
    ).convert("RGB")

    _, eval_transform = get_transforms(
        "baseline"
    )

    tensor = eval_transform(
        image
    ).unsqueeze(0).to(device)

    display_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    display_image = np.asarray(
        display_image
    ).astype(np.float32) / 255.0

    return tensor, display_image


# ==========================================
# 5. SUPERPONER MAPA SOBRE IMAGEN
# ==========================================

def overlay_heatmap(
    image,
    heatmap,
    alpha=0.45
):

    colored_heatmap = plt.get_cmap(
        "jet"
    )(heatmap)[..., :3]

    overlay = (
        (1 - alpha) * image
        + alpha * colored_heatmap
    )

    return np.clip(
        overlay,
        0,
        1
    )


# ==========================================
# 6. EXPLICAR UNA IMAGEN
# ==========================================

def explain_image(
    image_path,
    resnet,
    vit,
    device
):

    tensor, original = prepare_image(
        image_path,
        device
    )

    resnet_map, resnet_pred, resnet_probs = (
        grad_cam(
            resnet,
            tensor
        )
    )

    vit_map, vit_pred, vit_probs = (
        attention_rollout(
            vit,
            tensor
        )
    )

    resnet_overlay = overlay_heatmap(
        original,
        resnet_map
    )

    vit_overlay = overlay_heatmap(
        original,
        vit_map
    )

    return {

        "original": original,

        "resnet_map": resnet_map,

        "vit_map": vit_map,

        "resnet_overlay": resnet_overlay,

        "vit_overlay": vit_overlay,

        "resnet_pred": resnet_pred,

        "vit_pred": vit_pred,

        "resnet_probs": resnet_probs,

        "vit_probs": vit_probs
    }


# ==========================================
# 7. GUARDAR FIGURA COMPARATIVA
# ==========================================

def save_explanation(
    explanation,
    output_path,
    true_class,
    category
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 5)
    )

    resnet_pred = explanation["resnet_pred"]

    vit_pred = explanation["vit_pred"]

    resnet_conf = explanation[
        "resnet_probs"
    ][resnet_pred]

    vit_conf = explanation[
        "vit_probs"
    ][vit_pred]

    axes[0].imshow(
        explanation["original"]
    )

    axes[0].set_title(
        f"Original\nReal: {true_class}"
    )

    axes[1].imshow(
        explanation["resnet_overlay"]
    )

    axes[1].set_title(
        f"ResNet18 - Grad-CAM\n"
        f"{CLASSES[resnet_pred]} "
        f"({resnet_conf:.1%})"
    )

    axes[2].imshow(
        explanation["vit_overlay"]
    )

    axes[2].set_title(
        f"ViT - Attention Rollout\n"
        f"{CLASSES[vit_pred]} "
        f"({vit_conf:.1%})"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        category.replace("_", " "),
        fontsize=13
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ==========================================
# 8. SELECCIONAR EJEMPLOS
# ==========================================

def select_examples(
    split,
    examples_per_category
):

    predictions_dir = (
        RESULTS / "predictions" / split
    )

    resnet = pd.read_csv(
        predictions_dir /
        f"{RESNET_NAME}_predictions.csv"
    )

    vit = pd.read_csv(
        predictions_dir /
        f"{VIT_NAME}_predictions.csv"
    )

    merged = resnet[
        ["image", "y_true", "y_pred"]
    ].merge(

        vit[
            ["image", "y_true", "y_pred"]
        ],

        on="image",

        suffixes=("_resnet", "_vit"),

        validate="one_to_one"
    )

    if len(merged) != len(resnet) or len(merged) != len(vit):

        raise ValueError(
            "Las imágenes no coinciden."
        )

    if not (
        merged["y_true_resnet"]
        == merged["y_true_vit"]
    ).all():

        raise ValueError(
            "Las etiquetas no coinciden."
        )

    merged["y_true"] = merged[
        "y_true_resnet"
    ]

    resnet_correct = (
        merged["y_true"]
        == merged["y_pred_resnet"]
    )

    vit_correct = (
        merged["y_true"]
        == merged["y_pred_vit"]
    )

    merged["category"] = np.select(

        [
            resnet_correct & vit_correct,

            resnet_correct & ~vit_correct,

            ~resnet_correct & vit_correct
        ],

        [
            "ambos_correctos",

            "solo_resnet_correcto",

            "solo_vit_correcto"
        ],

        default="ambos_incorrectos"
    )

    selected = []

    categories = [
        "ambos_correctos",
        "solo_resnet_correcto",
        "solo_vit_correcto",
        "ambos_incorrectos"
    ]

    # Muestreo reproducible por clase
    # y tipo de acierto/error.

    for category in categories:

        for class_id in range(len(CLASSES)):

            subset = merged[

                (merged["category"] == category)

                & (merged["y_true"] == class_id)
            ]

            if len(subset) == 0:
                continue

            sample = subset.sample(

                n=min(
                    examples_per_category,
                    len(subset)
                ),

                random_state=SEED
            )

            selected.append(sample)

    if not selected:

        raise RuntimeError(
            "No se encontraron ejemplos."
        )

    return pd.concat(
        selected,
        ignore_index=True
    )


# ==========================================
# 9. EJECUCIÓN PRINCIPAL
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["val", "test", "local"],
        default="val"
    )

    parser.add_argument(
        "--examples-per-category",
        type=int,
        default=2
    )

    args = parser.parse_args()

    if args.examples_per_category < 1:
        parser.error(
            "Debes seleccionar al menos un ejemplo."
        )

    device = get_device()

    print(
        "\n=== HERITAGE GUARDIAN EXPLAINABILITY ==="
    )

    print("Dispositivo:", device)

    print("Split:", args.split)

    root = (
        LOCAL_TEST
        if args.split == "local"
        else DATA / args.split
    )

    output_dir = (
        RESULTS /
        "figures" /
        "explainability" /
        args.split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

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

    examples = select_examples(

        args.split,

        args.examples_per_category
    )

    print(
        "\nEjemplos seleccionados:",
        len(examples)
    )

    records = []

    for index, row in examples.iterrows():

        image_path = root / row["image"]

        true_class = CLASSES[
            int(row["y_true"])
        ]

        category = row["category"]

        print(
            f"\n[{index + 1}/{len(examples)}] "
            f"{row['image']}"
        )

        explanation = explain_image(
            image_path,
            resnet,
            vit,
            device
        )

        category_dir = output_dir / category

        category_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        filename = (
            f"{index:02d}_{true_class}_"
            f"{Path(row['image']).stem}"
        )

        output_path = (
            category_dir / f"{filename}.png"
        )

        save_explanation(

            explanation,

            output_path,

            true_class,

            category
        )

        # Conservar los mapas originales.

        np.savez_compressed(

            category_dir / f"{filename}_maps.npz",

            grad_cam=explanation["resnet_map"],

            attention_rollout=explanation["vit_map"]
        )

        resnet_pred = explanation["resnet_pred"]
        vit_pred = explanation["vit_pred"]

        same_as_evaluation = (

            resnet_pred == row["y_pred_resnet"]

            and vit_pred == row["y_pred_vit"]
        )

        if not same_as_evaluation:
            print(
                "Aviso: la predicción difiere "
                "de la evaluación original."
            )

        print(
            "Real:",
            true_class
        )

        print(
            "ResNet:",
            CLASSES[resnet_pred]
        )

        print(
            "ViT:",
            CLASSES[vit_pred]
        )

        print(
            "Guardado:",
            output_path.name
        )

        records.append({

            "image": row["image"],

            "true_class": true_class,

            "category": category,

            "resnet_prediction": CLASSES[
                resnet_pred
            ],

            "vit_prediction": CLASSES[
                vit_pred
            ],

            "resnet_prob_mural": float(
                explanation["resnet_probs"][1]
            ),

            "vit_prob_mural": float(
                explanation["vit_probs"][1]
            ),

            "matches_original_evaluation":
                same_as_evaluation,

            "figure": output_path.relative_to(
                RESULTS
            ).as_posix()
        })

    pd.DataFrame(records).to_csv(

        output_dir / "explainability_index.csv",

        index=False
    )

    print("\nInterpretabilidad completada.")

    print("Resultados:", output_dir)


if __name__ == "__main__":

    main()
