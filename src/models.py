
import torch
import torch.nn as nn
import timm

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from src.config import CLASSES, IMAGE_SIZE


NUM_CLASSES = len(CLASSES)

MODEL_NAMES = [
    "resnet18",
    "vit_small"
]


def build_resnet18(
    pretrained=True,
    freeze_backbone=False
):

    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = resnet18(weights=weights)

    if freeze_backbone:

        for param in model.parameters():
            param.requires_grad = False

    # Reemplazar clasificador original
    # de 1000 clases por nuestras 2 clases.

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model


def build_vit_small(
    pretrained=True,
    freeze_backbone=False
):

    model = timm.create_model(

        "vit_small_patch16_224",

        pretrained=pretrained,

        num_classes=NUM_CLASSES
    )

    if freeze_backbone:

        for param in model.parameters():
            param.requires_grad = False

        for param in model.head.parameters():
            param.requires_grad = True

    return model


def build_model(
    model_name,
    pretrained=True,
    freeze_backbone=False
):

    if model_name == "resnet18":

        return build_resnet18(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    elif model_name == "vit_small":

        return build_vit_small(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )

    else:

        raise ValueError(
            f"Modelo desconocido: {model_name}"
        )


def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def main():

    device = get_device()

    print("\n=== HERITAGE GUARDIAN ===")
    print("Dispositivo:", device)

    # Imagen ficticia para verificar
    # las dimensiones de los modelos.

    dummy_input = torch.randn(
        2, 3, IMAGE_SIZE, IMAGE_SIZE
    ).to(device)

    for model_name in MODEL_NAMES:

        print(f"\n=== {model_name.upper()} ===")

        # No descargamos pesos en esta prueba.
        # Solo comprobamos la arquitectura.

        model = build_model(
            model_name,
            pretrained=False
        ).to(device)

        model.eval()

        total, trainable = count_parameters(
            model
        )

        print(
            f"Parámetros totales: {total:,}"
        )

        print(
            f"Parámetros entrenables: "
            f"{trainable:,}"
        )

        with torch.inference_mode():

            logits = model(dummy_input)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

        print(
            "Input shape:",
            dummy_input.shape
        )

        print(
            "Output shape:",
            logits.shape
        )

        print(
            "Probabilidades de prueba:"
        )

        print(probabilities)

        print(
            "Suma de probabilidades:",
            probabilities.sum(dim=1)
        )

        assert logits.shape == (
            2, NUM_CLASSES
        ), "Dimensión de salida incorrecta"

        print("Modelo verificado correctamente.")

        del model

    print("\nAmbos modelos están preparados.")


if __name__ == "__main__":

    main()
