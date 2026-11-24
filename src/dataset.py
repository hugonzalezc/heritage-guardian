
import random

import numpy as np
import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)

from src.config import (
    DATA,
    IMAGE_SIZE,
    SEED,
    CLASSES
)


# Normalización habitual de ImageNet

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transforms(mode="improved"):

    if mode == "baseline":

        train_transform = transforms.Compose([

            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize(MEAN, STD)
        ])

    elif mode in ["improved", "balanced"]:

        train_transform = transforms.Compose([

            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.85, 1.0),
                ratio=(0.85, 1.15)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(8),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.02
            ),

            transforms.ToTensor(),

            transforms.Normalize(MEAN, STD)
        ])

    else:

        raise ValueError(
            f"Modo desconocido: {mode}"
        )

    # Validación y test sin augmentation aleatorio

    eval_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(MEAN, STD)
    ])

    return train_transform, eval_transform


def create_sampler(dataset, generator=None):

    targets = torch.tensor(dataset.targets)

    class_counts = torch.bincount(targets)

    class_weights = 1.0 / class_counts.float()

    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(

        weights=sample_weights,

        num_samples=len(dataset),

        replacement=True,

        generator=generator
    )

    return sampler


def create_dataloaders(
    batch_size=32,
    num_workers=0,
    mode="improved",
    seed=SEED
):

    seed_everything(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_transform, eval_transform = (
        get_transforms(mode)
    )

    datasets = {

        "train": ImageFolder(
            DATA / "train",
            transform=train_transform
        ),

        "val": ImageFolder(
            DATA / "val",
            transform=eval_transform
        ),

        "test": ImageFolder(
            DATA / "test",
            transform=eval_transform
        )
    }

    expected_mapping = {
        name: idx
        for idx, name in enumerate(CLASSES)
    }

    for split, dataset in datasets.items():

        if dataset.class_to_idx != expected_mapping:

            raise ValueError(
                f"Etiquetas incorrectas en {split}: "
                f"{dataset.class_to_idx}"
            )

    sampler = None

    if mode == "balanced":

        sampler = create_sampler(
            datasets["train"],
            generator=generator
        )

    common_args = {

        "batch_size": batch_size,

        "num_workers": num_workers,

        "pin_memory": torch.cuda.is_available(),

        "worker_init_fn": seed_worker,

        "persistent_workers": num_workers > 0
    }

    loaders = {

        "train": DataLoader(

            datasets["train"],

            shuffle=sampler is None,

            sampler=sampler,

            generator=generator,

            **common_args
        ),

        "val": DataLoader(

            datasets["val"],

            shuffle=False,

            **common_args
        ),

        "test": DataLoader(

            datasets["test"],

            shuffle=False,

            **common_args
        )
    }

    return loaders, datasets


def main():

    print("\n=== DATASET HERITAGE GUARDIAN ===")

    print(
        "CUDA disponible:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    loaders, datasets = create_dataloaders(

        batch_size=32,

        num_workers=0,

        mode="balanced"
    )

    print(
        "\nClases:",
        datasets["train"].class_to_idx
    )

    for split in ["train", "val", "test"]:

        print(
            f"{split}: "
            f"{len(datasets[split])} imágenes | "
            f"{len(loaders[split])} batches"
        )

    images, labels = next(
        iter(loaders["train"])
    )

    print("\n=== PRIMER BATCH ===")

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Tipo de datos:", images.dtype)

    print(
        "Distribución del batch:",
        torch.bincount(labels, minlength=2).tolist()
    )

    # Comprobamos el oversampling durante
    # una época de muestreo sin cargar imágenes.

    sampled_indices = list(
        loaders["train"].sampler
    )

    sampled_labels = torch.tensor([

        datasets["train"].targets[index]

        for index in sampled_indices
    ])

    counts = torch.bincount(
        sampled_labels,
        minlength=2
    )

    print("\n=== OVERSAMPLING ===")

    print("Grafiti:", counts[0].item())
    print("Mural:", counts[1].item())

    print("\nDataLoaders preparados correctamente.")


if __name__ == "__main__":

    main()
