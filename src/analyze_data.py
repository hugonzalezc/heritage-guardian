
import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torchvision.datasets import ImageFolder

from src.config import DATA, RESULTS, SEED


random.seed(SEED)

FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"

FIGURES.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def analyze_dataset():

    records = []

    for split in ["train", "val", "test"]:

        dataset = ImageFolder(DATA / split)

        print(f"\nAnalizando {split}...")

        for path, label in dataset.samples:

            with Image.open(path) as img:

                width, height = img.size

                records.append({
                    "split": split,
                    "class": dataset.classes[label],
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height,
                    "path": str(path)
                })

    df = pd.DataFrame(records)

    df.to_csv(
        METRICS / "dataset_analysis.csv",
        index=False
    )

    return df


def plot_distribution(df):

    plt.figure(figsize=(9, 5))

    ax = sns.countplot(
        data=df,
        x="split",
        hue="class",
        order=["train", "val", "test"],
        palette={
            "grafiti": "#6366f1",
            "mural": "#0d9488"
        }
    )

    for container in ax.containers:
        ax.bar_label(container)

    plt.title("Distribución de clases por partición")
    plt.xlabel("Partición")
    plt.ylabel("Cantidad de imágenes")
    plt.legend(title="Clase")
    plt.tight_layout()

    plt.savefig(
        FIGURES / "class_distribution.png",
        dpi=200
    )

    plt.close()


def plot_resolutions(df):

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="width",
        y="height",
        hue="class",
        alpha=0.5
    )

    plt.title("Resoluciones originales del dataset")
    plt.xlabel("Ancho (px)")
    plt.ylabel("Alto (px)")
    plt.tight_layout()

    plt.savefig(
        FIGURES / "image_resolutions.png",
        dpi=200
    )

    plt.close()


def plot_examples(df):

    train_df = df[df["split"] == "train"]

    fig, axes = plt.subplots(
        2, 5, figsize=(16, 7)
    )

    for row, class_name in enumerate(
        ["grafiti", "mural"]
    ):

        samples = train_df[
            train_df["class"] == class_name
        ].sample(
            n=5,
            random_state=SEED
        )

        for col, (_, sample) in enumerate(
            samples.iterrows()
        ):

            with Image.open(sample["path"]) as img:

                axes[row, col].imshow(
                    img.convert("RGB")
                )

            axes[row, col].set_title(
                class_name.capitalize()
            )

            axes[row, col].axis("off")

    plt.suptitle(
        "Heritage Guardian - Ejemplos del dataset",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES / "dataset_examples.png",
        dpi=200
    )

    plt.close()


def main():

    df = analyze_dataset()

    print("\n=== DISTRIBUCIÓN DEL DATASET ===")

    print(
        pd.crosstab(
            df["split"],
            df["class"],
            margins=True
        )
    )

    print("\n=== RESOLUCIONES ===")

    print(
        df.groupby("class")[
            ["width", "height"]
        ].describe().round(2)
    )

    print("\n=== DISTRIBUCIÓN PORCENTUAL ===")

    print(
        df["class"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    plot_distribution(df)
    plot_resolutions(df)
    plot_examples(df)

    print("\nAnálisis completado.")
    print(f"Total de imágenes: {len(df)}")
    print(f"Gráficos guardados en: {FIGURES}")


if __name__ == "__main__":
    main()
