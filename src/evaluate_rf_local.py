
import joblib
import numpy as np

from src.config import CHECKPOINTS, RESULTS
from src.models import get_device

from src.random_forest import (
    EXPERIMENT,
    create_feature_extractor,
    extract_embeddings,
    predict_random_forest,
    save_results,
    update_comparison
)


def main():

    print(
        "\n=== HERITAGE GUARDIAN RANDOM FOREST LOCAL ==="
    )

    device = get_device()

    print("Dispositivo:", device)

    # ======================================
    # 1. CARGAR RANDOM FOREST ENTRENADO
    # ======================================

    checkpoint_path = (
        CHECKPOINTS /
        "classical" /
        f"{EXPERIMENT}.joblib"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"No existe: {checkpoint_path}"
        )

    print("\nCargando Random Forest entrenado...")

    classifier = joblib.load(checkpoint_path)

    # ======================================
    # 2. CARGAR EXTRACTOR PREENTRENADO
    # ======================================

    print("Cargando ResNet18 preentrenada...")

    extractor = create_feature_extractor(device)

    # ======================================
    # 3. EXTRAER EMBEDDINGS LOCALES
    # ======================================

    X, y, paths = extract_embeddings(
        extractor,
        "local",
        device
    )

    if X.shape[1] != classifier.n_features_in_:

        raise ValueError(
            "La dimensión de los embeddings "
            "no coincide con el clasificador."
        )

    embedding_dir = RESULTS / "embeddings"

    embedding_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(

        embedding_dir / "local_embeddings.npz",

        X=X,

        y=y,

        paths=np.asarray(paths)
    )

    # ======================================
    # 4. GENERAR PREDICCIONES
    # ======================================

    predictions = predict_random_forest(

        classifier,

        X,

        y,

        paths
    )

    # ======================================
    # 5. GUARDAR MÉTRICAS Y GRÁFICOS
    # ======================================

    metrics = save_results(
        predictions,
        "local"
    )

    update_comparison(
        metrics,
        "local"
    )

    print(
        "\nEvaluación local de Random Forest completada."
    )

    print(
        "Embeddings:",
        embedding_dir / "local_embeddings.npz"
    )

    print(
        "Métricas:",
        RESULTS / "metrics" / "local"
    )


if __name__ == "__main__":

    main()
