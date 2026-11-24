
import csv
import hashlib
import shutil

from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image
from tqdm import tqdm

from src.config import ROOT, DATA, LOCAL_TEST, RESULTS


# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

RAW_LOCAL = ROOT / "data" / "valparaiso_raw"

METADATA = ROOT / "data" / "valparaiso_metadata.csv"

AMBIGUOUS = ROOT / "data" / "valparaiso_ambiguous"

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff"
}

VALID_LABELS = {
    "grafiti",
    "mural",
    "dudoso"
}


# ==========================================
# 2. HASH DE IMÁGENES
# ==========================================

def calculate_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(65536),
            b""
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


# ==========================================
# 3. VALIDAR IMÁGENES
# ==========================================

def validate_image(path):

    try:

        with Image.open(path) as image:

            image.verify()

        with Image.open(path) as image:

            image.load()

            width, height = image.size

        return True, width, height

    except Exception:

        return False, 0, 0


# ==========================================
# 4. LEER METADATA
# ==========================================

def load_metadata():

    if not METADATA.exists():

        raise FileNotFoundError(
            f"No existe: {METADATA}"
        )

    with open(
        METADATA,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        rows = list(csv.DictReader(file))

    if not rows:

        raise ValueError("El CSV está vacío.")

    required_columns = {
        "id",
        "filename_stem",
        "source",
        "label"
    }

    if not required_columns.issubset(rows[0]):

        raise ValueError(
            "Faltan columnas en el CSV."
        )

    ids = set()
    stems = set()

    for row in rows:

        row["label"] = row["label"].strip().lower()

        row["filename_stem"] = (
            row["filename_stem"].strip()
        )

        if row["label"] not in VALID_LABELS:

            raise ValueError(
                f"Etiqueta inválida: {row}"
            )

        if row["id"] in ids:

            raise ValueError(
                f"ID repetido: {row['id']}"
            )

        if row["filename_stem"].lower() in stems:

            raise ValueError(
                f"Nombre repetido: {row['filename_stem']}"
            )

        ids.add(row["id"])

        stems.add(
            row["filename_stem"].lower()
        )

    return rows


# ==========================================
# 5. INDEXAR FOTOGRAFÍAS LOCALES
# ==========================================

def index_local_images():

    if not RAW_LOCAL.exists():

        raise FileNotFoundError(
            f"No existe: {RAW_LOCAL}"
        )

    index = defaultdict(list)

    for path in RAW_LOCAL.iterdir():

        if (
            path.is_file()

            and path.suffix.lower()
            in VALID_EXTENSIONS
        ):

            index[path.stem.lower()].append(path)

    return index


# ==========================================
# 6. HASHES DEL DATASET ORIGINAL
# ==========================================

def index_original_dataset():

    hashes = {}

    print(
        "\nComprobando duplicados con "
        "el dataset original..."
    )

    for split in ["train", "val", "test"]:

        split_path = DATA / split

        if not split_path.is_dir():

            raise FileNotFoundError(
                f"No existe: {split_path}"
            )

        paths = [

            path

            for path in split_path.rglob("*")

            if path.is_file()

            and path.suffix.lower()
            in VALID_EXTENSIONS
        ]

        for path in tqdm(
            paths,
            desc=f"Hashes {split}"
        ):

            file_hash = calculate_hash(path)

            hashes[file_hash] = (
                path.relative_to(DATA).as_posix()
            )

    return hashes


# ==========================================
# 7. PREPARAR DATASET LOCAL
# ==========================================

def main():

    print(
        "\n=== HERITAGE GUARDIAN LOCAL DATASET ==="
    )

    metadata = load_metadata()

    local_index = index_local_images()

    original_hashes = index_original_dataset()

    expected_counts = Counter(
        row["label"]
        for row in metadata
    )

    print("\nRegistros CSV:", len(metadata))

    print(
        "Distribución esperada:",
        dict(expected_counts)
    )

    # Evitar mezclar resultados de ejecuciones
    # anteriores con el nuevo dataset.

    for folder in [LOCAL_TEST, AMBIGUOUS]:

        if folder.exists():

            existing_files = [

                path

                for path in folder.rglob("*")

                if path.is_file()
            ]

            if existing_files:

                raise RuntimeError(

                    f"La carpeta {folder} "
                    "ya contiene archivos. "
                    "Revisa su contenido antes "
                    "de continuar."
                )

    seen_hashes = {}

    records = []

    planned_copies = []

    expected_stems = {
        row["filename_stem"].lower()
        for row in metadata
    }

    # Validar antes de copiar.

    for row in metadata:

        stem = row["filename_stem"].lower()

        label = row["label"]

        record = dict(row)

        record.update({

            "filename": "",

            "width": "",

            "height": "",

            "sha256": "",

            "status": "",

            "destination": "",

            "validation_notes": ""
        })

        matches = local_index.get(stem, [])

        if len(matches) == 0:

            record["status"] = "missing"

            records.append(record)

            print(f"[FALTANTE] {stem}")

            continue

        if len(matches) > 1:

            record["status"] = "multiple_files"

            record["validation_notes"] = (
                "Más de un archivo con este nombre."
            )

            records.append(record)

            print(f"[NOMBRE REPETIDO] {stem}")

            continue

        source_path = matches[0]

        record["filename"] = source_path.name

        valid, width, height = validate_image(
            source_path
        )

        if not valid:

            record["status"] = "corrupted"

            records.append(record)

            print(
                f"[CORRUPTA] {source_path.name}"
            )

            continue

        record["width"] = width

        record["height"] = height

        file_hash = calculate_hash(source_path)

        record["sha256"] = file_hash

        # Duplicados con train, val o test.

        if file_hash in original_hashes:

            record["status"] = "duplicate_original"

            record["validation_notes"] = (
                original_hashes[file_hash]
            )

            records.append(record)

            print(
                f"[DUPLICADO ORIGINAL] "
                f"{source_path.name}"
            )

            continue

        # Duplicados dentro de Valparaíso.

        if file_hash in seen_hashes:

            previous = seen_hashes[file_hash]

            record["status"] = "duplicate_local"

            record["validation_notes"] = (
                previous["filename"]
            )

            if previous["label"] != label:

                record["status"] = (
                    "conflicting_duplicate"
                )

            records.append(record)

            print(
                f"[DUPLICADO LOCAL] "
                f"{source_path.name}"
            )

            continue

        seen_hashes[file_hash] = {

            "filename": source_path.name,

            "label": label
        }

        # Determinar destino.

        if label == "dudoso":

            destination = (
                AMBIGUOUS / source_path.name
            )

        else:

            destination = (

                LOCAL_TEST /

                label /

                source_path.name
            )

        record["status"] = "valid"

        record["destination"] = (
            destination.relative_to(ROOT).as_posix()
        )

        planned_copies.append(
            (source_path, destination)
        )

        records.append(record)

    # Detectar fotografías no registradas.

    unregistered = [

        path.name

        for stem, paths in local_index.items()

        if stem not in expected_stems

        for path in paths
    ]

    if unregistered:

        print("\nArchivos sin registro en CSV:")

        for filename in unregistered:

            print(filename)

    # Guardar informe de validación.

    report_dir = RESULTS / "metrics"

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (

        report_dir /

        "local_dataset_manifest.csv"
    )

    with open(

        report_path,

        "w",

        encoding="utf-8-sig",

        newline=""

    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=list(records[0].keys())
        )

        writer.writeheader()

        writer.writerows(records)

    status_counts = Counter(

        record["status"]

        for record in records
    )

    print("\n=== VALIDACIÓN ===")

    for status, count in status_counts.items():

        print(f"{status}: {count}")

    print(
        "Archivos sin registro:",
        len(unregistered)
    )

    # Detenerse si existe alguna inconsistencia.

    if (

        any(
            record["status"] != "valid"

            for record in records
        )

        or unregistered

    ):

        print(
            "\nSe encontraron inconsistencias."
        )

        print(
            "No se ha copiado ninguna imagen."
        )

        print(
            "Revisa el informe:", report_path
        )

        raise SystemExit(1)

    # Crear carpetas definitivas.

    for folder in [

        LOCAL_TEST / "grafiti",

        LOCAL_TEST / "mural",

        AMBIGUOUS
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    # Copiar, nunca mover los originales.

    for source, destination in tqdm(

        planned_copies,

        desc="Copiando imágenes"
    ):

        shutil.copy2(
            source,
            destination
        )

    counts = Counter(

        record["label"]

        for record in records

        if record["status"] == "valid"
    )

    print("\n=== DATASET LOCAL PREPARADO ===")

    print("Grafiti:", counts["grafiti"])

    print("Mural:", counts["mural"])

    print("Dudoso:", counts["dudoso"])

    print(

        "Total test binario:",

        counts["grafiti"] + counts["mural"]
    )

    print("Total imágenes:", sum(counts.values()))

    print("\nTest:", LOCAL_TEST)

    print("Casos ambiguos:", AMBIGUOUS)

    print("Manifest:", report_path)

    print("\nPreparación completada.")


if __name__ == "__main__":

    main()
