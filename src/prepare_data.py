import argparse
import csv
import hashlib
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
import gdown
from PIL import Image
from .config import RAW, DATA, RESULTS, SEED, CLASSES

URLS = {'graffiti.zip': 'https://drive.google.com/uc?id=1ZuYCYdNjxWZABpG0QQDPzCUeWsmmq8PE', 'muralism.zip': 'https://drive.google.com/uc?id=10fccfg3XeHE4DcuONhxwgbpavIVrjpw3'}
EXT = {'.jpg', '.jpeg', '.png', '.webp'}
SPLITS = ['train', 'val', 'test']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    seen = defaultdict(list)
    for split in SPLITS:
        for cls in CLASSES:
            folder = DATA / split / cls
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXT)
            print(f'{split}/{cls}: {len(files)}')
            for p in files:
                try:
                    with Image.open(p) as im: im.verify()
                except Exception as e:
                    raise ValueError(f'Imagen inválida: {p}') from e
                seen[digest(p)].append((split, cls, p))
    leaks = [v for v in seen.values() if len({x[0] for x in v}) > 1]
    print(f'Grupos de duplicados exactos entre splits: {len(leaks)}')
    for group in leaks[:10]: print([str(x[2]) for x in group])
    if leaks: raise SystemExit('Resolver duplicados antes de entrenar; no se modificó el dataset.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if DATA.exists():
        print('Dataset existente: se conserva sin cambios.')
        audit(); return
    if args.audit_only: raise SystemExit('No existe el dataset estructurado.')
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        archive = RAW / name
        if not archive.exists():
            if not gdown.download(url, str(archive), quiet=False):
                raise RuntimeError(f'No se pudo descargar {name}')
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if not (RAW / member.filename).resolve().is_relative_to(RAW.resolve()):
                    raise ValueError('Ruta insegura en ZIP')
            z.extractall(RAW)
    graffiti, muralista = RAW / 'graffiti', RAW / 'muralista'
    if not graffiti.is_dir() or not muralista.is_dir():
        raise RuntimeError('Revisar estructura de ZIP: se esperan graffiti/ y muralista/')
    pools = {'grafiti': sorted(p for p in graffiti.rglob('*') if p.is_file() and p.suffix.lower() in EXT), 'mural': []}
    for p in sorted(muralista.iterdir()):
        if p.is_file() and p.suffix.lower() in EXT:
            pools['grafiti' if p.name.startswith('vandalico_') else 'mural'].append(p)
    seen = {}; clean = {c: [] for c in CLASSES}
    for cls in CLASSES:
        for p in pools[cls]:
            with Image.open(p) as im: im.verify()
            h = digest(p)
            if h in seen:
                if seen[h] != cls: raise ValueError(f'Etiquetas contradictorias: {p}')
                continue
            seen[h] = cls; clean[cls].append((p,h))
        if not clean[cls]: raise ValueError(f'Clase vacía: {cls}')
    rng = random.Random(SEED); rows = []
    for cls in ['mural','grafiti']:
        items = clean[cls]; rng.shuffle(items)
        n = len(items); a,b = int(.70*n),int(.85*n)
        for split, subset in zip(SPLITS,[items[:a],items[a:b],items[b:]]):
            folder = DATA / split / cls; folder.mkdir(parents=True, exist_ok=True)
            for i,(p,h) in enumerate(subset):
                dest = folder / f'{cls}_{i:05d}{p.suffix.lower()}'
                shutil.copy2(p,dest)
                rows.append([split,cls,str(p.relative_to(RAW)),str(dest.relative_to(DATA)),h])
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS/'dataset_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['split','class','source','destination','sha256']); w.writerows(rows)
    audit()

if __name__ == '__main__': main()
