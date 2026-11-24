from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
DATA = ROOT / 'data' / 'dataset_estructurado'
LOCAL_TEST = ROOT / 'data' / 'test_valparaiso'
RESULTS = ROOT / 'results'
CHECKPOINTS = ROOT / 'checkpoints'
SEED = 42
CLASSES = ['grafiti', 'mural']
IMAGE_SIZE = 224
