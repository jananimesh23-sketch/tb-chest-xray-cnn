from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "tb_cnn_v2_augmented.keras"

IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 1
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

DECISION_THRESHOLD = 0.35
