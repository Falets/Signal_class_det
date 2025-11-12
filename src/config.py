import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# грузим .env из корня проекта, если есть
load_dotenv(BASE_DIR / ".env")

CSV_PATH = os.getenv("CSV_PATH", str(BASE_DIR / "data.xlsx"))
MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "50"))
LR = float(os.getenv("LR", "0.001"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.0001"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))  # под винду лучше 0 или 1
DEVICE = os.getenv("DEVICE", "cpu")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
