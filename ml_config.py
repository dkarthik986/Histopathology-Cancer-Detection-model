from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

MODEL_PATH = MODEL_DIR / "model.h5"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
DATASET_MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"

IMAGE_SIZE = (96, 96)
INPUT_SHAPE = IMAGE_SIZE + (3,)
CLASS_NAMES = ("benign", "cancer")
CLASS_LABELS = {
    False: CLASS_NAMES[0],
    True: CLASS_NAMES[1],
}

SEED = 42
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "5"))
TRAIN_SAMPLES_PER_CLASS = int(os.getenv("TRAIN_SAMPLES_PER_CLASS", "1000"))
VALID_SAMPLES_PER_CLASS = int(os.getenv("VALID_SAMPLES_PER_CLASS", "250"))
TEST_SAMPLES_PER_CLASS = int(os.getenv("TEST_SAMPLES_PER_CLASS", "250"))

SAMPLES_PER_SPLIT = {
    "train": TRAIN_SAMPLES_PER_CLASS,
    "valid": VALID_SAMPLES_PER_CLASS,
    "test": TEST_SAMPLES_PER_CLASS,
}

SOURCE_DATASET = os.getenv("SOURCE_DATASET", "1aurent/PatchCamelyon")
