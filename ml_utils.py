from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from ml_config import (
    CLASS_NAMES,
    DATA_DIR,
    IMAGE_SIZE,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    SOURCE_DATASET,
    STATIC_DIR,
    TEMPLATE_DIR,
    UPLOAD_DIR,
)


def ensure_directories() -> None:
    for directory in (DATA_DIR, MODEL_DIR, STATIC_DIR, TEMPLATE_DIR, UPLOAD_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "valid", "test"):
        for class_name in CLASS_NAMES:
            (DATA_DIR / split_name / class_name).mkdir(parents=True, exist_ok=True)


def preprocess_image(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)

    return np.expand_dims(image_array, axis=0)


def load_trained_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. Run 'python train.py' first."
        )

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def human_label_from_score(score: float) -> str:
    return "Cancer" if score >= 0.5 else "Benign"


def confidence_from_score(score: float) -> float:
    return score if score >= 0.5 else 1.0 - score


def save_model_metadata(test_loss: float, test_accuracy: float) -> None:
    metadata = {
        "model_path": str(MODEL_PATH),
        "image_size": list(IMAGE_SIZE),
        "class_names": list(CLASS_NAMES),
        "source_dataset": SOURCE_DATASET,
        "test_loss": round(float(test_loss), 6),
        "test_accuracy": round(float(test_accuracy), 6),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
