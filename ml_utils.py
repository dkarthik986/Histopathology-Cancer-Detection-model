from __future__ import annotations

import io
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


def preprocess_image(image_source: str | Path | bytes | io.BufferedIOBase) -> np.ndarray:
    if isinstance(image_source, bytes):
        image_source = io.BytesIO(image_source)

    if hasattr(image_source, "seek"):
        image_source.seek(0)

    with Image.open(image_source) as image:
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


def predict_score(model: tf.keras.Model, image_batch: np.ndarray) -> float:
    predictions = model(image_batch, training=False).numpy()
    return float(predictions.reshape(-1)[0])


def warm_up_model(model: tf.keras.Model) -> None:
    dummy_batch = np.zeros((1,) + IMAGE_SIZE + (3,), dtype=np.float32)
    _ = predict_score(model, dummy_batch)


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
