from __future__ import annotations

import json

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers

from ml_config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    EPOCHS,
    INPUT_SHAPE,
    MODEL_PATH,
    SEED,
)
from ml_utils import ensure_directories, save_model_metadata
from prepare_data import prepare_dataset


def list_split_files(split_name: str) -> tuple[list[str], list[float]]:
    split_dir = DATA_DIR / split_name
    file_paths: list[str] = []
    labels: list[float] = []

    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        for image_path in sorted(class_dir.glob("*.png")):
            file_paths.append(str(image_path))
            labels.append(float(label_index))

    if not file_paths:
        raise FileNotFoundError(f"No training images found in '{split_dir}'.")

    return file_paths, labels


def load_image(image_path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_png(image_bytes, channels=3)
    image = tf.image.resize(image, INPUT_SHAPE[:2])
    image = tf.cast(image, tf.float32)
    label = tf.reshape(tf.cast(label, tf.float32), (1,))
    return image, label


def build_dataset(split_name: str, shuffle: bool) -> tf.data.Dataset:
    file_paths, labels = list_split_files(split_name)
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(file_paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model() -> tf.keras.Model:
    model = Sequential(
        [
            layers.Input(shape=INPUT_SHAPE),
            layers.Rescaling(1.0 / 255.0),
            layers.RandomFlip("horizontal"),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def main() -> None:
    ensure_directories()
    tf.keras.utils.set_random_seed(SEED)

    split_counts = prepare_dataset()
    print("Prepared dataset counts:")
    print(json.dumps(split_counts, indent=2))

    train_ds = build_dataset("train", shuffle=True)
    valid_ds = build_dataset("valid", shuffle=False)
    test_ds = build_dataset("test", shuffle=False)

    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_accuracy, test_auc = model.evaluate(test_ds, verbose=2)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    save_model_metadata(test_loss=test_loss, test_accuracy=test_accuracy)

    print(
        json.dumps(
            {
                "model_path": str(MODEL_PATH),
                "test_loss": round(float(test_loss), 6),
                "test_accuracy": round(float(test_accuracy), 6),
                "test_auc": round(float(test_auc), 6),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
