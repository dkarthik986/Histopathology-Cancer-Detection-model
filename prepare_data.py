from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

from ml_config import (
    CLASS_LABELS,
    CLASS_NAMES,
    DATASET_MANIFEST_PATH,
    DATA_DIR,
    SAMPLES_PER_SPLIT,
    SOURCE_DATASET,
)
from ml_utils import ensure_directories


def count_saved_images(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.png"))


def current_split_counts(split_name: str) -> dict[str, int]:
    split_dir = DATA_DIR / split_name
    return {
        class_name: count_saved_images(split_dir / class_name)
        for class_name in CLASS_NAMES
    }


def export_split(split_name: str, per_class_target: int) -> dict[str, int]:
    split_dir = DATA_DIR / split_name
    counts = current_split_counts(split_name)

    if all(count >= per_class_target for count in counts.values()):
        print(f"{split_name}: using existing images {counts}")
        return counts

    print(
        f"{split_name}: downloading from '{SOURCE_DATASET}' until each class has "
        f"{per_class_target} images..."
    )
    dataset = load_dataset(SOURCE_DATASET, split=split_name, streaming=True)

    for item in dataset:
        label_name = CLASS_LABELS[bool(item["label"])]
        if counts[label_name] >= per_class_target:
            if all(count >= per_class_target for count in counts.values()):
                break
            continue

        image_index = counts[label_name]
        image_path = split_dir / label_name / f"{split_name}_{label_name}_{image_index:05d}.png"
        item["image"].convert("RGB").save(image_path)
        counts[label_name] += 1

        if all(count >= per_class_target for count in counts.values()):
            break

    if not all(count >= per_class_target for count in counts.values()):
        raise RuntimeError(
            f"Unable to prepare split '{split_name}'. Final counts: {counts}"
        )

    print(f"{split_name}: prepared {counts}")
    return counts


def prepare_dataset() -> dict[str, dict[str, int]]:
    ensure_directories()
    manifest = {
        "source_dataset": SOURCE_DATASET,
        "splits": {},
    }

    for split_name, per_class_target in SAMPLES_PER_SPLIT.items():
        manifest["splits"][split_name] = export_split(split_name, per_class_target)

    DATASET_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved dataset manifest to {DATASET_MANIFEST_PATH}")
    return manifest["splits"]


if __name__ == "__main__":
    prepare_dataset()
