from __future__ import annotations

import argparse

from ml_utils import (
    confidence_from_score,
    human_label_from_score,
    load_trained_model,
    preprocess_image,
)

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to the image file")
args = parser.parse_args()

model = load_trained_model()
image = preprocess_image(args.image)
score = float(model.predict(image, verbose=0)[0][0])
label = human_label_from_score(score)
confidence = confidence_from_score(score)

print(f"Prediction: {label}")
print(f"Cancer probability: {score:.4f}")
print(f"Confidence: {confidence:.4f}")
