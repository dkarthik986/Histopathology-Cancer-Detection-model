from __future__ import annotations

import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from ml_config import STATIC_DIR, TEMPLATE_DIR
from ml_utils import (
    confidence_from_score,
    ensure_directories,
    human_label_from_score,
    load_trained_model,
    preprocess_image,
    predict_score,
    warm_up_model,
)

ensure_directories()
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
model = load_trained_model()
warm_up_model(model)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            uploaded_file = request.files.get("image")
            if uploaded_file is None or not uploaded_file.filename:
                return render_template(
                    "index.html",
                    error="Please choose an image to analyze.",
                )

            filename = secure_filename(uploaded_file.filename) or "uploaded_image.png"
            image = preprocess_image(uploaded_file.stream)
            score = predict_score(model, image)
            label = human_label_from_score(score)
            confidence = f"{confidence_from_score(score) * 100:.2f}%"
            cancer_probability = f"{score * 100:.2f}%"

            return render_template(
                "result.html",
                label=label,
                confidence=confidence,
                cancer_probability=cancer_probability,
                filename=filename,
            )
        except Exception as exc:
            app.logger.exception("Prediction failed")
            return render_template(
                "index.html",
                error=f"Prediction failed: {exc}",
            )

    return render_template("index.html", error=None)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
