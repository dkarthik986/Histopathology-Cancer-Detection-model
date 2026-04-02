from __future__ import annotations

import os
import time

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from ml_config import STATIC_DIR, TEMPLATE_DIR
from ml_utils import (
    configure_tensorflow_runtime,
    confidence_from_score,
    ensure_directories,
    human_label_from_score,
    load_trained_model,
    preprocess_image,
    predict_score,
    warm_up_model,
)

ensure_directories()
configure_tensorflow_runtime()
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
model = None


def get_model():
    global model
    if model is None:
        load_started = time.perf_counter()
        app.logger.info("Loading model...")
        model = load_trained_model()
        warm_up_model(model)
        app.logger.info(
            "Model loaded and warmed in %.2fs",
            time.perf_counter() - load_started,
        )
    return model


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            request_started = time.perf_counter()
            uploaded_file = request.files.get("image")
            if uploaded_file is None or not uploaded_file.filename:
                return render_template(
                    "index.html",
                    error="Please choose an image to analyze.",
                )

            filename = secure_filename(uploaded_file.filename) or "uploaded_image.png"
            active_model = get_model()

            preprocess_started = time.perf_counter()
            image = preprocess_image(uploaded_file.stream)
            app.logger.info(
                "Preprocessing completed in %.2fs for %s",
                time.perf_counter() - preprocess_started,
                filename,
            )

            predict_started = time.perf_counter()
            score = predict_score(active_model, image)
            app.logger.info(
                "Prediction completed in %.2fs for %s",
                time.perf_counter() - predict_started,
                filename,
            )

            label = human_label_from_score(score)
            confidence = f"{confidence_from_score(score) * 100:.2f}%"
            cancer_probability = f"{score * 100:.2f}%"
            app.logger.info(
                "Request completed in %.2fs for %s",
                time.perf_counter() - request_started,
                filename,
            )

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
