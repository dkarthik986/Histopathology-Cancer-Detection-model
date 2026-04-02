# Histopathology Image Cancer Prediction

This repository trains a TensorFlow/Keras CNN from scratch on public PatchCamelyon histopathology patches and serves predictions with Flask.

## Expected dataset format

Training expects RGB patch images arranged like this:

```text
data/
  train/
    benign/
    cancer/
  valid/
    benign/
    cancer/
  test/
    benign/
    cancer/
```

`prepare_data.py` builds that structure automatically from the public Hugging Face dataset `1aurent/PatchCamelyon`.

## Local workflow

```bash
python -m pip install -r requirements.txt
python prepare_data.py
python train.py
python predict.py --image data/test/cancer/test_cancer_00000.png
python app.py
```

The trained model is saved to `model/model.h5`, and both the CLI predictor and Flask app use the same `96x96` RGB preprocessing path.
