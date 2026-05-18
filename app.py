import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import mlflow

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "model/breast_cancer_model.h5")
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"No model found at {MODEL_PATH}")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

@app.route("/ready", methods=["GET"])
def ready():
    if model is None:
        return jsonify({"status": "model not loaded"}), 503
    return jsonify({"status": "ready"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model not loaded"}), 503
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400
    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]
    label = "malignant" if prediction > 0.5 else "benign"
    confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
