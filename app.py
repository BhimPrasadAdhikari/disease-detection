import os
import numpy as np
import tensorflow as tf
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Public GCS URL (you can override via env var if needed)
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://storage.googleapis.com/disease_detection_bucket/plant_disease_model.h5"
)
MODEL_PATH = "plant_disease_model.h5"


def download_model():
    """Download the model from GCS if not already present"""
    if not os.path.exists(MODEL_PATH):
        print(f"📥 Downloading model from {MODEL_URL} ...")
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully.")


# Download and load model
download_model()
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (ensure they match your training dataset)
class_names = ['Healthy', 'Powdery', 'Rust']


def preprocess_image(img_path):
    """Load and preprocess image for prediction"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🌿 Plant Disease Detection API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    # Preprocess and predict
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    # Remove file after prediction (optional)
    os.remove(file_path)

    return jsonify({
        "prediction": predicted_class,
        "confidence": f"{confidence:.2f}%",
        "probabilities": {
            class_names[i]: f"{prediction[0][i] * 100:.2f}%"
            for i in range(len(class_names))
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
