import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from google.cloud import storage
import shutil

app = Flask(__name__)

# GCS bucket details (override via env vars if needed)
BUCKET_NAME = os.getenv("BUCKET_NAME", "disease_detection_bucket")
MODEL_DIR = os.getenv("MODEL_DIR", "plant_disease_model_saved")
LOCAL_MODEL_DIR = "/tmp/plant_disease_model_saved"  # Cloud Run writable tmp


def download_model():
    """Download model directory from GCS if not already present"""
    if not os.path.exists(LOCAL_MODEL_DIR):
        print(f"📥 Downloading model from gs://{BUCKET_NAME}/{MODEL_DIR} ...")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)

        # download all blobs with this prefix
        blobs = bucket.list_blobs(prefix=MODEL_DIR)
        for blob in blobs:
            local_path = os.path.join("/tmp", blob.name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)

        print("✅ Model downloaded successfully.")
    else:
        print("⚡ Using cached model from /tmp")


# Download and load model
download_model()
model = tf.keras.models.load_model(LOCAL_MODEL_DIR)

# Class names (must match training dataset)
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

    try:
        # Preprocess
        img_array = preprocess_image(file_path)

        # Call the SavedModel 'serve' endpoint
        infer = model.signatures["serve"]
        prediction = infer(tf.constant(img_array))  # returns dict
        # The output tensor is the first (and only) value
        prediction = list(prediction.values())[0].numpy()

        # Decode prediction
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)

        return jsonify({
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "probabilities": {
                class_names[i]: f"{prediction[0][i] * 100:.2f}%"
                for i in range(len(class_names))
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
