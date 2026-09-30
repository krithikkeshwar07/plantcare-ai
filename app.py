
import os
import threading

import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# Use only the disease classification model.
MODEL_ID = os.getenv(
    "PLANT_MODEL_ID",
    "VaigandlaHemanth/leaf-disease-clip-vit",
)

# Keep CPU memory usage as low as reasonably possible.
DEVICE = torch.device("cpu")
torch.set_num_threads(1)

_model = None
_processor = None
_model_lock = threading.Lock()


def get_model():
    """Load the model once, only when a prediction is requested."""
    global _model, _processor

    with _model_lock:
        if _model is None or _processor is None:
            _processor = AutoImageProcessor.from_pretrained(MODEL_ID)

            _model = AutoModelForImageClassification.from_pretrained(
                MODEL_ID,
                low_cpu_mem_usage=False,
            )
            _model.to(DEVICE)
            _model.eval()

    return _processor, _model


def get_advice(label):
    """Return general guidance based on the predicted class."""
    name = label.replace("_", " ").strip()
    lower = name.lower()

    if "healthy" in lower:
        return {
            "disease_type": "Healthy",
            "precautions": [
                "Continue regular watering appropriate for the crop.",
                "Inspect leaves regularly for spots or discoloration.",
                "Maintain good airflow and remove dead plant material.",
            ],
            "solution": (
                "The model predicts a healthy leaf. Continue normal care "
                "and monitor the plant for any changes."
            ),
        }

    if "early blight" in lower:
        precautions = [
            "Avoid wetting the leaves when watering.",
            "Remove fallen or heavily affected leaves safely.",
            "Keep enough space between plants for airflow.",
        ]
        solution = (
            "Check the plant for characteristic spots and seek local "
            "agricultural advice before choosing a treatment."
        )

    elif "late blight" in lower:
        precautions = [
            "Avoid overhead watering where possible.",
            "Keep the plant area clean and improve airflow.",
            "Check nearby plants for similar symptoms.",
        ]
        solution = (
            "Late blight can spread quickly. Seek prompt advice from "
            "a local agricultural expert about suitable treatment."
        )

    elif "bacterial" in lower:
        precautions = [
            "Avoid handling wet plants unnecessarily.",
            "Clean gardening tools between plants.",
            "Avoid splashing water between affected plants.",
        ]
        solution = (
            "Confirm the diagnosis with an agricultural expert. "
            "Management depends on the crop and bacterial disease."
        )

    elif "virus" in lower or "mosaic" in lower:
        precautions = [
            "Check nearby plants for similar symptoms.",
            "Control insect pests using locally recommended methods.",
            "Clean tools after working with affected plants.",
        ]
        solution = (
            "Viral diseases usually need careful management rather than "
            "a simple cure. Consult a local agricultural expert."
        )

    elif "rust" in lower:
        precautions = [
            "Monitor the undersides of leaves for new symptoms.",
            "Avoid unnecessary leaf wetness.",
            "Remove badly affected fallen leaves where appropriate.",
        ]
        solution = (
            "Confirm the rust diagnosis and ask an agricultural expert "
            "about crop-appropriate disease management."
        )

    elif "powdery mildew" in lower:
        precautions = [
            "Maintain airflow around plants.",
            "Avoid overcrowding.",
            "Inspect new growth regularly.",
        ]
        solution = (
            "Confirm the symptoms and use only locally recommended "
            "treatments suitable for the crop."
        )

    else:
        precautions = [
            "Inspect other leaves for similar symptoms.",
            "Avoid overwatering and maintain good plant hygiene.",
            "Keep the plant under observation and take another clear photo.",
        ]
        solution = (
            "This is an AI-generated screening result. Confirm the "
            "disease before applying pesticides or other treatments."
        )

    return {
        "disease_type": name,
        "precautions": precautions,
        "solution": solution,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "PlantCare AI server is running.",
        "model_loaded": _model is not None,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    # Accept the usual image field and a file field as a fallback.
    uploaded_file = (
        request.files.get("image")
        or request.files.get("file")
    )

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({
            "error": "Please select a leaf image first."
        }), 400

    try:
        image = Image.open(uploaded_file.stream)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((768, 768))

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "The uploaded file is not a valid image. Please try JPG or PNG."
        }), 400

    try:
        processor, model = get_model()

        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            confidence_tensor, predicted_index = probabilities.max(dim=-1)

        index = int(predicted_index.item())
        confidence = float(confidence_tensor.item()) * 100

        labels = model.config.id2label
        label = labels.get(index, labels.get(str(index), str(index)))

        advice = get_advice(label)

        # This is a screening result, not a confirmed diagnosis.
        limitation = (
            "AI prediction only. Image classification may be incorrect, "
            "especially for unfamiliar crops, lighting, or symptoms. "
            "Confirm the result with an agricultural expert before treatment."
        )

        return jsonify({
            "success": True,
            "prediction": label,
            "disease": label,
            "disease_type": advice["disease_type"],
            "confidence": round(confidence, 2),
            "confidence_percentage": round(confidence, 2),
            "precautions": advice["precautions"],
            "solution": advice["solution"],
            "limitation": limitation,
        })

    except Exception:
        app.logger.exception("PlantCare AI prediction failed")
        return jsonify({
            "error": (
                "AI analysis failed. The model may still be downloading, "
                "or the server may have run out of memory. Please try again."
            )
        }), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image is too large. Please upload an image under 10 MB."
    }), 413


@app.route("/favicon.ico")
def favicon():
    return "", 204


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
