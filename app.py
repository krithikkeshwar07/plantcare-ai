
import os
import io
import threading

import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPModel,
    CLIPProcessor,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

MODEL_ID = os.getenv(
    "PLANT_MODEL_ID",
    "VaigandlaHemanth/leaf-disease-clip-vit",
)
LEAF_CHECKER_ID = os.getenv(
    "LEAF_CHECKER_ID",
    "openai/clip-vit-base-patch32",
)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_processor = None
_leaf_checker = None
_leaf_processor = None

_model_lock = threading.Lock()
_leaf_lock = threading.Lock()


# --------------------------------------------------
# LOAD THE DISEASE CLASSIFICATION MODEL
# --------------------------------------------------

def get_model():
    global _model, _processor

    if _model is not None and _processor is not None:
        return _processor, _model

    with _model_lock:
        if _model is None or _processor is None:
            processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

            model.to(_device)
            model.eval()

            _processor = processor
            _model = model

    return _processor, _model


# --------------------------------------------------
# LOAD A SEPARATE MODEL TO CHECK WHETHER IT IS A LEAF
# --------------------------------------------------

def get_leaf_checker():
    global _leaf_checker, _leaf_processor

    if _leaf_checker is not None and _leaf_processor is not None:
        return _leaf_processor, _leaf_checker

    with _leaf_lock:
        if _leaf_checker is None or _leaf_processor is None:
            processor = CLIPProcessor.from_pretrained(LEAF_CHECKER_ID)
            model = CLIPModel.from_pretrained(LEAF_CHECKER_ID)

            model.to(_device)
            model.eval()

            _leaf_processor = processor
            _leaf_checker = model

    return _leaf_processor, _leaf_checker


# --------------------------------------------------
# CHECK WHETHER THE UPLOADED PHOTO SHOWS A LEAF
# --------------------------------------------------

def is_leaf_image(image):
    """
    Returns (accepted, reason).

    CLIP compares the image with leaf and non-leaf descriptions.
    This is a screening check, not a perfect guarantee.
    """

    processor, model = get_leaf_checker()

    leaf_descriptions = [
        "a close-up photograph of a single green plant leaf",
        "a photograph of a diseased leaf with spots or discoloration",
        "a photograph of a yellowing or damaged plant leaf",
        "a botanical photograph showing leaf veins and leaf edges",
        "a photograph of a crop leaf attached to a plant",
    ]

    non_leaf_descriptions = [
        "a photograph of a person or human face",
        "a photograph of an animal or bird",
        "a photograph of a car, vehicle, or machine",
        "a photograph of a building or indoor room",
        "a photograph of food, fruit, or a cooked meal",
        "a photograph of a landscape or a mountain",
        "a photograph of a flower without a clear leaf",
        "a photograph of an object, document, or computer screen",
        "a photograph of soil, rocks, or water without a visible leaf",
    ]

    leaf_inputs = processor(
        text=leaf_descriptions,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    # Compare the same image against the two sets of descriptions.
    leaf_text_inputs = processor(
        text=leaf_descriptions,
        return_tensors="pt",
        padding=True,
    )
    nonleaf_text_inputs = processor(
        text=non_leaf_descriptions,
        return_tensors="pt",
        padding=True,
    )

    image_inputs = processor(
        images=image,
        return_tensors="pt",
    )

    image_inputs = {
        key: value.to(_device)
        for key, value in image_inputs.items()
    }
    leaf_text_inputs = {
        key: value.to(_device)
        for key, value in leaf_text_inputs.items()
    }
    nonleaf_text_inputs = {
        key: value.to(_device)
        for key, value in nonleaf_text_inputs.items()
    }

    with torch.inference_mode():
        image_features = model.get_image_features(**image_inputs)
        leaf_features = model.get_text_features(**leaf_text_inputs)
        nonleaf_features = model.get_text_features(**nonleaf_text_inputs)

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        leaf_features = leaf_features / leaf_features.norm(
            dim=-1, keepdim=True
        )
        nonleaf_features = nonleaf_features / nonleaf_features.norm(
            dim=-1, keepdim=True
        )

        leaf_scores = image_features @ leaf_features.T
        nonleaf_scores = image_features @ nonleaf_features.T

        # Compare the average similarities across each group.
        leaf_score = leaf_scores.mean().item()
        nonleaf_score = nonleaf_scores.mean().item()

    # Conservative screening threshold. Tune after testing with
    # real leaf and non-leaf photos from your target crops.
    if leaf_score < 0.20 or leaf_score <= nonleaf_score:
        return False, (
            "This image does not appear to clearly show a plant leaf. "
            "Please upload a close-up photo of one crop leaf in natural light."
        )

    return True, "Leaf image accepted."


# --------------------------------------------------
# DISEASE-SPECIFIC PRECAUTIONS
# --------------------------------------------------

def advice_for(label):
    text = str(label).lower().replace("___", " ").replace("_", " ")

    if "healthy" in text:
        return [
            "Continue normal care and crop-appropriate watering.",
            "Inspect both sides of leaves regularly for new spots or pests.",
            "Take another photo in a few days if the plant still looks unusual.",
        ]

    if "early blight" in text:
        return [
            "Remove badly affected leaves when practical and dispose of them away from the crop.",
            "Water at soil level and avoid wetting foliage.",
            "Improve plant spacing and airflow where practical.",
            "Ask a local agricultural expert about approved treatment if symptoms continue spreading.",
        ]

    if "late blight" in text:
        return [
            "Seek local agricultural advice promptly because this disease can spread quickly in suitable conditions.",
            "Avoid overhead watering and handling wet plants.",
            "Dispose of severely affected plant material safely.",
            "Confirm the disease before applying any treatment.",
        ]

    if "powdery mildew" in text:
        return [
            "Remove heavily affected leaves when practical.",
            "Improve airflow and avoid excessively dense growth.",
            "Monitor new leaves for additional symptoms.",
            "Ask a local agricultural expert about approved treatment if it spreads.",
        ]

    if "bacterial spot" in text or "bacterial blight" in text:
        return [
            "Avoid splashing water between plants.",
            "Clean tools between plants.",
            "Remove badly affected plant material when practical.",
            "Confirm the cause before selecting a treatment.",
        ]

    if "virus" in text or "mosaic" in text or "yellow leaf curl" in text:
        return [
            "Seek agricultural advice to confirm whether a virus is present.",
            "Check for insect pests such as aphids or whiteflies.",
            "Clean tools after handling affected plants.",
            "Ask an agricultural expert whether isolation or removal is appropriate.",
        ]

    if "rust" in text or "septoria" in text or "leaf mold" in text:
        return [
            "Remove severely affected leaves when practical.",
            "Avoid wetting foliage and improve airflow.",
            "Inspect nearby plants for similar symptoms.",
            "Confirm the diagnosis locally before selecting a treatment.",
        ]

    return [
        "Take a clear close-up photo of one affected leaf in natural daylight.",
        "Check nearby leaves for similar symptoms.",
        "Water at soil level and avoid splashing foliage.",
        "Confirm the cause with a local agricultural expert before applying pesticides.",
    ]


def disease_type_for(label):
    clean = str(label).replace("___", " — ").replace("_", " ").strip()

    if " — " in clean:
        clean = clean.split(" — ", 1)[1].strip()

    if "healthy" in clean.lower():
        return "Healthy leaf"

    return clean[:1].upper() + clean[1:] if clean else "Uncertain leaf condition"


def solution_for(label):
    text = str(label).lower().replace("_", " ")

    if "healthy" in text:
        return (
            "The model predicts a healthy leaf. Continue normal crop care "
            "and monitor new growth."
        )

    if "late blight" in text:
        return (
            "Seek local agricultural advice promptly. Confirm the disease "
            "before using any locally approved treatment."
        )

    if "early blight" in text:
        return (
            "Remove badly affected leaves when practical, keep foliage dry, "
            "and improve airflow. Seek local advice if symptoms spread."
        )

    if "powdery mildew" in text:
        return (
            "Remove heavily affected leaves when practical and improve airflow. "
            "Seek local advice if symptoms continue."
        )

    if "bacterial" in text:
        return (
            "Reduce water splash between plants and clean tools. Confirm the "
            "cause before selecting a treatment."
        )

    if "virus" in text or "mosaic" in text:
        return (
            "Ask an agricultural expert to confirm the cause and advise "
            "whether the plant should be isolated or removed."
        )

    return (
        "The cause is uncertain. Monitor whether symptoms spread and ask "
        "a local agricultural expert to confirm the problem."
    )


# --------------------------------------------------
# WEB ROUTES
# --------------------------------------------------

@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_id": MODEL_ID,
        "leaf_checker_id": LEAF_CHECKER_ID,
        "device": str(_device),
        "disease_model_loaded": _model is not None,
        "leaf_checker_loaded": _leaf_checker is not None,
    })


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Please choose a leaf image first."
        }), 400

    file = request.files["image"]

    if not file or not file.filename:
        return jsonify({
            "error": "Please choose a leaf image first."
        }), 400

    try:
        raw_data = file.read()

        if not raw_data:
            return jsonify({
                "error": "The uploaded file is empty. Please choose another image."
            }), 400

        image = Image.open(io.BytesIO(raw_data))
        image = ImageOps.exif_transpose(image).convert("RGB")

        # Reject extremely small images.
        if image.width < 64 or image.height < 64:
            return jsonify({
                "error": (
                    "The image is too small to analyze. "
                    "Please upload a clear photo at least 64 × 64 pixels."
                )
            }), 400

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return jsonify({
            "error": (
                "This file is not a readable image. "
                "Please upload a valid JPG, PNG, or WEBP photo."
            )
        }), 400

    try:
        # Step 1: Check that the photo appears to contain a leaf.
        accepted, reason = is_leaf_image(image)

        if not accepted:
            return jsonify({
                "error": reason,
                "error_type": "not_a_leaf",
            }), 400

        # Step 2: Predict the possible leaf disease.
        processor, model = get_model()

        inputs = processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(_device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

            count = min(3, int(probabilities.shape[0]))
            values, indices = torch.topk(probabilities, k=count)

        id2label = getattr(model.config, "id2label", {}) or {}
        predictions = []

        for value, index in zip(values.tolist(), indices.tolist()):
            label = str(
                id2label.get(
                    index,
                    id2label.get(str(index), f"Class {index}"),
                )
            )

            predictions.append({
                "label": label.replace("___", " — ").replace("_", " "),
                "confidence": round(value * 100, 1),
            })

        if not predictions:
            return jsonify({
                "error": "No prediction was returned. Please try another photo."
            }), 502

        top = predictions[0]
        label = top["label"]
        confidence = top["confidence"]

        limitation = (
            "This is an AI screening result, not a confirmed diagnosis. "
            "Model predictions can be incorrect, especially for unfamiliar "
            "crops, lighting conditions, or symptoms."
        )

        if confidence < 35:
            limitation = (
                "Low confidence: this result may be incorrect. Retake a clear "
                "photo in natural daylight. Do not treat based on this result "
                "alone. " + limitation
            )

        return jsonify({
            "prediction": disease_type_for(label),
            "disease_type": disease_type_for(label),
            "confidence": confidence,
            "precautions": advice_for(label),
            "solution": solution_for(label),
            "limitation": limitation,
            "top_predictions": predictions,
        })

    except Exception as exc:
        app.logger.exception("Plant image analysis failed")

        return jsonify({
            "error": (
                "The AI model could not analyze this image. "
                "Check your internet connection, dependencies, and terminal "
                "logs, then try again."
            ),
            "detail": str(exc)[:300],
        }), 503


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(413)
def too_large(_error):
    return jsonify({
        "error": "Image is too large. Please upload an image smaller than 10 MB."
    }), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({
        "error": "The requested page or API endpoint was not found."
    }), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )