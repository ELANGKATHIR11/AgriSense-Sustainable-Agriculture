#!/usr/bin/env python3
"""
Edge AI Service API
Provides HTTP endpoints for edge AI chatbot and vision models
"""

import os
from pathlib import Path

# Import edge AI modules
from edge_ai_chatbot import get_chatbot
from edge_ai_vision import get_vision_model
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize models
print("Initializing Edge AI models...")
chatbot = get_chatbot()
vision_model = get_vision_model()
print("✅ Edge AI models initialized")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "OK",
            "service": "Edge AI Service",
            "models_loaded": True,
            "chatbot_ready": chatbot is not None,
            "vision_ready": vision_model is not None,
        }
    )


@app.route("/chatbot/query", methods=["POST"])
def chatbot_query():
    """Process chatbot query"""
    try:
        data = request.json
        query = data.get("query", "")
        crop_name = data.get("crop_name", None)
        context = data.get("context", {})

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Process query
        result = chatbot.process_query(query, crop_name)

        return jsonify(
            {
                "success": True,
                "result": result,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chatbot/cultivation-guide/<crop_name>", methods=["GET"])
def get_cultivation_guide(crop_name):
    """Get cultivation guide for a specific crop"""
    try:
        crop_normalized = crop_name.replace("_", " ")
        guide = chatbot.get_cultivation_guide(crop_normalized)

        return jsonify(
            {"success": True, "crop": crop_normalized, "guide": guide}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/vision/analyze", methods=["POST"])
def analyze_image():
    """Analyze plant image"""
    try:
        if "image" not in request.files:
            return jsonify({"error": "Image file is required"}), 400

        image_file = request.files["image"]
        crop_name = request.form.get("crop_name", None)

        # Save uploaded file temporarily
        upload_dir = Path(__file__).parent.parent.parent / "uploads"
        upload_dir.mkdir(exist_ok=True)

        import uuid

        filename = str(uuid.uuid4()) + "." + image_file.filename.split(".")[-1]
        image_path = upload_dir / filename
        image_file.save(str(image_path))

        # Analyze image
        result = vision_model.analyze_plant_image(str(image_path), crop_name)

        # Clean up temporary file
        try:
            os.remove(str(image_path))
        except:
            pass

        return jsonify(
            {
                "success": True,
                "analysis": result,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/vision/analyze-path", methods=["POST"])
def analyze_image_path():
    """Analyze plant image from file path"""
    try:
        data = request.json
        image_path = data.get("image_path", "")
        crop_name = data.get("crop_name", None)

        if not image_path:
            return jsonify({"error": "image_path is required"}), 400

        # Analyze image
        result = vision_model.analyze_plant_image(image_path, crop_name)

        return jsonify(
            {
                "success": True,
                "analysis": result,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/diseases/<crop_name>", methods=["GET"])
def get_crop_diseases(crop_name):
    """Get diseases for a specific crop"""
    try:
        crop_normalized = crop_name.replace("_", " ")
        diseases = vision_model.crop_diseases.get(crop_normalized, [])

        disease_details = []
        for disease_name in diseases:
            disease_info = vision_model.disease_knowledge.get(disease_name, {})
            disease_details.append({"name": disease_name, **disease_info})

        return jsonify(
            {
                "success": True,
                "crop": crop_normalized,
                "diseases": disease_details,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("EDGE_AI_PORT", 5002))
    print(f"🚀 Starting Edge AI Service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
