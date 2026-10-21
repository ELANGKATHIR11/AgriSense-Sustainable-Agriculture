import os
import io
import json
import base64
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from app.core.config import settings

class PlantDiseaseClassifier:
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.model = None
        self.classes = []
        self.details = {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.load_model()

    def load_model(self):
        model_path = os.path.join(settings.MODELS_DIR, "artifacts", "plant_disease_model.pth")
        meta_path = os.path.join(settings.MODELS_DIR, "metadata", "plant_disease_model.json")
        
        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            print(f"Disease model artifacts not found at {model_path}. Running in degraded mode.")
            return

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.classes = meta.get("classes", [])
            self.details = meta.get("details", {})
            
            # Load PyTorch architecture
            num_classes = len(self.classes)
            base_model = models.mobilenet_v3_small(weights=None)
            in_features = base_model.classifier[0].in_features
            base_model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.Hardswish(),
                nn.Dropout(p=0.2),
                nn.Linear(512, num_classes)
            )
            
            checkpoint = torch.load(model_path, map_location=self.device)
            base_model.load_state_dict(checkpoint['state_dict'])
            base_model.to(self.device)
            base_model.eval()
            self.model = base_model
            print(f"Plant Disease Model loaded successfully onto {self.device} ({settings.GPU_NAME}) with {num_classes} classes!")
        except Exception as e:
            print(f"Error loading plant disease model: {e}")

    def predict_image(self, image_bytes: bytes, crop_hint: str = None) -> dict:
        if self.model is None:
            raise RuntimeError("Plant disease model is not loaded.")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.topk(probs, 1)
            
            prob_val = float(top_prob.item())
            class_name = self.classes[int(top_idx.item())]

        detail = self.details.get(class_name, {})
        
        return {
            "isHealthy": detail.get("isHealthy", False),
            "cropIdentified": detail.get("cropIdentified", "Identified Crop"),
            "diseaseName": detail.get("diseaseName", class_name),
            "scientificName": detail.get("scientificName", "Pathology Assessment"),
            "confidencePct": round(prob_val * 100, 1),
            "severity": detail.get("severity", "moderate"),
            "symptomsObserved": detail.get("symptomsObserved", ["Foliar lesion inspection"]),
            "organicTreatment": detail.get("organicTreatment", ["Biological neem extract application"]),
            "chemicalTreatment": detail.get("chemicalTreatment", ["Mancozeb 75% WP @ 2g/L"]),
            "preventionSteps": detail.get("preventionSteps", ["Sanitize crop equipment"]),
            "source": f"AgriSense Deep Vision CNN (RTX 5060 dGPU)",
            "isModelPrediction": True,
            "analyzedAt": pd_timestamp()
        }

def pd_timestamp():
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

disease_classifier = PlantDiseaseClassifier()
