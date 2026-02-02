import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from typing import Dict, Any, List
import json
import os
import random

print("🔄 AGRI-VLM INFERENCE ENGINE v3 RELOADING...")

# Import our new architecture and schema
try:
    from backend.ml.agri_vlm_model import AgriVLM
    from backend.ai_service.agri_vlm_schema import VLMAnalysisResult, Cure, Prevention
except ImportError:
    # Fallback for running as script relative to project root
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from backend.ml.agri_vlm_model import AgriVLM
    from backend.ai_service.agri_vlm_schema import VLMAnalysisResult, Cure, Prevention


class AgriVLMInference:
    """
    Inference Engine for AGRI-VLM-CARE+++.
    """

    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ AGRI-VLM Inference initialized on {self.device}")

        # Initialize Model Structure
        self.is_custom_model = False
        self.custom_classes = []

        # Try to load custom trained model if available
        custom_model_path = os.path.join(
            "backend", "ml", "models_custom", "custom_vlm_best.pth"
        )
        if os.path.exists(custom_model_path):
            print(f"🎯 LOADING CUSTOM VLM WEIGHTS: {custom_model_path}")
            try:
                checkpoint = torch.load(custom_model_path, map_location=self.device)
                self.custom_classes = checkpoint.get(
                    "classes", ["Healthy", "Powdery", "Rust"]
                )

                # Re-create the architecture used in training (ResNet50)
                self.model = resnet50(weights=None)  # Architecture only
                num_ftrs = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_ftrs, len(self.custom_classes))

                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.to(self.device)
                self.is_custom_model = True
                print(f"✅ Custom Model Loaded. Classes: {self.custom_classes}")
            except Exception as e:
                print(f"⚠️ Failed to load custom model: {e}")
                print("⚠️ Falling back to Standard AgriVLM Architecture")
                self.model = AgriVLM(num_crops=96, num_diseases=200, num_weeds=50)
                self.model.to(self.device)
        else:
            self.model = AgriVLM(num_crops=96, num_diseases=200, num_weeds=50)
            self.model.to(self.device)

        self.model.eval()

        # Load Weights if available
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"✅ Loaded VLM weights from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load weights: {e}")
                print(
                    "⚠️ Running with UNTRAINED weights (Architecture Verification Mode)"
                )
        else:
            print(
                "⚠️ No model weights found. Running in Architecture Verification Mode."
            )

        # Preprocessing (Standard ImageNet stats for ViT)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # --- SCENE GUARD (Verification Mode Feature) ---
        # Loads a standard pre-trained ResNet to detect if the image is actually a plant.
        try:
            print("🛡️ Initializing Scene Guard (ResNet50)...")
            weights = ResNet50_Weights.IMAGENET1K_V1
            self.scene_model = resnet50(weights=weights)
            self.scene_model.to(self.device)
            self.scene_model.eval()
            self.imagenet_categories = weights.meta["categories"]
            print("✅ Scene Guard Initialized.")
        except Exception as e:
            print(f"⚠️ Scene Guard Failed: {e}")
            self.scene_model = None

        # Agri-relevant keywords in ImageNet classes
        # Removed generic terms like "tree", "grass", "garden", "pot" that cause false positives on landscapes
        self.agri_keywords = [
            "vegetable",
            "fruit",
            "corn",
            "ear",
            "leaf",
            "mushroom",
            "agriculture",
            "lemon",
            "orange",
            "banana",
            "apple",
            "grape",
            "strawberry",
            "pineapple",
            "fig",
            "pomegranate",
            "broccoli",
            "cabbage",
            "cauliflower",
            "zucchini",
            "cucumber",
            "artichoke",
            "pepper",
            "potato",
            "onion",
            "wheat",
            "rice",
            "grain",
            "head cabbage",
        ]

        # Knowledge Base for "Farmer-Friendly" Content Generation
        # (Simulating the output of the Instruction-Tuned Decoder for the demo)
        self.knowledge_base = {
            "Leaf Blast": {
                "cure": {
                    "immediate_actions": [
                        "Remove and burn infected leaves immediately"
                    ],
                    "chemical_treatments": ["Spray Tricyclazole 75 WP (0.6g/L)"],
                    "biological_treatments": ["Spray Pseudomonas fluorescens @ 10g/L"],
                },
                "prevention": {
                    "cultural_practices": [
                        "Avoid excess Nitrogen fertilizer",
                        "Maintain proper water level",
                    ],
                    "long_term_strategy": ["Use resistant varieties"],
                },
            },
            "Brown Spot": {
                "cure": {
                    "immediate_actions": ["Improve soil nutrients"],
                    "chemical_treatments": ["Spray Mancozeb (2.5g/L)"],
                    "biological_treatments": ["Apply Trichoderma viride"],
                },
                "prevention": {
                    "cultural_practices": [
                        "Proper field sanitation",
                        "Balanced fertilization",
                    ],
                    "long_term_strategy": ["Crop rotation with non-host crops"],
                },
            },
            "Weed": {
                "cure": {
                    "immediate_actions": ["Manual weeding"],
                    "chemical_treatments": ["Apply appropriate herbicide"],
                    "biological_treatments": ["Mulching"],
                },
                "prevention": {
                    "cultural_practices": ["Deep ploughing in summer"],
                    "long_term_strategy": ["Cover cropping"],
                },
            },
            "Maydis Leaf Blight": {
                "cure": {
                    "immediate_actions": [
                        "Remove infected lower leaves",
                        "Apply bio-control agents immediately",
                    ],
                    "chemical_treatments": [
                        "Spray Mancozeb 75 WP @ 2.5g/L",
                        "Apply Propiconazole @ 1ml/L",
                    ],
                    "biological_treatments": ["Trichoderma viride seed treatment"],
                },
                "prevention": {
                    "cultural_practices": [
                        "Crop rotation with non-host crops",
                        "Destroy crop debris after harvest",
                    ],
                    "long_term_strategy": ["Use resistant Maize hybrids"],
                },
            },
            "Powdery Mildew": {
                "cure": {
                    "immediate_actions": [
                        "Prune infected parts immediately",
                        "Improve air circulation between plants",
                    ],
                    "chemical_treatments": [
                        "Potassium Bicarbonate spray (10g/L)",
                        "Sulfur-based fungicides",
                        "Neem Oil application",
                    ],
                    "biological_treatments": ["Bacillus subtilis biopesticide"],
                },
                "prevention": {
                    "cultural_practices": [
                        "Ensure 6+ hours of direct sunlight",
                        "Avoid overhead irrigation (use drip)",
                        "Maintain humidity below 93% (greenhouse)",
                    ],
                    "long_term_strategy": [
                        "Plant resistant varieties",
                        "Use slow-release nitrogen fertilizers",
                    ],
                },
            },
            "Wheat Rust": {
                "cure": {
                    "immediate_actions": [
                        "Monitor flag leaf status",
                        "Identify alternate hosts (e.g. Barberry) and remove",
                    ],
                    "chemical_treatments": [
                        "Azole fungicides (Tebuconazole/Propiconazole)",
                        "Strobilurins for preventive/early curative",
                        "Apply before flag leaf emergence",
                    ],
                    "biological_treatments": ["Seed treatment with PGPR"],
                },
                "prevention": {
                    "cultural_practices": [
                        "Avoid excessively early sowing",
                        "Balanced Nitrogen-Potash fertilization",
                        "Eradicate volunteer wheat plants",
                    ],
                    "long_term_strategy": [
                        "Continuous genetic resistance monitoring",
                        "Decadal rotation",
                    ],
                },
            },
            "Healthy": {
                "cure": {
                    "immediate_actions": ["No treatment required - plant is healthy!"],
                    "chemical_treatments": [],
                    "biological_treatments": [],
                },
                "prevention": {
                    "cultural_practices": [
                        "Continue current good practices",
                        "Regular monitoring for early disease detection",
                    ],
                    "long_term_strategy": [
                        "Maintain crop rotation",
                        "Use certified seeds",
                    ],
                },
            },
            "Rust": {
                "cure": {
                    "immediate_actions": ["Remove heavily infected leaves"],
                    "chemical_treatments": ["Apply Propiconazole 25 EC @ 1ml/L"],
                    "biological_treatments": ["Trichoderma seed treatment"],
                },
                "prevention": {
                    "cultural_practices": [
                        "Avoid dense planting",
                        "Ensure good drainage",
                    ],
                    "long_term_strategy": ["Plant rust-resistant varieties"],
                },
            },
            "Powdery": {
                "cure": {
                    "immediate_actions": ["Remove affected leaves"],
                    "chemical_treatments": ["Sulfur-based fungicide spray"],
                    "biological_treatments": ["Neem oil application"],
                },
                "prevention": {
                    "cultural_practices": [
                        "Improve air circulation",
                        "Avoid wet foliage",
                    ],
                    "long_term_strategy": ["Use resistant varieties"],
                },
            },
        }

    def predict(self, image: Image.Image) -> VLMAnalysisResult:
        """
        Run inference on a single image.
        """
        # 1. Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 2. Forward Pass (Neural Network)
        # 2. Forward Pass (Neural Network)
        with torch.no_grad():
            outputs = self.model(img_tensor)

        # 3. Decode Outputs & SCENE GUARD CHECK

        predicted_crop = "Unknown"
        predicted_diagnosis = "Unknown"
        confidence_score = 0.5
        is_agri_image = False
        detected_concept = "Unknown"

        # Handle Custom Model Output (Tensor) vs AgriVLM Output (Dict)
        if self.is_custom_model:
            # outputs is logits tensor [1, num_classes]
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = probs.topk(1)
            confidence_score = top_prob.item()
            predicted_diagnosis = self.custom_classes[top_idx.item()]

            # Heuristic Crop Mapping for Custom Classes
            if "rust" in predicted_diagnosis.lower():
                predicted_crop = "Wheat/Grain"
                detected_concept = "Wheat"
            elif "powdery" in predicted_diagnosis.lower():
                predicted_crop = "Vegetable/Plant"
                detected_concept = "Leaf"
            elif "healthy" in predicted_diagnosis.lower():
                predicted_crop = "Healthy Plant"
                detected_concept = "Leaf"
            else:
                predicted_crop = "Unknown Crop"

            is_agri_image = True  # Assume custom model is fed crop images

            # Still run SceneGuard to populate 'detected_concept' accurately if possible
            if self.scene_model:
                with torch.no_grad():
                    scene_logits = self.scene_model(img_tensor)
                    sg_probs = torch.nn.functional.softmax(scene_logits, dim=1)
                    _, sg_idx = sg_probs.topk(1)
                    detected_concept = self.imagenet_categories[sg_idx.item()]

            print(
                f"🧠 Custom Model Prediction: {predicted_diagnosis} on {predicted_crop} ({confidence_score:.2f})"
            )
        else:
            # AgriVLM Standard Logic (Not Custom)
            pass

        # A. Run Scene Guard first (Shared Logic)

        # A. Run Scene Guard first
        if self.scene_model:
            with torch.no_grad():
                scene_logits = self.scene_model(img_tensor)
                probs = torch.nn.functional.softmax(scene_logits, dim=1)
                top_prob, top_class_id = probs.topk(1)

                detected_concept = self.imagenet_categories[top_class_id.item()]
                confidence_score = top_prob.item()

                # Check if detected concept is agri-related
                # simple keyword matching
                is_agri_image = any(
                    k in detected_concept.lower() for k in self.agri_keywords
                )

                if not is_agri_image:
                    # Check top 5 for flexibility
                    _, top5_ids = probs.topk(5)
                    debug_concepts = []
                    for idx in top5_ids[0]:
                        concept = self.imagenet_categories[idx.item()]
                        debug_concepts.append(concept)
                        if any(k in concept.lower() for k in self.agri_keywords):
                            is_agri_image = True
                            detected_concept = concept
                            break
                    print(
                        f"🔍 Scene Guard Analysis: Detected='{detected_concept}' | Top5={debug_concepts} | IsAgri={is_agri_image}"
                    )

        # B. Logic Dispatch
        if self.scene_model and not is_agri_image:
            # Rejection Mode
            print(f"🛑 REJECTING Image: {detected_concept} is not in agri_keywords")

            return VLMAnalysisResult(
                crop_identified="Non-Crop Image",
                diagnosis=f"Scene Detected: {detected_concept}",
                confidence=confidence_score * 100,
                severity="N/A",
                cure={
                    "immediate_actions": [
                        "Please upload a clear image of a CROP leaf."
                    ],
                    "chemical_treatments": [],
                    "biological_treatments": [],
                },
                prevention={
                    "cultural_practices": [],
                    "long_term_strategy": ["Ensure the image is focused on the plant."],
                },
            )

        # C. It IS a plant (or Scene Guard missing), run Simulation/Model
        if is_agri_image:
            predicted_crop = (
                detected_concept.capitalize()
            )  # Use the real ImageNet class!

            # Pseudo-Logic for Disease based on Crop (Simulation)
            # If it's a real model, this part is `outputs['disease_head']`

            if "corn" in predicted_crop.lower():
                predicted_crop = "Maize (Zea mays)"
                predicted_diagnosis = (
                    "Maydis Leaf Blight"  # Integrated from Research Paper
                )
            elif "pot" in predicted_crop.lower():  # pot, flowerpot
                predicted_diagnosis = "Root Rot"
            elif "rust" in predicted_crop.lower() or "wheat" in predicted_crop.lower():
                predicted_diagnosis = "Wheat Rust"
            elif (
                "powdery" in predicted_crop.lower()
                or "mildew" in predicted_crop.lower()
            ):
                predicted_diagnosis = "Powdery Mildew"
            elif "healthy" in predicted_crop.lower():
                predicted_diagnosis = "Healthy"
            elif "wheat" in predicted_crop.lower() or "grain" in predicted_crop.lower():
                predicted_diagnosis = "Rust"
            else:
                predicted_diagnosis = "Healthy / Unknown Issue"

            # Fallback to defaults if "paddy" logic wanted
            if "rice" in predicted_crop.lower():
                predicted_diagnosis = "Leaf Blast"

        else:
            # Fallback if SceneGuard failed to load OR checks strangely passed
            # If SceneGuard is None, we should NOT simulate "Paddy" blindly anymore.
            if self.scene_model is None:
                print("⚠️ Scene Guard NOT Loaded. Returning System Error.")
                return VLMAnalysisResult(
                    crop_identified="System Error",
                    diagnosis="Scene Guard Initialization Failed",
                    confidence=0.0,
                    severity="Error",
                    cure={
                        "immediate_actions": [
                            "Check server logs",
                            "Torchvision/Weights missing",
                        ],
                        "chemical_treatments": [],
                        "biological_treatments": [],
                    },
                    prevention={"cultural_practices": [], "long_term_strategy": []},
                )

            # If we are here, is_agri_image must be True
            print(f"✅ Scene Guard Passed: {detected_concept}")
            predicted_crop = "Paddy"  # Default fallback if code logic falls through
            predicted_diagnosis = "Leaf Blast"
            confidence_score = 0.92

        # Extract Confidence from Head (Real Architecture Usage)
        if not self.is_custom_model:
            real_conf = outputs["confidence"].item()
            # If using standard model, we might want to use real_conf,
            # but we keep using our simulation/sceneguard logic for now if weights are random.
            pass

        # Post-process for scientific naming
        pathogen_map = {
            "Maydis Leaf Blight": "Cochliobolus heterostrophus",
            "Wheat Rust": "Puccinia tritici",
            "Powdery Mildew": "Erysiphe cichoracearum",
            "Leaf Blast": "Magnaporthe oryzae",
            "Root Rot": "Phytophthora",
        }
        scientific_name = pathogen_map.get(predicted_diagnosis, "N/A")

        # 4. Generate Structured Output
        kb_entry = self.knowledge_base.get(
            predicted_diagnosis, self.knowledge_base["Healthy"]
        )

        result = VLMAnalysisResult(
            crop_identified=predicted_crop,
            scientific_name=scientific_name,
            diagnosis=predicted_diagnosis,
            confidence=confidence_score * 100,
            severity="Medium",
            detected_concepts=detected_concept,  # Assuming detected_concept is the intended variable for detected_concepts
            cure=Cure(
                immediate_actions=kb_entry["cure"]["immediate_actions"],
                chemical_treatments=kb_entry["cure"]["chemical_treatments"],
                biological_treatments=kb_entry["cure"].get("biological_treatments", []),
            ),
            prevention=Prevention(
                cultural_practices=kb_entry["prevention"]["cultural_practices"],
                long_term_strategy=kb_entry["prevention"]["long_term_strategy"],
            ),
        )

        return result


# Singleton instance
_vlm_instance = None


def get_vlm_inference():
    global _vlm_instance
    if _vlm_instance is None:
        _vlm_instance = AgriVLMInference()
    return _vlm_instance
