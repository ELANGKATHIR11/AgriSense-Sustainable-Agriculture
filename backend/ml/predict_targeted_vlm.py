import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def predict(image_path, model_path, classes_path):
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        print(json.dumps({"error": "Model or classes file not found"}))
        return

    # Load classes
    with open(classes_path, "r") as f:
        class_names = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Model structure
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model weights: {str(e)}"}))
        return

    model = model.to(device)
    model.eval()

    # Preprocess image
    loader = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    try:
        image = Image.open(image_path).convert("RGB")
        image = loader(image).unsqueeze(0)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image)
            percentages = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            _, preds = torch.max(outputs, 1)

            predicted_class = class_names[preds[0]]
            confidence = percentages[preds[0]].item()

        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "all_scores": {
                class_names[i]: percentages[i].item()
                for i in range(len(class_names))
            },
        }
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "models"),
    )

    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "targeted_vlm_model.pth")
    classes_path = os.path.join(args.model_dir, "targeted_vlm_classes.json")

    predict(args.image, model_path, classes_path)
