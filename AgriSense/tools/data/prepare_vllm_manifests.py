"""
Prepare vLLM fine-tuning manifests from existing datasets.
Outputs:
 - agrisense_app/backend/data/vllm_text_finetune.jsonl  (text prompt/completion pairs)
 - agrisense_app/backend/data/vllm_image_text.jsonl      (image_path + caption pairs)

This manifest is compatible with many LoRA/QLoRA training flows where each line is a JSON object.
"""
import csv
import json
from pathlib import Path

INTENTS_CSV = Path("agrisense_app/backend/data/chatbot_intents.csv")
VLM_CSV = Path("agrisense_app/backend/data/vlm_pairs.csv")
OUT_DIR = Path("agrisense_app/backend/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEXT_OUT = OUT_DIR / "vllm_text_finetune.jsonl"
IMAGE_OUT = OUT_DIR / "vllm_image_text.jsonl"


def make_text_manifest():
    with open(INTENTS_CSV, 'r', encoding='utf-8') as f_in, open(TEXT_OUT, 'w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        for r in reader:
            prompt = f"User: {r['utterance']}\nAssistant:"
            completion = f" {r['intent']}"
            obj = {"prompt": prompt, "completion": completion}
            f_out.write(json.dumps(obj) + "\n")
    print(f"✅ Wrote text fine-tune manifest: {TEXT_OUT}")


def make_image_manifest():
    with open(VLM_CSV, 'r', encoding='utf-8') as f_in, open(IMAGE_OUT, 'w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        for r in reader:
            img = r['image_path']
            caption = r['caption']
            obj = {"image": img, "text": caption}
            f_out.write(json.dumps(obj) + "\n")
    print(f"✅ Wrote image-text manifest: {IMAGE_OUT}")


def main():
    make_text_manifest()
    make_image_manifest()

if __name__ == '__main__':
    main()
