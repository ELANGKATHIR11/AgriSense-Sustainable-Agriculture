"""
Train AgriSense Chatbot with Phi2 Model + Existing Datasets
Uses CPU and NPU (DirectML) for accelerated training
Integrates phi2-agriculture model with 1150+ agricultural intent examples
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

# Try to import required libraries
try:
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers not installed: pip install transformers datasets")

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("⚠️ onnxruntime not installed for NPU: pip install onnxruntime-directml")


class AgriChatbotTrainer:
    """Train agricultural chatbot using Phi2 model with local datasets"""

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir or Path(__file__).parent)
        self.models_dir = self.base_dir / "models"
        self.datasets_dir = self.base_dir / "datasets"
        self.phi2_model_path = self.models_dir / "phi2-agriculture"

        # Check hardware capabilities
        self.device = self.detect_hardware()
        print(f"\n🖥️ Training device: {self.device}")

    def detect_hardware(self) -> str:
        """Detect available hardware (CPU, GPU, NPU)"""
        if torch.cuda.is_available():
            print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            return "cuda"

        # Check for DirectML (NPU support on Windows)
        if HAS_ONNX:
            providers = ort.get_available_providers()
            if "DmlExecutionProvider" in providers:
                print("✅ DirectML (NPU) detected")
                return "dml"

        print("✅ Using CPU")
        return "cpu"

    def load_intent_dataset(self) -> pd.DataFrame:
        """Load chatbot intents dataset"""
        intent_file = self.datasets_dir / "chatbot_intents.csv"

        if not intent_file.exists():
            print(f"❌ Intent dataset not found: {intent_file}")
            return None

        df = pd.read_csv(intent_file)
        print(f"\n📊 Loaded {len(df)} intent examples")
        print(f"   Categories: {df['intent'].nunique()}")
        print("   Intent distribution:")
        print(df["intent"].value_counts().head(10))

        return df

    def load_vlm_text_dataset(self) -> List:
        """Load VLM text datasets for additional training"""
        text_datasets = []

        # Load JSONL files
        jsonl_files = [
            self.datasets_dir / "vllm_text_finetune.jsonl",
            self.datasets_dir / "vllm_image_text.jsonl",
        ]

        for jsonl_file in jsonl_files:
            if jsonl_file.exists():
                with open(jsonl_file, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        text_datasets.append(data)

        print(f"\n📚 Loaded {len(text_datasets)} VLM text examples")
        return text_datasets

    def prepare_training_data(
        self, intents_df: pd.DataFrame, vlm_data: List = None
    ) -> Dataset:
        """Prepare dataset for Phi2 fine-tuning"""

        # Convert intents to conversational format
        conversations = []

        # Process intent data
        for _, row in intents_df.iterrows():
            intent = row["intent"]
            utterance = row["utterance"]

            # Create instruction-response format for phi2
            prompt = f"User: {utterance}\nIntent: {intent}\nAssistant:"
            conversations.append({"text": prompt})

        # Add VLM text data if available
        if vlm_data:
            for item in vlm_data:
                if "text" in item:
                    conversations.append({"text": item["text"]})

        print(f"\n✅ Prepared {len(conversations)} training examples")

        # Convert to HuggingFace Dataset
        return Dataset.from_list(conversations)

    def load_phi2_model(self):
        """Load Phi2 model and tokenizer"""
        if not self.phi2_model_path.exists():
            print(f"\n❌ Phi2 model not found at: {self.phi2_model_path}")
            print("Please ensure phi2-agriculture model is downloaded")
            return None, None

        print(f"\n📥 Loading Phi2 model from: {self.phi2_model_path}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.phi2_model_path), trust_remote_code=True
            )

            # Set padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with appropriate settings for CPU/NPU
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                str(self.phi2_model_path), **model_kwargs
            )

            print("✅ Model loaded successfully")
            print(f"   Parameters: {model.num_parameters() / 1e9:.2f}B")

            return model, tokenizer

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None

    def train(self, epochs=3, batch_size=4, learning_rate=2e-5):
        """Train the chatbot model"""

        if not HAS_TRANSFORMERS:
            print("\n❌ transformers library required")
            print("Install: pip install transformers datasets accelerate")
            return

        print("\n" + "=" * 60)
        print("🚀 AgriSense Chatbot Training with Phi2")
        print("=" * 60)

        # Load datasets
        intents_df = self.load_intent_dataset()
        if intents_df is None:
            return

        vlm_data = self.load_vlm_text_dataset()

        # Prepare training data
        train_dataset = self.prepare_training_data(intents_df, vlm_data)

        # Load model
        model, tokenizer = self.load_phi2_model()
        if model is None:
            return

        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

        print("\n🔄 Tokenizing dataset...")
        tokenized_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        # Split into train/val
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_data = split_dataset["train"]
        val_data = split_dataset["test"]

        print(f"   Train samples: {len(train_data)}")
        print(f"   Val samples: {len(val_data)}")

        # Training arguments
        output_dir = self.models_dir / "phi2-agrisense-finetuned"

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=str(output_dir / "logs"),
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=100,
            fp16=self.device == "cuda",  # Use FP16 only on GPU
            dataloader_num_workers=0 if self.device == "cpu" else 2,
            remove_unused_columns=False,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False  # Causal LM, not masked LM
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
        )

        # Train!
        print("\n🏋️ Starting training...")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")

        try:
            train_result = trainer.train()

            # Save model
            print("\n💾 Saving fine-tuned model...")
            trainer.save_model()
            tokenizer.save_pretrained(str(output_dir))

            # Save training metrics
            metrics_file = output_dir / "training_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(train_result.metrics, f, indent=2)

            print("\n✅ Training complete!")
            print(f"   Model saved to: {output_dir}")
            print(f"   Final loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")

            # Create training summary
            self.create_training_summary(output_dir, train_result.metrics, intents_df)

        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback

            traceback.print_exc()

    def create_training_summary(
        self, output_dir: Path, metrics: Dict, intents_df: pd.DataFrame
    ):
        """Create training summary document"""

        summary = """# Phi2 AgriSense Chatbot Training Summary

## Training Completed
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Statistics
- **Total intents**: {len(intents_df)}
- **Unique categories**: {intents_df['intent'].nunique()}
- **Intent categories**: {', '.join(intents_df['intent'].unique()[:10])}...

## Model Information
- **Base model**: phi2-agriculture
- **Fine-tuned for**: Agricultural question answering and intent classification
- **Output model**: {output_dir.name}

## Training Configuration
- **Device**: {self.device}
- **Final training loss**: {metrics.get('train_loss', 'N/A')}
- **Evaluation loss**: {metrics.get('eval_loss', 'N/A')}

## Usage

### Load the fine-tuned model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{output_dir}",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")

# Test
query = "How much water does rice need per week?"
inputs = tokenizer(query, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Next Steps
1. Test the model with various agricultural queries
2. Integrate with backend API endpoints
3. Deploy for inference
4. Continue monitoring and fine-tuning as needed

## Intent Categories Trained
{chr(10).join('- ' + cat for cat in sorted(intents_df['intent'].unique()))}
"""

        summary_file = output_dir / "TRAINING_SUMMARY.md"
        with open(summary_file, "w") as f:
            f.write(summary)

        print(f"📄 Training summary: {summary_file}")


def main():
    """Main training function"""
    print(
        """
    ╔═══════════════════════════════════════════════════════════╗
    ║  AgriSense Chatbot Training with Phi2                    ║
    ║  Train on 1150+ agricultural intent examples             ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    )

    trainer = AgriChatbotTrainer()

    # Training configuration
    print("\n⚙️ Training Configuration:")
    print("   - Epochs: 3 (adjust for your needs)")
    print("   - Batch size: 4 (CPU-friendly, increase if using GPU)")
    print("   - Learning rate: 2e-5")
    print("\n💡 Tip: Training on CPU will take 2-4 hours. Use GPU if available.")
    print("💡 NPU support via DirectML - install: pip install onnxruntime-directml")

    proceed = input("\n🚀 Start training? (y/n): ").lower()

    if proceed == "y":
        trainer.train(epochs=3, batch_size=4, learning_rate=2e-5)
    else:
        print("Training cancelled.")


if __name__ == "__main__":
    main()
