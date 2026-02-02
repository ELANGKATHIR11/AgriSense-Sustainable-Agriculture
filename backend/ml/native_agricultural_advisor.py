"""
Native Agricultural Advisor Service
Replaces OpenAI GPT-3.5 Turbo with local inference
Uses Phi-2 OR falls back to rule-based responses if model not available
"""

import json
import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
PHI2_DIR = BASE_DIR / "models" / "phi2-agriculture"


class NativeAgriculturalAdvisor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_available = self._load_model()

    def _load_model(self):
        """Try to load Phi-2 model if available"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not PHI2_DIR.exists():
                print("[WARN] Phi-2 model not found, using fallback responses")
                return False

            print("[INFO] Loading Phi-2 model...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(PHI2_DIR))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(PHI2_DIR), low_cpu_mem_usage=True
            )
            print("[INFO] Phi-2 model loaded successfully!")
            return True

        except ImportError:
            print("[WARN] transformers/torch not installed, using fallback")
            return False
        except Exception as e:
            print(f"[WARN] Could not load Phi-2: {e}")
            return False

    def get_advice(self, query, context=None):
        """Get agricultural advice for a query"""

        if self.model_available:
            return self._get_phi2_response(query, context)
        else:
            return self._get_fallback_response(query, context)

    def _get_phi2_response(self, query, context):
        """Generate response using Phi-2"""
        try:
            # Create prompt
            prompt = f"You are an agricultural expert. Answer this farming question:\n\nQuestion: {query}\n\nAnswer:"

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer (after "Answer:")
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response.strip()

            return {
                "advice": answer,
                "recommendations": self._extract_recommendations(answer),
                "confidence": 0.85,
                "sources": ["Phi-2 Agricultural Model"],
                "model": "phi-2",
            }

        except Exception as e:
            print(f"[ERROR] Phi-2 inference failed: {e}")
            return self._get_fallback_response(query, context)

    def _get_fallback_response(self, query, context):
        """Rule-based fallback responses"""

        query_lower = query.lower()

        # Rice farming
        if "rice" in query_lower or "paddy" in query_lower:
            return {
                "advice": "For rice cultivation: Maintain 2-3 inches of standing water during vegetative stage. Apply balanced NPK fertilizer (4:2:1 ratio). Monitor for brown planthopper and rice blast disease. Ensure proper field leveling for uniform water distribution.",
                "recommendations": [
                    "Maintain 2-3 inches standing water during vegetative stage",
                    "Use NPK fertilizer in 4:2:1 ratio",
                    "Monitor for brown planthopper and rice blast",
                    "Level fields properly for uniform water distribution",
                    "Consider System of Rice Intensification (SRI) for higher yields",
                ],
                "confidence": 0.90,
                "sources": [
                    "Agricultural Knowledge Base",
                    "Rice Cultivation Guidelines",
                ],
                "model": "rule-based",
            }

        # Wheat farming
        elif "wheat" in query_lower:
            return {
                "advice": "Wheat requires well-drained soil with pH 6.0-7.0. Apply nitrogen in split doses: 50% at sowing, 25% at tillering, 25% at booting stage. Monitor for rust diseases and aphids. Practice crop rotation with legumes for soil health.",
                "recommendations": [
                    "Ensure soil pH between 6.0-7.0 with good drainage",
                    "Split nitrogen: 50% sowing, 25% tillering, 25% booting",
                    "Monitor for rust diseases and aphids regularly",
                    "Rotate with legumes to improve soil fertility",
                    "Apply zinc sulfate if deficiency observed",
                ],
                "confidence": 0.88,
                "sources": [
                    "Wheat Cultivation Standards",
                    "CIMMYT Guidelines",
                ],
                "model": "rule-based",
            }

        # Organic/Sustainable
        elif "organic" in query_lower or "sustainable" in query_lower:
            return {
                "advice": "Organic farming focuses on soil health, biodiversity, and natural inputs. Use compost, green manures, and biological pest control. Practice crop rotation and conservation tillage to improve soil structure and reduce erosion.",
                "recommendations": [
                    "Prepare compost from farm waste and kitchen scraps",
                    "Use green manures like dhaincha or sunhemp",
                    "Implement biological pest control (neem oil, beneficial insects)",
                    "Practice crop rotation to break pest cycles",
                    "Use conservation tillage to reduce soil erosion",
                    "Maintain buffer zones for biodiversity",
                ],
                "confidence": 0.85,
                "sources": [
                    "Organic Farming Standards",
                    "Sustainable Agriculture Practices",
                ],
                "model": "rule-based",
            }

        # Pest/Disease
        elif "pest" in query_lower or "disease" in query_lower:
            return {
                "advice": "Integrated Pest Management (IPM) combines cultural, biological, and chemical methods. Start with resistant varieties, monitor pest populations regularly, and use pesticides only when economic thresholds are reached.",
                "recommendations": [
                    "Plant pest-resistant crop varieties",
                    "Regular field scouting to monitor pests",
                    "Use biological controls (beneficial insects, neem products)",
                    "Apply chemical pesticides only at economic thresholds",
                    "Maintain proper field sanitation",
                    "Use pheromone traps for monitoring",
                ],
                "confidence": 0.87,
                "sources": ["IPM Guidelines", "Entomology Research"],
                "model": "rule-based",
            }

        # General advice
        else:
            return {
                "advice": "Modern sustainable agriculture combines traditional wisdom with scientific innovations. Focus on soil health through regular testing and organic matter addition. Implement efficient irrigation systems like drip or sprinkler to conserve water. Practice crop rotation to maintain soil fertility and break pest cycles.",
                "recommendations": [
                    "Test soil regularly and amend based on results",
                    "Implement drip irrigation for 30-50% water savings",
                    "Use crop rotation to maintain soil fertility",
                    "Adopt conservation agriculture practices",
                    "Monitor weather forecasts for timely operations",
                    "Consider agroforestry for additional income",
                ],
                "confidence": 0.75,
                "sources": [
                    "Agricultural Best Practices",
                    "Sustainable Farming Guidelines",
                ],
                "model": "rule-based",
            }

    def _extract_recommendations(self, text):
        """Extract bullet points or recommendations from text"""
        recommendations = []

        # Split by common delimiters
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                recommendations.append(line.lstrip("-•* "))
            elif line and len(line) < 200:  # Short actionable statements
                if any(
                    word in line.lower()
                    for word in [
                        "use",
                        "apply",
                        "maintain",
                        "monitor",
                        "ensure",
                    ]
                ):
                    recommendations.append(line)

        return recommendations[:6]  # Return up to 6 recommendations


# Create singleton instance
advisor = NativeAgriculturalAdvisor()


def get_agricultural_advice(query, context=None):
    """Main API function for getting agricultural advice"""
    return advisor.get_advice(query, context)


# Test if running directly
# Handle command line arguments for health check

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--health", action="store_true", help="Check health status")
    args = parser.parse_args()

    # Health check mode
    if args.health:
        advisor = NativeAgriculturalAdvisor()
        if advisor.model_available:
            print("Status: Healthy, Model: Phi-2")
            sys.exit(0)
        else:
            print("Status: Healthy, Model: Rule-based")
            sys.exit(0)

    # API Mode: Read from stdin, write to stdout
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        if not input_data:
            # If no input, maybe just testing manually
            print(
                json.dumps(
                    {
                        "error": "No input provided",
                        "advice": "Please provide a query.",
                    }
                )
            )
            sys.exit(1)

        data = json.loads(input_data)
        query = data.get("query")
        context = data.get("context")

        if not query:
            print(
                json.dumps(
                    {
                        "error": "Missing query",
                        "advice": "Query parameter is required.",
                    }
                )
            )
            sys.exit(1)

        # Get advice
        response = get_agricultural_advice(query, context)

        # Print JSON response to stdout
        print(json.dumps(response))
        sys.exit(0)

    except Exception as e:
        # Handle errors gracefully
        print(
            json.dumps(
                {
                    "error": str(e),
                    "advice": "I encountered an internal error. Please try again.",
                    "model": "error-handler",
                }
            )
        )
        sys.exit(1)
