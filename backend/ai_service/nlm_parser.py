import json
import logging
from typing import Dict, Any, Optional
from llm_engine import LLMEngine

logger = logging.getLogger("AgriSense-AI")

NLM_SYSTEM_PROMPT = """
You are the Neuro-Language Mapper (NLM).
Convert the user's query into a strict JSON object.
Use this schema:
{
  "intent": "crop_recommendation" | "yield_prediction" | "disease_detection" | "general_advice",
  "entities": {
    "crop": "string or null",
    "location": "string or null",
    "season": "string or null",
    "parameter": "string or null"
  }
}
Do not include any text other than the JSON.
"""


class NLMEngine:
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def parse_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Uses the local GGUF LLM to parse the user query into JSON.
        """
        try:
            # For smaller models like TinyLlama/Phi-3, we need to be very explicit in the prompt
            prompt = f"""
            User Query: "{user_query}"
            
            Extract the intent and entities as JSON.
            Example: "How much water for rice?" -> {{"intent": "water_requirement", "entities": {{"crop": "rice", "parameter": "water"}}}}
            
            JSON:
            """

            response_text = self.llm.generate_response(prompt, NLM_SYSTEM_PROMPT)

            # Robust JSON extraction
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = response_text[start:end]
                # Fix common weak-model JSON errors if necessary (basic cleanup)
                json_str = json_str.replace("'", '"')
                return json.loads(json_str)
            else:
                logger.warning(
                    f"NLM failed to find JSON in response: {response_text[:50]}..."
                )
                return {"intent": "general_advice", "entities": {}}
        except Exception as e:
            logger.error(f"NLM Error: {e}")
            return {"intent": "general_advice", "entities": {}}
