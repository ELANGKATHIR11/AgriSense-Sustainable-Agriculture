"""
AGRI-OS GenAI Contract — RAG-Only LLM Wrapper
===============================================
Wraps Phi-3-mini via Ollama with strict evidence-grounded constraints.

The LLM ONLY explains decisions using provided evidence.
It NEVER diagnoses, prescribes, or overrides the Decision Governor.

Tone modulation:
- HIGH confidence → direct, actionable language
- LOW confidence → cautious, hedging language

Multilingual: passes language param, relies on Phi-3's multilingual
capability + existing i18n patterns.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .schemas import DecisionAction, GovernorDecision, VRAGResult

logger = logging.getLogger("agrios.genai")

# System prompt enforcing RAG-only behavior
SYSTEM_PROMPT = """You are an agricultural decision explanation assistant for the AGRI-OS system.

STRICT RULES:
1. You ONLY explain the decision that has ALREADY been made by the Decision Governor.
2. You NEVER diagnose diseases yourself — the vision pipeline did that.
3. You NEVER prescribe treatments yourself — the action templates provide those.
4. You NEVER override or contradict the Governor's decision.
5. You ONLY use the evidence provided to you — no external knowledge.
6. If confidence is LOW, use cautious language: "This may indicate...", "Consider monitoring..."
7. If confidence is HIGH, use direct language: "The analysis shows...", "We recommend..."

RESPONSE FORMAT (JSON):
{
    "summary": "Brief summary of what was detected and what action is recommended",
    "evidence_used": ["List of evidence points that support this decision"],
    "confidence_level": "HIGH | MEDIUM | LOW",
    "what_to_do_next": "Clear next steps for the farmer",
    "what_NOT_to_do": "Things to avoid based on the current situation"
}

Respond in the language specified by the user. Be concise and practical — farmers need clarity, not jargon."""

# Tone templates based on confidence
TONE_HIGH = "Based on strong evidence, the analysis clearly shows"
TONE_MEDIUM = "The analysis suggests, with moderate confidence, that"
TONE_LOW = "While the evidence is limited, initial observations may indicate that"


class AgriGenAI:
    """
    RAG-only LLM wrapper for explaining Governor decisions.

    Uses Phi-3-mini via Ollama (existing integration) or falls back
    to template-based explanations when Ollama is unavailable.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "phi3:mini",
    ) -> None:
        self.ollama_url = ollama_base_url.rstrip("/")
        self.model = model
        self._available: Optional[bool] = None

    async def _check_availability(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    async def explain(
        self,
        governor_decision: GovernorDecision,
        vrag_evidence: List[VRAGResult],
        sensor_context: Dict[str, Any],
        language: str = "en",
    ) -> str:
        """
        Generate a natural-language explanation of the Governor's decision.

        Parameters
        ----------
        governor_decision : GovernorDecision from the Decision Governor
        vrag_evidence : VRAG retrieval results
        sensor_context : sensor readings and crop context
        language : target language code

        Returns
        -------
        str: JSON-formatted explanation
        """
        # Build context prompt
        user_prompt = self._build_prompt(governor_decision, vrag_evidence, sensor_context, language)

        # Try Ollama first
        if await self._check_availability():
            try:
                result = await self._call_ollama(user_prompt, language)
                if result:
                    return result
            except Exception as e:
                logger.warning("Ollama call failed, falling back to template: %s", e)

        # Fallback: template-based explanation
        return self._template_explain(governor_decision, vrag_evidence, sensor_context, language)

    def _build_prompt(
        self,
        decision: GovernorDecision,
        evidence: List[VRAGResult],
        sensor_ctx: Dict[str, Any],
        language: str,
    ) -> str:
        """Build the user prompt with all context for the LLM."""
        parts = [
            f"Language: {language}",
            f"\nDECISION: {decision.action.value}",
            f"Confidence: [{decision.confidence_band.lower:.2f}, {decision.confidence_band.median:.2f}, {decision.confidence_band.upper:.2f}]",
            f"Regret Score: {decision.regret_score:.4f}",
            "\nEVIDENCE CHAIN:",
        ]

        for e in decision.evidence:
            parts.append(f"  • {e}")

        if evidence:
            parts.append("\nVRAG RETRIEVAL RESULTS:")
            for r in evidence[:5]:
                parts.append(f"  • {r.evidence_text} (similarity: {r.similarity_score:.3f})")

        if sensor_ctx:
            parts.append("\nSENSOR CONTEXT:")
            for k, v in sensor_ctx.items():
                parts.append(f"  • {k}: {v}")

        if decision.treatment:
            parts.append("\nTREATMENT TEMPLATE:")
            for k, v in decision.treatment.items():
                parts.append(f"  • {k}: {v}")

        parts.append("\nPlease explain this decision to a farmer in simple, practical terms.")
        return "\n".join(parts)

    async def _call_ollama(self, user_prompt: str, language: str) -> Optional[str]:
        """Call Ollama API with the Phi-3 model."""
        import httpx

        lang_note = ""
        if language != "en":
            lang_note = f"\nIMPORTANT: Respond in {language} language."

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + lang_note},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 500,
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("message", {}).get("content", "")
        return None

    def _template_explain(
        self,
        decision: GovernorDecision,
        evidence: List[VRAGResult],
        sensor_ctx: Dict[str, Any],
        language: str,
    ) -> str:
        """
        Template-based fallback when Ollama is unavailable.
        Produces structured JSON explanation without LLM.
        """
        band = decision.confidence_band

        # Select tone
        if band.lower >= 0.8:
            conf_level = "HIGH"
            tone = TONE_HIGH
        elif band.lower >= 0.5:
            conf_level = "MEDIUM"
            tone = TONE_MEDIUM
        else:
            conf_level = "LOW"
            tone = TONE_LOW

        # Build summary
        action_desc = {
            DecisionAction.ACT: "immediate action is recommended",
            DecisionAction.WAIT: "we recommend waiting for more data before acting",
            DecisionAction.OBSERVE: "continued observation is advised",
            DecisionAction.DO_NOTHING: "no action is needed at this time",
        }

        summary = f"{tone} {action_desc.get(decision.action, 'further evaluation is needed')}."

        # Evidence used
        evidence_used = [e for e in decision.evidence if not e.startswith("Confidence band")]
        if evidence:
            evidence_used.extend([r.evidence_text for r in evidence[:3]])

        # What to do next
        next_steps = {
            DecisionAction.ACT: "Follow the treatment plan provided. Monitor the crop closely for 7 days after treatment.",
            DecisionAction.WAIT: "Take another photo in 24-48 hours. Check sensor readings. Consult local extension if unsure.",
            DecisionAction.OBSERVE: "Continue monitoring. Do not apply any treatments yet. Report if symptoms worsen.",
            DecisionAction.DO_NOTHING: "No intervention needed. Continue regular monitoring schedule.",
        }

        # What NOT to do
        dont_do = {
            DecisionAction.ACT: "Do not exceed recommended dosages. Do not mix chemicals without guidance.",
            DecisionAction.WAIT: "Do not apply preventive treatments without confirmation. Do not panic.",
            DecisionAction.OBSERVE: "Do not apply any treatments based on uncertain detection. Do not ignore worsening symptoms.",
            DecisionAction.DO_NOTHING: "Do not apply unnecessary treatments. Avoid over-watering or over-fertilizing.",
        }

        explanation = {
            "summary": summary,
            "evidence_used": evidence_used,
            "confidence_level": conf_level,
            "what_to_do_next": next_steps.get(decision.action, "Consult an expert."),
            "what_NOT_to_do": dont_do.get(decision.action, "Do not ignore symptoms."),
        }

        if decision.treatment:
            explanation["treatment_details"] = decision.treatment

        return json.dumps(explanation, indent=2, ensure_ascii=False)
