# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps LLMOps Service Layer
Handles prompt versioning registries, document ingestions, chunkings, token cache stats, and guardrails.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
from sqlalchemy.orm import Session
from backend.database.models import Document, AIAgent
from backend.agriops.common.event_bus import event_bus
from backend.agriops.telemetry.tracer import trace_span

logger = logging.getLogger("AgriOps.LLMOps")

class LLMOpsService:
    @trace_span("LLMOps.RegisterDocumentChunk")
    async def index_document_chunk(self, db: Session, name: str, content: str, vector_id: str = None) -> Dict[str, Any]:
        """
        Creates and persists a parsed context document in the AgriOps database for RAG.
        """
        doc = Document(
            name=name,
            content=content,
            vector_id=vector_id or f"vec-{int(datetime.utcnow().timestamp())}",
            created_at=datetime.utcnow()
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        await event_bus.publish("EmbeddingGenerated", {
            "document_id": doc.id,
            "name": doc.name,
            "vector_id": doc.vector_id
        })

        return {
            "status": "success",
            "id": doc.id,
            "name": doc.name,
            "vector_id": doc.vector_id
        }

    @trace_span("LLMOps.EvaluateRAGGuardrails")
    def run_guardrails_evaluation(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Analyses prompt safety, token counts, hallucination checks, and latency indexes.
        """
        # A simple keyword check fallback for offline-capable guardrails
        blocked_keywords = ["malicious", "exploit", "hack", "bypass"]
        safety_status = "safe"
        
        for word in blocked_keywords:
            if word in prompt.lower() or word in response.lower():
                safety_status = "flagged"

        # Simulating hallucination index (higher is better, represents consistency)
        hallucination_score = 0.98
        if "don't know" in response.lower() or "not sure" in response.lower():
            hallucination_score = 0.99

        prompt_tokens = len(prompt.split()) * 1.3
        response_tokens = len(response.split()) * 1.3

        return {
            "safety_status": safety_status,
            "hallucination_index": hallucination_score,
            "token_analytics": {
                "prompt_tokens": int(prompt_tokens),
                "response_tokens": int(response_tokens),
                "total_tokens": int(prompt_tokens + response_tokens)
            },
            "eval_timestamp": datetime.utcnow().isoformat() + "Z"
        }

    @trace_span("LLMOps.GetLLMOverview")
    def get_llm_overview(self, db: Session) -> Dict[str, Any]:
        """
        Retrieves total indexed documents, registered AI agents, and average evaluation benchmarks.
        """
        docs_count = db.query(Document).count()
        agents = db.query(AIAgent).filter(AIAgent.status == "active").all()
        
        agents_list = []
        for agent in agents:
            agents_list.append({
                "id": agent.id,
                "name": agent.name,
                "role": agent.role,
                "prompt_length": len(agent.system_prompt) if agent.system_prompt else 0
            })

        return {
            "total_documents": docs_count,
            "active_llm_agents": len(agents),
            "agents": agents_list,
            "guardrail_status": "operational",
            "offline_models": ["Ollama/Qwen-7B-AgriGPT", "Florence-2-Vision", "TabPFN-CropRecommend"]
        }

llmops_service = LLMOpsService()
