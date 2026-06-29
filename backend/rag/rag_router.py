# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps MRAG API Router
Exposes endpoints for multimodal, text, vision, and agent memory operations.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from backend.database.session import get_db
from backend.rag.mrag_orchestrator import mrag_orchestrator
from backend.vision.vrag_service import vrag_service
from backend.agriops.telemetry.tracer import trace_span

router = APIRouter(prefix="/mrag", tags=["AgriOps MRAG"])

class RetrieveRequest(BaseModel):
    query: str
    collection: str = "documents"
    k: int = 3
    filter: Optional[str] = None

class ImageRAGRequest(BaseModel):
    imageBase64: str
    mode: str = "disease"

class MemoryStoreRequest(BaseModel):
    agent_id: str
    memory_text: str
    metadata: Optional[dict] = None

@router.post("/retrieve")
@trace_span("MRAG.RetrieveText")
async def retrieve_text_context(payload: RetrieveRequest):
    """
    Performs standard semantic vector search on LanceDB collections.
    """
    results = mrag_orchestrator.search_collection(
        collection_name=payload.collection,
        query=payload.query,
        k=payload.k,
        metadata_filter=payload.filter
    )
    return {"results": results}

@router.post("/query")
@trace_span("MRAG.QueryFusion")
async def query_mrag(
    query: str = Body(...),
    sensor_context: Optional[dict] = Body(default=None)
):
    """
    Executes hybrid multimodal retrieval fusion combining vector search and sensor readings.
    """
    context = mrag_orchestrator.get_orchestrated_mrag_context(query, sensor_context)
    return context

@router.post("/vrag")
@trace_span("MRAG.VisionSearch")
async def vision_rag_search(payload: ImageRAGRequest):
    """
    Upgraded VRAG endpoint executing vision vector matching against LanceDB image collections.
    """
    res = await vrag_service.search_similar_images(payload.imageBase64, payload.mode)
    return res

@router.post("/memory/store")
@trace_span("MRAG.StoreMemory")
async def store_agent_memory(payload: MemoryStoreRequest):
    """
    Indexes long-term episodic memory for AgriOps Swarm agents in LanceDB.
    """
    mrag_orchestrator.index_document(
        collection_name="agent_memory",
        doc_id=f"mem-{payload.agent_id}-{int(pa.datetime.now().timestamp())}" if hasattr(pa, "datetime") else f"mem-{payload.agent_id}",
        text=payload.memory_text,
        metadata={
            "agent_id": payload.agent_id,
            **(payload.metadata or {})
        }
    )
    return {"status": "success", "message": "Agent memory registered in LanceDB"}

@router.get("/memory/retrieve")
@trace_span("MRAG.RecallMemory")
async def retrieve_agent_memory(agent_id: str, query: str, k: int = 3):
    """
    Recalls semantic memory tracks for a specific agent.
    """
    results = mrag_orchestrator.search_collection(
        collection_name="agent_memory",
        query=query,
        k=k,
        metadata_filter=f"agent_id = '{agent_id}'"
    )
    return {"memories": results}
