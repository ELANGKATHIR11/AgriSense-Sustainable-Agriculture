# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps Centralized Knowledge Platform Router
Exposes unified endpoints for all retrieval queries.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from backend.database.session import get_db
from backend.agriops.common.knowledge_service import knowledge_service

router = APIRouter(prefix="/knowledge", tags=["AgriOps Knowledge Platform"])


class KnowledgeQueryRequest(BaseModel):
    query: str
    sensor_context: Optional[dict] = None
    collections: Optional[List[str]] = None


@router.post("/retrieve")
async def retrieve_knowledge_context(
    payload: KnowledgeQueryRequest, db: Session = Depends(get_db)
):
    """
    Performs unified retrieval routing, context aggregation, and falls back to live web search on low confidence.
    """
    try:
        res = await knowledge_service.retrieve_unified_context(
            db=db,
            query=payload.query,
            sensor_context=payload.sensor_context,
            collections=payload.collections,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Knowledge retrieval error: {str(e)}"
        )
