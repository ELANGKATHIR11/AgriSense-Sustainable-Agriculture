# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps Router
Serves unified metrics and actions across all 5 layers to the frontend dashboard.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.agriops.common.event_bus import event_bus
from backend.agriops.telemetry.tracer import trace_span
from backend.agriops.dataops.service import dataops_service
from backend.agriops.mlops.service import mlops_service
from backend.agriops.llmops.service import llmops_service
from backend.agriops.agentops.service import agentops_service
from backend.agriops.aiops.service import aiops_service

router = APIRouter(prefix="/agriops", tags=["AgriOps Operations"])


@router.get("/overview")
@trace_span("AgriOps.GetOverview")
async def get_agriops_overview(db: Session = Depends(get_db)):
    """
    Returns consolidated dashboard parameters from all 5 operational layers.
    """
    try:
        hw = aiops_service.get_hardware_metrics()
        data_summary = dataops_service.get_dataset_registry_summary(db)
        ml_summary = mlops_service.get_performance_analytics(db)
        llm_summary = llmops_service.get_llm_overview(db)
        agent_summary = agentops_service.get_agent_swarm_overview(db)
        events = event_bus.get_history(20)

        return {
            "status": "operational",
            "layer_aiops": {"hardware": hw, "diagnostics": "healthy"},
            "layer_dataops": data_summary,
            "layer_mlops": ml_summary,
            "layer_llmops": llm_summary,
            "layer_agentops": agent_summary,
            "events_log": events,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to gather AgriOps overview: {str(e)}"
        )


@router.post("/diagnose")
async def run_platform_diagnostics(db: Session = Depends(get_db)):
    """
    Manually triggers AIOps platform diagnostics and processes self-healing loops if degraded.
    """
    alerts = await aiops_service.run_diagnostics(db)
    return {"status": "completed", "alerts_triggered": len(alerts), "alerts": alerts}


@router.get("/events")
async def get_events_history():
    """
    Fetches the operational event queue history.
    """
    return {"events": event_bus.get_history(100)}


@router.post("/mlops/promote")
async def promote_model_challenger(
    model_id: str = Body(..., embed=True), db: Session = Depends(get_db)
):
    """
    Promotes a model Challenger from staging to active.
    """
    res = await mlops_service.promote_model(db, model_id)
    if res.get("status") == "error":
        raise HTTPException(status_code=404, detail=res["message"])
    return res


@router.post("/agentops/run")
async def run_agentops_task(
    agent_name: str = Body(...),
    task_title: str = Body(...),
    description: str = Body(...),
    db: Session = Depends(get_db),
):
    """
    Dispatches a workspace task to the multi-agent bus.
    """
    res = await agentops_service.run_agent_task(db, agent_name, task_title, description)
    return res
