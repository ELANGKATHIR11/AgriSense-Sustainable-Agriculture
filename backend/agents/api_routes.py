from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from backend.orchestrator.swarm import orchestrator
from backend.memory.project_memory import memory_system

# Import all agents
from backend.agents.executive import CEOAgent, CTOAgent, COOAgent, ProgramManagerAgent
from backend.agents.planning import PlannerAgent, TaskDecomposerAgent, WorkflowAgent
from backend.agents.architecture import ArchitectAgent, SolutionArchitectAgent, APIArchitectAgent, DatabaseArchitectAgent
from backend.agents.developer import FullStackAgent, FrontendAgent, BackendAgent, APIAgent, DatabaseAgent
from backend.agents.ai import MLAgent, MLOpsAgent, LLMAgent, VLMAgent, VisionAgent, DigitalTwinAgent, AgenticAIAgent
from backend.agents.data import DataEngineerAgent, DataScientistAgent, DataValidationAgent, DataVersioningAgent
from backend.agents.qa import QAAgent, UnitTestAgent, IntegrationTestAgent, E2ETestAgent, RegressionAgent
from backend.agents.security import SecurityAgent, PenTestAgent, ComplianceAgent
from backend.agents.operations import DevOpsAgent, SREAgent, MonitoringAgent, IncidentResponseAgent
from backend.agents.iot import IoTAgent, MQTTAgent, EdgeAIAgent
from backend.agents.knowledge import DocumentationAgent, RAGAgent, MemoryAgent
from backend.agents.autonomy import BugFixAgent, RefactorAgent, DependencyAgent, SelfHealingAgent
from backend.agents.research import WebResearchAgent, TechnologyScoutAgent, CompetitorAnalysisAgent
from backend.agents.review import CodeReviewAgent, ArchitectureReviewAgent, SecurityReviewAgent, PerformanceReviewAgent

router = APIRouter(tags=["ASO Multi-Agent System"])

# Register all ASO agents on load
agents_to_register = [
    # Executive
    CEOAgent(), CTOAgent(), COOAgent(), ProgramManagerAgent(),
    # Planning
    PlannerAgent(), TaskDecomposerAgent(), WorkflowAgent(),
    # Architecture
    ArchitectAgent(), SolutionArchitectAgent(), APIArchitectAgent(), DatabaseArchitectAgent(),
    # Development
    FullStackAgent(), FrontendAgent(), BackendAgent(), APIAgent(), DatabaseAgent(),
    # AI
    MLAgent(), MLOpsAgent(), LLMAgent(), VLMAgent(), VisionAgent(), DigitalTwinAgent(), AgenticAIAgent(),
    # Data
    DataEngineerAgent(), DataScientistAgent(), DataValidationAgent(), DataVersioningAgent(),
    # QA
    QAAgent(), UnitTestAgent(), IntegrationTestAgent(), E2ETestAgent(), RegressionAgent(),
    # Security
    SecurityAgent(), PenTestAgent(), ComplianceAgent(),
    # Operations
    DevOpsAgent(), SREAgent(), MonitoringAgent(), IncidentResponseAgent(),
    # IoT
    IoTAgent(), MQTTAgent(), EdgeAIAgent(),
    # Knowledge
    DocumentationAgent(), RAGAgent(), MemoryAgent(),
    # Autonomy
    BugFixAgent(), RefactorAgent(), DependencyAgent(), SelfHealingAgent(),
    # Research
    WebResearchAgent(), TechnologyScoutAgent(), CompetitorAnalysisAgent(),
    # Review
    CodeReviewAgent(), ArchitectureReviewAgent(), SecurityReviewAgent(), PerformanceReviewAgent()
]

for agent in agents_to_register:
    orchestrator.register_agent(agent)

@router.on_event("startup")
async def startup_event():
    orchestrator.start()

# ── API Models ───────────────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task: str

class StoreMemoryRequest(BaseModel):
    agent_name: str
    task: str
    result: Dict[str, Any]


# ── Core Endpoints ───────────────────────────────────────────────────────────

@router.post("/agents/task")
async def submit_task(req: TaskRequest):
    """
    Queue a task to be processed asynchronously by the swarm orchestrator.
    """
    return await orchestrator.submit_task(req.task)

@router.get("/agents/status")
async def get_status():
    """
    Get the status of the Swarm Orchestrator queue.
    """
    return {
        "is_running": orchestrator.is_running,
        "queue_size": orchestrator.task_queue.qsize(),
        "active_workflows": len(orchestrator.active_workflows)
    }

@router.get("/agents/list")
async def list_agents():
    """
    List all registered specialist agents, roles, and current statuses.
    """
    return [
        {
            "id": agent.agent_id,
            "name": name,
            "role": agent.role,
            "skills": agent.skills,
            "status": agent.status
        }
        for name, agent in orchestrator.agents.items()
    ]

@router.get("/agents/history")
async def get_history(limit: int = 15):
    """
    Get the historic swarm workflow runs from SQLite.
    """
    return memory_system.get_history(limit=limit)

@router.post("/swarm/execute")
async def execute_swarm(req: TaskRequest):
    """
    Synchronously trigger a full ASO software engineering swarm workflow.
    """
    return await orchestrator.execute_swarm_pipeline(req.task)

# ── Memory Endpoints ─────────────────────────────────────────────────────────

@router.get("/memory/search")
async def search_memory(query: str = Query(..., description="Query string for semantic/keyword retrieval")):
    """
    Search agent execution memory logs in SQLite.
    """
    history = memory_system.get_history(limit=100)
    filtered = []
    q = query.lower()
    for h in history:
        if q in h["task"].lower() or q in h["agent"].lower() or q in json.dumps(h["result"]).lower():
            filtered.append(h)
    return {"query": query, "matches": filtered}

@router.post("/memory/store")
async def store_memory(req: StoreMemoryRequest):
    """
    Manually log a custom action or fact to the shared memory log.
    """
    memory_system.log_task(req.agent_name, req.task, req.result)
    return {"message": "Memory logged successfully"}

# ── Observability Endpoints ──────────────────────────────────────────────────

@router.get("/metrics")
async def get_metrics():
    """
    Retrieve operational and performance metrics for the ASO multi-agent swarm.
    """
    history = memory_system.get_history(limit=100)
    total_tasks = len(history)
    success_count = sum(1 for h in history if "failed" not in str(h["result"]).lower())
    
    # Calculate mock/simulated costs or token usage for monitoring
    mock_tokens = total_tasks * 4200
    mock_cost = round(mock_tokens * 0.0000015, 4) # Edge Ollama is free, but this mimics cloud telemetry
    
    # Calculate latencies
    latencies = []
    for h in history:
        res = h.get("result", {})
        if isinstance(res, dict) and "steps" in res:
            # Swarm pipeline
            try:
                start = datetime.fromisoformat(res["started_at"].replace("Z", ""))
                end = datetime.fromisoformat(res["completed_at"].replace("Z", ""))
                latencies.append((end - start).total_seconds())
            except Exception:
                pass
    
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.85
    
    return {
        "aso_telemetry": {
            "total_runs": total_tasks,
            "success_rate": round((success_count / total_tasks * 100), 2) if total_tasks > 0 else 100.0,
            "average_pipeline_latency_seconds": avg_latency,
            "total_simulated_tokens": mock_tokens,
            "simulated_cloud_cost_saved_usd": mock_cost,
            "online_agents": len(orchestrator.agents)
        }
    }
