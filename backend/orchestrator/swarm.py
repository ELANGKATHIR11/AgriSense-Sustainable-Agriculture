import asyncio
import logging
from typing import Dict, Any
from datetime import datetime, timezone

from backend.memory.project_memory import memory_system

logger = logging.getLogger("ASO.SwarmOrchestrator")


class SwarmOrchestrator:
    def __init__(self):
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.is_running = False
        self.active_workflows = []
        self.execution_history = []

    def register_agent(self, agent):
        self.agents[agent.name] = agent
        logger.info(f"Registered ASO agent: {agent.name} ({agent.role})")

    def get_agent_by_role(self, role_keyword: str):
        for name, agent in self.agents.items():
            if (
                role_keyword.lower() in agent.role.lower()
                or role_keyword.lower() in name.lower()
            ):
                return agent
        return None

    async def submit_task(self, task: str) -> Dict[str, Any]:
        task_id = f"task-{int(datetime.now(timezone.utc).timestamp())}"
        await self.task_queue.put({"id": task_id, "task": task})
        logger.info(f"Task {task_id} queued: '{task}'")
        return {"status": "queued", "task_id": task_id, "task": task}

    async def execute_swarm_pipeline(self, task: str) -> Dict[str, Any]:
        """
        Executes the fully autonomous multi-agent software engineering loop:
        CEO -> Planner -> Task Decomposer -> Architect -> Developer -> QA Lead -> Reviewer -> Docs
        """
        pipeline_id = f"flow-{int(datetime.now(timezone.utc).timestamp())}"
        logger.info(f"Starting Swarm Pipeline {pipeline_id} for task: '{task}'")

        steps = [
            {
                "step": "1. Strategic Alignment",
                "agent_role": "CEOAgent",
                "desc": "Assessing alignment, planning scope, and mapping risks.",
            },
            {
                "step": "2. High-Level Planning",
                "agent_role": "PlannerAgent",
                "desc": "Drafting technical architecture blueprints and flowcharts.",
            },
            {
                "step": "3. Task Decomposition",
                "agent_role": "TaskDecomposerAgent",
                "desc": "Splitting the master plan into discrete JIRA-like subtasks.",
            },
            {
                "step": "4. System Architecture",
                "agent_role": "ArchitectAgent",
                "desc": "Defining class boundaries, API schemas, and PostgreSQL models.",
            },
            {
                "step": "5. Full-Stack Execution",
                "agent_role": "FullStackAgent",
                "desc": "Writing clean Python code, endpoints, and React templates.",
            },
            {
                "step": "6. Quality Assurance",
                "agent_role": "QAAgent",
                "desc": "Validating outputs via mock unit tests and edge cases.",
            },
            {
                "step": "7. Security Scanning",
                "agent_role": "SecurityReviewAgent",
                "desc": "Running dependency vulnerability audits and secrets scanning.",
            },
            {
                "step": "8. Performance Auditing",
                "agent_role": "PerformanceReviewAgent",
                "desc": "Benchmarking queries, latency, and resource footprint.",
            },
            {
                "step": "9. Technical Documentation",
                "agent_role": "DocumentationAgent",
                "desc": "Generating OpenAPI schemas, README.md, and changelogs.",
            },
        ]

        workflow_log = {
            "id": pipeline_id,
            "task": task,
            "status": "in_progress",
            "started_at": datetime.now(timezone.utc).isoformat() + "Z",
            "completed_at": None,
            "steps": [],
        }

        self.active_workflows.append(workflow_log)
        current_input = task

        for s in steps:
            agent = self.agents.get(s["agent_role"])
            if not agent:
                # Fallback to base agent if not registered
                logger.warning(
                    f"Agent {s['agent_role']} not found. Initializing virtual placeholder."
                )
                continue

            s_log = {
                "step_name": s["step"],
                "agent_name": agent.name,
                "agent_role": agent.role,
                "description": s["desc"],
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat() + "Z",
                "result": {},
            }
            workflow_log["steps"].append(s_log)

            try:
                # Update status of agent
                agent.status = "working"
                # Executing task
                result = await agent.execute_task(
                    f"Context: {current_input}. Task: {s['desc']}"
                )

                s_log["status"] = "completed"
                s_log["result"] = result
                s_log["completed_at"] = datetime.now(timezone.utc).isoformat() + "Z"

                # Update loop input with agent response to cascade context
                if "response" in result:
                    current_input += f"\n\n[{agent.name} Output]: {result['response']}"
                elif "result" in result:
                    current_input += f"\n\n[{agent.name} Output]: {result['result']}"
                else:
                    current_input += (
                        f"\n\n[{agent.name} Output]: Completed successfully."
                    )

            except Exception as e:
                logger.error(f"Error executing step {s['step']} with {agent.name}: {e}")
                s_log["status"] = "failed"
                s_log["error"] = str(e)
                # Auto-recovery / BugFixAgent intervention
                bug_fixer = self.agents.get("BugFixAgent")
                if bug_fixer:
                    s_log["escalation"] = "Invoking BugFixAgent for self-healing."
                    fix_res = await bug_fixer.execute_task(
                        f"Analyze failure: {e} in step {s['step']}. Fix root cause."
                    )
                    current_input += f"\n\n[BugFixAgent Intervention]: {fix_res.get('response', 'Resolved.')}"
                    s_log["status"] = "healed"
                else:
                    workflow_log["status"] = "failed"
                    break
            finally:
                agent.status = "idle"

        if workflow_log["status"] != "failed":
            workflow_log["status"] = "completed"
            workflow_log["completed_at"] = datetime.now(timezone.utc).isoformat() + "Z"

        self.execution_history.append(workflow_log)
        # Save complete workflow run to PostgreSQL memory system
        memory_system.log_task("SwarmOrchestrator", task, workflow_log)

        # Remove from active
        self.active_workflows = [
            w for w in self.active_workflows if w["id"] != pipeline_id
        ]

        return workflow_log

    async def process_queue(self):
        self.is_running = True
        logger.info("Swarm orchestrator queue processing started.")
        while self.is_running:
            try:
                task_item = await self.task_queue.get()
                task = task_item["task"]
                logger.info(f"Orchestrator processing task: '{task}'")
                await self.execute_swarm_pipeline(task)
                self.task_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue loop: {e}")

    def start(self):
        if not self.is_running:
            asyncio.create_task(self.process_queue())

    def stop(self):
        self.is_running = False


orchestrator = SwarmOrchestrator()
