from backend.agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            role="Planner",
            skills=[
                "Execution planning",
                "Strategy formulation",
                "Timeline estimation",
            ],
        )


class TaskDecomposerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="TaskDecomposerAgent",
            role="Task Decomposer",
            skills=["Decomposition", "Dependency analysis", "Task scheduling"],
        )


class WorkflowAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="WorkflowAgent",
            role="Workflow Coordinator",
            skills=["Workflow design", "Automation", "State transition management"],
        )
