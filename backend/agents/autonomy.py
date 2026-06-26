from backend.agents.base_agent import BaseAgent

class BugFixAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="BugFixAgent",
            role="Bug Fix Specialist",
            skills=["Traceback analysis", "Root cause mapping", "Patch application"]
        )

class RefactorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="RefactorAgent",
            role="Code Refactoring Agent",
            skills=["Dead code removal", "Architectural realignment", "Clean code patterns"]
        )

class DependencyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DependencyAgent",
            role="Dependency Manager",
            skills=["Package upgrades", "Vulnerability mapping", "Conflict resolution"]
        )

class SelfHealingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SelfHealingAgent",
            role="Self-Healing System",
            skills=["Auto recovery", "Rollbacks", "Health validation"]
        )
