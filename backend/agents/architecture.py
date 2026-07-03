from backend.agents.base_agent import BaseAgent


class ArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ArchitectAgent",
            role="Chief Architect",
            skills=["System architecture", "Design review", "Dependency governance"],
        )


class SolutionArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SolutionArchitectAgent",
            role="Solution Architect",
            skills=["Enterprise architecture", "System boundaries", "API contracts"],
        )


class APIArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="APIArchitectAgent",
            role="API Architect",
            skills=["API governance", "OpenAPI standards", "Contract validation"],
        )


class DatabaseArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DatabaseArchitectAgent",
            role="Database Architect",
            skills=["Schema design", "Query optimization", "Migration planning"],
        )
