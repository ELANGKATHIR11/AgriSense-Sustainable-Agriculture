from backend.agents.base_agent import BaseAgent

class CEOAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CEOAgent",
            role="Chief Executive Officer",
            skills=["Strategic planning", "Task delegation", "Decision making", "Risk assessment", "Mission alignment"]
        )

class CTOAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CTOAgent",
            role="Chief Technology Officer",
            skills=["Architecture design", "Technology selection", "Code review", "Standards governance"]
        )

class COOAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="COOAgent",
            role="Chief Operating Officer",
            skills=["Workflow management", "Agent coordination", "Resource allocation"]
        )

class ProgramManagerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ProgramManagerAgent",
            role="Program Manager",
            skills=["Roadmaps", "Sprint planning", "Milestones", "Progress tracking"]
        )
