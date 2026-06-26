from backend.agents.base_agent import BaseAgent

class SecurityAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SecurityAgent",
            role="Security Architect",
            skills=["Security governance", "Risk modeling", "Policy enforcement"]
        )

class PenTestAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="PenTestAgent",
            role="Penetration Tester",
            skills=["Offensive testing", "Vulnerability exploitation", "Static analysis scan"]
        )

class ComplianceAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ComplianceAgent",
            role="Compliance Officer",
            skills=["Security compliance", "Standards auditing", "License checks"]
        )
