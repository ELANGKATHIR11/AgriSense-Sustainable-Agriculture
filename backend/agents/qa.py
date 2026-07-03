from backend.agents.base_agent import BaseAgent


class QAAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="QAAgent",
            role="QA Lead",
            skills=["Quality governance", "Test strategy", "Defect tracking"],
        )


class UnitTestAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="UnitTestAgent",
            role="Test Engineer",
            skills=["Unit testing", "Pytest", "Jest", "Code coverage"],
        )


class IntegrationTestAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="IntegrationTestAgent",
            role="Integration Test Engineer",
            skills=["System testing", "Integration tests", "API validation"],
        )


class E2ETestAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="E2ETestAgent",
            role="E2E Test Engineer",
            skills=["Playwright testing", "Selenium", "User flow automation"],
        )


class RegressionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="RegressionAgent",
            role="Regression Test Engineer",
            skills=["Stability testing", "Regression runs", "Snapshot testing"],
        )
