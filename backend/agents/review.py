from backend.agents.base_agent import BaseAgent


class CodeReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CodeReviewAgent",
            role="Code Reviewer",
            skills=["Static code analysis", "Linter compliance", "Readability checks"],
        )


class ArchitectureReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ArchitectureReviewAgent",
            role="Architecture Compliance Officer",
            skills=["Design conformity", "Strict layering", "Boundary validation"],
        )


class SecurityReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SecurityReviewAgent",
            role="Security Reviewer",
            skills=["CVE lookup", "Secrets scanning", "Sanitization reviews"],
        )


class PerformanceReviewAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="PerformanceReviewAgent",
            role="Performance Reviewer",
            skills=["Latency profiling", "SQL performance", "Memory leak checks"],
        )
