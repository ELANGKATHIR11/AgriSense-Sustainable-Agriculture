from backend.agents.base_agent import BaseAgent


class FullStackAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="FullStackAgent",
            role="Full Stack Engineer",
            skills=[
                "Python",
                "FastAPI",
                "React",
                "TypeScript",
                "Problem Solving",
                "End-to-End Implementation",
            ],
        )


class FrontendAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="FrontendAgent",
            role="Frontend Engineer",
            skills=[
                "React",
                "TypeScript",
                "Vite",
                "Tailwind CSS",
                "UI/UX",
                "Components",
                "Design Systems",
            ],
        )


class BackendAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="BackendAgent",
            role="Backend Engineer",
            skills=["Python", "FastAPI", "Services", "Middleware", "FastAPI APIs"],
        )


class APIAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="APIAgent",
            role="API Engineer",
            skills=[
                "Endpoint implementation",
                "API optimization",
                "OpenAPI validation",
            ],
        )


class DatabaseAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DatabaseAgent",
            role="Database Engineer",
            skills=["SQLAlchemy", "PostgreSQL", "Queries", "ORM", "Persistence"],
        )
