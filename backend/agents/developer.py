# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

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
