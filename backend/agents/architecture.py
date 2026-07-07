# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

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
