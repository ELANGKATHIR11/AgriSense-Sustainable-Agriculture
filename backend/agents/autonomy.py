# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

from backend.agents.base_agent import BaseAgent


class BugFixAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="BugFixAgent",
            role="Bug Fix Specialist",
            skills=["Traceback analysis", "Root cause mapping", "Patch application"],
        )


class RefactorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="RefactorAgent",
            role="Code Refactoring Agent",
            skills=[
                "Dead code removal",
                "Architectural realignment",
                "Clean code patterns",
            ],
        )


class DependencyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DependencyAgent",
            role="Dependency Manager",
            skills=["Package upgrades", "Vulnerability mapping", "Conflict resolution"],
        )


class SelfHealingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SelfHealingAgent",
            role="Self-Healing System",
            skills=["Auto recovery", "Rollbacks", "Health validation"],
        )
