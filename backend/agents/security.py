# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

from backend.agents.base_agent import BaseAgent


class SecurityAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SecurityAgent",
            role="Security Architect",
            skills=["Security governance", "Risk modeling", "Policy enforcement"],
        )


class PenTestAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="PenTestAgent",
            role="Penetration Tester",
            skills=[
                "Offensive testing",
                "Vulnerability exploitation",
                "Static analysis scan",
            ],
        )


class ComplianceAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="ComplianceAgent",
            role="Compliance Officer",
            skills=["Security compliance", "Standards auditing", "License checks"],
        )
