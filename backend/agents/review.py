# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

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
