# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

from backend.agents.base_agent import BaseAgent


class WebResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="WebResearchAgent",
            role="Web Research Specialist",
            skills=["Web mining", "Knowledge aggregation", "Online scraping"],
        )


class TechnologyScoutAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="TechnologyScoutAgent",
            role="Technology Scout",
            skills=[
                "New tools discovery",
                "Framework benchmarking",
                "Feasibility reviews",
            ],
        )


class CompetitorAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CompetitorAnalysisAgent",
            role="Market Analyst",
            skills=["Market research", "Competitive audits", "Feature comparison"],
        )
