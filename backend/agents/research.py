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
