from backend.agents.base_agent import BaseAgent


class DataEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DataEngineerAgent",
            role="Data Engineer",
            skills=["ETL pipelines", "Feature stores", "Data pipelines"],
        )


class DataScientistAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DataScientistAgent",
            role="Data Scientist",
            skills=["Statistical analysis", "Experiment design", "Data modeling"],
        )


class DataValidationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DataValidationAgent",
            role="Data Validation Agent",
            skills=[
                "Dataset verification",
                "Anomalies detection",
                "Data quality checks",
            ],
        )


class DataVersioningAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DataVersioningAgent",
            role="Data Versioning Agent",
            skills=["DVC", "Dataset lifecycle", "Metadata management"],
        )
