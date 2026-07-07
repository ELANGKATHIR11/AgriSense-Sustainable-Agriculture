# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

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
