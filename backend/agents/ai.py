from backend.agents.base_agent import BaseAgent

class MLAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MLAgent",
            role="Machine Learning Engineer",
            skills=["Training pipelines", "Model development", "PyTorch", "Scikit-Learn"]
        )

class MLOpsAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MLOpsAgent",
            role="MLOps Engineer",
            skills=["Model deployment", "Monitoring", "Drift detection", "Retraining pipelines"]
        )

class LLMAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="LLMAgent",
            role="LLM Engineer",
            skills=["AgriGPT", "Prompt engineering", "RAG integrations"]
        )

class VLMAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="VLMAgent",
            role="VLM Engineer",
            skills=["Vision-language systems", "SmolVLM", "Florence-2"]
        )

class VisionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="VisionAgent",
            role="Computer Vision Engineer",
            skills=["Disease detection", "Weed detection", "YOLO", "OpenCV"]
        )

class DigitalTwinAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DigitalTwinAgent",
            role="Digital Twin Engineer",
            skills=["WaterTwin", "SoilTwin", "CropTwin", "Simulation", "Forecasting"]
        )

class AgenticAIAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="AgenticAIAgent",
            role="Agentic AI Engineer",
            skills=["Multi-agent systems", "Swarm intelligence", "Orchestrator development"]
        )
