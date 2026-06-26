from backend.agents.base_agent import BaseAgent

class DevOpsAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DevOpsAgent",
            role="DevOps Lead",
            skills=["Infrastructure provisioning", "CI/CD pipelines", "Docker", "Kubernetes"]
        )

class SREAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="SREAgent",
            role="Site Reliability Engineer",
            skills=["Uptime enforcement", "Reliability metrics", "High availability design"]
        )

class MonitoringAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MonitoringAgent",
            role="Observability Engineer",
            skills=["Metrics logs", "Grafana dashboards", "Distributed tracing"]
        )

class IncidentResponseAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="IncidentResponseAgent",
            role="Incident Response Agent",
            skills=["Outage response", "Incident runbooks", "Post-mortem reports"]
        )
