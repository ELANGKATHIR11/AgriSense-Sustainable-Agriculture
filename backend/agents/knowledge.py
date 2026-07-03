from backend.agents.base_agent import BaseAgent


class DocumentationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DocumentationAgent",
            role="Documentation Specialist",
            skills=[
                "API docs",
                "Readmes",
                "Markdown documentation",
                "Technical writing",
            ],
        )


class RAGAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="RAGAgent",
            role="RAG Specialist",
            skills=[
                "LanceDB indexing",
                "Vector embeddings",
                "Semantic retrieval",
                "Context loading",
            ],
        )


class MemoryAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MemoryAgent",
            role="Memory Manager",
            skills=["Long term memory", "Session histories", "Episodic retrieval"],
        )
