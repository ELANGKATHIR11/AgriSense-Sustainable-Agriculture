# -*- coding: utf-8 -*-
import json
from datetime import datetime
from backend.rag.mrag_orchestrator import mrag_orchestrator

class ProjectMemory:
    def __init__(self, db_path=None):
        # db_path is accepted for backwards compatibility, but we use LanceDB
        pass

    def log_task(self, agent_name: str, task: str, result: dict):
        doc_id = f"mem-{agent_name}-{int(datetime.utcnow().timestamp())}"
        text_content = f"Agent: {agent_name}. Task: {task}. Result: {json.dumps(result)}"
        mrag_orchestrator.index_document(
            collection_name="agent_memory",
            doc_id=doc_id,
            text=text_content,
            metadata={
                "agent_name": agent_name,
                "task": task,
                "result": result
            }
        )

    def get_history(self, limit=10):
        try:
            tbl = mrag_orchestrator.db.open_table("agent_memory")
            # Retrieve all records
            rows = tbl.to_arrow().to_pylist()
            # Sort by timestamp descending
            rows.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            formatted = []
            for r in rows[:limit]:
                meta = json.loads(r.get("metadata", "{}"))
                formatted.append({
                    "agent": meta.get("agent_name"),
                    "task": meta.get("task"),
                    "result": meta.get("result"),
                    "time": r.get("timestamp")
                })
            return formatted
        except Exception:
            return []

memory_system = ProjectMemory()
