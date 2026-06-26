import sqlite3
import json
import os

class ProjectMemory:
    def __init__(self, db_path="agents_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memory_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT,
                    task TEXT,
                    result TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def log_task(self, agent_name: str, task: str, result: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memory_log (agent_name, task, result) VALUES (?, ?, ?)",
                (agent_name, task, json.dumps(result))
            )

    def get_history(self, limit=10):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT agent_name, task, result, timestamp FROM memory_log ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [{"agent": row[0], "task": row[1], "result": json.loads(row[2]), "time": row[3]} for row in cursor.fetchall()]

memory_system = ProjectMemory()
