# backend/tools/agridb_tool.py
"""AgriDB tool stub.
Provides a simple interface for the DiseaseVisionAgent to query the
PostgreSQL AgriDB instance. In a real deployment you would use a proper
database driver (e.g., psycopg2) and secure credentials. This stub returns
placeholder data to keep the project offline‑only.
"""

from typing import Dict, Any

def query_agridb(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a simple query against the AgriDB.

    Expected ``payload`` keys:
        - ``query``: SQL string (SELECT only – safe placeholder).
        - ``params``: Optional tuple of parameters for the query.

    Returns a dictionary with a ``result`` list and a ``source`` tag.
    """
    # Placeholder implementation – no real DB connection.
    sql = payload.get("query", "SELECT 1")
    params = payload.get("params", ())
    # Simulate a result set.
    simulated_result = [{"column": "value", "sql": sql, "params": params}]
    return {"result": simulated_result, "source": "agridb_tool_stub"}

# Register name for Antigravity SDK.
query_agridb.name = "query_agridb"
