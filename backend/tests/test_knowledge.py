# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps Centralized Knowledge Service Verification Tests
"""

import sys
import os
import asyncio
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.main import app
from backend.database.session import SessionLocalSync
from backend.agriops.common.knowledge_service import knowledge_service
from backend.database.models import AuditLog

client = TestClient(app)


def test_unified_retrieval_routing():
    print("Running test_unified_retrieval_routing...")
    db = SessionLocalSync()
    try:
        # 1. Query a known agricultural term
        res = asyncio.run(
            knowledge_service.retrieve_unified_context(
                db=db,
                query="composting vermicompost guidelines",
                sensor_context={"moisture": 35},
            )
        )

        assert "query" in res
        assert "context" in res
        assert len(res["context"]) > 0
        assert res["latency_ms"] > 0

        # Verify that it got logged in AuditLog PostgreSQL table
        audit = (
            db.query(AuditLog)
            .filter(AuditLog.action == "KNOWLEDGE_RETRIEVAL")
            .order_by(AuditLog.timestamp.desc())
            .first()
        )
        assert audit is not None
        details = audit.details
        assert "latency_ms" in details
        print("[OK] test_unified_retrieval_routing passed")
    finally:
        db.close()


def test_live_search_fallback_on_low_confidence():
    print("Running test_live_search_fallback_on_low_confidence...")
    db = SessionLocalSync()
    try:
        # Query something obscure that won't be in standard FAISS/documents database
        res = asyncio.run(
            knowledge_service.retrieve_unified_context(
                db=db,
                query="exotic hyper-spectral space-agriculture weed variants",
                sensor_context={"moisture": 10},
            )
        )

        # Should fall back to live web search
        assert res["source_used"] == "live_web_search"
        assert len(res["context"]) > 0
        assert any(item["source"] == "live_web_search" for item in res["context"])
        print("[OK] test_live_search_fallback_on_low_confidence passed")
    finally:
        db.close()


def test_knowledge_api_endpoint():
    print("Running test_knowledge_api_endpoint...")
    response = client.post(
        "/api/knowledge/retrieve",
        json={
            "query": "drip irrigation intervals",
            "sensor_context": {"temperature": 32},
        },
    )
    assert response.status_code == 200
    res_data = response.json()
    assert "context" in res_data
    assert "source_used" in res_data
    print("[OK] test_knowledge_api_endpoint passed")


if __name__ == "__main__":
    print("--- AgriOps Knowledge Platform Integration Verification Starting ---")
    try:
        test_unified_retrieval_routing()
        test_live_search_fallback_on_low_confidence()
        test_knowledge_api_endpoint()
        print("All Knowledge Platform verification tests completed successfully!")
        sys.exit(0)
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
