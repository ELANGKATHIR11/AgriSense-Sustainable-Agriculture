# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps LanceDB MRAG Platform Verification Tests
"""

import sys
import os
import asyncio
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from backend.main import app
from backend.rag.mrag_orchestrator import mrag_orchestrator
from backend.vision.vrag_service import vrag_service

client = TestClient(app)


def test_lancedb_initialization():
    print("Running test_lancedb_initialization...")
    tables = mrag_orchestrator.get_table_names()
    required = [
        "documents",
        "images",
        "diseases",
        "crops",
        "weeds",
        "weather",
        "agent_memory",
    ]
    for req in required:
        assert req in tables
    print("[OK] test_lancedb_initialization passed")


def test_lancedb_text_search():
    print("Running test_lancedb_text_search...")
    # Add a mock document
    mrag_orchestrator.index_document(
        collection_name="documents",
        doc_id="doc-test-999",
        text="Organic composting works best with vermicompost and loamy soil structures.",
        metadata={"category": "organic"},
    )

    # Query it
    res = mrag_orchestrator.search_collection("documents", "vermicompost", k=1)
    assert len(res) > 0
    assert "vermicompost" in res[0]["text"]
    assert res[0]["metadata"]["category"] == "organic"
    print("[OK] test_lancedb_text_search passed")


def test_mrag_orchestrated_query():
    print("Running test_mrag_orchestrated_query...")
    context = mrag_orchestrator.get_orchestrated_mrag_context(
        query="Late Blight treatment guidelines", sensor_context={"soilMoisture": 42}
    )
    assert "retrieved_context" in context
    assert context["sources_count"] > 0
    print("[OK] test_mrag_orchestrated_query passed")


def test_vrag_search():
    print("Running test_vrag_search...")
    with pytest.raises(NotImplementedError):
        asyncio.run(
            vrag_service.search_similar_images(
                "dummy_base64_foliage_image", mode="disease"
            )
        )
    print("[OK] test_vrag_search passed (raised NotImplementedError)")


def test_agent_memory_store_retrieve():
    print("Running test_agent_memory_store_retrieve...")
    # Store memory
    mrag_orchestrator.index_document(
        collection_name="agent_memory",
        doc_id="mem-agent-999",
        text="Completed irrigation loop in field 4.",
        metadata={"agent_id": "agent-999"},
    )

    # Query memory
    res = mrag_orchestrator.search_collection(
        collection_name="agent_memory",
        query="irrigation loop",
        k=1,
        metadata_filter="agent_id = 'agent-999'",
    )
    assert len(res) > 0
    assert "irrigation" in res[0]["text"]
    print("[OK] test_agent_memory_store_retrieve passed")


def test_mrag_api_endpoints():
    print("Running test_mrag_api_endpoints...")

    # 1. Retrieve endpoint
    response = client.post(
        "/api/mrag/retrieve",
        json={"query": "soil moisture", "collection": "documents", "k": 1},
    )
    assert response.status_code == 200
    assert "results" in response.json()

    # 2. Query endpoint
    response = client.post(
        "/api/mrag/query",
        json={"query": "late blight on squash", "sensor_context": {"moisture": 38}},
    )
    assert response.status_code == 200
    assert "retrieved_context" in response.json()

    # 3. VRAG endpoint
    response = client.post(
        "/api/mrag/vrag", json={"imageBase64": "dummy_base_64", "mode": "disease"}
    )
    assert response.status_code == 200
    assert "matches" in response.json()
    print("[OK] test_mrag_api_endpoints passed")


if __name__ == "__main__":
    print("--- AgriOps LanceDB MRAG Verification Starting ---")
    try:
        test_lancedb_initialization()
        test_lancedb_text_search()
        test_mrag_orchestrated_query()
        test_vrag_search()
        test_agent_memory_store_retrieve()
        test_mrag_api_endpoints()
        print("All MRAG system verification tests completed successfully!")
        sys.exit(0)
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
