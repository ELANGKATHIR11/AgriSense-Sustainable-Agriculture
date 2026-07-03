# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AgriOps Layer & API Verification Tests - Standard Executable
"""

import sys
import os
import asyncio

# Add project root to sys.path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# 1. Initialize tables first before importing app to prevent seeding errors
from backend.database.base import Base
from backend.database.connection import sync_engine
from backend.database.session import SessionLocalSync


def setup_db():
    Base.metadata.create_all(bind=sync_engine)
    return SessionLocalSync()


db_session = setup_db()

# 2. Now import app and client
from fastapi.testclient import TestClient
from backend.main import app
from backend.database.models import (
    ModelRegistry,
)
from backend.agriops.common.event_bus import event_bus
from backend.agriops.telemetry.tracer import telemetry
from backend.agriops.dataops.service import dataops_service
from backend.agriops.mlops.service import mlops_service
from backend.agriops.llmops.service import llmops_service
from backend.agriops.agentops.service import agentops_service
from backend.agriops.aiops.service import aiops_service

client = TestClient(app)


def test_agriops_event_bus():
    print("Running test_agriops_event_bus...")
    events_log = []

    async def handler(event):
        events_log.append(event)

    event_bus.subscribe("TestEvent", handler)
    asyncio.run(event_bus.publish("TestEvent", {"data": "test_payload"}))

    assert len(events_log) == 1
    assert events_log[0].name == "TestEvent"
    assert events_log[0].payload["data"] == "test_payload"
    print("[OK] test_agriops_event_bus passed")


def test_telemetry_span_logging():
    print("Running test_telemetry_span_logging...")

    @telemetry.trace_span("TestSpan")
    def dummy_func():
        return "ok"

    res = dummy_func()
    assert res == "ok"
    traces = telemetry.get_traces()
    assert any(t["name"] == "TestSpan" for t in traces)
    print("[OK] test_telemetry_span_logging passed")


def test_dataops_validation_and_drift(db):
    print("Running test_dataops_validation_and_drift...")
    payload = {
        "nitrogen": 45,
        "phosphorus": 35,
        "potassium": 40,
        "temperature": 28.5,
        "humidity": 60.2,
        "ph": 6.3,
        "soil_moisture": 38.0,
    }
    res = asyncio.run(dataops_service.validate_and_ingest_metrics(db, payload))
    assert res["valid"] is True
    assert res["quality_score"] == 1.0
    print("[OK] test_dataops_validation_and_drift passed")


def test_mlops_champion_challenger(db):
    print("Running test_mlops_champion_challenger...")
    # Clean up pre-existing test rows to avoid state contamination
    db.query(ModelRegistry).filter(
        ModelRegistry.id.in_(["mr-test-01", "mr-test-02"])
    ).delete()
    db.commit()

    # Seed active model and staging model of type 'crop_recommendation'
    m1 = ModelRegistry(
        id="mr-test-01",
        name="Crop-Challenger-Test-v1",
        version="v1.0.0",
        type="crop_recommendation",
        framework="TabPFN",
        status="staging",
        accuracy=0.985,
        f1_score=0.982,
        prediction_count=0,
    )
    db.add(m1)
    m2 = ModelRegistry(
        id="mr-test-02",
        name="Crop-Champion-Test-v0",
        version="v0.9.0",
        type="crop_recommendation",
        framework="TabPFN",
        status="active",
        accuracy=0.950,
        f1_score=0.948,
        prediction_count=100,
    )
    db.add(m2)
    db.commit()

    eval_res = asyncio.run(
        mlops_service.evaluate_champion_challenger(db, "crop_recommendation")
    )
    assert eval_res["promotion_recommended"] is True

    promo_res = asyncio.run(mlops_service.promote_model(db, "mr-test-01"))
    assert promo_res["status"] == "success"

    db.refresh(m1)
    db.refresh(m2)
    assert m1.status == "active"
    assert m2.status == "archived"
    print("[OK] test_mlops_champion_challenger passed")


def test_llmops_document_indexing(db):
    print("Running test_llmops_document_indexing...")
    res = asyncio.run(
        llmops_service.index_document_chunk(
            db,
            name="SoilCompostingGuide.txt",
            content="Composting instructions for loamy soil settings.",
            vector_id="vec-test-123",
        )
    )
    assert res["status"] == "success"
    assert res["vector_id"] == "vec-test-123"
    print("[OK] test_llmops_document_indexing passed")


def test_agentops_task_running(db):
    print("Running test_agentops_task_running...")
    res = asyncio.run(
        agentops_service.run_agent_task(
            db,
            agent_name="AgriGPT-FieldWorker",
            task_title="Nitrogen Adjustment Task",
            input_details="Inject urea compost into fields 2 and 3.",
        )
    )
    assert res["status"] == "success"
    assert res["execution_time_ms"] > 0
    print("[OK] test_agentops_task_running passed")


def test_aiops_hardware_and_diagnostics(db):
    print("Running test_aiops_hardware_and_diagnostics...")
    metrics = aiops_service.get_hardware_metrics()
    assert "cpu_percentage" in metrics
    assert "gpu_vram_total_mb" in metrics

    alerts = asyncio.run(aiops_service.run_diagnostics(db))
    assert isinstance(alerts, list)
    print("[OK] test_aiops_hardware_and_diagnostics passed")


def test_agriops_overview_api():
    print("Running test_agriops_overview_api...")
    response = client.get("/api/agriops/overview")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "operational"
    assert "layer_aiops" in json_data
    assert "layer_dataops" in json_data
    assert "layer_mlops" in json_data
    assert "layer_llmops" in json_data
    assert "layer_agentops" in json_data
    print("[OK] test_agriops_overview_api passed")


if __name__ == "__main__":
    print("--- AgriOps Verification Suite Starting ---")
    try:
        test_agriops_event_bus()
        test_telemetry_span_logging()
        test_dataops_validation_and_drift(db_session)
        test_mlops_champion_challenger(db_session)
        test_llmops_document_indexing(db_session)
        test_agentops_task_running(db_session)
        test_aiops_hardware_and_diagnostics(db_session)
        test_agriops_overview_api()
        print("All test runs completed successfully!")
        sys.exit(0)
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
