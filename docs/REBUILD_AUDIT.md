# AGRISENSE — SYSTEM REBUILD AUDIT & TRACEABILITY MATRIX

**Document ID:** `AUDIT-2026-10-21`  
**Target Specification:** `Backend.md`  
**Authoritative Frontend Source:** `src/App.tsx`, `src/api/client.ts`, `src/pages/*.tsx`

---

## 1. Actual Frontend Routes and API Contracts

All frontend communication routes through `src/api/client.ts` using `fetch()` against `/api/*` endpoints.

| Route / Method | Frontend Caller Method | Request Payload | Return Contract Type | Used in Page |
| :--- | :--- | :--- | :--- | :--- |
| `GET /api/sensors` | `agriApi.getSensors()` | None | `SensorsResponse` | Dashboard, IoT, Irrigation |
| `GET /api/status` | `agriApi.getStatus()` | None | `SystemStatusResponse` | App, Assistant |
| `POST /api/devices/:id/config` | `agriApi.updateDeviceConfig(id, payload)` | `{ samplingIntervalSec, moistureThresholdPct, calibrationOffsetPct }` | `{ success: boolean, device: DeviceTelemetry }` | IoT Monitoring |
| `GET /api/irrigation/status` | `agriApi.getSensors()` / direct | None | `IrrigationState` | Dashboard, Irrigation |
| `POST /api/irrigation/control` | `agriApi.controlIrrigation(payload)` | `{ zoneId, action: 'start'\|'stop'\|'toggle_auto', durationMin }` | `{ confirmed: boolean, zone: IrrigationZoneState, irrigation: IrrigationState }` | Irrigation |
| `POST /api/recommendations/crop` | `agriApi.recommendCrops(payload)` | `{ nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall }` | `{ recommendations: CropRecommendationItem[], modelSource: string }` | Irrigation |
| `POST /api/recommendations/fertilizer` | `agriApi.recommendFertilizer(payload)` | `{ cropType, nitrogen, phosphorus, potassium, soilPh, soilMoisture }` | `FertilizerResponse` | Irrigation |
| `POST /api/disease/analyze` | `agriApi.analyzeDisease(payload)` | `{ imageBase64?, mimeType?, cropHint?, samplePreset?, language }` | `DiseaseDiagnosisResult` | Disease Detection |
| `GET /api/analytics/digital-twin` | `agriApi.getDigitalTwin(zoneId, days)` | Query parameters: `?zoneId=...&days=...` | `DigitalTwinResponse` | Analytics Twin |
| `POST /api/assistant/chat` | `agriApi.askAssistant(payload)` | `{ message, language }` | `{ reply, languageUsed, isFallbackLanguage, sources, timestamp }` | Assistant |
| `GET /api/operations/overview` | `agriApi.getOperationsOverview()` | None | `OperationsOverviewResponse` | Operations Center |
| `POST /api/operations/settings` | `agriApi.updatePlatformSettings(payload)` | `Partial<PlatformSettings>` | `{ success: boolean, settings: PlatformSettings }` | Operations Center |
| `POST /api/operations/mlops/deploy` | `agriApi.deployMlModel(payload)` | `{ modelId, action: 'rollback'\|'promote' }` | `{ success: boolean, model: MlModelEntry, models: MlModelEntry[] }` | Operations Center |
| `POST /api/operations/hardware-command` | `agriApi.sendHardwareCommand(payload)` | `{ deviceId, command: 'wake_ping'\|'zero_calibrate'\|'gateway_selftest' }` | `{ success: boolean, device: DeviceTelemetry, timestamp: string }` | Operations Center |
| `POST /api/operations/dataops/validate` | `agriApi.runDataOpsValidation()` | None | `{ success: boolean, checkedNodes, staleNodes, timestamp }` | Operations Center |

---

## 2. Frontend Data Dependencies & State Lifecycle

1. **Global App State (`src/App.tsx`)**:
   - `sensorsData: SensorsResponse | null`: Root telemetry object polled every 30 seconds.
   - `systemStatus: SystemStatusResponse | null`: Service health indicators.
   - `darkMode: boolean`: Stored in `localStorage.agrisense_theme`.
   - `activePage: PageRoute`: Synchronized with `window.location.pathname` and navigation clicks.
2. **Component Level Data**:
   - `Farm3DScene.tsx`: Consumes `devices: DeviceTelemetry[]`, `zones: IrrigationZoneState[]`, and optional `twinSeries: DigitalTwinSeriesPoint[]`.
   - `RootZone3DProfile.tsx`: Consumes single `zone: IrrigationZoneState` and NPK numeric concentrations.
   - `TelemetryChart.tsx`: Consumes generic `TelemetryChartPoint[]` array.

---

## 3. Existing Mocks, Hardcoded Data & Stubs Identified in `server.ts`

| Component / Functionality | Current Implementation in `server.ts` | Target Rebuild Implementation |
| :--- | :--- | :--- |
| **Telemetry Ingestion** | In-memory `devices` array with randomized delta jitter | PostgreSQL `sensor_readings` hyper-table / table + real MQTT ingestion |
| **Telemetry History** | Statically generated `history24h` array | Real time-bucketed aggregation queries in PostgreSQL |
| **Irrigation Control** | Directly mutates memory object and appends mock audit log | Hardware command queue + acknowledgement state machine + DB audit log |
| **Crop Recommendations** | Hardcoded heuristics in JavaScript pretending to be Random Forest | Scikit-learn / XGBoost pipeline trained on `unified_training_dataset.csv` |
| **Disease Diagnosis** | Calls remote Google Gemini API, with hardcoded rule fallback | Local Vision Model (ResNet/DeiT or PyTorch/ONNX PlantVillage) + local rules |
| **Digital Twin** | Synthesized formula in loop without empirical model weights | Penman-Monteith ET0 + Richards equation soil-water balance solver in Python |
| **AI Assistant** | Calls remote Google Gemini 2.5 Flash API | Local RAG pipeline (Ollama / Qwen2.5 / Phi-3 + pgvector / FAISS + BM25) |
| **Operations Center** | In-memory lists of models, datasets, and hardcoded CI/CD records | Real database-backed model registry, dataset validator, and system metrics |

---

## 4. Existing Models and Compatibility

1. **Tabular Agricultural Models**:
   - Provenance dataset: `datasets/raw/unified_training_dataset.csv` (8,850 records, 59 Indian crop varieties with N, P, K, pH, rainfall, temperature, humidity, and water requirements).
   - Training pipeline: Train reproducible Random Forest & LightGBM/XGBoost models in Python using `scikit-learn` and save artifacts to `models/artifacts/crop_recommendation_rf.joblib`.
2. **Plant Pathology Model**:
   - Offline Computer Vision architecture: Torchvision ResNet50 / MobileNetV3 feature extractor or fine-tuned PlantVillage classifier.
   - Structured fallback: Agronomic IPM knowledge base with diagnostic rule engine.
3. **Local LLM / RAG Stack**:
   - Local LLM: Ollama HTTP integration (`http://localhost:11434`) targeting `qwen2.5-coder:3b` or `phi3:mini`.
   - Vector store: Local FAISS index / SQLite / PostgreSQL pgvector with BM25 hybrid ranking.

---

## 5. Available Datasets & Provenance

1. `unified_training_dataset.csv`:
   - Size: 2,169,769 bytes (8,850 rows, 27 columns).
   - Features: `crop`, `crop_type`, `season`, `N`, `P`, `K`, `pH`, `rainfall`, `temperature`, `humidity`, `soil_moisture`, `ndvi`, `irrigation_method`, `yield_target`, `water_target`, `area`.
   - Provenance: Indian Agricultural Research Institute (ICAR) & state agricultural universities agronomic trial aggregates.
2. `leafSpecimens.ts`:
   - High-fidelity SVG vector specimens for *Alternaria solani* (Early Blight), *Xanthomonas oryzae* (Bacterial Blight), and *Oryza sativa* healthy control.

---

## 6. Architecture & Integration Plan

```
┌────────────────────────────────────────────────────────────────┐
│                       React 19 Frontend Shell                  │
│                     (Vite / Port 3000 / 5173)                  │
└───────────────────────────────┬────────────────────────────────┘
                                │ HTTP /api/*
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                    Node.js Express Gateway                     │
│                  (server.ts / Port 8000/5000)                  │
│  - Serves static assets & Vite integration                     │
│  - Gateway rate-limiting, auth headers, request correlation IDs│
│  - Reverse-proxies ML, RAG, and Digital Twin calls to FastAPI  │
│  - Exposes honest hardware status                              │
└───────────────────────────────┬────────────────────────────────┘
                                │ Internal HTTP (Port 8004)
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                   Python 3.11/3.12 FastAPI                     │
│                       (Port 8004)                              │
│  - Agricultural ML Pipelines (Scikit-learn / PyTorch)          │
│  - Agronomic Calculation Engines (Fertilizer Deficit, IPM)     │
│  - Biophysical Digital Twins (WaterTwin, SoilTwin, CropTwin)   │
│  - Local RAG & Document Retrieval Engine                       │
│  - 5 Ops Disciplines (MLOps, DataOps, LLMOps, AIOps, AgentOps) │
│  - Database Engine (SQLAlchemy 2.x + PostgreSQL / SQLite)      │
└────────────────────────────────────────────────────────────────┘
```
