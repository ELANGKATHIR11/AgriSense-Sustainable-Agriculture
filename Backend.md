# AGRISENSE — COMPLETE OFFLINE BACKEND REBUILD MASTER PROMPT

## ROLE

Act as a principal software architect, senior Python/FastAPI engineer, Node.js/Express engineer, agricultural ML researcher, data engineer, MLOps engineer, LLMOps engineer, AIOps engineer, AgentOps engineer, IoT/edge engineer, PostgreSQL database architect, security engineer, and QA automation lead.

Your assignment is to reconstruct the entire AgriSense backend from scratch while preserving and fully integrating the existing new React frontend.

This is an implementation task, not a proposal, demonstration, prototype, or code-generation exercise that stops at scaffolding. Inspect the repositories, write the actual code, install and validate dependencies where the environment permits, run tests, fix defects, and leave the project in a reproducible working state.

## 1. SOURCE MATERIALS AND AUTHORITY

Inspect these source archives completely:

1. `Agrisense_Complete_Project_Analysis.zip`
2. `agrisense-—-smart-agriculture-&-iot-platform(1).zip`

Also inspect the available project documentation:

- `AGRISENSE_BLUEPRINT1.md`
- `AGRIOS_MANIFEST.md`
- `PROJECT_BLUEPRINT.md`

Use the old project-analysis archive to understand the existing architecture, API specifications, trained models, datasets, feature traceability, security issues, and known defects.

Use the new frontend ZIP as the authoritative source for the UI that must work.

Inspect, at minimum:

- `src/App.tsx`
- `src/api/client.ts`
- All seven page components under `src/pages/`
- All components, hooks, utilities, i18n resources, and device integrations
- `package.json`, `vite.config.ts`, `server.ts`, and `.env.example`
- Existing forms, chart inputs, response interfaces, action buttons, and error states

Do not assume that documentation accurately describes the current implementation. Verify claims against source files, model artifacts, datasets, tests, and executable behavior.

Create `docs/REBUILD_AUDIT.md` documenting:
- Actual frontend routes and API contracts.
- All frontend data dependencies.
- Existing mocks, hardcoded data, stubs, and simulated operations.
- Existing models and their compatibility.
- Available datasets and their provenance.
- Broken imports, circular dependencies, and missing assets.
- Features that can be verified as functional.
- Features that require new data, hardware, model weights, or credentials.

Do not copy the old backend wholesale. Reimplement its useful functionality in a clean architecture and reuse existing assets only after verification.

## 2. NON-NEGOTIABLE REQUIREMENTS

1. No mocked production data.
2. No hardcoded fake sensor telemetry, fabricated model metrics, synthetic production audit events, fabricated hardware acknowledgements, or preset AI responses.
3. No random values masquerading as measurements or model predictions.
4. No cloud API calls, cloud-hosted LLMs, remote embedding services, or internet-dependent inference.
5. No placeholder API handlers returning successful responses without performing the requested operation.
6. No buttons that report success without an actual state transition or verified hardware acknowledgement.
7. No empty CRUD handlers, TODO-only business logic, fake model training, or simulated drift detection presented as real.
8. Every displayed metric must come from a database query, actual runtime measurement, actual model evaluation, a documented calculation, or an explicitly identified forecast.
9. If a required model, dataset, broker, device, or local LLM is unavailable, report the actual unavailable state. Return a suitable error or clearly labeled degraded response; never fabricate a successful result.
10. Every feature must have automated tests, meaningful validation, structured logs, and an auditable operational state.
11. No secrets committed to source control.
12. Do not claim the entire application is working until the acceptance tests actually pass.

A development-only fixture loader may exist for testing, but it must be disabled by default, isolated from production, explicitly enabled, and visibly identified as test data.

## 3. TARGET ARCHITECTURE

Implement a locally runnable, service-oriented modular monolith with clear boundaries.

### Python ML and agronomic service

- Python 3.12.x
- FastAPI
- Pydantic v2
- SQLAlchemy 2.x
- Alembic migrations
- psycopg 3 or asyncpg, selected consistently for database access
- NumPy, pandas, scikit-learn, joblib, XGBoost, PyTorch and compatible computer-vision dependencies as required by verified artifacts
- Pytest, HTTPX, Ruff, Black and static type checking

Responsibilities:
- Agricultural inference and recommendations
- Sensor normalization and validation
- Digital-twin simulations
- Dataset ingestion and validation
- Model training, evaluation, registry integration and deployment
- Local LLM inference and RAG
- Agent orchestration
- Agricultural decision logic
- Data quality and anomaly detection

Run FastAPI on configurable localhost port `8004` by default.

### Node.js and Express gateway

Use a Node.js LTS release compatible with the project dependencies and Express.

Responsibilities:
- Serve the existing frontend build.
- Provide the `/api/*` routes consumed by `src/api/client.ts`.
- Validate and normalize gateway requests.
- Apply authentication, authorization, rate limits, request IDs, payload limits and security headers.
- Forward Python inference requests to FastAPI.
- Handle SSE or WebSocket connections when required.
- Return consistent JSON errors.
- Maintain no independent in-memory copy of authoritative business state.

The Express gateway is an actual service, not a collection of fabricated API responses.

### PostgreSQL

Use PostgreSQL as the authoritative operational database.

Support local development with Docker Compose or a native local PostgreSQL installation.

Use:
- SQLAlchemy models and Alembic migrations.
- UUID primary keys where appropriate.
- UTC timestamps.
- Numeric types suitable for agricultural measurements.
- Foreign keys and explicit transaction boundaries.
- Indexes for time-series queries and common filters.
- Connection pooling and bounded query timeouts.
- Parameterized queries.
- Database constraints and idempotency keys.
- Database-backed audit trails.
- A documented backup and restoration process.

Never silently switch to in-memory persistence or SQLite if PostgreSQL fails. Report database unavailability accurately.

### Local infrastructure

Support fully offline runtime services:

- PostgreSQL
- Local MQTT broker such as Mosquitto
- Local Ollama or another supported local model runtime
- Local embedding models
- FAISS or PostgreSQL with pgvector for retrieval
- Local task execution through a durable PostgreSQL-backed queue, or locally configured Redis/Celery if justified
- Prometheus-compatible metrics and local log collection

Do not require Redis, Docker, a GPU, or an MQTT broker for every development command. Make optional services explicit and isolate their dependency boundaries.

## 4. FULLY OFFLINE OPERATION

Implement two distinct offline guarantees.

### Runtime offline

After dependencies, model weights, and datasets have been acquired, the application must start and operate without internet access.

Enforce this by:
- Binding services to configurable local addresses.
- Using local model directories.
- Setting appropriate Hugging Face offline environment flags.
- Disabling telemetry and cloud-provider integrations.
- Preventing automatic model downloads.
- Prohibiting external network access from inference and RAG code.
- Using only local database, broker, and model-runtime endpoints.
- Disabling automatic external weather requests and replacing them with locally ingested weather observations, cached verified data, or an explicit unavailable state.

### Initial provisioning

Document a separate provisioning process for acquiring dependencies, pretrained models, and permitted agricultural datasets before disconnecting from the internet.

Verify model checksums, licenses, dataset provenance, version compatibility, and required storage.

Do not claim that an undownloaded model is available offline.

Remove the mandatory Gemini SDK dependency from the assistant's inference path. Do not silently fall back to Gemini, DeepSeek, or another cloud service.

Provide a configurable local LLM adapter with:
- Local model discovery.
- Model health checks.
- Context and generation limits.
- Token streaming where supported.
- Explicit model loading and unloading.
- CPU fallback where supported.
- Memory-aware model scheduling.
- Structured errors for missing weights or insufficient resources.

## 5. IMPLEMENT THE EXISTING FRONTEND CONTRACTS

Preserve the frontend layout, styling, routes, i18n resources, charts, 3D components, and existing user workflows unless a change is necessary to correct a documented defect.

Treat `src/api/client.ts` as the initial API contract. Extend it only through coordinated frontend and backend changes.

Implement the following endpoints and preserve their TypeScript-compatible response shapes.

### A. Dashboard and telemetry

`GET /api/sensors`

Return the `SensorsResponse` interface, including:
- Summary metrics.
- Device telemetry.
- Sensor history for the previous 24 hours.
- Weather summary with explicit provenance and freshness.
- Crop-health summary derived from actual data.
- Tank and irrigation state.
- Last successful data timestamp.

Calculate summaries from PostgreSQL. Do not return static device arrays or synthetic history.

`GET /api/status`

Return real application, database, MQTT, edge-gateway, and local AI service status. Distinguish configured, connected, available, degraded, and unavailable states.

Add the necessary internal endpoints for live snapshots, recent sensor readings, alerts, health probes, and dashboard aggregation.

### B. Device configuration and IoT

`POST /api/devices/:id/config`

Persist sampling intervals, moisture thresholds, and calibration offsets. Validate ranges, authorize the operator, record an audit event, and propagate configuration to connected devices when supported.

A database update alone must not be reported as successful hardware reconfiguration if the hardware has not acknowledged it.

Support:
- `POST /ingest`
- `POST /edge/ingest`
- `POST /arduino/ingest`
- `GET /live`
- `GET /recent`
- `GET /sensors/recent`
- `GET /edge/health`

Add a versioned telemetry schema and device identity.

Validate timestamps, measurement ranges, duplicate packets, sensor identity, calibration, and data freshness. Store both normalized readings and relevant quality flags.

Implement MQTT publishing and subscribing using a real broker connection. Persist pending outbound commands and ingestion offsets so that reconnects do not silently lose important operations.

Implement local buffering and synchronization for edge deployments.

The frontend's `isSimulated` field must accurately distinguish real hardware data from any explicitly enabled test fixture.

### C. Irrigation and actuation

`POST /api/irrigation/control`

Support:
- `start`
- `stop`
- `toggle_auto`

Validate zone identity, operator authorization, command parameters, safety interlocks, stale telemetry, tank level, maximum continuous runtime, and pump/valve constraints.

Persist command requests and status transitions. Track queued, dispatched, acknowledged, rejected, timed-out, and failed states.

Require explicit user confirmation for consequential actions.

Use a real MQTT or configured device adapter for actuation. Never claim that a valve opened merely because an API request was accepted.

If no physical device is connected, return an explicit hardware-unavailable response and preserve a truthful command record.

Implement tank-level history, rainwater logging, irrigation events, and alert acknowledgement with PostgreSQL persistence.

### D. Crop recommendation

`POST /api/recommendations/crop`

Input:
- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- Soil pH
- Rainfall

Output: the existing `CropRecommendationItem[]` contract and actual `modelSource`.

Use verified crop recommendation datasets and compatible trained models from the old repository when available.

Requirements:
- Correct feature ordering and preprocessing.
- Crop label decoding.
- Top-k ranking.
- Calibrated confidence when supported by evaluation.
- Crop metadata, scientific names, water requirements, duration, yield information, and drip-savings estimates backed by documented data or calculations.
- Explicit treatment of out-of-distribution inputs.
- Model and dataset version traceability.

Do not use arbitrary scores or manually constructed recommendations disguised as ML output.

If a verified model does not support the requested output fields, derive fields only from a documented agronomic source or validated calculation and label their provenance.

### E. Fertilizer recommendation

`POST /api/recommendations/fertilizer`

Accept crop type, NPK values, soil pH, and soil moisture.

Return:
- Nutrient deficits.
- Product dosage and application schedules.
- Organic amendments.
- Moisture warnings.
- Timestamp and provenance.

Implement agronomic calculations using documented nutrient requirements, soil-test interpretation, nutrient availability, product composition, units, and crop growth stage.

Account for fertilizer nutrient percentages, existing soil nutrients, application efficiency, and safe dosage limits.

Separate deterministic agronomic calculations from ML estimates. Do not invent a fertilizer model if no compatible model exists.

Require additional information or explicitly document assumptions when inputs are insufficient for a reliable dosage recommendation.

### F. Plant disease diagnosis

`POST /api/disease/analyze`

Accept:
- `imageBase64`
- `mimeType`
- `cropHint`
- `samplePreset`
- `language`

Return the exact `DiseaseDiagnosisResult` structure, including:
- `isHealthy`
- `cropIdentified`
- `diseaseName`
- `scientificName`
- `confidencePct`
- `severity`
- `symptomsObserved`
- `organicTreatment`
- `chemicalTreatment`
- `preventionSteps`
- `source`
- `isModelPrediction`
- `analyzedAt`

Implement real image decoding, validation, preprocessing, model inference, class mapping, and confidence handling.

Inspect the actual disease model artifacts before selecting the inference architecture. Use compatible local CNN or vision models and verified model weights.

If the old SCOLD VLM implementation contains placeholder inference, do not reuse its synthetic results.

If no usable model exists, report that the diagnostic model is unavailable. Never label a preset image's known identity as a prediction made by the model.

Only return treatment advice supported by an approved local knowledge base. Keep uncertain diagnoses explicitly uncertain and advise appropriate expert verification.

### G. Digital twins and analytics

`GET /api/analytics/digital-twin?zoneId=...&days=...`

Return the existing `DigitalTwinResponse` contract.

Implement separate WaterTwin, SoilTwin, and CropTwin services.

WaterTwin:
- Soil-water balance.
- Root-zone depletion.
- Evapotranspiration.
- Field capacity and wilting point.
- Irrigation demand.
- Rainfall and infiltration where available.
- Water-loss estimates.

SoilTwin:
- Nutrient balance.
- Soil pH and salinity risk.
- Organic carbon and nutrient mineralization only where supported by suitable inputs and models.
- Explicit uncertainty.

CropTwin:
- Crop growth stage.
- Growing-degree days.
- Canopy and biomass estimates where supported.
- Harvest timing.
- Yield prediction using a validated model.

Use physics-informed deterministic calculations first, then apply validated ML residual corrections when available.

Generate historical series from actual observations and forecasts from actual models or equations. Clearly label forecast status, assumptions, confidence, and missing-data limitations.

Do not fabricate historical measurements to populate chart series.

### H. Local agricultural assistant

`POST /api/assistant/chat`

Input:
- `message`
- `language`

Return:
- `reply`
- `languageUsed`
- `isFallbackLanguage`
- `sources`
- `timestamp`

Implement a local RAG pipeline with:
- Local agricultural document ingestion.
- Text extraction and deduplication.
- Chunking with source metadata.
- Local embedding generation.
- FAISS or pgvector retrieval.
- BM25 lexical retrieval.
- Hybrid ranking or reciprocal rank fusion.
- Optional local reranking.
- Local LLM response generation.
- Citation/source attribution.
- Session-aware conversation history persisted in PostgreSQL.
- Supported multilingual responses, including the frontend's configured languages.

Index available agricultural manuals, crop guides, verified Q&A data, and permitted research documents from the old repository.

If a local model is unavailable, return a clearly labeled retrieval-only answer when suitable evidence exists, or a structured unavailable response. Do not return fabricated AI-generated content while claiming that an LLM produced it.

Never generate unsupported pesticide dosage, chemical safety, or medical-style certainty. Make the assistant distinguish retrieved evidence, calculations, and uncertain recommendations.

## 6. OPERATIONS CENTER AND ALL FIVE OPS DISCIPLINES

The existing frontend has an extensive Operations Center. Fully implement the operations API rather than returning static overview objects.

### A. Operations overview

`GET /api/operations/overview`

Return every field in `OperationsOverviewResponse`:

- `systemHealth`
- `settings`
- `mlOps`
- `dataOps`
- `llmOps`
- `cicd`
- `security`
- `dependencyGraph`
- Actual collection timestamps and service statuses

Build the response from live system metrics, PostgreSQL queries, the model registry, dataset catalog, local LLM runtime, job history, audit records, and configured integrations.

Persist operational settings using PostgreSQL. Validate changes and audit every consequential modification.

Implement:

`POST /api/operations/settings`

Use optimistic concurrency or equivalent safeguards to avoid silently overwriting concurrent configuration changes.

### B. MLOps

Implement a real model registry and lifecycle manager.

Support:
- Model registration.
- Versioned artifacts and checksums.
- Model metadata and training dataset lineage.
- Training runs and evaluation results.
- Actual inference latency and error metrics.
- Drift detection against a recorded reference dataset.
- Retraining jobs.
- Holdout evaluation.
- Model promotion and rollback.
- Model loading and unloading.
- Model resource utilization.
- Registry and artifact integrity checks.

Implement:

`POST /api/operations/mlops/deploy`

Input:
- `modelId`
- `action`: `promote` or `rollback`

A promotion must validate artifact integrity, compatibility, test results, evaluation thresholds, and approval requirements. A rollback must restore an actual previously registered model version.

Use atomic version-pointer updates and preserve rollback history.

Implement the additional APIs required by the operations interface and legacy MLOps console, including model status, history, versions, retraining, retrain-all, drift checks, promotion, rollback, and resource monitoring.

A retraining action must enqueue and execute a genuine training job against a real dataset. If the job cannot run, show a failure with the reason.

Never claim a model was retrained simply because a button was clicked.

### C. DataOps

Implement:
- Dataset registry.
- Source and license metadata.
- Schema validation.
- Missing-value analysis.
- Duplicate detection.
- Unit and range checks.
- Timestamp and freshness validation.
- Sensor anomaly detection.
- Quarantine and reprocessing.
- Data lineage.
- Versioned preprocessing.
- Training/inference feature consistency.
- Dataset quality reports.
- Data drift reports.
- Retention and archival.
- Reproducible imports and exports.

Implement:

`POST /api/operations/dataops/validate`

Return actual counts of checked streams and quarantined stale streams.

Persist validation runs, violations, and remediation actions.

Do not mark an entire pipeline healthy if a required validation stage failed.

### D. LLMOps

Implement:
- Local model registry.
- Provider/model readiness.
- Actual generation request counts.
- Prompt template versions and hashes.
- Inference latency and failure metrics.
- Input/output token counts where the local runtime provides them.
- RAG document and chunk counts.
- Retrieval quality evaluation.
- Citation and grounding checks.
- Safety and agronomic guardrails.
- Model/context resource monitoring.
- Versioned system prompts.
- Request tracing and audit metadata.

All LLMOps metrics must derive from actual execution and stored events. If a metric is unavailable, expose `null` or a documented unavailable status instead of a hardcoded value.

### E. AIOps

Implement local observability using structured logs, Prometheus metrics, health checks, and persisted operational events.

Monitor:
- API error rates and latency.
- PostgreSQL availability and connection pool health.
- MQTT broker status.
- Edge-device freshness.
- Model loading and inference errors.
- CPU, RAM, disk, and GPU usage where supported.
- Queue depth and failed jobs.
- LLM runtime health.
- Model and dataset drift alerts.
- Repeated exceptions and dependency failures.

Use bounded retries, timeouts, circuit breakers, graceful shutdown, and resource limits.

Automatic recovery must be conservative, auditable, and limited to predefined safe operations. Do not let the system automatically perform arbitrary code changes or dangerous hardware actions.

### F. AgentOps

Implement a local, governed agent framework with specialized agents:

1. Farm Data Quality Agent.
2. Crop Advisory Agent.
3. Irrigation Planning Agent.
4. Plant Health Triage Agent.
5. Model Evaluation Agent.
6. Data Pipeline Agent.
7. Agricultural Knowledge Retrieval Agent.
8. Operations Diagnostics Agent.
9. Quality Assurance Agent.
10. Orchestrator/Supervisor Agent.

Each agent must have:
- A defined purpose.
- Typed input and output schemas.
- Explicit permissions.
- Registered tools.
- Timeouts and resource limits.
- Task state and execution history.
- Correlation IDs.
- Error handling and bounded retries.
- Recorded evidence and provenance.
- Human approval for consequential actions.

Implement durable task execution and explicit task states: queued, running, blocked, completed, and failed.

Use deterministic code for calculations, database operations, safety checks, and deployment decisions. Use local LLM reasoning only where language understanding or synthesis adds value.

Agents must not bypass agronomic safety rules, database authorization, model promotion gates, or hardware interlocks.

## 7. ADDITIONAL OPERATIONS API CONTRACTS

The existing `src/api/client.ts` methods must work without unnecessary frontend rewrites.

Implement at minimum:

- `GET /api/operations/overview`
- `POST /api/operations/settings`
- `POST /api/operations/mlops/deploy`
- `POST /api/operations/hardware-command`
- `POST /api/operations/dataops/validate`

Inspect `OperationsCenterPage.tsx` completely and implement every other endpoint required by its remaining actions, tabs, exports, settings, security views, CI/CD views, and diagnostics.

The hardware-command API must support:
- `wake_ping`
- `zero_calibrate`
- `gateway_selftest`

Validate the device and command, dispatch through the real adapter, record the outcome, and distinguish command acceptance from device acknowledgement.

Inspect all other pages and components for direct `fetch`, `WebSocket`, `EventSource`, and network calls. Include every discovered endpoint in a generated `docs/FRONTEND_API_CONTRACT.md`.

Do not stop at the five explicitly listed operations endpoints.

## 8. AGRICULTURAL DATA AND MODEL REUSE

Inspect all model and dataset catalogs in the old repository.

Build a reproducible inventory containing:
- File path and SHA-256.
- Dataset/model version.
- Source URL or origin.
- License and usage restrictions.
- Target task.
- Feature schema.
- Label mapping.
- Preprocessing requirements.
- Training and evaluation provenance.
- Runtime compatibility.
- Known limitations.

Reuse verified existing models for:
- Crop recommendation.
- Fertilizer recommendation.
- Irrigation and water optimization.
- Yield prediction.
- Disease detection.
- Weed classification.
- Sensor anomaly detection.
- Agricultural RAG.

Repair model compatibility through controlled migrations, environment pinning, or reproducible retraining. Never blindly load untrusted serialized model artifacts.

If a model is incompatible or missing, implement the actual training pipeline and evaluation process when suitable data is available.

Where data is missing, identify the exact missing artifact and the acquisition or training requirement. Do not invent datasets, metrics, scientific findings, or pretrained weights.

Prefer reproducible training with fixed seeds, documented splits, leakage prevention, and separate holdout evaluation.

For Indian agricultural data, preserve source attribution, geographical coverage, units, crop names, seasonality, and temporal coverage. Do not mix incompatible units or regional datasets without a documented normalization process.

## 9. DATABASE SCHEMA

Create normalized PostgreSQL tables, including appropriately designed versions of:

- users
- roles
- permissions
- user_roles
- farms
- farm_zones
- devices
- device_configurations
- sensor_readings
- sensor_quality_events
- device_commands
- device_command_events
- irrigation_zones
- irrigation_events
- tank_readings
- rainwater_events
- alerts
- alert_acknowledgements
- recommendations
- recommendation_outcomes
- disease_diagnoses
- assistant_sessions
- assistant_messages
- knowledge_documents
- knowledge_chunks
- embedding_metadata
- datasets
- dataset_versions
- data_validation_runs
- data_quality_violations
- model_registry
- model_versions
- model_artifacts
- training_runs
- evaluation_runs
- drift_checks
- deployment_events
- inference_events
- agent_definitions
- agent_tasks
- agent_execution_steps
- system_settings
- service_logs
- audit_events
- scheduled_jobs
- job_execution_history

Add any additional tables required by the implementation.

Do not store large binary image files directly in ordinary telemetry tables. Use controlled local file storage with database metadata, checksums, retention policies, and access controls.

Define foreign keys, deletion behavior, indexes, uniqueness constraints, and migration strategy.

Never expose credentials, private model files, uploaded images, or internal stack traces to unauthorized clients.

## 10. API DESIGN AND SECURITY

Use `/api/v1` for newly designed Python API endpoints while preserving all existing frontend gateway paths.

Provide:
- OpenAPI documentation.
- Consistent request and response schemas.
- Pagination, sorting, and filtering.
- Input validation.
- Request correlation IDs.
- Idempotency for consequential actions.
- Structured error responses.
- Explicit timeouts and payload-size limits.
- CORS restricted to configured local frontend origins.
- JWT authentication and refresh-token rotation where appropriate.
- Role-based access control.
- Audit logging for sensitive actions.
- Password hashing using a modern password-hashing algorithm.
- Rate limiting.
- Image-upload validation.
- SQL injection prevention.
- Secure local file handling.
- Secret redaction in logs.
- Dependency and artifact integrity checks.

Provide a documented local bootstrap procedure for the first administrator without committing default credentials.

Do not add unauthenticated administrative reset, arbitrary command execution, unrestricted filesystem access, or automatic destructive migrations.

## 11. FRONTEND INTEGRATION RULES

Preserve the existing new frontend.

Do not replace it with a newly generated dashboard.

Do not remove pages, change its design system, discard its internationalization, or break the 3D components just to simplify backend work.

Make the smallest required frontend changes to:
- Configure the API base URL.
- Handle authentication if introduced.
- Display real loading, empty, unavailable, and error states.
- Distinguish measured, predicted, and simulated data.
- Support real-time telemetry updates.
- Confirm hardware operations and report their actual outcomes.
- Display genuine model and pipeline states.
- Avoid treating a preset specimen as a successful diagnostic prediction.

The existing API client must be the primary contract source. Any necessary interface changes must be made consistently in both TypeScript and Python.

Maintain the current UI routes, supported languages, chart fields, action semantics, and operations tabs.

## 12. PROJECT STRUCTURE

Create a clean repository structure similar to:

`backend/python/`
- `app/main.py`
- `app/api/v1/`
- `app/core/`
- `app/db/`
- `app/models/`
- `app/schemas/`
- `app/repositories/`
- `app/services/`
- `app/agronomy/`
- `app/ml/`
- `app/mlops/`
- `app/dataops/`
- `app/llmops/`
- `app/aiops/`
- `app/agentops/`
- `app/iot/`
- `app/digital_twins/`
- `app/rag/`
- `app/security/`
- `app/tasks/`
- `alembic/`
- `tests/`

`backend/node/`
- `src/app.ts`
- `src/server.ts`
- `src/routes/`
- `src/controllers/`
- `src/services/`
- `src/middleware/`
- `src/clients/`
- `src/schemas/`
- `tests/`

`frontend/`
- Preserve the supplied React application and update only what is necessary.

`datasets/`
- `raw/`
- `processed/`
- `manifests/`
- `validation_reports/`

`models/`
- `registry/`
- `artifacts/`
- `metadata/`

`knowledge/`
- `raw/`
- `processed/`
- `indexes/`

`infra/`
- `docker-compose.yml`
- `postgres/`
- `mosquitto/`
- `prometheus/`

`scripts/`
- Database initialization and migration checks.
- Dataset inventory and validation.
- Model verification and evaluation.
- Offline dependency checks.
- Development startup and shutdown.
- Backup and restore.
- End-to-end smoke tests.

`docs/`
- Architecture.
- API contracts.
- Database schema.
- Model inventory.
- Dataset lineage.
- Offline installation.
- Security.
- Testing.
- Operations runbooks.

Adjust the structure when needed, but maintain explicit separation of responsibilities.

## 13. TESTING REQUIREMENTS

Write real automated tests, not tests that merely assert that a route returns HTTP 200.

### Backend unit tests

Test:
- Sensor normalization and validation.
- Database repositories and transactions.
- Crop recommendation preprocessing.
- Fertilizer calculations and safety limits.
- Irrigation calculations and interlocks.
- Disease image preprocessing and real model invocation.
- Digital-twin equations.
- Dataset quality rules.
- RAG retrieval and source attribution.
- Model registry transitions.
- Drift calculations.
- Agent permissions and execution states.

### Integration tests

Use a dedicated PostgreSQL test database.

Test:
- Database migrations from an empty database.
- Database persistence across service restarts.
- API-to-database workflows.
- MQTT message ingestion and command acknowledgements.
- Model artifact loading and inference.
- RAG indexing and retrieval.
- MLOps deployment and rollback.
- DataOps validation and quarantine.
- LLMOps metrics derived from actual requests.
- Agent execution history and permissions.

### Frontend contract tests

For every method in `src/api/client.ts`, verify:
- Correct route and HTTP method.
- Request payload validation.
- Response compatibility.
- Error handling.
- Loading and failure behavior.

Inspect every page for any additional network endpoints and include them.

### End-to-end tests

Exercise the following workflow:

1. Start PostgreSQL, FastAPI, Express, and required local services.
2. Run database migrations.
3. Register a device and farm zone.
4. Submit sensor telemetry.
5. Verify the reading was stored in PostgreSQL.
6. Retrieve the updated dashboard.
7. Generate a crop recommendation using a real model.
8. Generate a fertilizer recommendation.
9. Run disease inference using a real local model.
10. Query the agricultural assistant using the local RAG/LLM stack.
11. Retrieve digital-twin outputs.
12. Run DataOps validation.
13. Inspect actual model and LLM operational metrics.
14. Execute a model evaluation and test promotion/rollback.
15. Verify hardware command handling with an explicitly identified test adapter or a real connected device.
16. Restart the services and verify persistence.
17. Run the frontend build and TypeScript checks.
18. Run an offline test with external networking disabled.

Tests involving unavailable physical hardware must explicitly identify the test adapter. They must not count as verification of real physical actuation.

### Quality gates

- No unexpected test failures.
- No unresolved TypeScript errors.
- No database migration failures.
- No circular imports.
- No unhandled exceptions in tested workflows.
- No fabricated successful inference.
- No external network dependency during offline runtime tests.
- No secrets in source code.
- No unexplained API contract mismatches.

Measure and report test coverage honestly. Do not claim 100% coverage unless it is measured and demonstrated.

## 14. PERFORMANCE AND RESOURCE MANAGEMENT

Support CPU-only operation and optional GPU acceleration.

Detect available resources at runtime. Configure model loading according to the available CPU, RAM, GPU, and VRAM.

Implement:
- Lazy model loading.
- Warmup where beneficial.
- Bounded inference concurrency.
- Batch inference where appropriate.
- Model caching.
- Safe unloading under resource pressure.
- Request deadlines.
- Database query optimization.
- Pagination and bounded time-series queries.
- Bounded background job queues.
- Streaming for large responses.
- Image dimension and file-size limits.

Never require a GPU simply to start the application.

Do not promise a fixed inference latency or accuracy without benchmarking the actual machine and model artifacts.

## 15. IMPLEMENTATION PHASES

Execute these phases in order.

### Phase 1 — Repository and frontend audit

Extract both archives. Inspect the frontend contracts and old project assets. Generate the audit, API contract, feature traceability matrix, and model/dataset inventory.

### Phase 2 — Foundation

Create the repository structure, configuration, logging, security primitives, PostgreSQL schema, Alembic migrations, Express gateway, and FastAPI application.

### Phase 3 — Functional API integration

Implement all frontend API endpoints, response models, repositories, error handling, and persistent operations.

### Phase 4 — Agricultural ML

Load and verify existing models. Implement preprocessing, inference, model metadata, agronomic calculations, and model availability checks.

### Phase 5 — IoT and edge

Implement MQTT, HTTP ingestion, WebSerial-compatible ingestion, device management, telemetry normalization, durable command processing, and safe hardware acknowledgements.

### Phase 6 — Digital twins and analytics

Implement physics-based water, soil, and crop simulations, historical aggregation, and validated ML correction layers.

### Phase 7 — Offline LLM and RAG

Provision local models, build the knowledge ingestion pipeline, create local indexes, implement grounded responses, and validate offline execution.

### Phase 8 — MLOps, DataOps, LLMOps, AIOps, and AgentOps

Implement persistent operational state, genuine evaluation jobs, drift analysis, lifecycle management, metrics, recovery policies, and governed agents.

### Phase 9 — Frontend integration

Connect all seven frontend pages to the new APIs. Remove old in-memory backend behavior, wire real loading/error states, and verify every action and chart.

### Phase 10 — Validation and release

Run migrations, linting, type checks, unit tests, integration tests, frontend build, API contract tests, and offline end-to-end tests. Fix failures before marking the corresponding feature complete.

At the end of each phase, update `docs/REBUILD_STATUS.md` with:
- Implemented files.
- Actual commands executed.
- Test results.
- Unresolved failures.
- Required models, datasets, hardware, or dependencies.
- The next phase's prerequisites.

Do not stop after generating a plan or a folder structure. Continue implementing the project.

## 16. REQUIRED DELIVERABLES

Deliver:
1. A complete runnable FastAPI backend.
2. A complete runnable Express gateway.
3. A PostgreSQL schema and Alembic migrations.
4. All frontend API routes and compatible response schemas.
5. Real ML inference and reproducible training pipelines where supported.
6. A genuine local RAG and LLM service.
7. Functional MLOps, DataOps, LLMOps, AIOps, and AgentOps implementations.
8. IoT and edge adapters with honest hardware status.
9. Digital-twin services.
10. A local infrastructure configuration.
11. A verified dependency lock or pinned dependency manifests.
12. Automated unit, integration, contract, and end-to-end tests.
13. Offline installation and startup instructions.
14. Backup, restore, security, and operational runbooks.
15. A complete model and dataset provenance catalog.
16. A feature traceability matrix linking frontend component → API → service → database/model → tests.
17. An honest verification report listing passed tests, failed tests, missing assets, and unverified hardware.

## 17. FINAL ACCEPTANCE CRITERIA

The rebuild is complete only when:

- The supplied frontend runs against the new Express gateway.
- Every API method used by every frontend page is implemented.
- PostgreSQL is the source of truth for operational state.
- Sensor histories and audit records survive restarts.
- Recommendations execute real verified models or documented calculations.
- Disease detection performs actual image inference.
- The agricultural assistant uses a local model and local retrieval when configured.
- Digital-twin outputs are derived from actual observations, documented equations, and/or verified models.
- MLOps lifecycle actions change actual registry state and use real evaluation results.
- DataOps validation operates on actual datasets and sensor records.
- LLMOps metrics are based on actual local inference activity.
- AIOps reports actual resource and dependency status.
- AgentOps tasks execute registered tools under explicit permissions.
- Hardware actions never report unverified physical success.
- No cloud service is required at runtime.
- The application continues working after internet access is disabled.
- Tests demonstrate that no production code path silently substitutes mock data for missing services.
- All outstanding limitations are documented rather than concealed.

Begin by extracting and auditing both archives, mapping every frontend API call, identifying every fake or simulated behavior, and inventorying the real model and dataset assets. Then implement the system incrementally, running tests and fixing defects throughout.

**Priority order:** correctness → data integrity → agricultural safety → complete frontend compatibility → genuine AI/ML functionality → offline reliability → security → performance.

Do not sacrifice correctness or honesty to make the dashboard appear fully operational.