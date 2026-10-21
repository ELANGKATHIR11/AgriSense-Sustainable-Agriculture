from pydantic import BaseModel
from typing import List, Optional

class PlatformSettingsSchema(BaseModel):
    timezone: str = "Asia/Kolkata"
    moistureCriticalAlertPct: float = 32.0
    staleNodeTimeoutMin: int = 15
    maxContinuousIrrigationMin: int = 45
    safetyInterlockRequired: bool = True
    quarantineStaleNodes: bool = True
    enable3DHardwareShadows: bool = True
    telemetryRetentionDays: int = 90
    logLevel: str = "INFO"

class MlModelEntrySchema(BaseModel):
    id: str
    name: str
    version: str
    previousVersion: str
    task: str
    framework: str
    stage: str
    datasetVersion: str
    primaryMetricName: str
    primaryMetricValue: float
    secondaryMetricLabel: str
    avgLatencyMs: float
    driftScorePsi: float
    driftStatus: str
    lastEvaluatedIso: str

class AuditLogEntrySchema(BaseModel):
    id: str
    timestamp: str
    actor: str
    role: str
    category: str
    action: str
    target: str
    status: str
    correlationId: str
    details: str

class ServiceLogEntrySchema(BaseModel):
    id: str
    timestamp: str
    level: str
    service: str
    correlationId: str
    message: str

class DependencyNodeSchema(BaseModel):
    id: str
    label: str
    layer: str
    status: str
    metrics: str
    downstream: List[str]

class OperationsOverviewResponseSchema(BaseModel):
    timestamp: str
    systemHealth: dict
    settings: PlatformSettingsSchema
    mlOps: dict
    dataOps: dict
    llmOps: dict
    cicd: dict
    security: dict
    dependencyGraph: List[DependencyNodeSchema]

class HardwareCommandRequestSchema(BaseModel):
    deviceId: str
    command: str

class MlDeployRequestSchema(BaseModel):
    modelId: str
    action: str
