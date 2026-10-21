export interface DeviceReadings {
  soilMoisturePct: number;
  soilTempC: number;
  airTempC: number;
  airHumidityPct: number;
  soilPh: number;
  nitrogenMgKg: number;
  phosphorusMgKg: number;
  potassiumMgKg: number;
  ecDsM: number;
  solarLux: number;
}

export interface DeviceTelemetry {
  id: string;
  name: string;
  type: 'ESP32_SOIL_NODE' | 'ESP32_CLIMATE_NODE' | 'RPI_EDGE_GATEWAY';
  protocol: 'MQTT/TLS' | 'LoRaWAN-MQTT' | 'HTTP/REST';
  zoneId: string;
  zoneName: string;
  cropType: string;
  status: 'online' | 'stale' | 'offline';
  firmware: string;
  batteryPct: number;
  signalRssi: number;
  lastSeenMinutesAgo: number;
  samplingIntervalSec: number;
  moistureThresholdPct: number;
  calibrationOffsetPct?: number;
  coordinates3D?: { x: number; y: number; z: number };
  readings: DeviceReadings;
  isSimulated: boolean;
}

export interface IrrigationZoneState {
  zoneId: string;
  zoneName: string;
  cropType: string;
  valveStatus: 'active' | 'idle';
  soilMoisturePct: number;
  targetMoisturePct: number;
  flowRateLpm: number;
  lastIrrigatedIso: string;
  recommendedDurationMin: number;
  waterSavedLitersToday: number;
  autoScheduleEnabled: boolean;
}

export interface FarmEvent {
  id: string;
  timestamp: string;
  type: 'warning' | 'stale' | 'info';
  zoneId: string;
  code: string;
  value: string;
  threshold: string;
}

export interface IrrigationState {
  tankLevelPct: number;
  tankCapacityLiters: number;
  currentVolumeLiters: number;
  pumpStatus: 'active' | 'idle';
  pumpPressureBar: number;
  waterQualityPh: number;
  waterQualityTdsPpm: number;
  dailyWaterSavedPct: number;
  zones: IrrigationZoneState[];
  recentEvents: FarmEvent[];
}

export interface SensorHistoryPoint {
  timestamp: string;
  hourLabel: string;
  soilMoisturePct: number;
  soilTempC: number;
  airHumidityPct: number;
  evapotranspirationMm: number;
}

export interface SensorsResponse {
  summary: {
    totalSensors: number;
    onlineSensors: number;
    staleSensors: number;
    avgSoilMoisturePct: number;
    avgSoilTempC: number;
    avgAirHumidityPct: number;
    avgAirTempC: number;
    weatherSummary: {
      conditionCode: string;
      tempC: number;
      humidityPct: number;
      windSpeedKmh: number;
      rainProbabilityPct: number;
      solarRadiationWm2: number;
      source: string;
    };
    cropHealthSummary: {
      overallStatus: 'healthy' | 'attention_needed';
      healthyPlotsCount: number;
      attentionPlotsCount: number;
      activeAlertCode: string;
    };
  };
  devices: DeviceTelemetry[];
  history24h: SensorHistoryPoint[];
  irrigation: IrrigationState;
  timestamp: string;
}

export interface SystemStatusResponse {
  status: string;
  backendVersion: string;
  mqttBroker: string;
  edgeGateway: string;
  aiService: string;
  aiConfigured: boolean;
  totalDevices: number;
  onlineDevices: number;
  staleDevices: number;
  timestamp: string;
}

export interface CropRecommendationItem {
  cropKey: string;
  cropName: string;
  scientificName: string;
  rank: number;
  confidencePct: number;
  waterRequirementMm: number;
  growthDurationDays: number;
  expectedYieldTonHa: number;
  dripSavingsPct: number;
  limitingFactor: string;
}

export interface FertilizerScheduleItem {
  productKey: string;
  productName: string;
  dosageKgHa: number;
  timingKey: string;
  applicationNote: string;
}

export interface FertilizerResponse {
  cropType: string;
  deficitsKgHa: {
    nitrogen: number;
    phosphorus: number;
    potassium: number;
  };
  schedule: FertilizerScheduleItem[];
  organicAmendment: string;
  moistureWarning: string | null;
  timestamp: string;
}

export interface DiseaseDiagnosisResult {
  isHealthy: boolean;
  cropIdentified: string;
  diseaseName: string;
  scientificName: string;
  confidencePct: number;
  severity: 'none' | 'low' | 'moderate' | 'high';
  symptomsObserved: string[];
  organicTreatment: string[];
  chemicalTreatment: string[];
  preventionSteps: string[];
  source: string;
  isModelPrediction: boolean;
  analyzedAt: string;
}

export interface DigitalTwinSeriesPoint {
  dateIso: string;
  dateShort: string;
  observedMoisturePct: number;
  predictedMoisturePct: number;
  soilTempC: number;
  irrigationMm: number;
  cropNdvi: number;
  evapotranspirationMm: number;
}

export interface DigitalTwinResponse {
  zoneId: string;
  days: number;
  waterTwin: {
    modelStatus: string;
    rootZoneDepletionMm: number;
    fieldCapacityPct: number;
    wiltingPointPct: number;
    dailyEt0Mm: number;
    nextIrrigationHours: number;
    seepageLossPct: number;
    confidencePct: number;
    isSimulatedForecast: boolean;
  };
  soilTwin: {
    modelStatus: string;
    cationExchangeCapacityMeq: number;
    organicCarbonPct: number;
    microbialActivityIndex: number;
    salinityRisk: string;
    npkMineralizationRate: string;
    confidencePct: number;
    isSimulatedForecast: boolean;
  };
  cropTwin: {
    modelStatus: string;
    phenologyStageKey: string;
    daysAfterSowing: number;
    growingDegreeDays: number;
    leafAreaIndex: number;
    canopyCoverPct: number;
    projectedHarvestDays: number;
    projectedYieldTonHa: number;
    confidencePct: number;
    isSimulatedForecast: boolean;
  };
  series: DigitalTwinSeriesPoint[];
  timestamp: string;
}

export interface MlModelEntry {
  id: string;
  name: string;
  version: string;
  previousVersion: string;
  task: string;
  framework: string;
  stage: 'production' | 'staging' | 'archived';
  datasetVersion: string;
  primaryMetricName: string;
  primaryMetricValue: number;
  secondaryMetricLabel: string;
  avgLatencyMs: number;
  driftScorePsi: number;
  driftStatus: 'nominal' | 'watch' | 'retrain_recommended';
  lastEvaluatedIso: string;
}

export interface AuditLogEntry {
  id: string;
  timestamp: string;
  actor: string;
  role: string;
  category:
    | 'IRRIGATION'
    | 'IOT_HARDWARE'
    | 'MLOPS'
    | 'DATAOPS'
    | 'LLMOPS'
    | 'SECURITY'
    | 'SETTINGS';
  action: string;
  target: string;
  status: 'CONFIRMED' | 'REJECTED' | 'WARNING';
  correlationId: string;
  details: string;
}

export interface ServiceLogEntry {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARN' | 'ERROR';
  service: string;
  correlationId: string;
  message: string;
}

export interface PlatformSettings {
  timezone: string;
  moistureCriticalAlertPct: number;
  staleNodeTimeoutMin: number;
  maxContinuousIrrigationMin: number;
  safetyInterlockRequired: boolean;
  quarantineStaleNodes: boolean;
  enable3DHardwareShadows: boolean;
  telemetryRetentionDays: number;
  logLevel: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';
}

export interface DependencyNode {
  id: string;
  label: string;
  layer: string;
  status: 'healthy' | 'warning' | 'active';
  metrics: string;
  downstream: string[];
}

export interface OperationsOverviewResponse {
  timestamp: string;
  systemHealth: {
    uptimeSeconds: number;
    memoryRssMb: number;
    heapUsedMb: number;
    totalApiRequests: number;
    totalApiErrors: number;
    avgApiLatencyMs: number;
    services: {
      id: string;
      name: string;
      status: 'healthy' | 'degraded' | 'fallback_ready';
      latencyMs: number;
      version: string;
      dependency: string;
    }[];
  };
  settings: PlatformSettings;
  mlOps: {
    models: MlModelEntry[];
  };
  dataOps: {
    freshnessPct: number;
    staleNodeCount: number;
    schemaCompliancePct: number;
    outOfRangeCount: number;
    datasets: {
      id: string;
      name: string;
      version: string;
      records: number;
      freshness: string;
      missingnessPct: number;
      license: string;
      status: string;
    }[];
    validationRules: {
      id: string;
      name: string;
      passed: boolean;
      checkedCount: number;
      violationCount: number;
    }[];
  };
  llmOps: {
    totalRequests: number;
    visionRequests: number;
    assistantRequests: number;
    fallbackCount: number;
    avgLatencyMs: number;
    estimatedInputTokens: number;
    estimatedOutputTokens: number;
    activeSystemPromptVersion: string;
    guardrailPolicy: string;
    ragChunksIndexed: number;
    lastInferenceIso: string;
    providerStatus: string;
    ragDocuments: {
      id: string;
      title: string;
      chunks: number;
      embeddingModel: string;
      status: string;
    }[];
  };
  cicd: {
    externalProviderConnected: boolean;
    externalProviderNote: string;
    branch: string;
    runtimeEnv: string;
    checks: {
      id: string;
      name: string;
      status: string;
      durationSec: number;
      lastRunIso: string;
    }[];
  };
  security: {
    rbacRoles: {
      role: string;
      permissions: string;
      scope: string;
    }[];
    auditTrail: AuditLogEntry[];
    serviceLogs: ServiceLogEntry[];
  };
  dependencyGraph: DependencyNode[];
}

async function handleJsonResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errMsg = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      if (data?.error) errMsg = data.error;
    } catch {
      // ignore json parse error
    }
    throw new Error(errMsg);
  }
  return res.json() as Promise<T>;
}

export const agriApi = {
  getSensors: async (): Promise<SensorsResponse> => {
    const res = await fetch('/api/sensors');
    return handleJsonResponse<SensorsResponse>(res);
  },

  getStatus: async (): Promise<SystemStatusResponse> => {
    const res = await fetch('/api/status');
    return handleJsonResponse<SystemStatusResponse>(res);
  },

  updateDeviceConfig: async (
    id: string,
    payload: {
      samplingIntervalSec: number;
      moistureThresholdPct: number;
      calibrationOffsetPct?: number;
    }
  ): Promise<{ success: boolean; device: DeviceTelemetry }> => {
    const res = await fetch(`/api/devices/${encodeURIComponent(id)}/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleJsonResponse(res);
  },

  controlIrrigation: async (payload: {
    zoneId: string;
    action: 'start' | 'stop' | 'toggle_auto';
    durationMin?: number;
  }): Promise<{ confirmed: boolean; zone: IrrigationZoneState; irrigation: IrrigationState }> => {
    const res = await fetch('/api/irrigation/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleJsonResponse(res);
  },

  recommendCrops: async (payload: {
    nitrogen: number;
    phosphorus: number;
    potassium: number;
    temperature: number;
    humidity: number;
    ph: number;
    rainfall: number;
  }): Promise<{ recommendations: CropRecommendationItem[]; modelSource: string }> => {
    const res = await fetch('/api/recommendations/crop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleJsonResponse(res);
  },

  recommendFertilizer: async (payload: {
    cropType: string;
    nitrogen: number;
    phosphorus: number;
    potassium: number;
    soilPh: number;
    soilMoisture: number;
  }): Promise<FertilizerResponse> => {
    const res = await fetch('/api/recommendations/fertilizer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleJsonResponse(res);
  },

  analyzeDisease: async (payload: {
    imageBase64?: string;
    mimeType?: string;
    cropHint?: string;
    samplePreset?: string;
    language: string;
  }): Promise<DiseaseDiagnosisResult> => {
    const res = await fetch('/api/disease/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleJsonResponse(res);
  },

  getDigitalTwin: async (
    zoneId: string,
    days: number,
    scenario?: {
      rainDeltaPct?: number;
      tempDeltaC?: number;
      irrigationEffPct?: number;
      fertilizerEfficiencyPct?: number;
    }
  ): Promise<DigitalTwinResponse> => {
    const params = new URLSearchParams({
      zoneId,
      days: days.toString(),
    });
    if (scenario) {
      if (typeof scenario.rainDeltaPct === 'number') {
        params.append('rainDeltaPct', scenario.rainDeltaPct.toString());
      }
      if (typeof scenario.tempDeltaC === 'number') {
        params.append('tempDeltaC', scenario.tempDeltaC.toString());
      }
      if (typeof scenario.irrigationEffPct === 'number') {
        params.append('irrigationEffPct', scenario.irrigationEffPct.toString());
      }
      if (typeof scenario.fertilizerEfficiencyPct === 'number') {
        params.append('fertilizerEfficiencyPct', scenario.fertilizerEfficiencyPct.toString());
      }
    }
    const res = await fetch(`/api/analytics/digital-twin?${params.toString()}`);
    return handleJsonResponse(res);
  },

  askAssistant: async (payload: {
    message: string;
    language: string;
  }): Promise<{
    reply: string;
    languageUsed: string;
    isFallbackLanguage: boolean;
    sources: string[];
    timestamp: string;
  }> => {
    const res = await fetch('/api/assistant/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleJsonResponse(res);
  },

  getOperationsOverview: async (): Promise<OperationsOverviewResponse> => {
    const res = await fetch('/api/operations/overview');
    return handleJsonResponse<OperationsOverviewResponse>(res);
  },

  updatePlatformSettings: async (
    payload: Partial<PlatformSettings>
  ): Promise<{ success: boolean; settings: PlatformSettings }> => {
    const res = await fetch('/api/operations/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleJsonResponse(res);
  },

  deployMlModel: async (payload: {
    modelId: string;
    action: 'rollback' | 'promote';
  }): Promise<{ success: boolean; model: MlModelEntry; models: MlModelEntry[] }> => {
    const res = await fetch('/api/operations/mlops/deploy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleJsonResponse(res);
  },

  sendHardwareCommand: async (payload: {
    deviceId: string;
    command: 'wake_ping' | 'zero_calibrate' | 'gateway_selftest';
  }): Promise<{ success: boolean; device: DeviceTelemetry; timestamp: string }> => {
    const res = await fetch('/api/operations/hardware-command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleJsonResponse(res);
  },

  runDataOpsValidation: async (): Promise<{
    success: boolean;
    checkedNodes: number;
    staleNodes: number;
    timestamp: string;
  }> => {
    const res = await fetch('/api/operations/dataops/validate', {
      method: 'POST',
    });
    return handleJsonResponse(res);
  },
};
