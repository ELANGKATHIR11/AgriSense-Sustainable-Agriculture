/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface SensorReading {
  id: string;
  deviceId: string;
  timestamp: string;
  soilMoisture: number;
  temperature: number;
  humidity: number;
  pH: number;
  nitrogen: number;
  phosphorus: number;
  potassium: number;
}

export interface ModelRegistryEntry {
  id: string;
  name: string;
  version: string;
  type: 'crop_recommendation' | 'irrigation_optimization' | 'yield_prediction' | 'disease_detection';
  framework: string;
  status: 'active' | 'staging' | 'retired';
  accuracy: number;
  f1Score: number;
  lastRetrained: string;
  predictionCount: number;
  latencyMs: number;
}

export interface PredictionLog {
  id: string;
  timestamp: string;
  modelName: string;
  inputs: Record<string, any>;
  output: string | number | Record<string, any>;
  latencyMs: number;
  confidence: number;
  driftScore: number;
}

export interface DiseaseDetectionResult {
  disease: string;
  confidence: number;
  severity: 'low' | 'medium' | 'high';
  symptoms: string[];
  recommendations: string[];
  farmer_explanation?: string;
}

export interface CropRecommendationInput {
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  pH: number;
  temperature: number;
  humidity: number;
  rainfall: number;
}

export interface CropRecommendationResult {
  crops: Array<{
    name: string;
    suitability: number; // 0-100
    description: string;
    optimalConditions: string;
  }>;
  optimalPH: string;
  nutritionStatus: string;
}

export interface IrrigationInput {
  moisture: number;
  temperature: number;
  humidity: number;
  cropType: string;
}

export interface IrrigationResult {
  waterRequiredLiters: number;
  moistureStatus: string;
  advice: string;
  durationMinutes: number;
  irrigationSchedule: string;
}

export interface YieldInput {
  cropType: string;
  areaAcres: number;
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  avgRainfall: number;
  avgTemp: number;
}

export interface YieldResult {
  predictedYieldTons: number;
  confidenceMin: number;
  confidenceMax: number;
  marketValueEstimate: number;
  yieldBreakdown: string;
}

export interface WeatherDay {
  date: string;
  temperature: number;
  humidity: number;
  rainfall: number;
  condition: 'sunny' | 'rainy' | 'cloudy' | 'stormy';
  windSpeed: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  content: string;
  timestamp: string;
}

// --- DIGITAL TWIN CONTRACTS ---

export interface WaterTwinState {
  currentMoisture: number;
  predictedMoisture5Days: number[];
  evapotranspirationET0: number; // calculated via FAO-56 Penman-Monteith
  waterDeficitLiters: number;
  irrigationRecommendation: string;
  drainageRate: number;
  waterBalanceHistory: Array<{ day: string; rainfall: number; irrigation: number; et0: number; activeMoisture: number }>;
}

export interface SoilTwinState {
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  pH: number;
  electricalConductivity: number; // dS/m
  organicCarbon: number; // %
  healthScore: number; // 0-100
  nutrientDeficitForecast: string;
  depletionTimeline: Array<{ month: string; nitrogen: number; phosphorus: number; potassium: number }>;
}

export interface CropTwinState {
  cropType: string;
  sowingDate: string;
  growthStage: 'Germination' | 'Vegetative' | 'Flowering' | 'Yield Formation' | 'Ripening';
  biomassIndex: number; // kg/hectare
  cropHealthScore: number; // 0-100
  predictedYieldMultiplier: number;
  harvestForecastDate: string;
  growthTimeline: Array<{ stage: string; expectedBiomass: number; actualBiomass: number }>;
}

export interface WeatherTwinState {
  currentTemp: number;
  currentHumidity: number;
  windSpeed: number;
  rainProbability: number;
  heatStressIndex: 'Normal' | 'Moderate' | 'Severe' | 'Extreme';
  forecast: WeatherDay[];
  riskIndicators: string[];
}

export interface DiseaseTwinState {
  riskScore: number; // 0-100
  outbreakProbability: number; // %
  environmentalPropensity: number; // % humidity * temperature multiplier
  susceptibleCrops: string[];
  preventiveActionRequired: string[];
}

export interface FarmTwinState {
  overallHealthScore: number; // 0-100
  riskIndex: number; // 0-100
  yieldIndex: number; // 0-100
  sustainabilityIndex: number; // 0-100
  timestamp: string;
  waterTwin: WaterTwinState;
  soilTwin: SoilTwinState;
  cropTwin: CropTwinState;
  weatherTwin: WeatherTwinState;
  diseaseTwin: DiseaseTwinState;
  physicsModel?: {
    evapotranspirationET0: number;
    windSpeed: number;
    waterDeficitLiters: number;
    confidenceInterval: number[];
    uncertaintyMarginLiters: number;
  };
}

export interface ScenarioInput {
  scenarioId: 'drought_5_days' | 'fertilizer_boost_20' | 'downpour' | 'mildew_outbreak' | 'optimal_ops';
  soilMoistureChange: number;
  fertilizerNpkMultiplier: number;
  rainfallEventMm: number;
  tempShiftCelsius: number;
  humidityShiftPercent: number;
}

export interface ScenarioOutcomeTimeline {
  day: number;
  soilMoisture: number;
  cropHealth: number;
  yieldImpact: number;
  riskScore: number;
}

export interface ScenarioResult {
  scenarioName: string;
  outcomeSummary: string;
  projectedYieldTonsPerAcre: number;
  pestRiskScore: number;
  waterStressIndex: number;
  timeline: ScenarioOutcomeTimeline[];
}

