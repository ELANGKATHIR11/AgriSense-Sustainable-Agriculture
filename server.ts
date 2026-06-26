/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Body parsing with enlarged limit for base64 image uploads (plant disease images)
app.use(express.json({ limit: "15mb" }));
app.use(express.urlencoded({ limit: "15mb", extended: true }));

// --- Live DB/State Store ---
// Pre-seed mock sensor readings
let sensorReadings = [
  { id: "1", deviceId: "ESP32-S01", timestamp: new Date(Date.now() - 3600000 * 5).toISOString(), soilMoisture: 42.5, temperature: 27.8, humidity: 62.1, pH: 6.4, nitrogen: 45, phosphorus: 38, potassium: 42 },
  { id: "2", deviceId: "ESP32-S01", timestamp: new Date(Date.now() - 3600000 * 4).toISOString(), soilMoisture: 41.2, temperature: 28.1, humidity: 61.5, pH: 6.4, nitrogen: 46, phosphorus: 37, potassium: 43 },
  { id: "3", deviceId: "ESP32-S01", timestamp: new Date(Date.now() - 3600000 * 3).toISOString(), soilMoisture: 39.8, temperature: 28.5, humidity: 60.2, pH: 6.3, nitrogen: 45, phosphorus: 39, potassium: 41 },
  { id: "4", deviceId: "ESP32-S01", timestamp: new Date(Date.now() - 3600000 * 2).toISOString(), soilMoisture: 38.3, temperature: 28.9, humidity: 59.4, pH: 6.3, nitrogen: 47, phosphorus: 38, potassium: 42 },
  { id: "5", deviceId: "ESP32-S01", timestamp: new Date(Date.now() - 3600000 * 1).toISOString(), soilMoisture: 37.1, temperature: 29.2, humidity: 58.8, pH: 6.3, nitrogen: 44, phosphorus: 38, potassium: 44 },
];

let modelRegistry = [
  { id: "cm-01", name: "CropRecommendation-XGBoost", version: "v2.1.0", type: "crop_recommendation", framework: "XGBoost", status: "active", accuracy: 0.942, f1Score: 0.938, lastRetrained: "2026-05-28T04:22:00Z", predictionCount: 1450, latencyMs: 14 },
  { id: "cm-02", name: "CropRecommendation-LightGBM", version: "v2.0.8", type: "crop_recommendation", framework: "LightGBM", status: "staging", accuracy: 0.939, f1Score: 0.935, lastRetrained: "2026-06-01T11:15:00Z", predictionCount: 120, latencyMs: 8 },
  { id: "ir-01", name: "IrrigationIrrig-RandomForest", version: "v1.4.2", type: "irrigation_optimization", framework: "RandomForest", status: "active", accuracy: 0.915, f1Score: 0.908, lastRetrained: "2026-05-20T08:00:00Z", predictionCount: 890, latencyMs: 22 },
  { id: "yd-01", name: "YieldPredictor-CatBoost", version: "v3.0.1", type: "yield_prediction", framework: "CatBoost", status: "active", accuracy: 0.887, f1Score: 0.879, lastRetrained: "2026-05-25T14:30:00Z", predictionCount: 620, latencyMs: 19 },
  { id: "vs-01", name: "PlantDisease-SmolVLM-3B", version: "v1.0.0", type: "disease_detection", framework: "HuggingFace SmolVLM", status: "active", accuracy: 0.923, f1Score: 0.919, lastRetrained: "2026-05-10T12:00:00Z", predictionCount: 1120, latencyMs: 185 },
];

let predictionLogs = [
  { id: "pl-1", timestamp: new Date(Date.now() - 60000 * 30).toISOString(), modelName: "CropRecommendation-XGBoost", inputs: { N: 50, P: 40, K: 40, pH: 6.5, temp: 28, hum: 60, rainfall: 120 }, output: "Maize (92% confidence)", latencyMs: 15, confidence: 0.92, driftScore: 0.04 },
  { id: "pl-2", timestamp: new Date(Date.now() - 60000 * 20).toISOString(), modelName: "IrrigationIrrig-RandomForest", inputs: { moisture: 35, temp: 29, hum: 58 }, output: "Irrigation recommended: 1200L/acre", latencyMs: 20, confidence: 0.94, driftScore: 0.08 },
  { id: "pl-3", timestamp: new Date(Date.now() - 60000 * 10).toISOString(), modelName: "PlantDisease-SmolVLM-3B", inputs: { imageUploaded: true }, output: "Tomato Late Blight (91% confidence)", latencyMs: 190, confidence: 0.91, driftScore: 0.01 },
];

// --- Dynamic Simulation Methods ---
function simulateCropRecommendation(input: any) {
  const { nitrogen: N, phosphorus: P, potassium: K, pH, temperature: T, humidity: H, rainfall: R } = input;
  
  const cropsList: Array<{ name: string; score: number; desc: string; optimal: string }> = [
    { name: "Rice", score: 0, desc: "High water retaining crop, thrives in clayey loam.", optimal: "N: 80, P: 40, K: 40, pH: 5.5-6.5, Temp: 20-35C, Rain: >150" },
    { name: "Maize", score: 0, desc: "Adaptable coarse grain, medium nutrient feeder.", optimal: "N: 60, P: 35, K: 35, pH: 5.8-7.2, Temp: 18-30C, Rain: 60-110" },
    { name: "Chickpea", score: 0, desc: "Legume fixing Nitrogen, prefers well-drained loam.", optimal: "N: 20, P: 60, K: 30, pH: 6.0-7.5, Temp: 15-25C, Rain: 35-60" },
    { name: "Cotton", score: 0, desc: "Fiber crop, desires warm climate and heavy feeding.", optimal: "N: 100, P: 50, K: 50, pH: 6.0-8.0, Temp: 22-38C, Rain: 50-80" },
    { name: "Soybeans", score: 0, desc: "High protein oilseed, requires moderate climate.", optimal: "N: 30, P: 70, K: 45, pH: 6.0-7.0, Temp: 20-30C, Rain: 70-120" },
    { name: "Golden Melon", score: 0, desc: "Premium horticulture crop, susceptible to extreme floods.", optimal: "N: 50, P: 30, K: 80, pH: 6.0-6.8, Temp: 24-35C, Rain: 40-70" },
  ];

  cropsList.forEach((crop) => {
    let matchScore = 100;
    
    if (crop.name === "Rice" && (pH < 5.0 || pH > 7.0)) matchScore -= 25;
    if (crop.name === "Maize" && (pH < 5.5 || pH > 7.5)) matchScore -= 20;
    if (crop.name === "Chickpea" && (pH < 6.0 || pH > 8.0)) matchScore -= 30;

    if (crop.name === "Rice" && (N < 50)) matchScore -= 20;
    if (crop.name === "Cotton" && (N < 70)) matchScore -= 30;
    if (crop.name === "Chickpea" && N > 50) matchScore -= 20;

    if (crop.name === "Golden Melon" && K < 60) matchScore -= 40;

    if (crop.name === "Rice" && R < 120) matchScore -= 40;
    if (crop.name === "Chickpea" && R > 80) matchScore -= 35;

    crop.score = Math.max(15, Math.min(98, Math.round(matchScore + (Math.random() * 8 - 4))));
  });

  const sortedCrops = cropsList.sort((a, b) => b.score - a.score);
  return {
    crops: sortedCrops.slice(0, 3).map(c => ({
      name: c.name,
      suitability: c.score,
      description: c.desc,
      optimalConditions: c.optimal
    })),
    optimalPH: pH >= 6.0 && pH <= 7.0 ? "Healthy neutral optimal pH zone" : pH < 6.0 ? "Acidic sub-level; consider agricultural lime addition." : "Alkaline level; soil sulfur recommended.",
    nutritionStatus: `Nitrogen level (${N} ppm) ${N > 40 ? "is adequate" : "is low; boost organic mulching"}. Phosphorus (${P} ppm) and Potassium (${K} ppm) are balanced for primary horticulture.`
  };
}

function simulateIrrigationOptimization(input: any) {
  const { moisture, temperature, humidity, cropType } = input;
  let baselineMoisture = 45;
  let moistureDiff = Math.max(0, baselineMoisture - moisture);
  
  let evapFactor = 1.0;
  if (temperature > 30) evapFactor += 0.25;
  if (humidity < 50) evapFactor += 0.15;

  let waterRequiredLiters = Math.round(moistureDiff * 80 * evapFactor);
  if (moisture > 45) waterRequiredLiters = 0;

  let status = "Adequate";
  let duration = 0;
  let advice = "No watering sequence needed at this moisture level.";

  if (waterRequiredLiters > 0) {
    status = moisture < 20 ? "CRITICAL UNDERWATERED" : "Moderate Moisture Stress";
    duration = Math.round(waterRequiredLiters / 40);
    advice = `Triggering immediate micro-drip sequence for ${cropType || "crops"}. High soil ambient matrix suction corrected.`;
  }

  return {
    waterRequiredLiters,
    moistureStatus: status,
    advice,
    durationMinutes: duration,
    irrigationSchedule: waterRequiredLiters > 0 ? "Daily, split over dawn/dusk intervals" : "Standby until soil moisture dips below 35%."
  };
}

function simulateYieldPrediction(input: any) {
  const { areaAcres, avgRainfall, avgTemp, cropType } = input;
  
  let baseYieldPerAcre = 2.4;
  if (cropType === "Rice") baseYieldPerAcre = 3.6;
  if (cropType === "Maize") baseYieldPerAcre = 4.2;
  if (cropType === "Cotton") baseYieldPerAcre = 1.8;
  if (cropType === "Soybeans") baseYieldPerAcre = 2.0;

  let tempDiff = Math.abs(avgTemp - 25);
  let tempFactor = Math.max(0.7, 1.0 - (tempDiff * 0.02));

  let rainFactor = 1.0;
  if (avgRainfall < 50) rainFactor = 0.6;
  else if (avgRainfall > 150) rainFactor = cropType === "Rice" ? 1.2 : 0.75;

  const area = areaAcres || 1;
  const predictedYield = parseFloat((baseYieldPerAcre * area * tempFactor * rainFactor).toFixed(2));
  const confidenceMin = parseFloat((predictedYield * 0.9).toFixed(2));
  const confidenceMax = parseFloat((predictedYield * 1.08).toFixed(2));
  const marketPricePerTon = cropType === "Maize" ? 220 : cropType === "Rice" ? 310 : cropType === "Cotton" ? 480 : 350;

  return {
    predictedYieldTons: predictedYield,
    confidenceMin,
    confidenceMax,
    marketValueEstimate: Math.round(predictedYield * marketPricePerTon),
    yieldBreakdown: `Estimated via Agrisense YieldPredictor v3.0 (CatBoost core). Factors analyzed: Climate suitability ${Math.round(tempFactor * 100)}%, moisture stress coefficient ${Math.round(rainFactor * 100)}% with ${area} acres under calculation.`
  };
}

// --- API Router Endpoints ---

// Get pre-loaded/active IoT readings
app.get("/api/sensors", (req, res) => {
  res.json({ readings: sensorReadings });
});

// Post action for simulated ESP32 device
app.post("/api/sensors/ingest", (req, res) => {
  const { deviceId, soilMoisture, temperature, humidity, pH, nitrogen, phosphorus, potassium } = req.body;
  const newReading = {
    id: String(sensorReadings.length + 1),
    deviceId: deviceId || "ESP32-S02",
    timestamp: new Date().toISOString(),
    soilMoisture: Number(soilMoisture ?? 40),
    temperature: Number(temperature ?? 28),
    humidity: Number(humidity ?? 60),
    pH: Number(pH ?? 6.5),
    nitrogen: Number(nitrogen ?? 40),
    phosphorus: Number(phosphorus ?? 40),
    potassium: Number(potassium ?? 40)
  };
  sensorReadings.unshift(newReading);
  if (sensorReadings.length > 50) sensorReadings.pop();
  res.json({ message: "Data ingested successfully", logged: newReading });
});

// Run Tabular ML prediction (Crop Recommendation)
app.post("/api/crop-recommend", async (req, res) => {
  try {
    const input = req.body;
    const mathResult = simulateCropRecommendation(input);
    
    // Log in MLOps prediction log registry
    const logId = "pl-" + Math.random().toString(36).substring(4, 9);
    const newLog = {
      id: logId,
      timestamp: new Date().toISOString(),
      modelName: "CropRecommendation-XGBoost",
      inputs: input,
      output: `${mathResult.crops[0]?.name} (${mathResult.crops[0]?.suitability}% confidence)`,
      latencyMs: 14 + Math.round(Math.random() * 5),
      confidence: (mathResult.crops[0]?.suitability || 100) / 100,
      driftScore: parseFloat((Math.random() * 0.08).toFixed(3))
    };
    predictionLogs.unshift(newLog);

    res.json(mathResult);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Run Tabular ML prediction (Irrigation Optimization)
app.post("/api/irrigation-optimize", async (req, res) => {
  try {
    const input = req.body;
    const mathResult = simulateIrrigationOptimization(input);

    const logId = "pl-" + Math.random().toString(36).substring(4, 9);
    predictionLogs.unshift({
      id: logId,
      timestamp: new Date().toISOString(),
      modelName: "IrrigationIrrig-RandomForest",
      inputs: input,
      output: `Water req: ${mathResult.waterRequiredLiters}L`,
      latencyMs: 18 + Math.round(Math.random() * 8),
      confidence: mathResult.waterRequiredLiters > 0 ? 0.92 : 0.97,
      driftScore: parseFloat((Math.random() * 0.08).toFixed(3))
    });

    res.json(mathResult);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Run Tabular ML prediction (Yield Prediction)
app.post("/api/yield-predict", async (req, res) => {
  try {
    const input = req.body;
    const mathResult = simulateYieldPrediction(input);

    const logId = "pl-" + Math.random().toString(36).substring(4, 9);
    predictionLogs.unshift({
      id: logId,
      timestamp: new Date().toISOString(),
      modelName: "YieldPredictor-CatBoost",
      inputs: input,
      output: `${mathResult.predictedYieldTons} tons forecast`,
      latencyMs: 15 + Math.round(Math.random() * 5),
      confidence: 0.89,
      driftScore: parseFloat((Math.random() * 0.05).toFixed(3))
    });

    res.json(mathResult);
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Run MLOps Data
app.get("/api/mlops", (req, res) => {
  res.json({
    metrics: {
      averageAccuracy: 0.917,
      inferenceCount: 1450 + 890 + 620 + predictionLogs.length,
      averageLatencyMs: 32,
      activeModelsCount: modelRegistry.filter(m => m.status === "active").length,
      anomalousInferences: 2,
      driftIndex: 0.045
    },
    registry: modelRegistry,
    logs: predictionLogs
  });
});

// Run Action for Model Retraining (Simulation)
app.post("/api/mlops/retrain", (req, res) => {
  const { modelId } = req.body;
  const modelIdx = modelRegistry.findIndex(m => m.id === modelId);
  if (modelIdx !== -1) {
    modelRegistry[modelIdx].lastRetrained = new Date().toISOString();
    modelRegistry[modelIdx].accuracy = parseFloat((modelRegistry[modelIdx].accuracy + (Math.random() * 0.012 - 0.004)).toFixed(3));
    modelRegistry[modelIdx].f1Score = parseFloat((modelRegistry[modelIdx].f1Score + (Math.random() * 0.01 - 0.003)).toFixed(3));
    modelRegistry[modelIdx].version = `v${parseFloat(modelRegistry[modelIdx].version.replace('v', '')) + 0.1}`;
    res.json({ message: "Retraining completed", updated: modelRegistry[modelIdx] });
  } else {
    res.status(404).json({ error: "Model not found in MLOps registry" });
  }
});

// Vision Crop Disease and Weed Detection (SmolVLM 3B native simulation)
app.post("/api/disease-detect", async (req, res) => {
  const { imageBase64, mode } = req.body;
  
  if (!imageBase64) {
    return res.status(400).json({ error: "Image content is required for CV inference" });
  }

  const fallbackPredictions = [
    { disease: "Tomato Leaf Mold", confidence: 94.5, severity: "medium", symptoms: ["Yellow spots on upper leaf surfaces", "Olive-green velvet-like mold on under-leaves", "Curling foliage"], recommendations: ["Improve ventilation in greenhouse", "Avoid overhead crop watering", "Apply copper-based biological fungicide"] },
    { disease: "Powdery Mildew on Squash", confidence: 88.2, severity: "low", symptoms: ["White talcum-like powdery spots on leaves and stems", "Premature leaf defoliation", "Stunted vegetative growth"], recommendations: ["Ensure full direct sunlight", "Space plants adequately to circulate air", "Apply neem oil extract or potassium bicarbonate"] },
    { disease: "Broadleaf Weed (Pigweed)", confidence: 91.0, severity: "high", symptoms: ["Erect red/green weed clusters competing with primary crop rootzones", "Aggressive moisture depletion", "Rapid seed dispersal heads spreading"], recommendations: ["Targeted localized weed extraction", "Instate organic cover compost layers to suppress germination", "Utilize selective organic pre-emergents"] },
    { disease: "Healthy Vegetation", confidence: 97.4, severity: "low", symptoms: ["Vibrant chloroplast color", "Good structural turgor pressure", "No trace of pathogenic necrosis or insect scoring"], recommendations: ["Persist current irrigation optimization rules", "Maintain biological companion planting", "Document baseline metrics in Agrisense dashboard"] }
  ];

  const mockAns = fallbackPredictions[Math.floor(Math.random() * fallbackPredictions.length)];
  predictionLogs.unshift({
    id: "pl-" + Math.random().toString(36).substring(4, 9),
    timestamp: new Date().toISOString(),
    modelName: "PlantDisease-SmolVLM-3B",
    inputs: { imageUploaded: true },
    output: `${mockAns.disease} (${mockAns.confidence}% confidence)`,
    latencyMs: 140,
    confidence: mockAns.confidence / 100,
    driftScore: 0.02
  });
  
  return res.json(mockAns);
});

// AgriGPT Interactive Assistant Agent (Offline/Simulated Expert Advisory)
app.post("/api/chat", async (req, res) => {
  const { messages } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "Valid message history is required" });
  }

  const lastUserMessage = messages[messages.length - 1]?.content?.toLowerCase() || "";
  let answer = "I am AgriGPT, your Agri-Intelligence Agent. I can assist you with fertilizer ratios, ESP32 sensors, crops, or irrigation optimization. What would you like to build or analyze?";
  
  if (lastUserMessage.includes("nitrogen") || lastUserMessage.includes("npk") || lastUserMessage.includes("fertilizer")) {
    answer = "#### Nitrogen & NPK Nutrition Note:\n- **Nitrogen (N)**: Crucial for photosynthetic leaf/shoot production.\n- **Phosphorus (P)**: Supports robust early root systems & flowering.\n- **Potassium (K)**: Enforces cellular turgor, water regulation, and pathogen resistance.\n\n*Agronomic recommendation*: Utilize cover combinations (cloves, legumes) or organic blood meal when nitrogen levels drop below 40 ppm.";
  } else if (lastUserMessage.includes("wheat") || lastUserMessage.includes("rice") || lastUserMessage.includes("crop")) {
    answer = "#### Crop Suitability Recommendations:\nOur tabular ML models evaluate soil parameters such as **pH** (optimal range: 6.0 - 7.0), moisture, and temperature. For waterlogged fields or heavy clay, high-suitability crops include **Rice** (80-90% suitability), whereas well-drained loamy soils favor **Maize** and **Soybeans**.";
  } else if (lastUserMessage.includes("esp32") || lastUserMessage.includes("sensor") || lastUserMessage.includes("hardware")) {
    answer = "#### ESP32 Wiring & Hardware Guide:\n- **Soil Moisture**: Capacitive sensor v1.2 connected to GPIO34 pin (ADC1).\n- **Temp/Humidity DHT22**: Out pin routed to GPIO15.\n- **Solenoid Relay**: Gate terminal connected to GPIO12.\n\nUse the simulated telemetry panel or check out the C++ code firmware under **IoT Telemetry Control Hub** to copy the complete flashing script.";
  } else if (lastUserMessage.includes("disease") || lastUserMessage.includes("pathogen")) {
    answer = "#### Disease detection computer vision model:\nOur SmolVLM-3B model analyzes crop leaves in real-time. If you notice yellowing spots or powdery patterns, it often indicates Tomato Leaf Mold or Powdery Mildew. You can test this instantly by dragging and dropping sample crop leaves into the **Disease Vision** tab!";
  }

  res.json({ text: answer });
});

// ==========================================
// --- AGRISENSE DIGITAL TWIN SIMULATOR ENGINE ---
// ==========================================

// Physics-Based FAO-56 Penman-Monteith Evapotranspiration ET0 Model
function calculateFAO56ET0(T: number, RH: number, windSpeed: number, Rn: number = 15): number {
  // Saturation vapor pressure (kPa)
  const es = 0.6108 * Math.exp((17.27 * T) / (T + 237.3));
  // Actual vapor pressure (kPa)
  const ea = es * (RH / 100);
  // Slope of vapor pressure curve (kPa/C)
  const delta = (4098 * es) / Math.pow(T + 237.3, 2);
  // Psychrometric constant (kPa/C)
  const gamma = 0.066;
  
  // FAO-56 Penman-Monteith simplified daily reference evapotranspiration
  const numerator = 0.408 * delta * Rn + gamma * (900 / (T + 273)) * windSpeed * (es - ea);
  const denominator = delta + gamma * (1 + 0.34 * windSpeed);
  
  const et0 = numerator / denominator;
  return parseFloat(Math.max(1.0, Math.min(10.0, et0)).toFixed(2));
}

// Residual AI Correction Layer: Physics Prediction + AI Correction = Final Prediction
function predictMoistureWithAICorrection(currentMoisture: number, et0: number, rain: number, irrigation: number, daysAhead: number): number[] {
  let predictions: number[] = [];
  let tempM = currentMoisture;
  
  for (let d = 1; d <= daysAhead; d++) {
    // 1. Physics Engine state transition model
    // Moisture depletion proportional to ET0; saturated topsoil drains; rain/irrigation refills.
    const drainage = tempM > 45 ? 1.2 : 0.4;
    const physicsMoisturePredict = tempM + (rain * 0.4) + (irrigation * 0.3) - (et0 * 0.8) - drainage;
    
    // 2. Tabular AI (XGBoost/LightGBM proxy) Residual Correction for local non-linear variables
    // Simulating moisture adsorption, soil porosity, and microclimate canopy shading variables
    const aiCorrection = Math.sin(d * 0.6) * 0.4 - (tempM < 25 ? -0.5 : 0.2);
    
    // Final residual prediction integration
    tempM = parseFloat(Math.max(12.0, Math.min(90.0, physicsMoisturePredict + aiCorrection)).toFixed(1));
    predictions.push(tempM);
  }
  return predictions;
}

// Anomaly Detection Partition Engine (Isolation Forest approximation)
function detectSensorAnomalies(humidity: number, moisture: number, temp: number, pH: number, nitrogen: number): { hasAnomaly: boolean; alerts: string[] } {
  const alerts: string[] = [];
  // Check bounds defining three-sigma path partitions
  if (moisture < 15.0) {
    alerts.push("CRITICAL ANOMALY: Soil desiccation detected (< 15%). Potential sensor dry-air exposure.");
  } else if (moisture > 92.0) {
    alerts.push("CRITICAL ANOMALY: High water table inundation (> 92%). Potential waterlogging or valve leak.");
  }
  
  if (temp < 4.0 || temp > 47.0) {
    alerts.push(`CRITICAL ANOMALY: Extreme thermal outlier logged (${temp}°C). Potential thermal exposure.`);
  }
  
  if (pH < 4.5 || pH > 9.2) {
    alerts.push(`CRITICAL ANOMALY: Extreme chemical soil drift detected (pH: ${pH}).`);
  }
  
  if (nitrogen < 5 || nitrogen > 150) {
    alerts.push(`CRITICAL ANOMALY: Abnormal NPK reading (N: ${nitrogen} ppm). Electrode connectivity check recommended.`);
  }
  
  return {
    hasAnomaly: alerts.length > 0,
    alerts
  };
}

// Master Digital Twin Consolidated In-Memory Store
let currentTwinState = {
  overallHealthScore: 88,
  riskIndex: 12,
  yieldIndex: 94,
  sustainabilityIndex: 92,
  timestamp: new Date().toISOString(),
  waterTwin: {
    currentMoisture: 38.3,
    predictedMoisture5Days: [36.8, 35.1, 33.4, 31.2, 29.5],
    evapotranspirationET0: 4.82,
    waterDeficitLiters: 1250,
    irrigationRecommendation: "Trigger standard early-morning microflushes on Grid Zone 4: 1250 Liters required",
    drainageRate: 0.6,
    waterBalanceHistory: [
      { day: "Mon", rainfall: 0.0, irrigation: 0, et0: 4.2, activeMoisture: 42.5 },
      { day: "Tue", rainfall: 0.0, irrigation: 0, et0: 4.5, activeMoisture: 41.2 },
      { day: "Wed", rainfall: 0.0, irrigation: 0, et0: 4.8, activeMoisture: 39.8 },
      { day: "Thu", rainfall: 1.2, irrigation: 1200, et0: 4.4, activeMoisture: 44.5 },
      { day: "Fri", rainfall: 0.0, irrigation: 0, et0: 4.7, activeMoisture: 41.2 },
      { day: "Sat", rainfall: 0.0, irrigation: 0, et0: 4.9, activeMoisture: 38.3 }
    ]
  },
  soilTwin: {
    nitrogen: 45,
    phosphorus: 38,
    potassium: 42,
    pH: 6.4,
    electricalConductivity: 1.2,
    organicCarbon: 2.1,
    healthScore: 89,
    nutrientDeficitForecast: "Nitrogen depletion window predicted within 45 days of vegetative vigor.",
    depletionTimeline: [
      { month: "June", nitrogen: 45, phosphorus: 38, potassium: 42 },
      { month: "July", nitrogen: 43, phosphorus: 37, potassium: 40 },
      { month: "August", nitrogen: 39, phosphorus: 36, potassium: 38 },
      { month: "September", nitrogen: 34, phosphorus: 34, potassium: 35 }
    ]
  },
  cropTwin: {
    cropType: "Sweet Maize",
    sowingDate: "2026-04-12",
    growthStage: "Vegetative" as 'Germination' | 'Vegetative' | 'Flowering' | 'Yield Formation' | 'Ripening',
    biomassIndex: 1850,
    cropHealthScore: 86,
    predictedYieldMultiplier: 1.05,
    harvestForecastDate: "2026-08-30",
    growthTimeline: [
      { stage: "Germination", expectedBiomass: 200, actualBiomass: 210 },
      { stage: "Vegetative", expectedBiomass: 1500, actualBiomass: 1850 },
      { stage: "Flowering", expectedBiomass: 3500, actualBiomass: 0 },
      { stage: "Yield Formation", expectedBiomass: 6000, actualBiomass: 0 }
    ]
  },
  weatherTwin: {
    currentTemp: 28.5,
    currentHumidity: 62.0,
    windSpeed: 8.4,
    rainProbability: 25,
    heatStressIndex: "Moderate" as 'Normal' | 'Moderate' | 'Severe' | 'Extreme',
    forecast: [
      { date: "Wednesday", temperature: 28.5, humidity: 62.0, rainfall: 0.0, condition: "sunny" as const, windSpeed: 8.4 },
      { date: "Thursday", temperature: 29.1, humidity: 58.5, rainfall: 1.2, condition: "cloudy" as const, windSpeed: 9.6 },
      { date: "Friday", temperature: 24.8, humidity: 82.0, rainfall: 24.5, condition: "stormy" as const, windSpeed: 18.2 },
      { date: "Saturday", temperature: 26.2, humidity: 75.4, rainfall: 4.8, condition: "rainy" as const, windSpeed: 12.0 },
      { date: "Sunday", temperature: 27.9, humidity: 64.1, rainfall: 0.0, condition: "sunny" as const, windSpeed: 7.8 },
      { date: "Monday", temperature: 28.2, humidity: 60.5, rainfall: 0.0, condition: "sunny" as const, windSpeed: 8.0 },
      { date: "Tuesday", temperature: 28.8, humidity: 59.0, rainfall: 0.0, condition: "sunny" as const, windSpeed: 8.5 }
    ],
    riskIndicators: ["Storm convection event on Friday afternoon.", "Atmospheric mold propensity increases post-precipitation."]
  },
  diseaseTwin: {
    riskScore: 18,
    outbreakProbability: 15,
    environmentalPropensity: 22,
    susceptibleCrops: ["Tomato Leaf Mold", "Powdery Mildew on Squash"],
    preventiveActionRequired: ["Boost high-tunnel air circulation", "Transition watering to ground drip lines"]
  }
};

// POST /api/twin/update - Recompute digital twins based on incoming IoT telemetry & FAO-56
app.post("/api/twin/update", (req, res) => {
  try {
    const { soilMoisture, temperature, humidity, pH, nitrogen, phosphorus, potassium, rainfall, windSpeed } = req.body;
    
    // Fallbacks based on active telemetry if not supplied
    const currentM = typeof soilMoisture === "number" ? soilMoisture : currentTwinState.waterTwin.currentMoisture;
    const currentT = typeof temperature === "number" ? temperature : currentTwinState.weatherTwin.currentTemp;
    const currentH = typeof humidity === "number" ? humidity : currentTwinState.weatherTwin.currentHumidity;
    const currentpH = typeof pH === "number" ? pH : currentTwinState.soilTwin.pH;
    const currentN = typeof nitrogen === "number" ? nitrogen : currentTwinState.soilTwin.nitrogen;
    const currentP = typeof phosphorus === "number" ? phosphorus : currentTwinState.soilTwin.phosphorus;
    const currentK = typeof potassium === "number" ? potassium : currentTwinState.soilTwin.potassium;
    const currentWind = typeof windSpeed === "number" ? windSpeed : currentTwinState.weatherTwin.windSpeed;
    const currentRain = typeof rainfall === "number" ? rainfall : 0.0;
    
    // 1. Calculate FAO-56 PM Evapotranspiration
    const computedET0 = calculateFAO56ET0(currentT, currentH, currentWind);
    
    // 2. Perform Physical Water Balance simulation
    // PM predicted moisture
    const predMoisture = predictMoistureWithAICorrection(currentM, computedET0, currentRain, 0, 5);
    
    // 3. Compute Deficit
    let baselineTarget = 42.0;
    let deficitMoisture = Math.max(0, baselineTarget - currentM);
    let waterDeficitLiters = Math.round(deficitMoisture * 90);
    
    // 4. Determine Soil Health Index
    let pHScore = 100 - Math.min(60, Math.abs(currentpH - 6.5) * 45);
    let nitrogenFactor = currentN > 40 ? 100 : (currentN / 40) * 100;
    let computedSoilScore = Math.round((pHScore + nitrogenFactor + 90 + 95) / 4);

    // 5. Update state
    currentTwinState.timestamp = new Date().toISOString();
    currentTwinState.waterTwin.currentMoisture = currentM;
    currentTwinState.waterTwin.predictedMoisture5Days = predMoisture;
    currentTwinState.waterTwin.evapotranspirationET0 = computedET0;
    currentTwinState.waterTwin.waterDeficitLiters = waterDeficitLiters;
    currentTwinState.waterTwin.irrigationRecommendation = waterDeficitLiters > 0 
      ? `Moisture deficit flagged. Irrigate immediately with ${waterDeficitLiters} Liters on active grid.` 
      : "Moisture saturation within optimal range. No irrigation required.";
    
    currentTwinState.soilTwin.nitrogen = currentN;
    currentTwinState.soilTwin.phosphorus = currentP;
    currentTwinState.soilTwin.potassium = currentK;
    currentTwinState.soilTwin.pH = currentpH;
    currentTwinState.soilTwin.healthScore = computedSoilScore;
    
    currentTwinState.weatherTwin.currentTemp = currentT;
    currentTwinState.weatherTwin.currentHumidity = currentH;
    currentTwinState.weatherTwin.windSpeed = currentWind;
    currentTwinState.weatherTwin.heatStressIndex = currentT > 32 ? "Severe" : currentT > 28 ? "Moderate" : "Normal";
    
    // 6. Disease Outbreak Calculations based on microclimate
    let weatherIntensity = (currentH / 100) * (currentT / 25);
    let pathogenRisk = Math.min(98, Math.round(weatherIntensity * 45 + (currentM > 55 ? 25 : 0)));
    currentTwinState.diseaseTwin.riskScore = pathogenRisk;
    currentTwinState.diseaseTwin.outbreakProbability = Math.round(pathogenRisk * 0.9);
    currentTwinState.diseaseTwin.environmentalPropensity = Math.round(weatherIntensity * 100);

    // 7. Overall Farm Indices
    currentTwinState.overallHealthScore = Math.round((computedSoilScore + currentTwinState.cropTwin.cropHealthScore + (100 - pathogenRisk) + (currentM > 25 && currentM < 55 ? 100 : 50)) / 4);
    currentTwinState.riskIndex = Math.round((pathogenRisk + (currentM < 25 ? 45 : 0) + (currentT > 35 ? 30 : 0)) / 2);
    currentTwinState.yieldIndex = Math.round(currentTwinState.cropTwin.predictedYieldMultiplier * 85);
    
    // Check for sensor anomalies using Isolate Forest limits and append to alerts
    const detection = detectSensorAnomalies(currentH, currentM, currentT, currentpH, currentN);
    
    res.json({
      status: "updated",
      hasAnomaly: detection.hasAnomaly,
      anomalies: detection.alerts,
      twinState: currentTwinState
    });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/twin/state - Returns master Twin consolidation state
app.get("/api/twin/state", (req, res) => {
  res.json(currentTwinState);
});

// GET /api/twin/water - Returns specialized Water Twin State
app.get("/api/twin/water", (req, res) => {
  res.json(currentTwinState.waterTwin);
});

// GET /api/twin/soil - Returns specialized Soil Twin State
app.get("/api/twin/soil", (req, res) => {
  res.json(currentTwinState.soilTwin);
});

// GET /api/twin/crop - Returns specialized Crop Twin State
app.get("/api/twin/crop", (req, res) => {
  res.json(currentTwinState.cropTwin);
});

// GET /api/twin/weather - Returns specialized Weather Twin State
app.get("/api/twin/weather", (req, res) => {
  res.json(currentTwinState.weatherTwin);
});

// GET /api/twin/disease - Returns specialized Disease Twin State
app.get("/api/twin/disease", (req, res) => {
  res.json(currentTwinState.diseaseTwin);
});

// POST /api/twin/simulate - Scenarios what-if simulation execution
app.post("/api/twin/simulate", (req, res) => {
  try {
    const { scenarioId } = req.body;
    
    let scenarioName = "Ideal Operating Baseline";
    let outcomeSummary = "Optimal state parameter ranges; vegetative leaf development peaking.";
    let projectedYield = 4.2; // tons/acre
    let pestRisk = 12;
    let waterStress = 0.08;
    
    let timeline: any[] = [];
    
    if (scenarioId === "drought_5_days") {
      scenarioName = "Scenario 1: No Irrigation (5 Days)";
      outcomeSummary = "Severe moisture deficiency. Crop tissue suction triggers secondary root collapse without immediate drip valve override.";
      projectedYield = 2.4;
      pestRisk = 18;
      waterStress = 0.84;
      
      let moisture = 38.3;
      for (let i = 1; i <= 5; i++) {
        moisture -= 4.2;
        timeline.push({
          day: i,
          soilMoisture: parseFloat(moisture.toFixed(1)),
          cropHealth: Math.max(40, 86 - i * 8),
          yieldImpact: parseFloat((100 - i * 6.5).toFixed(1)),
          riskScore: Math.round(18 + i * 14)
        });
      }
    } else if (scenarioId === "fertilizer_boost_20") {
      scenarioName = "Scenario 2: Increase Nitrogen & Potassium by 20%";
      outcomeSummary = "Accelerated plant height expansion. Chloroplast densities optimized via high inorganic feeding ratios.";
      projectedYield = 4.8;
      pestRisk = 14;
      waterStress = 0.07;
      
      let cropH = 86;
      for (let i = 1; i <= 5; i++) {
        cropH = Math.min(99, cropH + 1.8);
        timeline.push({
          day: i,
          soilMoisture: 38.3,
          cropHealth: Math.round(cropH),
          yieldImpact: parseFloat((100 + i * 2.1).toFixed(1)),
          riskScore: 12
        });
      }
    } else if (scenarioId === "downpour") {
      scenarioName = "Scenario 3: Severe Climate Event (50mm Heavy Rainfall)";
      outcomeSummary = "Saturated soil structure inducing root hypoxia conditions. Runoff may wash off vital trace fertilizers.";
      projectedYield = 3.8;
      pestRisk = 65;
      waterStress = 0.0;
      
      let moisture = 38.3;
      let health = 86;
      for (let i = 1; i <= 5; i++) {
        if (i === 1) moisture = 85.0;
        else moisture -= 3.5;
        health = Math.max(70, health - 1.2);
        timeline.push({
          day: i,
          soilMoisture: parseFloat(moisture.toFixed(1)),
          cropHealth: Math.round(health),
          yieldImpact: parseFloat((100 - i * 1.5).toFixed(1)),
          riskScore: Math.round(18 + i * 8)
        });
      }
    } else if (scenarioId === "mildew_outbreak") {
      scenarioName = "Scenario 4: Fungal Disease Outbreak Propagation";
      outcomeSummary = "Active pathogen spots confirmed. Foliar necrotic lesions reduce total photosynthetic solar assimilation capability.";
      projectedYield = 2.8;
      pestRisk = 94;
      waterStress = 0.12;
      
      let health = 86;
      for (let i = 1; i <= 5; i++) {
        health -= 7.5;
        timeline.push({
          day: i,
          soilMoisture: 38.3,
          cropHealth: Math.round(health),
          yieldImpact: parseFloat((100 - i * 8.2).toFixed(1)),
          riskScore: Math.round(40 + i * 10)
        });
      }
    } else {
      // optimal_ops or others
      scenarioName = "Scenario 5: Optimal Automated Cyber-Operations";
      outcomeSummary = "All systems green. Active machine learning adjustments controlling water balance limits dynamically.";
      projectedYield = 4.95;
      pestRisk = 5;
      waterStress = 0.02;
      
      for (let i = 1; i <= 5; i++) {
        timeline.push({
          day: i,
          soilMoisture: 40.0 + Math.sin(i) * 0.5,
          cropHealth: 92 + Math.min(7, i),
          yieldImpact: parseFloat((100 + i * 2.8).toFixed(1)),
          riskScore: 5
        });
      }
    }
    
    res.json({
      scenarioName,
      outcomeSummary,
      projectedYieldTonsPerAcre: projectedYield,
      pestRiskScore: pestRisk,
      waterStressIndex: waterStress,
      timeline
    });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/twin/analytics - Consolidated KPI analysis
app.get("/api/twin/analytics", (req, res) => {
  res.json({
    kpis: {
      healthIndexHistory: [81, 83, 85, 84, 88],
      waterConservationLiters: 14500,
      carbonOffsetPercentage: 11.4,
      nitrogenUtilizationRate: 91.2
    },
    sustainabilityIndices: {
      waterUseEfficiency: 94,
      pesticideReductionIndex: 88,
      soilStructuralRetention: 91
    }
  });
});

// Handle static production files and SPA fallback
const distPath = path.join(process.cwd(), "dist");

async function initServer() {
  if (process.env.NODE_ENV !== "production") {
    console.log("Configuring Vite Development Server Middleware...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("Serving Production Static Assets of Agrisense...");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Agrisense Hub] Server successfully listening at http://0.0.0.0:${PORT}`);
  });
}

initServer();
