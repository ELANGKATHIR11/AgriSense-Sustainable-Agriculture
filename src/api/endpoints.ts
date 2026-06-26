// AgriSense API Endpoint Declarations
export const API_ENDPOINTS = {
  // Core ML endpoints
  dashboard:               "/api/dashboard",
  cropRecommendation:      "/api/crop-recommend",
  irrigationRecommendation: "/api/irrigation-optimize",
  yieldForecast:           "/api/yield-predict",

  // Vision / VLM endpoints (Florence-2)
  diseaseDetection:        "/api/vision/disease",
  weedDetection:           "/api/vision/weed",
  nutrientDetection:       "/api/vision/nutrient",
  pestDetection:           "/api/vision/pest",
  cropIdentification:      "/api/vision/crop",
  healthAnalysis:          "/api/vision/health",
  yieldPrediction:         "/api/vision/yield",

  // IoT Sensor endpoints
  sensorData:              "/api/sensors",
  sensorIngest:            "/api/sensors/ingest",

  // Agent Chat (Qwen2.5-Coder)
  chat:                    "/api/chat",

  // MLOps
  mlops:                   "/api/mlops",
  mlopsRetrain:            "/api/mlops/retrain",

  // Digital Twin
  weather:                 "/api/twin/weather",
  twinState:               "/api/twin/state",
  twinUpdate:              "/api/twin/update",
  twinSimulate:            "/api/twin/simulate",
  twinAnalytics:           "/api/twin/analytics",
};
