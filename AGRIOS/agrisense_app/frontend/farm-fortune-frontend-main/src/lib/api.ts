/**
 * AgriSense API Client
 * Comprehensive API integration for frontend components
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ============================================================
// Type Definitions
// ============================================================

export interface TankStatus {
  tank_id?: string;
  level_pct: number | null;
  volume_l: number | null;
  capacity_l?: number;
  last_update?: string;
  low_threshold?: number;
  is_low?: boolean;
}

export interface AlertItem {
  id?: string;
  ts: string;
  level?: 'info' | 'warning' | 'critical';
  category?: string;
  message: string;
  acked?: boolean;
  sent?: boolean;
  source?: string;
}

export interface WeatherCacheRow {
  ts: string;
  temp_c: number;
  humidity_pct: number;
  precip_mm: number;
  wind_kph?: number;
  condition?: string;
  icon?: string;
}

export interface ValveEvent {
  id?: string;
  ts: string;
  action: 'start' | 'stop';
  status: 'sent' | 'queued' | 'failed' | 'acked';
  zone_id?: string;
  duration_s?: number;
  duration_sec?: number;
}

export interface RainwaterSummary {
  tank_id: string;
  total_l: number;
  last_harvest_ts?: string;
  last_harvest_l?: number;
  avg_daily_l?: number;
}

export interface RainwaterEntry {
  ts: string;
  volume_l: number;
  source?: string;
}

export interface DashboardSummary {
  tank: TankStatus | null;
  alerts: AlertItem[];
  valve_events: ValveEvent[];
  soil_moisture_pct: number | null;
  weather_latest: WeatherCacheRow | null;
  impact: {
    saved_l: number;
    cost_rs: number;
    co2e_kg: number;
  } | null;
  tank_history: { level_pct: number; ts: string }[];
}

export interface LiveSensorData {
  timestamp?: string;
  air_temperature?: number;
  humidity?: number;
  soil_moisture_percentage?: number;
  soil_moisture_raw?: number;
  soil_temperature?: number; // Added
  ph_level?: number;
  nitrogen?: number;
  phosphorus?: number;
  potassium?: number;
  light_intensity_percentage?: number;
  ec_value?: number;
}

export interface DeviceStatus {
  device_id: string;
  name: string;
  status: 'online' | 'offline' | 'error';
  last_seen?: string;
  battery_pct?: number;
  signal_strength?: number;
  is_connected?: boolean; // Added
}

export interface ArduinoStatus {
  status: 'active' | 'inactive' | 'error' | 'connected' | 'disconnected';
  total_devices: number;
  last_reading_time?: string;
  recent_readings: {
    timestamp: string;
    temperature: number;
    humidity?: number;
    zone_id: string;
  }[];
}

export interface RecommendationResult {
  water_liters: number;
  fert_n_g: number;
  fert_p_g: number;
  fert_k_g: number;
  confidence?: number;
  tips?: string[];
  savings?: {
    water_pct: number;
    cost_rs: number;
  };
  // Extended UI fields
  water_source?: string;
  irrigation_cycles?: number;
  suggested_runtime_min?: number;
  assumed_flow_lpm?: number;
  best_time?: string;
  expected_savings_liters?: number;
  expected_cost_saving_rs?: number;
  expected_co2e_kg?: number;
  fertilizer_equivalents?: Record<string, string>;
  notes?: string[];
}

export interface IrrigationAck {
  ok: boolean;
  success?: boolean;
  status?: string;
  note?: string;
  message?: string;
  event_id?: string;
}

export interface SensorReading {
  ts: string;
  zone_id: string;
  soil_moisture_pct?: number;
  temperature_c?: number;
  humidity_pct?: number;
  light_lux?: number;
  ph?: number;
  ec?: number;
}

export interface ChatbotResponse {
  answer: string;
  confidence?: number;
  sources?: string[];
  follow_up_questions?: string[];
}

export interface CropCard {
  id: string;
  name: string;
  scientificName?: string;
  category?: string;
  description?: string;
  waterRequirement?: 'Low' | 'Medium' | 'High';
  season?: string;
  tempRange?: string;
  phRange?: string;
  growthPeriod?: string;
  tips: string[];
}

export interface RecommendationRequest {
  plant?: string;
  crop?: string;
  soil_type?: string;
  area_m2?: number;
  ph?: number;
  moisture_pct?: number;
  soil_moisture_pct?: number;
  temperature_c?: number;
  humidity_pct?: number;
  ec_dS_m?: number;
  n_ppm?: number;
  p_ppm?: number;
  k_ppm?: number;
}

// ============================================================
// API Client
// ============================================================

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Dashboard
  async dashboardSummary(
    zoneId: string = 'Z1',
    tankId: string = 'T1',
    alertLimit: number = 5,
    valveLimit: number = 5
  ): Promise<DashboardSummary> {
    const params = new URLSearchParams({
      zone_id: zoneId,
      tank_id: tankId,
      alert_limit: String(alertLimit),
      valve_limit: String(valveLimit),
    });
    return this.request<DashboardSummary>(`/dashboard/summary?${params}`);
  }

  // Tank
  async tankStatus(tankId: string = 'T1'): Promise<TankStatus> {
    return this.request<TankStatus>(`/tank/status?tank_id=${tankId}`);
  }

  async tankHistory(tankId: string = 'T1', limit: number = 20): Promise<{ level_pct: number; ts: string }[]> {
    return this.request<{ level_pct: number; ts: string }[]>(`/tank/history?tank_id=${tankId}&limit=${limit}`);
  }

  async postTankLevel(tankId: string, levelPct: number): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('/tank/level', {
      method: 'POST',
      body: JSON.stringify({ tank_id: tankId, level_pct: levelPct }),
    });
  }

  // Alerts - method matching Irrigation.tsx usage: api.alerts(zone, limit)
  async alerts(zoneId: string = 'Z1', limit: number = 10): Promise<{ items: AlertItem[] }> {
    const params = new URLSearchParams({
      zone_id: zoneId,
      limit: String(limit),
    });
    try {
      const result = await this.request<AlertItem[] | { items: AlertItem[] }>(`/alerts?${params}`);
      if (Array.isArray(result)) {
        return { items: result };
      }
      return result;
    } catch {
      return { items: [] };
    }
  }

  async getAlerts(limit: number = 10, unAckedOnly: boolean = false): Promise<AlertItem[]> {
    const params = new URLSearchParams({
      limit: String(limit),
      unacked_only: String(unAckedOnly),
    });
    return this.request<AlertItem[]>(`/alerts?${params}`);
  }

  async alertAck(alertTs: string): Promise<{ success: boolean }> {
    return this.request<{ success: boolean }>('/alerts/ack', {
      method: 'POST',
      body: JSON.stringify({ ts: alertTs }),
    });
  }

  // Irrigation - updated signature to match Irrigation.tsx: api.irrigationStart(zone, duration, force)
  async irrigationStart(zoneId: string = 'Z1', durationSec?: number, force: boolean = false): Promise<IrrigationAck> {
    return this.request<IrrigationAck>('/irrigation/start', {
      method: 'POST',
      body: JSON.stringify({ zone_id: zoneId, duration_sec: durationSec, force }),
    });
  }

  async irrigationStop(zoneId: string = 'Z1'): Promise<IrrigationAck> {
    return this.request<IrrigationAck>('/irrigation/stop', {
      method: 'POST',
      body: JSON.stringify({ zone_id: zoneId }),
    });
  }

  // Valve events - method matching Irrigation.tsx usage: api.valveEvents(zone, limit)
  async valveEvents(zoneId: string = 'Z1', limit: number = 10): Promise<{ items: ValveEvent[] }> {
    const params = new URLSearchParams({
      zone_id: zoneId,
      limit: String(limit),
    });
    try {
      const result = await this.request<ValveEvent[] | { items: ValveEvent[] }>(`/valves/events?${params}`);
      if (Array.isArray(result)) {
        return { items: result };
      }
      return result;
    } catch {
      return { items: [] };
    }
  }

  // Rainwater
  async rainwaterSummary(tankId: string = 'T1'): Promise<RainwaterSummary> {
    return this.request<RainwaterSummary>(`/rainwater/summary?tank_id=${tankId}`);
  }

  async rainwaterRecent(tankId: string = 'T1', limit: number = 5): Promise<{ items: RainwaterEntry[] }> {
    return this.request<{ items: RainwaterEntry[] }>(`/rainwater/recent?tank_id=${tankId}&limit=${limit}`);
  }

  // Sensors
  async sensorsRecent(limit: number = 10): Promise<SensorReading[]> {
    return this.request<SensorReading[]>(`/sensors/recent?limit=${limit}`);
  }

  async sensorsLive(): Promise<{ data: LiveSensorData | null }> {
    try {
      return await this.request<{ data: LiveSensorData | null }>('/live');
    } catch {
      return { data: null };
    }
  }

  async sensorsDeviceStatus(): Promise<{ devices: DeviceStatus[] }> {
    try {
      return await this.request<{ devices: DeviceStatus[] }>('/edge/health');
    } catch {
      return { devices: [] };
    }
  }

  async arduinoStatus(): Promise<ArduinoStatus | null> {
    try {
      const response = await this.request<SensorReading[]>('/sensors/recent');
      if (Array.isArray(response)) {
        const readings = response.map((r: SensorReading) => ({
          timestamp: r.ts,
          temperature: r.temperature_c || 0,
          humidity: r.humidity_pct,
          zone_id: r.zone_id,
        }));
        return {
          status: readings.length > 0 ? 'active' : 'inactive',
          total_devices: 1,
          last_reading_time: readings[0]?.timestamp,
          recent_readings: readings,
        };
      }
      return null;
    } catch {
      return null;
    }
  }

  // Recommendations


  // Recommendations
  async recommend(params: RecommendationRequest): Promise<RecommendationResult> {
    return this.request<RecommendationResult>('/recommend', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  // Chatbot
  // Chatbot
  async chatbotAsk(
    question: string, 
    options?: { 
      session_id?: string; 
      language?: string; 
      top_k?: number; 
      include_sources?: boolean; 
    }
  ): Promise<any> {
    return this.request<any>('/chatbot/ask', {
      method: 'POST',
      body: JSON.stringify({ 
        question,
        ...options 
      }),
    });
  }

  async chatbotGreeting(language: string = 'en'): Promise<{ greeting: string }> {
    return this.request<{ greeting: string }>(`/chatbot/greeting?language=${language}`);
  }

  // Health
  async health(): Promise<{ status: string; timestamp: string }> {
    return this.request<{ status: string; timestamp: string }>('/health');
  }

  // Weather
  async weatherRefresh(lat: number, lon: number): Promise<WeatherCacheRow> {
    return this.request<WeatherCacheRow>('/admin/weather/refresh', {
      method: 'POST',
      body: JSON.stringify({ lat, lon }),
    });
  }

  // Recent readings  
  async getRecent(limit: number = 10): Promise<SensorReading[]> {
    return this.request<SensorReading[]>(`/recent?limit=${limit}`);
  }

  // Crops - method matching Crops.tsx usage: api.crops()
  async crops(): Promise<{ items: CropCard[] }> {
    try {
      const result = await this.request<{ items?: CropCard[]; crops?: string[] }>('/crops');
      if (result.items) {
        return { items: result.items };
      }
      // Transform simple crops list to CropCard array
      if (result.crops) {
        return {
          items: result.crops.map((name, idx) => ({
            id: `crop-${idx}`,
            name,
            tips: [],
          })),
        };
      }
      return { items: [] };
    } catch {
      return { items: [] };
    }
  }

  async getCrops(): Promise<{ crops: string[] }> {
    return this.request<{ crops: string[] }>('/crops');
  }

  // Soil types
  async getSoilTypes(): Promise<{ soil_types: string[] }> {
    return this.request<{ soil_types: string[] }>('/soil/types');
  }

  // Disease Detection
  async detectDisease(data: { image_data: string; crop_type?: string; field_info?: any }): Promise<any> {
    return this.request('/api/disease/detect', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Weed Detection
  async detectWeeds(data: { image_data: string; crop_type: string; field_info?: any }): Promise<any> {
    return this.request('/api/weed/analyze', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Enhanced Crop Recommendation
  async predictCrop(data: CropInput): Promise<CropRecommendation[]> {
    return this.request<CropRecommendation[]>('/crops/recommend', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Water Optimization
  async waterOptimization(data: WaterOptimizationRequest): Promise<WaterOptimizationResponse> {
    return this.request<WaterOptimizationResponse>('/ml/water-optimization', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async waterOptimizationInfo(): Promise<WaterOptimizationInfo> {
    return this.request<WaterOptimizationInfo>('/ml/water-optimization/info');
  }

  // Yield Prediction
  async yieldPrediction(data: YieldPredictionRequest): Promise<YieldPredictionResponse> {
    return this.request<YieldPredictionResponse>('/ml/yield-prediction', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async yieldPredictionInfo(): Promise<YieldPredictionInfo> {
    return this.request<YieldPredictionInfo>('/ml/yield-prediction/info');
  }

  // Sensors Live Recommendations (Stub/Wrapper)
  async sensorsRecommendationsLive(): Promise<{ recommendations: BackendRecommendation; sensor_data?: LiveSensorData }> {
      try {
        const live = await this.sensorsLive();
        if (!live.data) throw new Error("No live sensor data");
        
        // Use the soil-type from the data if available, else default
        const payload: any = {
            temperature_c: live.data.air_temperature,
            humidity_pct: live.data.humidity,
            soil_moisture_pct: live.data.soil_moisture_percentage,
            ph: live.data.ph_level,
            // Add defaults for required fields if missing from live data
            soil_type: "loam",
            crop: "tomato", 
            area_m2: 100
        };
        
        const rec = await this.recommend(payload);
        return { recommendations: rec, sensor_data: live.data };
      } catch (e) {
          console.error("Live recommendation failed", e);
          throw e;
      }
  }

  // Aliases for Recommend.tsx compatibility
  async plants() { return this.crops(); }
  async soilTypes() { return this.getSoilTypes(); }
}

// Export alias for compatibility
export type BackendRecommendation = RecommendationResult;

export interface CropInput {
  pH: number;
  N: number;
  P: number;
  K: number;
  Fe: number;
  Mn: number;
  Zn: number;
  Cu: number;
  B: number;
  Water: number;
  Moisture: number;
  Temperature: number;
  Rainfall: number;
}

export interface CropRecommendation {
  rank: number;
  crop: string;
  suitability: number;
}

// ML Types
export interface WaterOptimizationRequest {
  soil_moisture: number;
  temperature: number;
  humidity: number;
  crop_type: string;
  soil_type: string;
  evapotranspiration: number;
  rainfall_forecast: number;
  plant_growth_stage: number;
  area_m2: number;
}

export interface WaterOptimizationResponse {
  irrigation_volume_per_m2: number;
  total_irrigation_liters: number;
  irrigation_urgency: number;
  recommended_frequency_days: number;
  confidence: number;
  model_version: string;
  recommendations: string[];
}

export interface WaterOptimizationInfo {
  model_available: boolean;
  model_version?: string;
  supported_crops: string[];
  supported_soils: string[];
  feature_descriptions?: Record<string, string>;
}

export interface YieldPredictionRequest {
  crop_type: string;
  area_hectares: number;
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  temperature: number;
  humidity: number;
  rainfall: number;
  irrigation: number;
  growing_days: number;
  soil_type: string;
  pest_pressure: number;
}

export interface YieldPredictionResponse {
  predicted_yield_kg_ha: number;
  total_production_kg: number;
  yield_category: string;
  regional_average_yield: number;
  confidence: number;
  model_version: string;
  recommendations: string[];
}

export interface YieldPredictionInfo {
  model_available: boolean;
  model_version?: string;
  supported_crops: string[];
  supported_soils: string[];
  crop_typical_yields?: Record<string, any>;
}

export interface PlantListItem {
    label: string;
    value: string;
}

// Export singleton instance
export const api = new ApiClient();

// Export class for custom instances
export { ApiClient };
