import {
  SensorData,
  SystemMetrics,
  ActivityLog,
  CropProfile,
  ChatMessage,
  DiseaseResult,
  CropInput,
  IrrigationStatus,
  CropDetail,
  MLDataset,
  MLModel,
  RecommendationResult,
  AnalysisResult,
} from "../types";

// Use proxy in dev (empty = same origin), or explicit URL for production
const envApi = import.meta.env.VITE_API_BASE_URL;
// Ensure no trailing slash and fall back to relative '/api' in browser (Vite proxy)
const API_BASE = envApi
  ? envApi.replace(/\/$/, "")
  : typeof window !== "undefined"
    ? "/api"
    : "http://localhost:5000/api";

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// --- MOCK FALLBACKS ---
const generateMockSensorData = (): SensorData => ({
  timestamp: new Date().toISOString(),
  temperature: 24 + Math.random() * 5,
  humidity: 55 + Math.random() * 10,
  soilMoisture: 60 + Math.random() * 20,
  soilTemperature: 20 + Math.random() * 3,
  phLevel: 6.5 + Math.random() * 0.5,
  nitrogen: 120 + Math.random() * 20,
  phosphorus: 40 + Math.random() * 10,
  potassium: 50 + Math.random() * 10,
  lightIntensity: 800 + Math.random() * 200,
});

let mockLogs: ActivityLog[] = [
  {
    id: 1,
    timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
    action: "Sensor Sync",
    status: "success",
    details: "Synced with 12 sensors",
  },
  {
    id: 2,
    timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
    action: "Model Retraining",
    status: "success",
    details: "Retrained Crop Recommendation model",
  },
];

let mockIrrigationState: IrrigationStatus = {
  pump_active: false,
  water_level: 78,
  flow_rate: 0,
  last_active: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
  mode: "Auto",
};

let mockDatasets: MLDataset[] = [
  {
    id: "1",
    name: "soil_samples_2023.csv",
    type: "CSV",
    size: "4.2 MB",
    records: 15420,
    uploaded_at: "2023-10-15",
    status: "Ready",
  },
  {
    id: "2",
    name: "disease_img_v2.zip",
    type: "Image",
    size: "256 MB",
    records: 5000,
    uploaded_at: "2023-11-02",
    status: "Ready",
  },
];

let mockModels: MLModel[] = [
  {
    id: "1",
    name: "Crop Recommender",
    version: "v2.1",
    type: "Classification",
    status: "Trained",
    accuracy: 0.92,
    last_trained: "2023-11-20",
    dataset_id: "1",
  },
  {
    id: "2",
    name: "Disease Detector",
    version: "v1.0",
    type: "Classification",
    status: "Trained",
    accuracy: 0.88,
    last_trained: "2023-11-15",
    dataset_id: "2",
  },
  {
    id: "3",
    name: "Yield Predictor",
    version: "v0.5",
    type: "Regression",
    status: "Trained",
    accuracy: 0.82,
    last_trained: "2023-09-15",
    dataset_id: "1",
  },
];

// --- API: Sensors ---
export const fetchLiveSensors = async (): Promise<SensorData> => {
  try {
    const res = await fetch(`${API_BASE}/iot/sensors/latest`);
    if (!res.ok) throw new Error("Not ok");
    const data = await res.json();
    return {
      timestamp: data.timestamp || new Date().toISOString(),
      temperature: data.temperature || data.air_temperature || 24,
      humidity: data.humidity ?? 60,
      soilMoisture: data.soilMoisture || data.soil_moisture || 45,
      soilTemperature: data.soilTemperature || data.soil_temperature || 22,
      phLevel: data.phLevel || data.ph_level || 6.5,
      nitrogen: data.nitrogen ?? 40,
      phosphorus: data.phosphorus ?? 20,
      potassium: data.potassium ?? 30,
      lightIntensity: data.lightIntensity || data.light_intensity || 800,
    };
  } catch {
    return generateMockSensorData();
  }
};

// --- API: Admin / Metrics ---
export const fetchSystemMetrics = async (): Promise<SystemMetrics> => {
  try {
    const res = await fetch(`${API_BASE}/admin/summary`);
    if (!res.ok) throw new Error("Not ok");
    const d = await res.json();
    const uptimeHours = (d.uptime || 0) / 3600;
    return {
      cpuUsage:
        typeof d.cpuLoad === "string" ? parseInt(d.cpuLoad, 10) || 15 : 15,
      memoryUsage: 40 + Math.floor(Math.random() * 15),
      diskUsage: d.disk_usage || 45,
      uptime: `${uptimeHours.toFixed(1)}h`,
      activeConnections: d.activeConnections ?? 14,
      modelStatus: "loaded",
    };
  } catch {
    return {
      cpuUsage: 30 + Math.floor(Math.random() * 15),
      memoryUsage: 40 + Math.floor(Math.random() * 10),
      diskUsage: 60,
      uptime: "124h",
      activeConnections: 14,
      modelStatus: "loaded",
    };
  }
};

export const fetchActivityLogs = async (): Promise<ActivityLog[]> => {
  try {
    const res = await fetch(`${API_BASE}/admin/activities`);
    if (!res.ok) throw new Error("Not ok");
    const arr = await res.json();
    if (!Array.isArray(arr)) return mockLogs;
    return arr
      .map((l: any) => ({
        id: l.id ?? Date.now(),
        timestamp:
          typeof l.timestamp === "string"
            ? l.timestamp
            : new Date(l.timestamp || Date.now()).toISOString(),
        action: l.action ?? "Unknown",
        details: l.details ?? l.descriptor ?? "-",
        status:
          l.status?.toLowerCase() === "success" ||
          l.status?.toLowerCase() === "warning" ||
          l.status?.toLowerCase() === "error"
            ? (l.status.toLowerCase() as "success" | "warning" | "error")
            : "success",
      }))
      .sort(
        (a: ActivityLog, b: ActivityLog) =>
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
      );
  } catch {
    return mockLogs;
  }
};

export const triggerAdminAction = async (action: string) => {
  try {
    await fetch(`${API_BASE}/admin/action`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
  } catch {
    mockLogs.unshift({
      id: Date.now(),
      timestamp: new Date().toISOString(),
      action: `Admin: ${action}`,
      status: "success",
      details: "Triggered manually",
    });
  }
  return true;
};

// --- API: Chat (LLM) ---
export const sendChatMessage = async (message: string): Promise<string> => {
  try {
    // Determine AI Service URL (local python service)
    const AI_SERVICE_URL = "http://localhost:8000/chat";
    
    const res = await fetch(AI_SERVICE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, context: {} }),
    });
    
    if (!res.ok) throw new Error("AI Service Error");
    const data = await res.json();
    return data.reply || "I am processing your request...";
  } catch (error) {
    console.error("AI Service failed:", error);
    // Fallback to legacy node backend if python service fails
    try {
        const res = await fetch(`${API_BASE}/llm/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, conversation_history: [] }),
        });
        if (!res.ok) throw new Error("Legacy API failed");
        const data = await res.json();
        return data.reply || data.advice || "System currently offline.";
    } catch {
        await delay(800);
        return "⚠️ Offline Mode: Based on standard agronomy rules, please ensure soil moisture is maintained. Connect to the server for personalized advice.";
    }
  }
};

export const analyzeCropImage = async (imageFile: File): Promise<any> => {
  try {
    const formData = new FormData();
    formData.append('file', imageFile);
    // Use the proxy path configured in vite.config.ts
    const response = await fetch(`${API_BASE}/analyze/image`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) throw new Error('Analysis failed');
    return await response.json();
  } catch (error) {
    console.error("VLM Analysis failed:", error);
    throw error;
  }
};

// --- API: Disease Detection (VLM) ---
export const detectDisease = async (
  imageFile: File,
): Promise<DiseaseResult> => {
  try {
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append(
      "prompt",
      "Analyze this plant image for diseases. Identify disease name and suggest treatment.",
    );
    const res = await fetch(`${API_BASE}/vlm/analyze-plant`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Not ok");
    const data = await res.json();
    const analysis = data.data || data;
    return {
      disease_name: analysis.disease_name || analysis.disease || "Unknown",
      confidence: analysis.confidence ?? 0.85,
      treatment:
        analysis.treatment ||
        analysis.recommendations ||
        "Consult agricultural expert.",
      image_url: analysis.image_url,
    };
  } catch {
    await delay(1500);
    return {
      disease_name: "Tomato Early Blight",
      confidence: 0.94,
      treatment:
        "Apply fungicides containing mancozeb or copper. Improve air circulation and avoid overhead watering.",
    };
  }
};

// --- API: Crop Recommendation (ML) ---
export const recommendCrop = async (input: CropInput): Promise<RecommendationResult> => {
  try {
    const res = await fetch(`${API_BASE}/crop-recommendation/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        N: input.nitrogen,
        P: input.phosphorus,
        K: input.potassium,
        temperature: input.temperature,
        humidity: input.humidity,
        ph: input.ph,
        rainfall: input.rainfall,
      }),
    });
    if (!res.ok) throw new Error("Not ok");
    const data = await res.json();
    const pred = data.data || data;
    
    // Normalize response
    const cropName = pred.prediction || pred.recommended_crop || pred.crop || "Maize (Corn)";
    const confidence = pred.confidence || 0.85;

    return {
      crop: cropName,
      confidence: confidence,
      details: `Recommendation based on NPK (${input.nitrogen}, ${input.phosphorus}, ${input.potassium}), Temp ${input.temperature}°C, Rain ${input.rainfall}mm.`
    };
  } catch {
    await delay(1000);
    return {
        crop: "Maize (Corn)",
        confidence: 0.75,
        details: "Mock fallback response due to API connection failure."
    };
  }
};

// --- API: Irrigation (Water Tank) ---
export const fetchIrrigationStatus = async (): Promise<IrrigationStatus> => {
  try {
    const res = await fetch(`${API_BASE}/iot/water-tank/status`);
    if (!res.ok) throw new Error("Not ok");
    const d = await res.json();
    return {
      pump_active: (d.pumpStatus || "OFF") === "ON",
      water_level: d.level ?? 78,
      flow_rate: d.pumpStatus === "ON" ? 12.5 : 0,
      last_active: d.lastUpdated || new Date().toISOString(),
      mode: "Auto",
    };
  } catch {
    if (mockIrrigationState.pump_active) {
      mockIrrigationState.water_level = Math.max(
        0,
        mockIrrigationState.water_level - 0.5,
      );
      mockIrrigationState.flow_rate = 12.5;
    } else mockIrrigationState.flow_rate = 0;
    return { ...mockIrrigationState };
  }
};

export const toggleIrrigationPump = async (
  active: boolean,
): Promise<IrrigationStatus> => {
  try {
    const res = await fetch(`${API_BASE}/iot/water-tank/pump`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: active ? "ON" : "OFF" }),
    });
    if (!res.ok) throw new Error("Not ok");
    const d = await res.json();
    mockIrrigationState = {
      pump_active: (d.action || (active ? "ON" : "OFF")) === "ON",
      water_level: mockIrrigationState.water_level,
      flow_rate: active ? 12.5 : 0,
      last_active: new Date().toISOString(),
      mode: "Auto",
    };
    return mockIrrigationState;
  } catch {
    mockIrrigationState.pump_active = active;
    mockIrrigationState.last_active = new Date().toISOString();
    return { ...mockIrrigationState };
  }
};

// --- API: Crop Library ---
// --- API: Crop Library ---
export const fetchCropLibrary = async (
  query: string = "",
): Promise<CropProfile[]> => {
  try {
    const res = await fetch(`${API_BASE}/crop-recommendation/crops`);
    if (!res.ok) throw new Error("Not ok");
    const data = await res.json();
    let list = Array.isArray(data.data)
      ? data.data
      : Array.isArray(data)
        ? data
        : [];
    list = list.map((c: any) => ({
      id: c.id ?? 0,
      name: c.name ?? "Unknown",
      scientificName: c.scientificName ?? c.scientific_name ?? "",
      type: c.type ?? c.category ?? "General",
      season: c.season ?? "General",
      phRange: c.phRange ?? c.soil_type ?? "6.0-7.0",
      tempRange: c.tempRange ?? "20-30°C",
      humidityRange: c.humidityRange ?? "50-70%",
      waterReq: c.waterReq ?? "Medium",
      duration: c.duration ?? "120 days",
      imageUrl: c.imageUrl ?? "",
    }));
    if (query) {
      const q = query.toLowerCase();
      list = list.filter(
        (c: CropProfile) =>
          c.name.toLowerCase().includes(q) ||
          (c.scientificName && c.scientificName.toLowerCase().includes(q)),
      );
    }
    return list;
  } catch {
    await delay(400);
    const base: CropProfile[] = [
      {
        id: 1,
        name: "Rice",
        scientificName: "Oryza sativa",
        type: "Cereal",
        season: "Kharif",
        phRange: "5.5-7.0",
        tempRange: "20-35°C",
        humidityRange: "60-80%",
        waterReq: "High",
        duration: "120-150 days",
        imageUrl: "",
      },
      {
        id: 2,
        name: "Wheat",
        scientificName: "Triticum",
        type: "Cereal",
        season: "Rabi",
        phRange: "6.0-7.0",
        tempRange: "10-25°C",
        humidityRange: "50-60%",
        waterReq: "Medium",
        duration: "120-140 days",
        imageUrl: "",
      },
      {
        id: 3,
        name: "Maize",
        scientificName: "Zea mays",
        type: "Cereal",
        season: "Kharif",
        phRange: "5.5-7.5",
        tempRange: "18-27°C",
        humidityRange: "50-70%",
        waterReq: "Medium",
        duration: "90-110 days",
        imageUrl: "",
      },
    ];
    if (query) {
      const q = query.toLowerCase();
      return base.filter((c) => c.name.toLowerCase().includes(q));
    }
    return base;
  }
};

// --- API: ML Studio (pipelined to backend / mock) ---
export const fetchDatasets = async (): Promise<MLDataset[]> => {
  try {
    const res = await fetch(`${API_BASE}/ml/datasets`);
    if (!res.ok) throw new Error("Not ok");
    const data = await res.json();
    return data.datasets || data;
  } catch {
    await delay(500);
    return mockDatasets;
  }
};

export const uploadDataset = async (file: File): Promise<MLDataset> => {
  try {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE}/ml/datasets`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) throw new Error("Not ok");
    return await res.json();
  } catch {
    await delay(1000);
    const ds: MLDataset = {
      id: Date.now().toString(),
      name: file.name,
      type: file.name.endsWith(".csv") ? "CSV" : "Image",
      size: `${(file.size / (1024 * 1024)).toFixed(2)} MB`,
      records: 0,
      uploaded_at: new Date().toISOString().split("T")[0],
      status: "Processing",
    };
    mockDatasets.push(ds);
    return ds;
  }
};

export const fetchModels = async (): Promise<MLModel[]> => {
  try {
    const res = await fetch(`${API_BASE}/ml/models`);
    if (!res.ok) throw new Error("Not ok");
    const data = await res.json();
    return data.models || data;
  } catch {
    await delay(400);
    return mockModels;
  }
};

export const triggerTraining = async (modelId: string): Promise<boolean> => {
  try {
    const res = await fetch(`${API_BASE}/ml/train/${modelId}`, {
      method: "POST",
    });
    if (res.ok) {
      const data = await res.json();
      console.log('Training triggered:', data);
      return data.status === 'started';
    }
    throw new Error("Not ok");
  } catch {
    await delay(800);
    mockModels = mockModels.map((m) =>
      m.id === modelId ? { ...m, status: "Training" as const } : m,
    );
    return true;
  }
};

// --- ML Service Aggregator ---
export const MLService = {
  recommendCrop: async (data: any) => {
    return recommendCrop(data);
  },
  analyze: async (data: {
    N: number;
    P: number;
    K: number;
    temperature: number;
    humidity: number;
    ph: number;
    rainfall: number;
    area: number;
  }): Promise<AnalysisResult> => {
    try {
      const res = await fetch(`${API_BASE}/ml/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("ML Analysis failed");
      const json = await res.json();
      return json.data;
    } catch (error: any) {
      console.error("ML Analysis error:", error);
      // Fallback mock
      return {
        water_requirement: 450.5,
        season: "Summer",
        crop_group: "Cereal",
        recommended_crop: "Rice",
        expected_yield: 12.4
      };
    }
  },
  predictYield: async (data: any) => {
    try {
      const res = await fetch(`${API_BASE}/yield-prediction/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Yield prediction failed");
      const response = await res.json();
      return response.data;
    } catch (error) {
      console.error("Yield prediction error:", error);
      throw error;
    }
  },
  predictWaterRequirement: async (data: any) => {
    try {
      const res = await fetch(`${API_BASE}/water-requirement/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Water requirement prediction failed");
      const response = await res.json();
      return response.data;
    } catch (error) {
      console.error("Water requirement prediction error:", error);
      throw error;
    }
  },
  classifySeason: async (data: any) => {
    try {
      const res = await fetch(`${API_BASE}/season-classification/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error("Season classification failed");
      const response = await res.json();
      return response.data;
    } catch (error) {
      console.error("Season classification error:", error);
      throw error;
    }
  },
  analyzePlantImage: async (image: File) => {
    return detectDisease(image);
  },
  analyzeTargetedDisease: async (image: File) => {
    // Mock for specialized model
    await delay(1000);
    return {
      prediction: "Rust",
      confidence: 92,
    };
  },
  predictDiseaseRisk: async (data: any) => {
    await delay(1000);
    // Simple mock logic
    const highRisk = data.humidity > 80 && data.temperature > 25;
    return {
      risk_level: highRisk ? "High" : "Low",
      risk_score: highRisk ? 0.85 : 0.2,
    };
  },
  chat: async (message: string, history: any[]) => {
    // Mock chat response
    await delay(1000);
    return {
      reply: `I understand you're asking about "${message}". As an AI assistant, I can help with crop advice, weather updates, and market prices.`
    };
  },
  analyzeWithVLM: async (formData: FormData, task: string) => {
    // Mock VLM analysis
    await delay(1500);
    return {
      success: true,
      data: {
        detected_weed: "Crabgrass",
        confidence: 0.88,
        coverage: 25,
        severity: "medium",
        cost: 1200,
        recommendations: ["Manual weeding", "Pre-emergent herbicide"]
      }
    };
  }
};

export const IoTService = {
  getDevices: async () => {
    await delay(500);
    return [
      { id: '1', name: 'Soil Sensor A', type: 'soil', location: 'Field A', status: 'active', topic: 'agrisense/sensor/soil/1' },
      { id: '2', name: 'Pump Controller', type: 'pump', location: 'Pump House', status: 'active', topic: 'agrisense/control/pump/1' }
    ];
  },
  createDevice: async (device: any) => {
    await delay(500);
    return { id: Date.now().toString(), ...device, status: 'active' };
  },
  deleteDevice: async (id: string) => {
    await delay(500);
    return { success: true };
  }
};

export const WaterTankService = {
  getTankStatus: async () => {
    await delay(500);
    return {
      level: 65,
      capacity: 5000,
      currentVolume: 3250,
      pumpStatus: 'OFF',
      lastUpdated: new Date().toISOString()
    };
  },
  getUsageHistory: async () => {
    await delay(500);
    return Array.from({ length: 7 }, (_, i) => ({
      date: new Date(Date.now() - (6 - i) * 86400000).toLocaleDateString(),
      usage: Math.floor(Math.random() * 500) + 100
    }));
  },
  controlPump: async (action: 'ON' | 'OFF') => {
    await delay(1000);
    return { success: true, status: action };
  }
};

export const CropService = {
  getAll: fetchCropLibrary,
};
