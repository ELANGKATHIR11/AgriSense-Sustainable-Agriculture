import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

// API Methods
export const apiService = {
  // Health & System
  health: () => api.get('/health'),
  
  // Sensors
  getSensorLive: () => api.get('/sensors/live').catch(() => ({ 
    success: true, 
    data: { 
      temperature: 25, 
      humidity: 60, 
      soil_moisture: 45, 
      ph: 6.5, 
      nitrogen: 40, 
      phosphorus: 30, 
      potassium: 35,
      light_intensity: 500,
      source: 'fallback' 
    } 
  })),
  
  getSensorHistory: (hours = 24) => api.get(`/sensors/history?hours=${hours}`),
  
  getDevices: () => api.get('/sensors/devices').catch(() => ({ 
    success: true, 
    devices: [], 
    count: 0 
  })),
  
  getDeviceStatus: () => api.get('/sensors/devices/status').catch(() => ({ 
    success: true, 
    total_devices: 0, 
    status_summary: { active: 0, inactive: 0 } 
  })),
  
  postSensorData: (data: any) => api.post('/sensors/data', data),
  
  // ML Predictions
  recommendCrop: (data: any) => api.post('/recommend', data),
  
  optimizeWater: (data: any) => api.post('/water/optimize', data),
  
  recommendFertilizer: (data: any) => api.post('/fertilizer/recommend', data),
  
  predictYield: (data: any) => api.post('/yield/predict', data),
  
  getCrops: () => api.get('/crops').catch(() => ({ 
    success: true, 
    crops: ['Rice', 'Wheat', 'Maize', 'Cotton'] 
  })),
  
  getModelsStatus: () => api.get('/models/status'),
  
  // AI Services
  chat: (data: any) => api.post('/ai/chat', data),
  
  detectDisease: (data: any) => api.post('/ai/disease/detect', data),
  
  detectWeed: (data: any) => api.post('/ai/weed/detect', data),
  
  assessPlantHealth: (data: any) => api.post('/ai/plant-health/assess', data),
  
  // Admin
  getAdminMetrics: () => api.get('/admin/metrics').catch(() => ({ 
    success: true, 
    metrics: { cpu: { percent: 0 }, memory: { percent: 0 }, disk: { percent: 0 } } 
  })),
  
  getAdminSummary: () => api.get('/admin/summary').catch(() => ({ 
    success: true, 
    summary: { total_devices: 0, active_devices: 0 } 
  })),
  
  getActivities: () => api.get('/admin/activities').catch(() => ({ 
    success: true, 
    activities: [] 
  })),
  
  performAdminAction: (data: any) => api.post('/admin/action', data),
  
  resetSystem: () => api.post('/admin/reset'),
  
  // Dashboard
  getDashboardSummary: () => api.get('/dashboard/summary').catch(() => ({ 
    success: true, 
    metrics: { soil_moisture: 45, temperature: 25, humidity: 60, ph_level: 6.5 } 
  })),
  
  getAlerts: () => api.get('/alerts').catch(() => ({ 
    success: true, 
    alerts: [] 
  })),
}

export default apiService
