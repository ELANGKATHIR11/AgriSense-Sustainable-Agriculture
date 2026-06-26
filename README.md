# 🌾 AGRISENSE Architecture & Backend Attachment Guide

Agrisense is an industry-grade full-stack agriculture orchestrator. This repository contains the **complete, polished, production-ready React 19 frontend** featuring an API-abstracted architecture for seamless integration with any FastAPI backend.

---

## 🏗️ Technical Directory Structure

The application is structured according to clean architecture, split into modular, maintainable, and type-safe domains:

```
src/
├── app/                 # Config, providers, and main styles
├── api/                 # API Client, endpoint URL maps, and Interceptors 
│   ├── apiClient.ts     # Generic fetch handler with pre/post interceptor queues
│   ├── endpoints.ts     # Central dictionary of FastAPI endpoint mappings
│   └── interceptors.ts  # Request/response interceptor queues
├── config/              # App config parameters and mock bypass flags
│   └── appConfig.ts     # Toggle USE_MOCK_DATA or point VITE_API_URL
├── types/               # Future-proof interfaces representing FastAPI contracts
│   └── index.ts         # Types file containing complete database-ready models
├── mocks/               # High-fidelity offline data layer representing AI models
│   ├── mockDashboard.ts # Dashboard KPIs
│   ├── mockDisease.ts   # Pathology vision diagnoses
│   ├── mockCrop.ts      # Tabular soil recommendations
│   ├── mockSensors.ts   # Telemetry & irrigation mock algorithms
│   ├── mockMLOps.ts     # MLOps drift indexes & model logs
│   ├── mockYield.ts     # CatBoost crop yield regressions
│   └── mockChat.ts      # AgriGPT chat history answers
├── services/            # Custom Domain Services orchestrating API vs. Local Mocks
│   ├── dashboardService.ts
│   ├── cropService.ts
│   ├── diseaseService.ts
│   ├── irrigationService.ts
│   ├── sensorService.ts
│   ├── chatService.ts
│   ├── weatherService.ts
│   ├── mlopsService.ts
│   └── yieldService.ts
├── hooks/               # TanStack React Query Hooks wrapping all Services
│   ├── useDashboard.ts
│   ├── useWeather.ts
│   ├── useDiseaseAnalysis.ts
│   ├── useCropRecommendation.ts
│   ├── useIrrigation.ts
│   ├── useSensors.ts
│   ├── useChat.ts
│   ├── useModels.ts
│   └── useYield.ts
├── store/               # Zustand Global Stores
│   ├── authStore.ts     # Session profiles
│   ├── themeStore.ts    # Light/dark toggles
│   ├── chatStore.ts     # AgriGPT conversations
│   ├── sensorStore.ts   # Cached telemetry lists
│   └── settingsStore.ts # Farmland credentials and alert thresholds
├── providers/           # QueryProvider layer centering query clients
│   └── QueryProvider.tsx
└── pages/               # Functional view screens mapped to sidebar router links
```

---

## 🔌 Integrating Future FastAPI Backend (STRICT ZERO-REFACTOR CONTRACT)

To attach a FastAPI backend to this React frontend, **zero frontend components need to be modified**. The entire app switches via two simple configuration triggers.

### Step-by-Step Attachment Guide

#### Step 1: Set the API Environment Variables
Create a `.env` file in your root folder (or define them in your environment shell):
```env
# Point to your active FastAPI host URL
VITE_API_URL=http://your-fastapi-backend:8000

# Instruct the Agrisense service layer to bypass mocks and perform live API queries
VITE_USE_MOCK_DATA=false
```

#### Step 2: Ensure FastAPI API Models Match the TypeScript Contracts
Your FastAPI controllers should structure responses that align with these exact JSON schemas defined in `src/types/index.ts`:

* **Crop Recommendation (`POST /api/crop-recommend`)**
  * Input: `{ nitrogen: float, phosphorus: float, potassium: float, pH: float, temperature: float, humidity: float, rainfall: float }`
  * Return Schema:
    ```json
    {
      "crops": [
        { "name": "Rice", "suitability": 94, "description": "High water retaining crop...", "optimalConditions": "N: 80, P: 40..." }
      ],
      "optimalPH": "Ideal pH level outline string",
      "nutritionStatus": "Text feedback summary regarding fertilizer values"
    }
    ```

* **Crop Pathology Vision (`POST /api/disease-detect`)**
  * Input: `{ imageBase64: string, mode: "disease" | "weed" }`
  * Return Schema:
    ```json
    {
      "disease": "Tomato Leaf Mold",
      "confidence": 94.5,
      "severity": "medium",
      "symptoms": ["Yellow spots on upper leaf surfaces..."],
      "recommendations": ["Improve greenhouse ventilation..."]
    }
    ```

* **Irrigation Recommendation (`POST /api/irrigation-optimize`)**
  * Input: `{ moisture: float, temperature: float, humidity: float, cropType: string }`
  * Return Schema:
    ```json
    {
      "waterRequiredLiters": 1200,
      "moistureStatus": "CRITICAL UNDERWATERED",
      "advice": "Triggering micro-drip valve array...",
      "durationMinutes": 30,
      "irrigationSchedule": "Daily dawn intervals"
    }
    ```

* **IoT Telemetry readings (`GET /api/sensors`)**
  * Return Schema:
    ```json
    {
      "readings": [
        {
          "id": "1",
          "deviceId": "ESP32-S01",
          "timestamp": "2026-06-03T10:00:00Z",
          "soilMoisture": 42.5,
          "temperature": 27.8,
          "humidity": 62.1,
          "pH": 6.4,
          "nitrogen": 45,
          "phosphorus": 38,
          "potassium": 42
        }
      ]
    }
    ```

---

## ⚡ Key Architecture Advantages

1. **Robust TanStack React Query Caching**: State transitions feel instantaneous because we cache API queries globally. Repetitive window shifting is protected via customizable `staleTime` windows.
2. **Global Zustand Synchronization**: Telemetry and profile configurations are managed outside the React component render loops, eliminating re-render fatigue.
3. **No Raw fetch() in Components**: Ensures high code-quality compliance. Front-end engineers operate exclusively on type-safe hook functions (like `const { readings, isLoading } = useSensors()`).
4. **Completely Local & Native**: No dependencies on external API-key providers or system SDKs remain on the frontend. It compiles instantly and runs sandboxed.
