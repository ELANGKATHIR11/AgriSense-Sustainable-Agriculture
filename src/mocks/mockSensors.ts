/**
 * AGRISENSE ESP32 Wi-Fi Sensor Telemetry Mock Data
 */

import { SensorReading } from "../types";

export const initialMockSensors: SensorReading[] = [
  { id: "1", deviceId: "ESP32-S01", timestamp: new Date(Date.now() - 3600000 * 5).toISOString(), soilMoisture: 42.5, temperature: 27.8, humidity: 62.1, pH: 6.4, nitrogen: 45, phosphorus: 38, potassium: 42 },
  { id: "2", deviceId: "ESP32-S01", timestamp: new Date(Date.now() - 3600000 * 4).toISOString(), soilMoisture: 41.2, temperature: 28.1, humidity: 61.5, pH: 6.4, nitrogen: 46, phosphorus: 37, potassium: 43 },
  { id: "3", deviceId: "ESP32-S01", timestamp: new Date(Date.now() - 3600000 * 3).toISOString(), soilMoisture: 39.8, temperature: 28.5, humidity: 60.2, pH: 6.3, nitrogen: 45, phosphorus: 39, potassium: 41 },
  { id: "4", deviceId: "ESP32-S01", timestamp: new Date(Date.now() - 3600000 * 2).toISOString(), soilMoisture: 38.3, temperature: 28.9, humidity: 59.4, pH: 6.3, nitrogen: 47, phosphorus: 38, potassium: 42 },
  { id: "5", deviceId: "ESP32-S01", timestamp: new Date(Date.now() - 3600000 * 1).toISOString(), soilMoisture: 37.1, temperature: 29.2, humidity: 58.8, pH: 6.3, nitrogen: 44, phosphorus: 38, potassium: 44 },
];

export const mockMoistureCalculations = (moisture: number, temperature: number) => {
  let waterRequired = 0;
  let status = "Adequate";
  let minutes = 0;

  if (moisture < 35) {
    waterRequired = Math.round((45 - moisture) * 90);
    status = moisture < 20 ? "CRITICAL MOISTURE STRESS" : "Mild Moisture Deficit";
    minutes = Math.round(waterRequired / 40);
  }

  return {
    waterRequiredLiters: waterRequired,
    moistureStatus: status,
    advice: waterRequired > 0 
      ? `Moisture limit (${moisture}%) falls below safety parameters. Trigger drip manifold for ${minutes} mins.`
      : "Soil turgor is optimal. No active watering sequence needed.",
    durationMinutes: minutes,
    irrigationSchedule: waterRequired > 0 ? "Daily, early dawn cycle" : "Standby"
  };
};
