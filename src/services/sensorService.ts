/**
 * AGRISENSE ESP32 Telemetry Logging Service
 */

import { apiClient } from "../api/apiClient";
import { API_ENDPOINTS } from "../api/endpoints";
import { SensorReading } from "../types";

export const sensorService = {
  async getSensors(): Promise<SensorReading[]> {
    const response = await apiClient.get<{ readings: SensorReading[] }>(API_ENDPOINTS.sensorData);
    return response.readings || [];
  },

  async ingestReading(reading: Partial<SensorReading>): Promise<SensorReading> {
    const response = await apiClient.post<{ logged: SensorReading }>(API_ENDPOINTS.sensorIngest, reading);
    return response.logged;
  }
};
