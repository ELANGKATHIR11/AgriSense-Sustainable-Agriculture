/**
 * AGRISENSE Digital Twin Core Service Layer
 * Intersects with Penman-Monteith physics engines & scenarios
 */

import { apiClient } from "../api/apiClient";
import { API_ENDPOINTS } from "../api/endpoints";
import { FarmTwinState, ScenarioInput, ScenarioResult } from "../types";

export const twinService = {
  async getTwinState(): Promise<FarmTwinState> {
    return apiClient.get<FarmTwinState>(API_ENDPOINTS.twinState);
  },

  async updateTwinState(telemetry: Partial<any>): Promise<any> {
    return apiClient.post<any>(API_ENDPOINTS.twinUpdate, telemetry);
  },

  async runScenario(scenarioId: string): Promise<ScenarioResult> {
    return apiClient.post<ScenarioResult>(API_ENDPOINTS.twinSimulate, { scenarioId });
  },

  async getAnalytics(): Promise<any> {
    return apiClient.get<any>(API_ENDPOINTS.twinAnalytics);
  }
};
