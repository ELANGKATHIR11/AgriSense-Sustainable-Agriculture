/**
 * AGRISENSE Tabular Regression Model Registry MLOps Service
 */

import { apiClient } from "../api/apiClient";
import { API_ENDPOINTS } from "../api/endpoints";
import { ModelRegistryEntry, PredictionLog } from "../types";

export const mlopsService = {
  async getMLOpsData() {
    return apiClient.get<any>(API_ENDPOINTS.mlops);
  },

  async triggerRetrain(modelId: string): Promise<ModelRegistryEntry> {
    interface RetrainResponse {
      updated: ModelRegistryEntry;
    }
    const response = await apiClient.post<RetrainResponse>(API_ENDPOINTS.mlopsRetrain, { modelId });
    return response.updated;
  }
};
