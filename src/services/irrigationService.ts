/**
 * AGRISENSE Evapotranspiration and Field Moisture Advice Service
 */

import { apiClient } from "../api/apiClient";
import { API_ENDPOINTS } from "../api/endpoints";
import { IrrigationInput, IrrigationResult } from "../types";

export const irrigationService = {
  async optimizeIrrigation(input: IrrigationInput): Promise<IrrigationResult> {
    return apiClient.post<IrrigationResult>(API_ENDPOINTS.irrigationRecommendation, input);
  }
};
