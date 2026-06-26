/**
 * AGRISENSE Tabular ML Crop Suitability & Soil Evaluation Service
 */

import { apiClient } from "../api/apiClient";
import { API_ENDPOINTS } from "../api/endpoints";
import { CropRecommendationInput, CropRecommendationResult } from "../types";

export const cropService = {
  async getRecommendation(input: CropRecommendationInput): Promise<CropRecommendationResult> {
    return apiClient.post<CropRecommendationResult>(API_ENDPOINTS.cropRecommendation, input);
  }
};
