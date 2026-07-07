/**
 * License: GNU Affero General Public License v3.0 (AGPL-3.0)
 * This file is part of AgriSense.
 * 
 * TERMS OF USE:
 * This project is licensed under the AGPL-3.0. Private modifications or private use
 * without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
 * AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
 * Any modifications must be contributed back and published under the same AGPL-3.0 license.
 */

/**
 * AGRISENSE Computer Vision Service
 * Routes image + mode to the correct backend vision endpoint.
 */

import { apiClient } from "../api/apiClient";
import { API_ENDPOINTS } from "../api/endpoints";
import { DiseaseDetectionResult } from "../types";

type VisionMode = "disease" | "weed" | "nutrient" | "pest" | "crop" | "health" | "yield";

const ENDPOINT_MAP: Record<VisionMode, string> = {
  disease:  API_ENDPOINTS.diseaseDetection,
  weed:     API_ENDPOINTS.weedDetection,
  nutrient: API_ENDPOINTS.nutrientDetection,
  pest:     API_ENDPOINTS.pestDetection,
  crop:     API_ENDPOINTS.cropIdentification,
  health:   API_ENDPOINTS.healthAnalysis,
  yield:    API_ENDPOINTS.yieldPrediction,
};

export const diseaseService = {
  async detectVision(imageBase64: string, mode: VisionMode = "disease"): Promise<DiseaseDetectionResult> {
    const endpoint = ENDPOINT_MAP[mode] ?? API_ENDPOINTS.diseaseDetection;
    return apiClient.post<DiseaseDetectionResult>(endpoint, { imageBase64, mode });
  },

  /** @deprecated Use detectVision with mode="disease" */
  async detectDisease(imageBase64: string, mode: VisionMode = "disease"): Promise<DiseaseDetectionResult> {
    return this.detectVision(imageBase64, mode);
  }
};
