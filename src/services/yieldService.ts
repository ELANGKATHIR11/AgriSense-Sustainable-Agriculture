/**
 * AGRISENSE Crop Yield Forecast Service
 */

import { apiClient } from "../api/apiClient";
import { API_ENDPOINTS } from "../api/endpoints";
import { YieldInput, YieldResult } from "../types";

export const yieldService = {
  async predictYield(input: YieldInput): Promise<YieldResult> {
    return apiClient.post<YieldResult>(API_ENDPOINTS.yieldForecast, input);
  }
};
