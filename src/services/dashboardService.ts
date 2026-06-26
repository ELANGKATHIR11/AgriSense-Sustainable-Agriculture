/**
 * AGRISENSE Dashboard Feature Service
 */

import { apiClient } from "../api/apiClient";
import { API_ENDPOINTS } from "../api/endpoints";

export const dashboardService = {
  async getDashboardData() {
    return apiClient.get<any>(API_ENDPOINTS.dashboard);
  }
};
