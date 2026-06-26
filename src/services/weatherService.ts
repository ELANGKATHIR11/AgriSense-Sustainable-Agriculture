/**
 * AGRISENSE Evapotranspiration and Rain Advisory Service
 */

import { apiClient } from "../api/apiClient";
import { API_ENDPOINTS } from "../api/endpoints";
import { WeatherDay } from "../types";

export const weatherService = {
  async getWeatherForecast(): Promise<WeatherDay[]> {
    const response = await apiClient.get<any>(API_ENDPOINTS.weather);
    return response.forecast;
  },

  async getWeatherAdvice() {
    const response = await apiClient.get<any>(API_ENDPOINTS.weather);
    return {
      soilMoistureStatus: "FAO-56 Evapotranspiration active",
      rainAlert: response.riskIndicators?.[0] || "No severe rain storms on the 5-day horizon.",
      temperatureAlert: response.heatStressIndex === "Normal" ? "Thermal comfort zone" : `Heat stress index: ${response.heatStressIndex}`
    };
  }
};
