/**
 * AGRISENSE Agronomic Weather & Crop Evapotranspiration Mock Data
 */

import { WeatherDay } from "../types";

export const mockWeatherForecast: WeatherDay[] = [
  { date: "Wednesday", temperature: 28.5, humidity: 62.0, rainfall: 0.0, condition: "sunny", windSpeed: 8.4 },
  { date: "Thursday", temperature: 29.1, humidity: 58.5, rainfall: 1.2, condition: "cloudy", windSpeed: 9.6 },
  { date: "Friday", temperature: 24.8, humidity: 82.0, rainfall: 24.5, condition: "stormy", windSpeed: 18.2 },
  { date: "Saturday", temperature: 26.2, humidity: 75.4, rainfall: 4.8, condition: "rainy", windSpeed: 12.0 },
  { date: "Sunday", temperature: 27.9, humidity: 64.1, rainfall: 0.0, condition: "sunny", windSpeed: 7.8 }
];

export const mockWeatherAdvice = {
  summary: "Approaching convective rain window on Friday afternoon. Expect high humidity index.",
  irrigationImpact: "Precipitation volume (24.5mm) will saturate topsoil. Shut off active solenoid valve cascades starting Friday midday.",
  diseaseRisk: "Pathogen Late Blight sporulation risks are high on Saturday due to 80%+ ambient humidity envelopes."
};
