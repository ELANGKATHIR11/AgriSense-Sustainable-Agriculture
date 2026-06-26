/**
 * AGRISENSE React Query Hooks - Microclimatic Weather Intel
 */

import { useQuery } from "@tanstack/react-query";
import { weatherService } from "../services/weatherService";

export function useWeather() {
  const forecastQuery = useQuery({
    queryKey: ["weatherForecast"],
    queryFn: () => weatherService.getWeatherForecast()
  });

  const adviceQuery = useQuery({
    queryKey: ["weatherAdvice"],
    queryFn: () => weatherService.getWeatherAdvice()
  });

  return {
    forecast: forecastQuery.data || [],
    advice: adviceQuery.data,
    isLoading: forecastQuery.isLoading || adviceQuery.isLoading,
    isError: forecastQuery.isError || adviceQuery.isError,
    refetch: () => {
      forecastQuery.refetch();
      adviceQuery.refetch();
    }
  };
}
