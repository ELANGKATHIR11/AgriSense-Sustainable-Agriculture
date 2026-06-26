/**
 * AGRISENSE React Query Hooks - Dashboard Telemetry
 */

import { useQuery } from "@tanstack/react-query";
import { dashboardService } from "../services/dashboardService";

export function useDashboard() {
  return useQuery({
    queryKey: ["dashboardData"],
    queryFn: () => dashboardService.getDashboardData(),
    staleTime: 45000, // Cache results for 45 seconds
    refetchInterval: 60000 // Refetch every minute
  });
}
