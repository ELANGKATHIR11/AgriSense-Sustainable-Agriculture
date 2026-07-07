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
 * AGRISENSE React Query Hooks - ESP32 Hardware Logging
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { sensorService } from "../services/sensorService";
import { SensorReading } from "../types";
import { useSensorStore } from "../store/sensorStore";

export function useSensors() {
  const queryClient = useQueryClient();
  const setReadings = useSensorStore((state) => state.setReadings);
  const appendReading = useSensorStore((state) => state.appendReading);

  const query = useQuery({
    queryKey: ["sensorsData"],
    queryFn: async () => {
      const data = await sensorService.getSensors();
      setReadings(data);
      return data;
    },
    refetchInterval: 30000 // Ingest refresh window
  });

  const ingestMutation = useMutation({
    mutationFn: (reading: Partial<SensorReading>) => sensorService.ingestReading(reading),
    onSuccess: (newReading) => {
      appendReading(newReading);
      queryClient.invalidateQueries({ queryKey: ["sensorsData"] });
    }
  });

  return {
    readings: query.data || [],
    isLoading: query.isLoading,
    isError: query.isError,
    refetch: query.refetch,
    ingest: ingestMutation.mutate,
    isIngesting: ingestMutation.isPending
  };
}
