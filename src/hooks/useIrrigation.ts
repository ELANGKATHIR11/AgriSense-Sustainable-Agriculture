/**
 * AGRISENSE React Query Hooks - Irrigation Optimization
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { irrigationService } from "../services/irrigationService";
import { IrrigationInput } from "../types";

export function useIrrigation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (input: IrrigationInput) => irrigationService.optimizeIrrigation(input),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mlopsData"] });
    }
  });
}
