/**
 * AGRISENSE React Query Hooks - Crop Tonnage Yield Prediction
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { yieldService } from "../services/yieldService";
import { YieldInput } from "../types";

export function useYield() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (input: YieldInput) => yieldService.predictYield(input),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mlopsData"] });
    }
  });
}
