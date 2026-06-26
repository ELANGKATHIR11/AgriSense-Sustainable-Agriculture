/**
 * AGRISENSE React Query Hooks - Crop Recommendation Form
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { cropService } from "../services/cropService";
import { CropRecommendationInput } from "../types";

export function useCropRecommendation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (input: CropRecommendationInput) => cropService.getRecommendation(input),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mlopsData"] });
    }
  });
}
