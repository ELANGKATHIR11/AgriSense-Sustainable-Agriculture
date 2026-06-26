/**
 * AGRISENSE React Query Hooks - Crop Pathology Detection
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { diseaseService } from "../services/diseaseService";

export function useDiseaseAnalysis() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ imageBase64, mode }: { imageBase64: string; mode: "disease" | "weed" }) =>
      diseaseService.detectDisease(imageBase64, mode),
    onSuccess: () => {
      // Invalidate predictions logs or ML stats so they pull updated feeds
      queryClient.invalidateQueries({ queryKey: ["mlopsData"] });
    }
  });
}
