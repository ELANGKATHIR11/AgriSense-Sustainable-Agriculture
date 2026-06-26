/**
 * AGRISENSE React Query Hooks - MLOps Analytics Core
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { mlopsService } from "../services/mlopsService";

export function useModels() {
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: ["mlopsData"],
    queryFn: () => mlopsService.getMLOpsData()
  });

  const retrainMutation = useMutation({
    mutationFn: (modelId: string) => mlopsService.triggerRetrain(modelId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["mlopsData"] });
    }
  });

  return {
    metrics: query.data?.metrics || { averageAccuracy: 0.91, inferenceCount: 3000, averageLatencyMs: 35, activeModelsCount: 4, anomalousInferences: 1, driftIndex: 0.04 },
    registry: query.data?.registry || [],
    logs: query.data?.logs || [],
    isLoading: query.isLoading,
    isError: query.isError,
    triggerRetrain: retrainMutation.mutateAsync,
    isRetraining: retrainMutation.isPending
  };
}
