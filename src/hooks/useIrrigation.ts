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
