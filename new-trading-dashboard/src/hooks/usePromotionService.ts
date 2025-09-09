import { useState, useEffect, useCallback } from 'react';
import promotionService, {
  PromotionPipeline,
  StrategyCandidate,
  ValidationResult
} from '@/services/promotionService';

interface UsePromotionServiceReturn {
  pipelines: PromotionPipeline[];
  candidates: StrategyCandidate[];
  stats: {
    totalCandidates: number;
    totalPromoted: number;
    totalRejected: number;
    successRate: number;
    activePipelines: number;
  };

  // Actions
  addCandidate: (candidate: StrategyCandidate) => void;
  promoteCandidate: (candidateId: string, pipelineId: string) => Promise<boolean>;
  getValidationResult: (candidateId: string) => ValidationResult | undefined;
  checkForPromotions: () => Promise<void>;
}

export const usePromotionService = (): UsePromotionServiceReturn => {
  const [pipelines, setPipelines] = useState<PromotionPipeline[]>([]);
  const [candidates, setCandidates] = useState<StrategyCandidate[]>([]);
  const [stats, setStats] = useState({
    totalCandidates: 0,
    totalPromoted: 0,
    totalRejected: 0,
    successRate: 0,
    activePipelines: 0
  });

  // Load initial data
  useEffect(() => {
    refreshData();
  }, []);

  // Periodic updates
  useEffect(() => {
    const interval = setInterval(() => {
      refreshData();
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const refreshData = useCallback(() => {
    setPipelines(promotionService.getPipelines());
    setCandidates(promotionService.getCandidates());
    setStats(promotionService.getPromotionStats());
  }, []);

  const addCandidate = useCallback((candidate: StrategyCandidate) => {
    promotionService.addCandidate(candidate);
    refreshData();
  }, [refreshData]);

  const promoteCandidate = useCallback(async (
    candidateId: string,
    pipelineId: string
  ): Promise<boolean> => {
    try {
      const result = await promotionService.promoteCandidate(candidateId, pipelineId);
      refreshData();
      return result;
    } catch (error) {
      console.error('Failed to promote candidate:', error);
      throw error;
    }
  }, [refreshData]);

  const getValidationResult = useCallback((candidateId: string) => {
    return promotionService.getValidationResult(candidateId);
  }, []);

  const checkForPromotions = useCallback(async () => {
    try {
      await promotionService.checkForPromotions();
      refreshData();
    } catch (error) {
      console.error('Failed to check for promotions:', error);
      throw error;
    }
  }, [refreshData]);

  return {
    pipelines,
    candidates,
    stats,

    // Actions
    addCandidate,
    promoteCandidate,
    getValidationResult,
    checkForPromotions
  };
};

export default usePromotionService;
