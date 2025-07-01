import { useState, useEffect, useCallback } from 'react';
import { useWeb3 } from './useWeb3';
import { arbitrageService } from '../services/api';
import { ArbitrageOpportunity, ArbitrageExecution } from '../types/arbitrage';

interface UseArbitrageReturn {
  opportunities: ArbitrageOpportunity[];
  executions: ArbitrageExecution[];
  loading: boolean;
  error: string | null;
  executeArbitrage: (opportunityId: string, amount: string) => Promise<string>;
  refreshOpportunities: () => Promise<void>;
}

export const useArbitrage = (): UseArbitrageReturn => {
  const { signer, chainId } = useWeb3();
  const [opportunities, setOpportunities] = useState<ArbitrageOpportunity[]>([]);
  const [executions, setExecutions] = useState<ArbitrageExecution[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshOpportunities = useCallback(async () => {
    if (!chainId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await arbitrageService.getOpportunities(chainId);
      setOpportunities(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load opportunities');
    } finally {
      setLoading(false);
    }
  }, [chainId]);

  const executeArbitrage = useCallback(async (
    opportunityId: string,
    amount: string
  ): Promise<string> => {
    if (!signer || !chainId) throw new Error('Wallet not connected');
    
    setLoading(true);
    setError(null);
    
    try {
      const txHash = await arbitrageService.executeArbitrage(
        signer,
        chainId,
        opportunityId,
        amount
      );
      
      // Add to executions
      const execution: ArbitrageExecution = {
        id: txHash,
        opportunityId,
        amount,
        txHash,
        status: 'pending',
        timestamp: Date.now(),
      };
      
      setExecutions(prev => [execution, ...prev]);
      await refreshOpportunities(); // Refresh after execution
      
      return txHash;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to execute arbitrage';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [signer, chainId, refreshOpportunities]);

  // Load data on mount and refresh periodically
  useEffect(() => {
    refreshOpportunities();
    
    // Refresh every 30 seconds
    const interval = setInterval(refreshOpportunities, 30000);
    return () => clearInterval(interval);
  }, [refreshOpportunities]);

  // Load execution history
  useEffect(() => {
    if (signer && chainId) {
      arbitrageService.getExecutionHistory(signer, chainId)
        .then(setExecutions)
        .catch(console.error);
    }
  }, [signer, chainId]);

  return {
    opportunities,
    executions,
    loading,
    error,
    executeArbitrage,
    refreshOpportunities,
  };
};
