import { useState, useEffect, useCallback } from 'react';
import { useWeb3 } from './useWeb3';
import { portfolioService } from '../services/api';
import { Portfolio, PortfolioPosition } from '../types/portfolio';

interface UsePortfolioReturn {
  portfolio: Portfolio | null;
  positions: PortfolioPosition[];
  loading: boolean;
  error: string | null;
  rebalance: () => Promise<void>;
  updateAllocation: (allocations: { [token: string]: number }) => Promise<void>;
  refreshPortfolio: () => Promise<void>;
}

export const usePortfolio = (): UsePortfolioReturn => {
  const { signer, chainId, account } = useWeb3();
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [positions, setPositions] = useState<PortfolioPosition[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshPortfolio = useCallback(async () => {
    if (!account || !chainId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const [portfolioData, positionsData] = await Promise.all([
        portfolioService.getPortfolio(account, chainId),
        portfolioService.getPositions(account, chainId)
      ]);
      
      setPortfolio(portfolioData);
      setPositions(positionsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load portfolio');
    } finally {
      setLoading(false);
    }
  }, [account, chainId]);

  const rebalance = useCallback(async () => {
    if (!signer || !chainId) throw new Error('Wallet not connected');
    
    setLoading(true);
    setError(null);
    
    try {
      await portfolioService.rebalancePortfolio(signer, chainId);
      await refreshPortfolio();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to rebalance portfolio';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [signer, chainId, refreshPortfolio]);

  const updateAllocation = useCallback(async (allocations: { [token: string]: number }) => {
    if (!signer || !chainId) throw new Error('Wallet not connected');
    
    setLoading(true);
    setError(null);
    
    try {
      await portfolioService.updateAllocation(signer, chainId, allocations);
      await refreshPortfolio();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update allocation';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [signer, chainId, refreshPortfolio]);

  // Load portfolio data when wallet connects
  useEffect(() => {
    refreshPortfolio();
    
    // Refresh every 60 seconds
    const interval = setInterval(refreshPortfolio, 60000);
    return () => clearInterval(interval);
  }, [refreshPortfolio]);

  return {
    portfolio,
    positions,
    loading,
    error,
    rebalance,
    updateAllocation,
    refreshPortfolio,
  };
};
