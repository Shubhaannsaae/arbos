import { useState, useEffect, useCallback } from 'react';
import { useWeb3 } from './useWeb3';
import { agentsService } from '../services/agents';
import { Agent, AgentConfig } from '../types/agents';

interface UseAgentsReturn {
  agents: Agent[];
  loading: boolean;
  error: string | null;
  createAgent: (config: AgentConfig) => Promise<string>;
  startAgent: (id: string) => Promise<void>;
  stopAgent: (id: string) => Promise<void>;
  updateAgent: (id: string, config: Partial<AgentConfig>) => Promise<void>;
  deleteAgent: (id: string) => Promise<void>;
  getAgentMetrics: (id: string) => Promise<any>;
}

export const useAgents = (): UseAgentsReturn => {
  const { signer, chainId } = useWeb3();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadAgents = useCallback(async () => {
    if (!signer || !chainId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const agentsList = await agentsService.getAgents(signer, chainId);
      setAgents(agentsList);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load agents');
    } finally {
      setLoading(false);
    }
  }, [signer, chainId]);

  const createAgent = useCallback(async (config: AgentConfig): Promise<string> => {
    if (!signer || !chainId) throw new Error('Wallet not connected');
    
    setLoading(true);
    setError(null);
    
    try {
      const agentId = await agentsService.createAgent(signer, chainId, config);
      await loadAgents(); // Refresh agents list
      return agentId;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create agent';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [signer, chainId, loadAgents]);

  const startAgent = useCallback(async (id: string) => {
    if (!signer || !chainId) throw new Error('Wallet not connected');
    
    setLoading(true);
    setError(null);
    
    try {
      await agentsService.startAgent(signer, chainId, id);
      await loadAgents();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start agent');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [signer, chainId, loadAgents]);

  const stopAgent = useCallback(async (id: string) => {
    if (!signer || !chainId) throw new Error('Wallet not connected');
    
    setLoading(true);
    setError(null);
    
    try {
      await agentsService.stopAgent(signer, chainId, id);
      await loadAgents();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop agent');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [signer, chainId, loadAgents]);

  const updateAgent = useCallback(async (id: string, config: Partial<AgentConfig>) => {
    if (!signer || !chainId) throw new Error('Wallet not connected');
    
    setLoading(true);
    setError(null);
    
    try {
      await agentsService.updateAgent(signer, chainId, id, config);
      await loadAgents();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update agent');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [signer, chainId, loadAgents]);

  const deleteAgent = useCallback(async (id: string) => {
    if (!signer || !chainId) throw new Error('Wallet not connected');
    
    setLoading(true);
    setError(null);
    
    try {
      await agentsService.deleteAgent(signer, chainId, id);
      await loadAgents();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete agent');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [signer, chainId, loadAgents]);

  const getAgentMetrics = useCallback(async (id: string) => {
    if (!signer || !chainId) throw new Error('Wallet not connected');
    
    try {
      return await agentsService.getAgentMetrics(signer, chainId, id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get agent metrics');
      throw err;
    }
  }, [signer, chainId]);

  // Load agents on mount and when dependencies change
  useEffect(() => {
    loadAgents();
  }, [loadAgents]);

  return {
    agents,
    loading,
    error,
    createAgent,
    startAgent,
    stopAgent,
    updateAgent,
    deleteAgent,
    getAgentMetrics,
  };
};
