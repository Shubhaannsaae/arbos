import { useState, useEffect, useCallback } from 'react';
import { useWeb3 } from './useWeb3';
import { chainlinkService } from '../services/chainlink';
import { ChainlinkData } from '../types';

interface UseChainlinkReturn {
  priceFeeds: { [token: string]: number };
  vrfResults: { [requestId: string]: number };
  ccipMessages: any[];
  automationStatus: boolean;
  loading: boolean;
  error: string | null;
  requestPrice: (token: string) => Promise<void>;
  requestRandomness: () => Promise<string>;
  sendCCIPMessage: (destinationChain: number, receiver: string, data: string) => Promise<string>;
}

export const useChainlink = (): UseChainlinkReturn => {
  const { provider, signer, chainId } = useWeb3();
  const [priceFeeds, setPriceFeeds] = useState<{ [token: string]: number }>({});
  const [vrfResults, setVrfResults] = useState<{ [requestId: string]: number }>({});
  const [ccipMessages, setCcipMessages] = useState<any[]>([]);
  const [automationStatus, setAutomationStatus] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requestPrice = useCallback(async (token: string) => {
    if (!signer || !chainId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const price = await chainlinkService.getLatestPrice(signer, chainId, token);
      setPriceFeeds(prev => ({ ...prev, [token]: price }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get price');
    } finally {
      setLoading(false);
    }
  }, [signer, chainId]);

  const requestRandomness = useCallback(async (): Promise<string> => {
    if (!signer || !chainId) throw new Error('Wallet not connected');
    
    setLoading(true);
    setError(null);
    
    try {
      const requestId = await chainlinkService.requestRandomness(signer, chainId);
      return requestId;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to request randomness';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [signer, chainId]);

  const sendCCIPMessage = useCallback(async (
    destinationChain: number,
    receiver: string,
    data: string
  ): Promise<string> => {
    if (!signer || !chainId) throw new Error('Wallet not connected');
    
    setLoading(true);
    setError(null);
    
    try {
      const messageId = await chainlinkService.sendCCIPMessage(
        signer,
        chainId,
        destinationChain,
        receiver,
        data
      );
      
      // Add to messages list
      setCcipMessages(prev => [...prev, {
        id: messageId,
        destinationChain,
        receiver,
        data,
        timestamp: Date.now(),
        status: 'pending'
      }]);
      
      return messageId;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send CCIP message';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setLoading(false);
    }
  }, [signer, chainId]);

  // Initialize and load data
  useEffect(() => {
    if (provider && chainId) {
      // Load initial price feeds for common tokens
      const commonTokens = ['ETH', 'BTC', 'LINK', 'USDC'];
      commonTokens.forEach(token => {
        requestPrice(token).catch(console.error);
      });
    }
  }, [provider, chainId, requestPrice]);

  return {
    priceFeeds,
    vrfResults,
    ccipMessages,
    automationStatus,
    loading,
    error,
    requestPrice,
    requestRandomness,
    sendCCIPMessage,
  };
};
