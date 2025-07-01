import { ethers } from 'ethers';
import { ArbitrageOpportunity, ArbitrageExecution } from '../types/arbitrage';
import { Portfolio, PortfolioPosition } from '../types/portfolio';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.arbos.io';

class APIService {
  private baseURL: string;

  constructor() {
    this.baseURL = API_BASE_URL;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    const response = await fetch(url, config);
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Arbitrage API
  async getOpportunities(chainId: number): Promise<ArbitrageOpportunity[]> {
    return this.request<ArbitrageOpportunity[]>(`/arbitrage/opportunities?chainId=${chainId}`);
  }

  async executeArbitrage(
    signer: ethers.JsonRpcSigner,
    chainId: number,
    opportunityId: string,
    amount: string
  ): Promise<string> {
    const account = await signer.getAddress();
    
    const response = await this.request<{ txData: any }>('/arbitrage/execute', {
      method: 'POST',
      body: JSON.stringify({
        chainId,
        opportunityId,
        amount,
        account
      })
    });

    // Execute the transaction using the signer
    const tx = await signer.sendTransaction(response.txData);
    return tx.hash;
  }

  async getExecutionHistory(
    signer: ethers.JsonRpcSigner,
    chainId: number
  ): Promise<ArbitrageExecution[]> {
    const account = await signer.getAddress();
    return this.request<ArbitrageExecution[]>(`/arbitrage/history?account=${account}&chainId=${chainId}`);
  }

  // Portfolio API
  async getPortfolio(account: string, chainId: number): Promise<Portfolio> {
    return this.request<Portfolio>(`/portfolio?account=${account}&chainId=${chainId}`);
  }

  async getPositions(account: string, chainId: number): Promise<PortfolioPosition[]> {
    return this.request<PortfolioPosition[]>(`/portfolio/positions?account=${account}&chainId=${chainId}`);
  }

  async rebalancePortfolio(
    signer: ethers.JsonRpcSigner,
    chainId: number
  ): Promise<string> {
    const account = await signer.getAddress();
    
    const response = await this.request<{ txData: any }>('/portfolio/rebalance', {
      method: 'POST',
      body: JSON.stringify({ account, chainId })
    });

    const tx = await signer.sendTransaction(response.txData);
    return tx.hash;
  }

  async updateAllocation(
    signer: ethers.JsonRpcSigner,
    chainId: number,
    allocations: { [token: string]: number }
  ): Promise<string> {
    const account = await signer.getAddress();
    
    const response = await this.request<{ txData: any }>('/portfolio/allocate', {
      method: 'POST',
      body: JSON.stringify({ account, chainId, allocations })
    });

    const tx = await signer.sendTransaction(response.txData);
    return tx.hash;
  }

  // Price API
  async getTokenPrice(token: string, chainId: number): Promise<number> {
    const response = await this.request<{ price: number }>(`/prices/${token}?chainId=${chainId}`);
    return response.price;
  }

  async getHistoricalPrices(
    token: string,
    chainId: number,
    timeframe: '1h' | '24h' | '7d' | '30d'
  ): Promise<Array<{ timestamp: number; price: number }>> {
    return this.request(`/prices/${token}/history?chainId=${chainId}&timeframe=${timeframe}`);
  }
}

export const apiService = new APIService();

// Specific service exports
export const arbitrageService = {
  getOpportunities: (chainId: number) => apiService.getOpportunities(chainId),
  executeArbitrage: (signer: ethers.JsonRpcSigner, chainId: number, opportunityId: string, amount: string) =>
    apiService.executeArbitrage(signer, chainId, opportunityId, amount),
  getExecutionHistory: (signer: ethers.JsonRpcSigner, chainId: number) =>
    apiService.getExecutionHistory(signer, chainId),
};

export const portfolioService = {
  getPortfolio: (account: string, chainId: number) => apiService.getPortfolio(account, chainId),
  getPositions: (account: string, chainId: number) => apiService.getPositions(account, chainId),
  rebalancePortfolio: (signer: ethers.JsonRpcSigner, chainId: number) =>
    apiService.rebalancePortfolio(signer, chainId),
  updateAllocation: (signer: ethers.JsonRpcSigner, chainId: number, allocations: { [token: string]: number }) =>
    apiService.updateAllocation(signer, chainId, allocations),
};

export const priceService = {
  getTokenPrice: (token: string, chainId: number) => apiService.getTokenPrice(token, chainId),
  getHistoricalPrices: (token: string, chainId: number, timeframe: '1h' | '24h' | '7d' | '30d') =>
    apiService.getHistoricalPrices(token, chainId, timeframe),
};
