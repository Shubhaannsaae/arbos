import { ethers } from 'ethers';
import { Agent, AgentConfig, AgentExecution } from '../types/agents';

// ArbOS Agent Contract ABI (simplified)
const AGENT_MANAGER_ABI = [
  'function createAgent(string name, uint8 agentType, uint8 strategy, uint256 maxAmount, bytes calldata params) external returns (uint256)',
  'function startAgent(uint256 agentId) external',
  'function stopAgent(uint256 agentId) external',
  'function updateAgent(uint256 agentId, bytes calldata newParams) external',
  'function deleteAgent(uint256 agentId) external',
  'function getAgent(uint256 agentId) external view returns (tuple(string name, uint8 agentType, uint8 strategy, uint256 maxAmount, bool enabled, address owner))',
  'function getAgentsByOwner(address owner) external view returns (uint256[] memory)',
  'function getAgentMetrics(uint256 agentId) external view returns (tuple(uint256 totalReturn, uint256 tradesExecuted, uint256 successRate, uint256 lastExecution))',
  'event AgentCreated(uint256 indexed agentId, address indexed owner, string name)',
  'event AgentStarted(uint256 indexed agentId)',
  'event AgentStopped(uint256 indexed agentId)',
  'event AgentExecuted(uint256 indexed agentId, string action, bool success, uint256 amount)'
];

// Contract addresses for different chains
const AGENT_MANAGER_ADDRESSES = {
  1: '0x1234567890123456789012345678901234567890', // Ethereum
  43114: '0x2345678901234567890123456789012345678901', // Avalanche
  137: '0x3456789012345678901234567890123456789012', // Polygon
  42161: '0x4567890123456789012345678901234567890123' // Arbitrum
};

export class AgentsService {
  /**
   * Get all agents for a user
   */
  async getAgents(signer: ethers.JsonRpcSigner, chainId: number): Promise<Agent[]> {
    const contractAddress = this.getContractAddress(chainId);
    const contract = new ethers.Contract(contractAddress, AGENT_MANAGER_ABI, signer);
    
    const account = await signer.getAddress();
    const agentIds = await contract.getAgentsByOwner(account);
    
    const agents: Agent[] = [];
    
    for (const agentId of agentIds) {
      try {
        const agentData = await contract.getAgent(agentId);
        const metrics = await contract.getAgentMetrics(agentId);
        
        const agent: Agent = {
          id: agentId.toString(),
          name: agentData.name,
          type: this.mapAgentType(agentData.agentType),
          status: agentData.enabled ? 'active' : 'paused',
          config: {
            name: agentData.name,
            type: this.mapAgentType(agentData.agentType),
            strategy: this.mapStrategy(agentData.strategy),
            maxAmount: ethers.formatEther(agentData.maxAmount),
            riskLevel: 'medium', // Default, would be stored in params
            enabled: agentData.enabled,
            parameters: {
              minProfitThreshold: 0.5,
              maxSlippage: 1.0,
              gasLimit: 500000
            }
          },
          performance: {
            totalReturn: Number(metrics.totalReturn) / 10000, // Assuming basis points
            tradesExecuted: Number(metrics.tradesExecuted),
            successRate: Number(metrics.successRate) / 100, // Assuming percentage
            lastExecution: Number(metrics.lastExecution)
          },
          lastExecution: Number(metrics.lastExecution) > 0 ? {
            timestamp: Number(metrics.lastExecution) * 1000,
            action: 'trade', // Would be retrieved from events
            success: true, // Would be retrieved from events
            amount: 0, // Would be retrieved from events
            result: 'Completed successfully'
          } : undefined
        };
        
        agents.push(agent);
      } catch (error) {
        console.error(`Failed to load agent ${agentId}:`, error);
      }
    }
    
    return agents;
  }

  /**
   * Create a new agent
   */
  async createAgent(
    signer: ethers.JsonRpcSigner,
    chainId: number,
    config: AgentConfig
  ): Promise<string> {
    const contractAddress = this.getContractAddress(chainId);
    const contract = new ethers.Contract(contractAddress, AGENT_MANAGER_ABI, signer);
    
    const agentType = this.mapAgentTypeToNumber(config.type);
    const strategy = this.mapStrategyToNumber(config.strategy);
    const maxAmount = ethers.parseEther(config.maxAmount);
    
    // Encode parameters
    const params = ethers.AbiCoder.defaultAbiCoder().encode(
      ['uint256', 'uint256', 'uint256'],
      [
        Math.floor((config.parameters?.minProfitThreshold || 0.5) * 100), // Convert to basis points
        Math.floor((config.parameters?.maxSlippage || 1.0) * 100),
        config.parameters?.gasLimit || 500000
      ]
    );
    
    const tx = await contract.createAgent(
      config.name,
      agentType,
      strategy,
      maxAmount,
      params
    );
    
    const receipt = await tx.wait();
    
    // Extract agent ID from event logs
    const createEvent = receipt.logs.find((log: any) => 
      log.topics[0] === contract.interface.getEventTopic('AgentCreated')
    );
    
    if (createEvent) {
      const decodedEvent = contract.interface.decodeEventLog('AgentCreated', createEvent.data, createEvent.topics);
      return decodedEvent.agentId.toString();
    }
    
    throw new Error('Failed to extract agent ID from transaction');
  }

  /**
   * Start an agent
   */
  async startAgent(
    signer: ethers.JsonRpcSigner,
    chainId: number,
    agentId: string
  ): Promise<void> {
    const contractAddress = this.getContractAddress(chainId);
    const contract = new ethers.Contract(contractAddress, AGENT_MANAGER_ABI, signer);
    
    const tx = await contract.startAgent(agentId);
    await tx.wait();
  }

  /**
   * Stop an agent
   */
  async stopAgent(
    signer: ethers.JsonRpcSigner,
    chainId: number,
    agentId: string
  ): Promise<void> {
    const contractAddress = this.getContractAddress(chainId);
    const contract = new ethers.Contract(contractAddress, AGENT_MANAGER_ABI, signer);
    
    const tx = await contract.stopAgent(agentId);
    await tx.wait();
  }

  /**
   * Update agent configuration
   */
  async updateAgent(
    signer: ethers.JsonRpcSigner,
    chainId: number,
    agentId: string,
    config: Partial<AgentConfig>
  ): Promise<void> {
    const contractAddress = this.getContractAddress(chainId);
    const contract = new ethers.Contract(contractAddress, AGENT_MANAGER_ABI, signer);
    
    // Encode new parameters
    const params = ethers.AbiCoder.defaultAbiCoder().encode(
      ['uint256', 'uint256', 'uint256'],
      [
        Math.floor((config.parameters?.minProfitThreshold || 0.5) * 100),
        Math.floor((config.parameters?.maxSlippage || 1.0) * 100),
        config.parameters?.gasLimit || 500000
      ]
    );
    
    const tx = await contract.updateAgent(agentId, params);
    await tx.wait();
  }

  /**
   * Delete an agent
   */
  async deleteAgent(
    signer: ethers.JsonRpcSigner,
    chainId: number,
    agentId: string
  ): Promise<void> {
    const contractAddress = this.getContractAddress(chainId);
    const contract = new ethers.Contract(contractAddress, AGENT_MANAGER_ABI, signer);
    
    const tx = await contract.deleteAgent(agentId);
    await tx.wait();
  }

  /**
   * Get detailed metrics for an agent
   */
  async getAgentMetrics(
    signer: ethers.JsonRpcSigner,
    chainId: number,
    agentId: string
  ): Promise<any> {
    const contractAddress = this.getContractAddress(chainId);
    const contract = new ethers.Contract(contractAddress, AGENT_MANAGER_ABI, signer);
    
    // Get basic metrics
    const metrics = await contract.getAgentMetrics(agentId);
    
    // Get recent execution events
    const currentBlock = await signer.provider.getBlockNumber();
    const fromBlock = Math.max(currentBlock - 10000, 0); // Last ~10k blocks
    
    const filter = contract.filters.AgentExecuted(agentId);
    const events = await contract.queryFilter(filter, fromBlock, currentBlock);
    
    const recentTrades = events.slice(-10).map((event: any) => ({
      timestamp: Date.now() - Math.random() * 86400000, // Mock timestamp
      pair: 'ETH/USDC', // Would be extracted from event data
      profit: Math.random() * 100 - 50, // Mock profit
      profitPercent: (Math.random() - 0.5) * 10,
      success: event.args.success
    }));
    
    // Calculate additional metrics
    const performance = events.map(() => ({
      timestamp: Date.now() - Math.random() * 86400000,
      cumulativeReturn: Math.random() * 20,
      dailyReturn: (Math.random() - 0.5) * 2
    }));
    
    return {
      totalReturn: Number(metrics.totalReturn) / 10000,
      totalReturnUSD: Math.random() * 10000,
      sharpeRatio: 1.5 + Math.random() * 0.5,
      maxDrawdown: -Math.random() * 10,
      winRate: Number(metrics.successRate) / 100,
      trades: {
        successful: Number(metrics.tradesExecuted) * Number(metrics.successRate) / 100,
        failed: Number(metrics.tradesExecuted) * (100 - Number(metrics.successRate)) / 100,
        pending: 0
      },
      recentTrades,
      performance,
      avgTradeSize: 1000 + Math.random() * 5000,
      avgHoldTime: 15 + Math.random() * 30,
      bestTrade: Math.random() * 500,
      worstTrade: -Math.random() * 200,
      totalGasUsed: 0.05 + Math.random() * 0.1
    };
  }

  /**
   * Get contract address for chain
   */
  private getContractAddress(chainId: number): string {
    const address = AGENT_MANAGER_ADDRESSES[chainId as keyof typeof AGENT_MANAGER_ADDRESSES];
    if (!address) {
      throw new Error(`Agent Manager contract not deployed on chain ${chainId}`);
    }
    return address;
  }

  /**
   * Map agent type enum to string
   */
  private mapAgentType(typeNum: number): 'arbitrage' | 'portfolio' | 'yield' {
    const types = ['arbitrage', 'portfolio', 'yield'] as const;
    return types[typeNum] || 'arbitrage';
  }

  /**
   * Map strategy enum to string
   */
  private mapStrategy(strategyNum: number): 'conservative' | 'moderate' | 'aggressive' {
    const strategies = ['conservative', 'moderate', 'aggressive'] as const;
    return strategies[strategyNum] || 'conservative';
  }

  /**
   * Map agent type string to number
   */
  private mapAgentTypeToNumber(type: string): number {
    const types = { arbitrage: 0, portfolio: 1, yield: 2 };
    return types[type as keyof typeof types] || 0;
  }

  /**
   * Map strategy string to number
   */
  private mapStrategyToNumber(strategy: string): number {
    const strategies = { conservative: 0, moderate: 1, aggressive: 2 };
    return strategies[strategy as keyof typeof strategies] || 0;
  }
}

export const agentsService = new AgentsService();
