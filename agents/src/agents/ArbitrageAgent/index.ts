import { ethers } from 'ethers';
import { logger } from '../../shared/utils/logger';
import { Validator } from '../../shared/utils/validator';
import { getAgentConfig, getNetworkConfig } from '../../config/agentConfig';
import { getElizaConfig, invokeBedrockModel } from '../../config/elizaConfig';
import { renderPromptTemplate } from '../../config/modelConfig';
import { ARBITRAGE_THRESHOLDS, isArbitrageViable } from '../../shared/constants/thresholds';
import { AgentContext, AgentDecision, AgentExecution, AgentState } from '../../shared/types/agent';
import { ArbitrageOpportunity } from '../../shared/types/market';

// Import actions
import { detectOpportunities } from './actions/detectOpportunities';
import { calculateProfitability } from './actions/calculateProfitability';
import { executeArbitrage } from './actions/executeArbitrage';

// Import evaluators
import { PriceEvaluator } from './evaluators/priceEvaluator';
import { RiskEvaluator } from './evaluators/riskEvaluator';

// Import providers
import { DexProvider } from './providers/dexProvider';
import { ChainlinkProvider } from './providers/chainlinkProvider';

export interface ArbitrageAgentConfig {
  minProfitThreshold: number;
  maxPositionSize: string;
  maxSlippage: number;
  supportedDexes: string[];
  supportedChains: number[];
  riskTolerance: 'low' | 'medium' | 'high';
  autoExecute: boolean;
  maxConcurrentTrades: number;
  crossChainEnabled: boolean;
}

export class ArbitrageAgent {
  private agentId: string;
  private config: ArbitrageAgentConfig;
  private state: AgentState;
  private providers: {
    dex: DexProvider;
    chainlink: ChainlinkProvider;
  };
  private evaluators: {
    price: PriceEvaluator;
    risk: RiskEvaluator;
  };
  private executionInterval: NodeJS.Timeout | null = null;
  private activeExecutions: Map<string, AgentExecution> = new Map();

  constructor(agentId: string, config: ArbitrageAgentConfig) {
    this.agentId = agentId;
    this.config = config;
    
    // Initialize providers
    this.providers = {
      dex: new DexProvider(config.supportedDexes, config.supportedChains),
      chainlink: new ChainlinkProvider(config.supportedChains)
    };

    // Initialize evaluators
    this.evaluators = {
      price: new PriceEvaluator(this.providers.chainlink),
      risk: new RiskEvaluator(config.riskTolerance)
    };

    // Initialize state
    this.state = this.initializeState();

    logger.agentStarted(this.agentId, 'arbitrage', {
      config: this.config,
      supportedChains: config.supportedChains
    });
  }

  private initializeState(): AgentState {
    const agentConfig = getAgentConfig('arbitrage');
    
    return {
      agentId: this.agentId,
      status: 'idle',
      healthScore: 100,
      errorCount: 0,
      warningCount: 0,
      memory: {
        shortTerm: {},
        longTerm: {},
        episodic: [],
        semantic: {}
      },
      configuration: agentConfig.constraints as any,
      performance: {
        agentId: this.agentId,
        period: {
          start: Date.now(),
          end: Date.now()
        },
        metrics: {
          totalExecutions: 0,
          successfulExecutions: 0,
          failedExecutions: 0,
          successRate: 0,
          averageExecutionTime: 0,
          totalGasUsed: ethers.BigNumber.from(0),
          averageGasUsed: ethers.BigNumber.from(0),
          totalProfit: ethers.BigNumber.from(0),
          totalLoss: ethers.BigNumber.from(0),
          netProfit: ethers.BigNumber.from(0),
          profitFactor: 0,
          sharpeRatio: 0,
          maxDrawdown: 0,
          winRate: 0
        },
        chainlinkUsage: {
          dataFeedCalls: 0,
          automationExecutions: 0,
          functionsRequests: 0,
          vrfRequests: 0,
          ccipMessages: 0,
          totalCost: ethers.BigNumber.from(0)
        }
      },
      resources: {
        cpuUsage: 0,
        memoryUsage: 0,
        networkLatency: 0,
        apiCallsRemaining: 1000,
        gasAllowanceRemaining: ethers.BigNumber.from('10000000000000000000') // 10 ETH
      }
    };
  }

  async start(): Promise<void> {
    if (this.state.status !== 'idle') {
      throw new Error('Agent is already running');
    }

    this.state.status = 'analyzing';
    
    // Initialize providers
    await this.providers.chainlink.initialize();
    await this.providers.dex.initialize();

    // Start execution loop
    const interval = getAgentConfig('arbitrage').executionInterval;
    this.executionInterval = setInterval(async () => {
      await this.executionLoop();
    }, interval);

    logger.info('ArbitrageAgent started', {
      agentId: this.agentId,
      interval,
      config: this.config
    });
  }

  async stop(): Promise<void> {
    this.state.status = 'idle';
    
    if (this.executionInterval) {
      clearInterval(this.executionInterval);
      this.executionInterval = null;
    }

    // Wait for active executions to complete
    const activeCount = this.activeExecutions.size;
    if (activeCount > 0) {
      logger.info(`Waiting for ${activeCount} active executions to complete`);
      
      const timeout = 300000; // 5 minutes
      const startTime = Date.now();
      
      while (this.activeExecutions.size > 0 && (Date.now() - startTime) < timeout) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    logger.agentStopped(this.agentId, 'arbitrage');
  }

  private async executionLoop(): Promise<void> {
    try {
      this.state.status = 'analyzing';
      
      // Check if we've reached concurrent execution limit
      if (this.activeExecutions.size >= this.config.maxConcurrentTrades) {
        logger.debug('Max concurrent trades reached, skipping cycle', {
          agentId: this.agentId,
          activeExecutions: this.activeExecutions.size,
          maxConcurrent: this.config.maxConcurrentTrades
        });
        return;
      }

      // Detect arbitrage opportunities
      const opportunities = await this.detectOpportunities();
      
      if (opportunities.length === 0) {
        this.state.status = 'idle';
        return;
      }

      // Analyze each opportunity
      for (const opportunity of opportunities) {
        if (this.activeExecutions.size >= this.config.maxConcurrentTrades) {
          break;
        }

        const decision = await this.analyzeOpportunity(opportunity);
        
        if (decision.action === 'execute' && this.config.autoExecute) {
          await this.initiateExecution(opportunity, decision);
        }
      }

      this.state.status = 'idle';

    } catch (error) {
      this.state.status = 'error';
      this.state.errorCount++;
      
      logger.agentError(this.agentId, 'arbitrage', error as Error, {
        executionLoop: true
      });

      // Pause execution if too many errors
      if (this.state.errorCount >= 5) {
        this.state.status = 'paused';
        logger.error('Agent paused due to excessive errors', {
          agentId: this.agentId,
          errorCount: this.state.errorCount
        });
      }
    }
  }

  private async detectOpportunities(): Promise<ArbitrageOpportunity[]> {
    try {
      const context: AgentContext = {
        agentId: this.agentId,
        agentType: 'arbitrage',
        userId: 'system', // System-initiated
        sessionId: `session_${Date.now()}`,
        networkIds: this.config.supportedChains,
        timestamp: Date.now(),
        gasPrice: await this.providers.dex.getGasPrice(1),
        nonce: 0
      };

      return await detectOpportunities(
        this.providers.dex,
        this.providers.chainlink,
        context,
        {
          supportedDexes: this.config.supportedDexes,
          supportedChains: this.config.supportedChains,
          minProfitThreshold: this.config.minProfitThreshold,
          maxPositionSize: ethers.utils.parseEther(this.config.maxPositionSize)
        }
      );

    } catch (error) {
      logger.error('Failed to detect opportunities', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
      return [];
    }
  }

  private async analyzeOpportunity(opportunity: ArbitrageOpportunity): Promise<AgentDecision> {
    try {
      // Calculate detailed profitability
      const profitability = await calculateProfitability(
        opportunity,
        this.providers.dex,
        this.providers.chainlink,
        {
          maxSlippage: this.config.maxSlippage,
          gasOptimization: true
        }
      );

      // Perform risk assessment
      const riskAssessment = await this.evaluators.risk.evaluateArbitrageRisk(
        opportunity,
        profitability
      );

      // Get AI decision from Bedrock
      const aiDecision = await this.getAIDecision(opportunity, profitability, riskAssessment);

      // Validate decision
      const isViable = isArbitrageViable(
        profitability.netProfitPercentage,
        parseFloat(ethers.utils.formatEther(profitability.netProfitUsd)),
        riskAssessment.overallRisk,
        aiDecision.confidence
      );

      const decision: AgentDecision = {
        action: isViable && aiDecision.recommendation === 'execute' ? 'execute' : 'reject',
        confidence: aiDecision.confidence,
        reasoning: aiDecision.reasoning,
        parameters: {
          opportunity,
          profitability,
          riskAssessment,
          aiAnalysis: aiDecision
        },
        riskScore: riskAssessment.overallRisk,
        expectedOutcome: profitability,
        alternatives: [
          {
            action: 'monitor',
            probability: 0.3,
            outcome: 'Continue monitoring for better conditions'
          },
          {
            action: 'reject',
            probability: 0.2,
            outcome: 'Skip due to insufficient profit or high risk'
          }
        ]
      };

      logger.decisionMade(this.agentId, decision.action, decision.confidence, {
        opportunityId: opportunity.id,
        tokenPair: opportunity.tokenPair,
        expectedProfit: ethers.utils.formatEther(profitability.netProfitUsd),
        riskScore: riskAssessment.overallRisk
      });

      return decision;

    } catch (error) {
      logger.error('Failed to analyze opportunity', {
        agentId: this.agentId,
        opportunityId: opportunity.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        action: 'reject',
        confidence: 0,
        reasoning: `Analysis failed: ${error instanceof Error ? error.message : String(error)}`,
        parameters: { opportunity },
        riskScore: 100,
        expectedOutcome: null,
        alternatives: []
      };
    }
  }

  private async getAIDecision(
    opportunity: ArbitrageOpportunity,
    profitability: any,
    riskAssessment: any
  ): Promise<{
    recommendation: 'execute' | 'reject' | 'monitor';
    confidence: number;
    reasoning: string;
  }> {
    try {
      const elizaConfig = getElizaConfig('arbitrage');
      
      const prompt = renderPromptTemplate('arbitrage_analysis', {
        tokenPair: opportunity.tokenPair,
        sourceExchange: opportunity.sourceExchange.name,
        sourceChain: getNetworkConfig(opportunity.sourceExchange.chainId).name,
        targetExchange: opportunity.targetExchange.name,
        targetChain: getNetworkConfig(opportunity.targetExchange.chainId).name,
        priceDifference: opportunity.priceDifferencePercentage.toFixed(3),
        liquidity: ethers.utils.formatEther(opportunity.sourceExchange.liquidity.add(opportunity.targetExchange.liquidity)),
        gasCosts: ethers.utils.formatEther(opportunity.estimatedGasCost),
        marketConditions: await this.getMarketConditions()
      });

      const response = await invokeBedrockModel({
        modelId: elizaConfig.modelId,
        prompt,
        maxTokens: elizaConfig.maxTokens,
        temperature: elizaConfig.temperature
      });

      // Parse AI response
      const aiResponse = JSON.parse(response);
      
      return {
        recommendation: aiResponse.recommendation || 'reject',
        confidence: aiResponse.confidence || 0,
        reasoning: aiResponse.analysis || 'No analysis provided'
      };

    } catch (error) {
      logger.error('AI decision failed, using fallback logic', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });

      // Fallback to rule-based decision
      const profitPercentage = opportunity.potentialProfitPercentage;
      const riskScore = riskAssessment.overallRisk;

      if (profitPercentage >= ARBITRAGE_THRESHOLDS.MIN_PROFIT_PERCENTAGE && 
          riskScore <= 50) {
        return {
          recommendation: 'execute',
          confidence: 0.7,
          reasoning: 'Fallback decision: Profit above threshold and acceptable risk'
        };
      }

      return {
        recommendation: 'reject',
        confidence: 0.8,
        reasoning: 'Fallback decision: Insufficient profit or high risk'
      };
    }
  }

  private async getMarketConditions(): Promise<string> {
    try {
      // Get gas prices across chains
      const gasPrices = await Promise.all(
        this.config.supportedChains.map(async (chainId) => {
          const gasPrice = await this.providers.dex.getGasPrice(chainId);
          return `${getNetworkConfig(chainId).name}: ${ethers.utils.formatUnits(gasPrice, 'gwei')} Gwei`;
        })
      );

      // Get network congestion data
      const congestionData = await this.providers.dex.getNetworkCongestion();

      return `Gas Prices: ${gasPrices.join(', ')}. Network congestion: ${JSON.stringify(congestionData)}`;

    } catch (error) {
      return 'Market conditions unavailable';
    }
  }

  private async initiateExecution(
    opportunity: ArbitrageOpportunity,
    decision: AgentDecision
  ): Promise<void> {
    const executionId = `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const execution: AgentExecution = {
      id: executionId,
      agentId: this.agentId,
      decision,
      startTime: Date.now(),
      status: 'pending',
      transactions: [],
      gasUsed: ethers.BigNumber.from(0),
      actualOutcome: null,
      errors: []
    };

    this.activeExecutions.set(executionId, execution);

    logger.executionStarted(this.agentId, executionId, 'arbitrage', {
      opportunityId: opportunity.id,
      tokenPair: opportunity.tokenPair
    });

    try {
      execution.status = 'executing';
      
      const result = await executeArbitrage(
        opportunity,
        this.providers.dex,
        this.providers.chainlink,
        {
          maxSlippage: this.config.maxSlippage,
          gasOptimization: true,
          mevProtection: true
        }
      );

      execution.status = 'completed';
      execution.endTime = Date.now();
      execution.transactions = result.transactions;
      execution.gasUsed = result.totalGasUsed;
      execution.actualOutcome = result.actualProfit;

      // Update performance metrics
      this.updatePerformanceMetrics(execution, true);

      logger.executionCompleted(this.agentId, executionId, true, {
        actualProfit: ethers.utils.formatEther(result.actualProfit),
        gasUsed: result.totalGasUsed.toString(),
        duration: execution.endTime - execution.startTime
      });

    } catch (error) {
      execution.status = 'failed';
      execution.endTime = Date.now();
      execution.errors.push(error instanceof Error ? error.message : String(error));

      this.updatePerformanceMetrics(execution, false);

      logger.executionCompleted(this.agentId, executionId, false, {
        error: error instanceof Error ? error.message : String(error),
        duration: execution.endTime! - execution.startTime
      });

    } finally {
      this.activeExecutions.delete(executionId);
    }
  }

  private updatePerformanceMetrics(execution: AgentExecution, success: boolean): void {
    const metrics = this.state.performance.metrics;
    
    metrics.totalExecutions++;
    
    if (success) {
      metrics.successfulExecutions++;
      if (execution.actualOutcome && ethers.BigNumber.isBigNumber(execution.actualOutcome)) {
        metrics.totalProfit = metrics.totalProfit.add(execution.actualOutcome);
      }
    } else {
      metrics.failedExecutions++;
    }

    metrics.successRate = (metrics.successfulExecutions / metrics.totalExecutions) * 100;
    metrics.winRate = metrics.successRate;
    
    if (execution.endTime) {
      const executionTime = execution.endTime - execution.startTime;
      metrics.averageExecutionTime = 
        (metrics.averageExecutionTime * (metrics.totalExecutions - 1) + executionTime) / metrics.totalExecutions;
    }

    metrics.totalGasUsed = metrics.totalGasUsed.add(execution.gasUsed);
    metrics.averageGasUsed = metrics.totalGasUsed.div(metrics.totalExecutions);
    
    metrics.netProfit = metrics.totalProfit.sub(metrics.totalLoss);
    
    if (metrics.totalLoss.gt(0)) {
      metrics.profitFactor = parseFloat(ethers.utils.formatEther(
        metrics.totalProfit.mul(100).div(metrics.totalLoss)
      )) / 100;
    }
  }

  // Public methods for external control
  async pauseAgent(): Promise<void> {
    this.state.status = 'paused';
    logger.info('ArbitrageAgent paused', { agentId: this.agentId });
  }

  async resumeAgent(): Promise<void> {
    if (this.state.status === 'paused') {
      this.state.status = 'idle';
      logger.info('ArbitrageAgent resumed', { agentId: this.agentId });
    }
  }

  getState(): AgentState {
    return { ...this.state };
  }

  getActiveExecutions(): AgentExecution[] {
    return Array.from(this.activeExecutions.values());
  }

  async updateConfiguration(newConfig: Partial<ArbitrageAgentConfig>): Promise<void> {
    this.config = { ...this.config, ...newConfig };
    
    // Reinitialize providers if needed
    if (newConfig.supportedChains || newConfig.supportedDexes) {
      await this.providers.dex.updateConfiguration({
        supportedDexes: this.config.supportedDexes,
        supportedChains: this.config.supportedChains
      });
    }

    logger.info('ArbitrageAgent configuration updated', {
      agentId: this.agentId,
      newConfig
    });
  }

  async getHealthStatus(): Promise<{
    status: string;
    healthScore: number;
    issues: string[];
    recommendations: string[];
  }> {
    const issues: string[] = [];
    const recommendations: string[] = [];
    
    // Check error rate
    if (this.state.performance.metrics.successRate < 80) {
      issues.push('Low success rate');
      recommendations.push('Review arbitrage strategies and risk parameters');
    }

    // Check resource usage
    if (this.state.resources.gasAllowanceRemaining.lt(ethers.utils.parseEther('1'))) {
      issues.push('Low gas allowance');
      recommendations.push('Refill gas allowance for continued operation');
    }

    // Check API limits
    if (this.state.resources.apiCallsRemaining < 100) {
      issues.push('Low API call quota');
      recommendations.push('Upgrade API plan or optimize call frequency');
    }

    const healthScore = Math.max(0, 100 - (issues.length * 20));

    return {
      status: this.state.status,
      healthScore,
      issues,
      recommendations
    };
  }
}

export default ArbitrageAgent;
