import { ethers, BigNumber } from 'ethers';
import { logger } from '../../shared/utils/logger';
import { Validator } from '../../shared/utils/validator';
import { getAgentConfig } from '../../config/agentConfig';
import { getElizaConfig, invokeBedrockModel } from '../../config/elizaConfig';
import { renderPromptTemplate } from '../../config/modelConfig';
import { PORTFOLIO_THRESHOLDS, shouldRebalancePortfolio } from '../../shared/constants/thresholds';
import { AgentContext, AgentDecision, AgentExecution, AgentState } from '../../shared/types/agent';
import { Portfolio, PortfolioPosition, AllocationTarget, PortfolioPerformance } from '../../shared/types/market';

// Import actions
import { rebalancePortfolio } from './actions/rebalancePortfolio';
import { optimizeAllocation } from './actions/optimizeAllocation';
import { assessRisk } from './actions/riskAssessment';

// Import evaluators
import { PerformanceEvaluator } from './evaluators/performanceEvaluator';
import { VolatilityEvaluator } from './evaluators/volatilityEvaluator';

// Import providers
import { MarketDataProvider } from './providers/marketDataProvider';
import { PortfolioProvider } from './providers/portfolioProvider';

export interface PortfolioAgentConfig {
  rebalanceThreshold: number;
  maxPositions: number;
  minPositionSize: BigNumber;
  maxPositionSize: BigNumber;
  allowedAssets: string[];
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  rebalanceFrequency: number;
  autoRebalance: boolean;
  benchmarkIndex: string;
  maxDrawdown: number;
  targetReturn: number;
}

export class PortfolioAgent {
  private agentId: string;
  private config: PortfolioAgentConfig;
  private state: AgentState;
  private providers: {
    marketData: MarketDataProvider;
    portfolio: PortfolioProvider;
  };
  private evaluators: {
    performance: PerformanceEvaluator;
    volatility: VolatilityEvaluator;
  };
  private executionInterval: NodeJS.Timeout | null = null;
  private activeExecutions: Map<string, AgentExecution> = new Map();
  private portfolios: Map<string, Portfolio> = new Map();

  constructor(agentId: string, config: PortfolioAgentConfig) {
    this.agentId = agentId;
    this.config = config;

    // Initialize providers
    this.providers = {
      marketData: new MarketDataProvider(),
      portfolio: new PortfolioProvider()
    };

    // Initialize evaluators
    this.evaluators = {
      performance: new PerformanceEvaluator(config.benchmarkIndex),
      volatility: new VolatilityEvaluator(config.riskTolerance)
    };

    // Initialize state
    this.state = this.initializeState();

    logger.agentStarted(this.agentId, 'portfolio', {
      config: this.config
    });
  }

  private initializeState(): AgentState {
    const agentConfig = getAgentConfig('portfolio');
    
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
        semantic: {
          marketCycles: [],
          allocationHistory: [],
          performanceMetrics: {}
        }
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
          totalGasUsed: BigNumber.from(0),
          averageGasUsed: BigNumber.from(0),
          totalProfit: BigNumber.from(0),
          totalLoss: BigNumber.from(0),
          netProfit: BigNumber.from(0),
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
          totalCost: BigNumber.from(0)
        }
      },
      resources: {
        cpuUsage: 0,
        memoryUsage: 0,
        networkLatency: 0,
        apiCallsRemaining: 1000,
        gasAllowanceRemaining: BigNumber.from('5000000000000000000') // 5 ETH
      }
    };
  }

  async start(): Promise<void> {
    if (this.state.status !== 'idle') {
      throw new Error('Portfolio agent is already running');
    }

    this.state.status = 'analyzing';

    // Initialize providers
    await this.providers.marketData.initialize();
    await this.providers.portfolio.initialize();

    // Load existing portfolios
    await this.loadPortfolios();

    // Start execution loop
    const interval = getAgentConfig('portfolio').executionInterval;
    this.executionInterval = setInterval(async () => {
      await this.executionLoop();
    }, interval);

    logger.info('PortfolioAgent started', {
      agentId: this.agentId,
      interval,
      portfolioCount: this.portfolios.size
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

    logger.agentStopped(this.agentId, 'portfolio');
  }

  private async executionLoop(): Promise<void> {
    try {
      this.state.status = 'analyzing';

      // Check each portfolio for rebalancing needs
      for (const [portfolioId, portfolio] of this.portfolios) {
        if (this.activeExecutions.size >= this.config.maxPositions) {
          break;
        }

        // Analyze portfolio performance and risk
        const analysis = await this.analyzePortfolio(portfolio);
        
        if (analysis.needsRebalancing) {
          const decision = await this.makePortfolioDecision(portfolio, analysis);
          
          if (decision.action === 'rebalance' && this.config.autoRebalance) {
            await this.initiateRebalancing(portfolio, decision);
          } else if (decision.action === 'optimize') {
            await this.initiateOptimization(portfolio, decision);
          }
        }

        // Update portfolio performance metrics
        await this.updatePortfolioMetrics(portfolio);
      }

      this.state.status = 'idle';

    } catch (error) {
      this.state.status = 'error';
      this.state.errorCount++;
      
      logger.agentError(this.agentId, 'portfolio', error as Error, {
        executionLoop: true
      });

      if (this.state.errorCount >= 5) {
        this.state.status = 'paused';
        logger.error('Portfolio agent paused due to excessive errors', {
          agentId: this.agentId,
          errorCount: this.state.errorCount
        });
      }
    }
  }

  private async loadPortfolios(): Promise<void> {
    try {
      const portfolios = await this.providers.portfolio.getUserPortfolios();
      
      portfolios.forEach(portfolio => {
        this.portfolios.set(portfolio.id, portfolio);
      });

      logger.info('Portfolios loaded', {
        agentId: this.agentId,
        count: this.portfolios.size
      });

    } catch (error) {
      logger.error('Failed to load portfolios', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async analyzePortfolio(portfolio: Portfolio): Promise<{
    needsRebalancing: boolean;
    deviationPercentage: number;
    riskMetrics: any;
    performanceMetrics: any;
    recommendation: string;
  }> {
    try {
      // Calculate current allocation vs target
      const currentAllocation = this.calculateCurrentAllocation(portfolio);
      const deviationPercentage = this.calculateAllocationDeviation(
        currentAllocation,
        portfolio.allocation
      );

      // Assess risk metrics
      const riskMetrics = await assessRisk(
        portfolio,
        this.providers.marketData,
        { riskTolerance: this.config.riskTolerance }
      );

      // Evaluate performance
      const performanceMetrics = await this.evaluators.performance.evaluatePortfolio(portfolio);

      // Check rebalancing criteria
      const timeSinceLastRebalance = Date.now() - (portfolio.rebalancing?.lastRebalance || 0);
      const needsRebalancing = shouldRebalancePortfolio(
        deviationPercentage,
        timeSinceLastRebalance,
        parseFloat(ethers.utils.formatEther(portfolio.totalValue))
      );

      // Generate AI recommendation
      const recommendation = await this.getAIRecommendation(
        portfolio,
        { deviationPercentage, riskMetrics, performanceMetrics }
      );

      return {
        needsRebalancing,
        deviationPercentage,
        riskMetrics,
        performanceMetrics,
        recommendation
      };

    } catch (error) {
      logger.error('Portfolio analysis failed', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        needsRebalancing: false,
        deviationPercentage: 0,
        riskMetrics: {},
        performanceMetrics: {},
        recommendation: 'Error in analysis'
      };
    }
  }

  private async makePortfolioDecision(
    portfolio: Portfolio,
    analysis: any
  ): Promise<AgentDecision> {
    try {
      // Determine action based on analysis
      let action = 'hold';
      let confidence = 0.5;

      if (analysis.needsRebalancing) {
        if (analysis.deviationPercentage > PORTFOLIO_THRESHOLDS.REBALANCE_THRESHOLD * 2) {
          action = 'rebalance';
          confidence = 0.9;
        } else if (analysis.riskMetrics.overallRisk > 70) {
          action = 'reduce_risk';
          confidence = 0.8;
        } else if (analysis.performanceMetrics.sharpeRatio < 0.5) {
          action = 'optimize';
          confidence = 0.7;
        }
      }

      const decision: AgentDecision = {
        action,
        confidence,
        reasoning: analysis.recommendation,
        parameters: {
          portfolio,
          analysis,
          targetAllocation: portfolio.allocation,
          maxDeviation: this.config.rebalanceThreshold
        },
        riskScore: analysis.riskMetrics.overallRisk || 50,
        expectedOutcome: {
          expectedReturn: analysis.performanceMetrics.expectedReturn || 0,
          riskReduction: action === 'reduce_risk' ? 20 : 0,
          costEstimate: this.estimateTransactionCosts(portfolio)
        },
        alternatives: [
          {
            action: 'hold',
            probability: 0.3,
            outcome: 'Maintain current allocation'
          },
          {
            action: 'gradual_rebalance',
            probability: 0.4,
            outcome: 'Phased rebalancing over time'
          }
        ]
      };

      logger.decisionMade(this.agentId, decision.action, decision.confidence, {
        portfolioId: portfolio.id,
        deviationPercentage: analysis.deviationPercentage,
        riskScore: analysis.riskMetrics.overallRisk
      });

      return decision;

    } catch (error) {
      logger.error('Portfolio decision making failed', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        action: 'hold',
        confidence: 0,
        reasoning: `Decision failed: ${error instanceof Error ? error.message : String(error)}`,
        parameters: { portfolio },
        riskScore: 100,
        expectedOutcome: null,
        alternatives: []
      };
    }
  }

  private async getAIRecommendation(
    portfolio: Portfolio,
    analysis: any
  ): Promise<string> {
    try {
      const elizaConfig = getElizaConfig('portfolio');
      
      const prompt = renderPromptTemplate('portfolio_optimization', {
        currentAllocation: JSON.stringify(portfolio.positions.map(p => ({
          asset: p.token.symbol,
          percentage: p.percentage,
          value: ethers.utils.formatEther(p.value)
        }))),
        marketData: JSON.stringify(await this.getMarketSummary()),
        riskMetrics: JSON.stringify(analysis.riskMetrics),
        riskTolerance: this.config.riskTolerance,
        investmentHorizon: 'long_term',
        targetReturn: this.config.targetReturn.toString(),
        constraints: JSON.stringify({
          maxPositions: this.config.maxPositions,
          allowedAssets: this.config.allowedAssets,
          maxDrawdown: this.config.maxDrawdown
        })
      });

      const response = await invokeBedrockModel({
        modelId: elizaConfig.modelId,
        prompt,
        maxTokens: elizaConfig.maxTokens,
        temperature: elizaConfig.temperature
      });

      const aiResponse = JSON.parse(response);
      return aiResponse.analysis || 'No specific recommendation provided';

    } catch (error) {
      logger.error('AI recommendation failed', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return 'Unable to generate AI recommendation';
    }
  }

  private async initiateRebalancing(
    portfolio: Portfolio,
    decision: AgentDecision
  ): Promise<void> {
    const executionId = `rebalance_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const execution: AgentExecution = {
      id: executionId,
      agentId: this.agentId,
      decision,
      startTime: Date.now(),
      status: 'pending',
      transactions: [],
      gasUsed: BigNumber.from(0),
      actualOutcome: null,
      errors: []
    };

    this.activeExecutions.set(executionId, execution);

    logger.executionStarted(this.agentId, executionId, 'rebalance', {
      portfolioId: portfolio.id
    });

    try {
      execution.status = 'executing';
      
      const result = await rebalancePortfolio(
        portfolio,
        this.providers.portfolio,
        this.providers.marketData,
        {
          maxSlippage: 0.5,
          gasOptimization: true,
          useChainlinkAutomation: true
        }
      );

      execution.status = 'completed';
      execution.endTime = Date.now();
      execution.transactions = result.transactions;
      execution.gasUsed = result.totalGasUsed;
      execution.actualOutcome = result;

      // Update portfolio state
      await this.updatePortfolioAfterRebalance(portfolio, result);

      // Update performance metrics
      this.updatePerformanceMetrics(execution, true);

      logger.executionCompleted(this.agentId, executionId, true, {
        portfolioId: portfolio.id,
        transactionCount: result.transactions.length,
        gasUsed: result.totalGasUsed.toString(),
        duration: execution.endTime - execution.startTime
      });

    } catch (error) {
      execution.status = 'failed';
      execution.endTime = Date.now();
      execution.errors.push(error instanceof Error ? error.message : String(error));

      this.updatePerformanceMetrics(execution, false);

      logger.executionCompleted(this.agentId, executionId, false, {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error),
        duration: execution.endTime! - execution.startTime
      });

    } finally {
      this.activeExecutions.delete(executionId);
    }
  }

  private async initiateOptimization(
    portfolio: Portfolio,
    decision: AgentDecision
  ): Promise<void> {
    const executionId = `optimize_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const execution: AgentExecution = {
      id: executionId,
      agentId: this.agentId,
      decision,
      startTime: Date.now(),
      status: 'pending',
      transactions: [],
      gasUsed: BigNumber.from(0),
      actualOutcome: null,
      errors: []
    };

    this.activeExecutions.set(executionId, execution);

    try {
      execution.status = 'executing';
      
      const result = await optimizeAllocation(
        portfolio,
        this.providers.marketData,
        {
          optimizationObjective: 'sharpe_ratio',
          constraints: {
            maxPositions: this.config.maxPositions,
            minPositionSize: this.config.minPositionSize,
            maxPositionSize: this.config.maxPositionSize
          },
          riskTolerance: this.config.riskTolerance
        }
      );

      execution.status = 'completed';
      execution.endTime = Date.now();
      execution.actualOutcome = result;

      // Store optimization results for future reference
      this.state.memory.semantic.allocationHistory.push({
        timestamp: Date.now(),
        oldAllocation: portfolio.allocation,
        newAllocation: result.optimizedAllocation,
        expectedImprovement: result.expectedImprovement
      });

      logger.executionCompleted(this.agentId, executionId, true, {
        portfolioId: portfolio.id,
        expectedImprovement: result.expectedImprovement,
        duration: execution.endTime - execution.startTime
      });

    } catch (error) {
      execution.status = 'failed';
      execution.endTime = Date.now();
      execution.errors.push(error instanceof Error ? error.message : String(error));

      logger.executionCompleted(this.agentId, executionId, false, {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error),
        duration: execution.endTime! - execution.startTime
      });

    } finally {
      this.activeExecutions.delete(executionId);
    }
  }

  private calculateCurrentAllocation(portfolio: Portfolio): Array<{ asset: string; percentage: number }> {
    return portfolio.positions.map(position => ({
      asset: position.token.symbol,
      percentage: position.percentage
    }));
  }

  private calculateAllocationDeviation(
    current: Array<{ asset: string; percentage: number }>,
    target: AllocationTarget[]
  ): number {
    let totalDeviation = 0;
    
    target.forEach(targetAllocation => {
      const currentAllocation = current.find(c => c.asset === targetAllocation.token.symbol);
      const currentPercentage = currentAllocation?.percentage || 0;
      const deviation = Math.abs(currentPercentage - targetAllocation.targetPercentage);
      totalDeviation += deviation;
    });

    return totalDeviation / target.length;
  }

  private async getMarketSummary(): Promise<any> {
    try {
      return await this.providers.marketData.getMarketSummary();
    } catch (error) {
      return { status: 'unavailable', error: error instanceof Error ? error.message : String(error) };
    }
  }

  private estimateTransactionCosts(portfolio: Portfolio): BigNumber {
    // Estimate gas costs for portfolio operations
    const baseGasPerTransaction = 150000;
    const estimatedTransactions = portfolio.positions.length * 2; // Buy and sell
    const totalGas = baseGasPerTransaction * estimatedTransactions;
    
    // Use average gas price across supported chains
    const averageGasPrice = BigNumber.from('20000000000'); // 20 Gwei
    
    return averageGasPrice.mul(totalGas);
  }

  private async updatePortfolioAfterRebalance(portfolio: Portfolio, rebalanceResult: any): Promise<void> {
    try {
      // Update portfolio positions based on rebalance results
      await this.providers.portfolio.updatePortfolioPositions(
        portfolio.id,
        rebalanceResult.newPositions
      );

      // Refresh portfolio data
      const updatedPortfolio = await this.providers.portfolio.getPortfolio(portfolio.id);
      if (updatedPortfolio) {
        this.portfolios.set(portfolio.id, updatedPortfolio);
      }

    } catch (error) {
      logger.error('Failed to update portfolio after rebalance', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async updatePortfolioMetrics(portfolio: Portfolio): Promise<void> {
    try {
      const performance = await this.evaluators.performance.evaluatePortfolio(portfolio);
      const volatility = await this.evaluators.volatility.calculateVolatility(portfolio);

      // Store updated metrics
      this.state.memory.semantic.performanceMetrics[portfolio.id] = {
        timestamp: Date.now(),
        performance,
        volatility,
        totalValue: parseFloat(ethers.utils.formatEther(portfolio.totalValue))
      };

    } catch (error) {
      logger.error('Failed to update portfolio metrics', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private updatePerformanceMetrics(execution: AgentExecution, success: boolean): void {
    const metrics = this.state.performance.metrics;
    
    metrics.totalExecutions++;
    
    if (success) {
      metrics.successfulExecutions++;
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
  }

  // Public methods for external control
  async addPortfolio(portfolio: Portfolio): Promise<void> {
    this.portfolios.set(portfolio.id, portfolio);
    
    logger.info('Portfolio added to agent', {
      agentId: this.agentId,
      portfolioId: portfolio.id,
      totalValue: ethers.utils.formatEther(portfolio.totalValue)
    });
  }

  async removePortfolio(portfolioId: string): Promise<void> {
    this.portfolios.delete(portfolioId);
    
    logger.info('Portfolio removed from agent', {
      agentId: this.agentId,
      portfolioId
    });
  }

  getPortfolios(): Portfolio[] {
    return Array.from(this.portfolios.values());
  }

  getState(): AgentState {
    return { ...this.state };
  }

  getActiveExecutions(): AgentExecution[] {
    return Array.from(this.activeExecutions.values());
  }

  async updateConfiguration(newConfig: Partial<PortfolioAgentConfig>): Promise<void> {
    this.config = { ...this.config, ...newConfig };
    
    logger.info('PortfolioAgent configuration updated', {
      agentId: this.agentId,
      newConfig
    });
  }

  async pauseAgent(): Promise<void> {
    this.state.status = 'paused';
    logger.info('PortfolioAgent paused', { agentId: this.agentId });
  }

  async resumeAgent(): Promise<void> {
    if (this.state.status === 'paused') {
      this.state.status = 'idle';
      logger.info('PortfolioAgent resumed', { agentId: this.agentId });
    }
  }

  async getHealthStatus(): Promise<{
    status: string;
    healthScore: number;
    issues: string[];
    recommendations: string[];
  }> {
    const issues: string[] = [];
    const recommendations: string[] = [];
    
    // Check success rate
    if (this.state.performance.metrics.successRate < 80) {
      issues.push('Low success rate for portfolio operations');
      recommendations.push('Review portfolio strategies and market conditions');
    }

    // Check portfolio performance
    for (const [portfolioId, portfolio] of this.portfolios) {
      const metrics = this.state.memory.semantic.performanceMetrics[portfolioId];
      if (metrics && metrics.performance.totalReturnPercentage < -this.config.maxDrawdown) {
        issues.push(`Portfolio ${portfolioId} exceeds maximum drawdown`);
        recommendations.push('Consider defensive rebalancing or risk reduction');
      }
    }

    // Check resource usage
    if (this.state.resources.gasAllowanceRemaining.lt(ethers.utils.parseEther('1'))) {
      issues.push('Low gas allowance');
      recommendations.push('Refill gas allowance for continued operation');
    }

    const healthScore = Math.max(0, 100 - (issues.length * 15));

    return {
      status: this.state.status,
      healthScore,
      issues,
      recommendations
    };
  }
}

export default PortfolioAgent;
