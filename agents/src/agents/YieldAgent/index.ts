import { ethers, BigNumber } from 'ethers';
import { logger } from '../../shared/utils/logger';
import { getAgentConfig } from '../../config/agentConfig';
import { getElizaConfig, invokeBedrockModel } from '../../config/elizaConfig';
import { renderPromptTemplate } from '../../config/modelConfig';
import { YIELD_THRESHOLDS, isYieldOpportunityAttractive } from '../../shared/constants/thresholds';
import { AgentContext, AgentDecision, AgentExecution, AgentState } from '../../shared/types/agent';
import { YieldOpportunity, YieldStrategy } from '../../shared/types/market';

// Import actions
import { findYieldOpportunities } from './actions/findYieldOpportunities';
import { migrateAssets } from './actions/migrateAssets';
import { optimizeYield } from './actions/optimizeYield';

// Import evaluators
import { YieldEvaluator } from './evaluators/yieldEvaluator';

// Import providers
import { ProtocolProvider } from './providers/protocolProvider';

export interface YieldAgentConfig {
  minAPY: number;
  maxRiskScore: number;
  supportedProtocols: string[];
  supportedChains: number[];
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  autoCompound: boolean;
  autoMigrate: boolean;
  maxPositionSize: BigNumber;
  diversificationMinimum: number;
  harvestFrequency: number; // in hours
}

export class YieldAgent {
  private agentId: string;
  private config: YieldAgentConfig;
  private state: AgentState;
  private providers: {
    protocol: ProtocolProvider;
  };
  private evaluators: {
    yield: YieldEvaluator;
  };
  private executionInterval: NodeJS.Timeout | null = null;
  private activeExecutions: Map<string, AgentExecution> = new Map();
  private yieldPositions: Map<string, YieldOpportunity[]> = new Map();

  constructor(agentId: string, config: YieldAgentConfig) {
    this.agentId = agentId;
    this.config = config;

    // Initialize providers
    this.providers = {
      protocol: new ProtocolProvider(config.supportedProtocols, config.supportedChains)
    };

    // Initialize evaluators
    this.evaluators = {
      yield: new YieldEvaluator(config.riskTolerance)
    };

    // Initialize state
    this.state = this.initializeState();

    logger.agentStarted(this.agentId, 'yield', {
      config: this.config
    });
  }

  private initializeState(): AgentState {
    const agentConfig = getAgentConfig('yield');
    
    return {
      agentId: this.agentId,
      status: 'idle',
      healthScore: 100,
      errorCount: 0,
      warningCount: 0,
      memory: {
        shortTerm: {},
        longTerm: {
          harvestHistory: [],
          migrationHistory: [],
          protocolPerformance: {}
        },
        episodic: [],
        semantic: {
          yieldStrategies: [],
          riskProfiles: {},
          protocolAnalysis: {}
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
        gasAllowanceRemaining: BigNumber.from('3000000000000000000') // 3 ETH
      }
    };
  }

  async start(): Promise<void> {
    if (this.state.status !== 'idle') {
      throw new Error('Yield agent is already running');
    }

    this.state.status = 'analyzing';

    // Initialize providers
    await this.providers.protocol.initialize();

    // Load existing yield positions
    await this.loadYieldPositions();

    // Start execution loop
    const interval = getAgentConfig('yield').executionInterval;
    this.executionInterval = setInterval(async () => {
      await this.executionLoop();
    }, interval);

    logger.info('YieldAgent started', {
      agentId: this.agentId,
      interval,
      positionsCount: this.getTotalPositionsCount()
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

    logger.agentStopped(this.agentId, 'yield');
  }

  private async executionLoop(): Promise<void> {
    try {
      this.state.status = 'analyzing';

      // 1. Check for harvest opportunities
      await this.checkHarvestOpportunities();

      // 2. Find new yield opportunities
      if (this.shouldSearchForNewOpportunities()) {
        await this.findNewYieldOpportunities();
      }

      // 3. Check for migration opportunities
      if (this.config.autoMigrate) {
        await this.checkMigrationOpportunities();
      }

      // 4. Optimize existing positions
      await this.optimizeExistingPositions();

      this.state.status = 'idle';

    } catch (error) {
      this.state.status = 'error';
      this.state.errorCount++;
      
      logger.agentError(this.agentId, 'yield', error as Error, {
        executionLoop: true
      });

      if (this.state.errorCount >= 5) {
        this.state.status = 'paused';
        logger.error('Yield agent paused due to excessive errors', {
          agentId: this.agentId,
          errorCount: this.state.errorCount
        });
      }
    }
  }

  private async loadYieldPositions(): Promise<void> {
    try {
      // Load existing yield positions for all supported protocols
      for (const protocol of this.config.supportedProtocols) {
        const positions = await this.providers.protocol.getUserPositions(protocol);
        if (positions.length > 0) {
          this.yieldPositions.set(protocol, positions);
        }
      }

      logger.info('Yield positions loaded', {
        agentId: this.agentId,
        protocolCount: this.yieldPositions.size,
        totalPositions: this.getTotalPositionsCount()
      });

    } catch (error) {
      logger.error('Failed to load yield positions', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async checkHarvestOpportunities(): Promise<void> {
    try {
      const harvestablePositions = [];

      for (const [protocol, positions] of this.yieldPositions) {
        for (const position of positions) {
          const shouldHarvest = await this.shouldHarvestPosition(position);
          if (shouldHarvest) {
            harvestablePositions.push({ protocol, position });
          }
        }
      }

      if (harvestablePositions.length > 0) {
        logger.info('Found harvestable positions', {
          count: harvestablePositions.length,
          positions: harvestablePositions.map(h => `${h.protocol}:${h.position.id}`)
        });

        for (const { protocol, position } of harvestablePositions) {
          await this.initiateHarvest(protocol, position);
        }
      }

    } catch (error) {
      logger.error('Failed to check harvest opportunities', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async findNewYieldOpportunities(): Promise<void> {
    try {
      const context: AgentContext = {
        agentId: this.agentId,
        agentType: 'yield',
        userId: 'system',
        sessionId: `session_${Date.now()}`,
        networkIds: this.config.supportedChains,
        timestamp: Date.now(),
        gasPrice: BigNumber.from('20000000000'), // 20 gwei
        nonce: 0
      };

      const opportunities = await findYieldOpportunities(
        this.providers.protocol,
        context,
        {
          supportedProtocols: this.config.supportedProtocols,
          supportedChains: this.config.supportedChains,
          minAPY: this.config.minAPY,
          maxRiskScore: this.config.maxRiskScore,
          maxPositionSize: this.config.maxPositionSize
        }
      );

      if (opportunities.length > 0) {
        logger.info('Found new yield opportunities', {
          count: opportunities.length,
          topAPY: Math.max(...opportunities.map(o => o.apy))
        });

        for (const opportunity of opportunities) {
          const decision = await this.analyzeYieldOpportunity(opportunity);
          
          if (decision.action === 'invest') {
            await this.initiateInvestment(opportunity, decision);
          }
        }
      }

    } catch (error) {
      logger.error('Failed to find new yield opportunities', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async checkMigrationOpportunities(): Promise<void> {
    try {
      for (const [currentProtocol, positions] of this.yieldPositions) {
        for (const position of positions) {
          const migrationAnalysis = await this.analyzeMigrationOpportunity(position, currentProtocol);
          
          if (migrationAnalysis.shouldMigrate) {
            await this.initiateMigration(position, currentProtocol, migrationAnalysis.targetProtocol);
          }
        }
      }

    } catch (error) {
      logger.error('Failed to check migration opportunities', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async optimizeExistingPositions(): Promise<void> {
    try {
      const context: AgentContext = {
        agentId: this.agentId,
        agentType: 'yield',
        userId: 'system',
        sessionId: `session_${Date.now()}`,
        networkIds: this.config.supportedChains,
        timestamp: Date.now(),
        gasPrice: BigNumber.from('20000000000'),
        nonce: 0
      };

      const allPositions = Array.from(this.yieldPositions.values()).flat();
      
      if (allPositions.length > 0) {
        const optimizationResult = await optimizeYield(
          allPositions,
          this.providers.protocol,
          context,
          {
            riskTolerance: this.config.riskTolerance,
            diversificationTarget: this.config.diversificationMinimum,
            maxPositionSize: this.config.maxPositionSize
          }
        );

        if (optimizationResult.optimizedPositions.length > 0) {
          await this.implementOptimization(optimizationResult);
        }
      }

    } catch (error) {
      logger.error('Failed to optimize existing positions', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async shouldHarvestPosition(position: YieldOpportunity): Promise<boolean> {
    try {
      const rewards = await this.providers.protocol.getClaimableRewards(position.protocol, position.id);
      
      if (rewards.length === 0) return false;

      // Calculate total reward value in USD
      const totalRewardValue = rewards.reduce((sum, reward) => {
        const rewardValueUsd = parseFloat(ethers.utils.formatEther(reward.amount)) * reward.priceUsd;
        return sum + rewardValueUsd;
      }, 0);

      // Check if rewards exceed minimum threshold
      const minHarvestValue = 50; // $50 minimum
      if (totalRewardValue < minHarvestValue) return false;

      // Check time since last harvest
      const timeSinceLastHarvest = Date.now() - (position.lastHarvest || 0);
      const harvestInterval = this.config.harvestFrequency * 60 * 60 * 1000; // Convert hours to ms

      return timeSinceLastHarvest >= harvestInterval;

    } catch (error) {
      logger.error('Failed to check if position should be harvested', {
        positionId: position.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return false;
    }
  }

  private async analyzeYieldOpportunity(opportunity: YieldOpportunity): Promise<AgentDecision> {
    try {
      // Evaluate yield opportunity using the evaluator
      const evaluation = await this.evaluators.yield.evaluateOpportunity(opportunity);

      // Get AI decision
      const aiDecision = await this.getAIDecision(opportunity, evaluation);

      // Check if opportunity meets criteria
      const meetsThresholds = isYieldOpportunityAttractive(
        opportunity.apy,
        evaluation.riskScore,
        evaluation.auditScore
      );

      const decision: AgentDecision = {
        action: meetsThresholds && aiDecision.recommendation === 'invest' ? 'invest' : 'reject',
        confidence: aiDecision.confidence,
        reasoning: aiDecision.reasoning,
        parameters: {
          opportunity,
          evaluation,
          aiAnalysis: aiDecision
        },
        riskScore: evaluation.riskScore,
        expectedOutcome: {
          expectedAPY: opportunity.apy,
          riskAdjustedReturn: opportunity.apy * (1 - evaluation.riskScore / 100),
          investmentAmount: this.calculateOptimalInvestmentAmount(opportunity, evaluation)
        },
        alternatives: [
          {
            action: 'monitor',
            probability: 0.3,
            outcome: 'Continue monitoring for better opportunities'
          },
          {
            action: 'reject',
            probability: 0.2,
            outcome: 'Skip due to high risk or low yield'
          }
        ]
      };

      logger.decisionMade(this.agentId, decision.action, decision.confidence, {
        opportunityId: opportunity.id,
        protocol: opportunity.protocol,
        apy: opportunity.apy,
        riskScore: evaluation.riskScore
      });

      return decision;

    } catch (error) {
      logger.error('Failed to analyze yield opportunity', {
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
    opportunity: YieldOpportunity,
    evaluation: any
  ): Promise<{
    recommendation: 'invest' | 'reject' | 'monitor';
    confidence: number;
    reasoning: string;
  }> {
    try {
      const elizaConfig = getElizaConfig('yield');
      
      const prompt = renderPromptTemplate('yield_analysis', {
        protocol: opportunity.protocol,
        apy: opportunity.apy.toFixed(2),
        tvl: ethers.utils.formatEther(opportunity.tvl),
        riskScore: evaluation.riskScore.toFixed(1),
        auditScore: evaluation.auditScore?.toFixed(1) || 'N/A',
        liquidityScore: evaluation.liquidityScore.toFixed(1),
        tokenPair: opportunity.tokenPair,
        chainId: opportunity.chainId,
        riskTolerance: this.config.riskTolerance
      });

      const response = await invokeBedrockModel({
        modelId: elizaConfig.modelId,
        prompt,
        maxTokens: elizaConfig.maxTokens,
        temperature: elizaConfig.temperature
      });

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
      if (opportunity.apy >= this.config.minAPY && evaluation.riskScore <= this.config.maxRiskScore) {
        return {
          recommendation: 'invest',
          confidence: 0.7,
          reasoning: 'Fallback decision: APY above threshold and acceptable risk'
        };
      }

      return {
        recommendation: 'reject',
        confidence: 0.8,
        reasoning: 'Fallback decision: Does not meet minimum criteria'
      };
    }
  }

  private calculateOptimalInvestmentAmount(opportunity: YieldOpportunity, evaluation: any): BigNumber {
    // Start with maximum position size
    let amount = this.config.maxPositionSize;

    // Adjust based on risk score
    const riskAdjustment = (100 - evaluation.riskScore) / 100;
    amount = amount.mul(Math.floor(riskAdjustment * 100)).div(100);

    // Adjust based on TVL (higher TVL = larger position allowed)
    const tvlUsd = parseFloat(ethers.utils.formatEther(opportunity.tvl));
    const tvlAdjustment = Math.min(tvlUsd / 10000000, 1); // Max adjustment at $10M TVL
    amount = amount.mul(Math.floor(tvlAdjustment * 100)).div(100);

    // Ensure minimum investment
    const minInvestment = ethers.utils.parseEther('100'); // $100 minimum
    if (amount.lt(minInvestment)) {
      amount = minInvestment;
    }

    return amount;
  }

  private async initiateInvestment(opportunity: YieldOpportunity, decision: AgentDecision): Promise<void> {
    const executionId = `invest_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
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

    logger.executionStarted(this.agentId, executionId, 'invest', {
      opportunityId: opportunity.id,
      protocol: opportunity.protocol,
      expectedAPY: opportunity.apy
    });

    try {
      execution.status = 'executing';
      
      const investmentAmount = decision.expectedOutcome?.investmentAmount || this.config.maxPositionSize.div(10);
      
      const result = await this.providers.protocol.depositToProtocol(
        opportunity.protocol,
        opportunity.id,
        investmentAmount,
        opportunity.chainId
      );

      execution.status = 'completed';
      execution.endTime = Date.now();
      execution.transactions = [result.transactionHash];
      execution.gasUsed = result.gasUsed;
      execution.actualOutcome = result;

      // Update positions
      await this.updateYieldPositions(opportunity.protocol);

      // Update performance metrics
      this.updatePerformanceMetrics(execution, true);

      logger.executionCompleted(this.agentId, executionId, true, {
        opportunityId: opportunity.id,
        investmentAmount: ethers.utils.formatEther(investmentAmount),
        transactionHash: result.transactionHash,
        duration: execution.endTime - execution.startTime
      });

    } catch (error) {
      execution.status = 'failed';
      execution.endTime = Date.now();
      execution.errors.push(error instanceof Error ? error.message : String(error));

      this.updatePerformanceMetrics(execution, false);

      logger.executionCompleted(this.agentId, executionId, false, {
        opportunityId: opportunity.id,
        error: error instanceof Error ? error.message : String(error),
        duration: execution.endTime! - execution.startTime
      });

    } finally {
      this.activeExecutions.delete(executionId);
    }
  }

  private async initiateHarvest(protocol: string, position: YieldOpportunity): Promise<void> {
    const executionId = `harvest_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const execution: AgentExecution = {
      id: executionId,
      agentId: this.agentId,
      decision: {
        action: 'harvest',
        confidence: 1.0,
        reasoning: 'Automated harvest based on threshold criteria',
        parameters: { protocol, position },
        riskScore: 10,
        expectedOutcome: null,
        alternatives: []
      },
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
      
      const result = await this.providers.protocol.harvestRewards(protocol, position.id, position.chainId);

      execution.status = 'completed';
      execution.endTime = Date.now();
      execution.transactions = [result.transactionHash];
      execution.gasUsed = result.gasUsed;
      execution.actualOutcome = result;

      // Update harvest history
      this.state.memory.longTerm.harvestHistory.push({
        timestamp: Date.now(),
        protocol,
        positionId: position.id,
        rewardsHarvested: result.rewardsHarvested,
        gasUsed: result.gasUsed
      });

      this.updatePerformanceMetrics(execution, true);

      logger.executionCompleted(this.agentId, executionId, true, {
        protocol,
        positionId: position.id,
        rewardsValue: result.totalValueUsd,
        duration: execution.endTime - execution.startTime
      });

    } catch (error) {
      execution.status = 'failed';
      execution.endTime = Date.now();
      execution.errors.push(error instanceof Error ? error.message : String(error));

      this.updatePerformanceMetrics(execution, false);

      logger.executionCompleted(this.agentId, executionId, false, {
        protocol,
        positionId: position.id,
        error: error instanceof Error ? error.message : String(error),
        duration: execution.endTime! - execution.startTime
      });

    } finally {
      this.activeExecutions.delete(executionId);
    }
  }

  private async analyzeMigrationOpportunity(
    position: YieldOpportunity,
    currentProtocol: string
  ): Promise<{
    shouldMigrate: boolean;
    targetProtocol: string | null;
    expectedImprovement: number;
  }> {
    try {
      const context: AgentContext = {
        agentId: this.agentId,
        agentType: 'yield',
        userId: 'system',
        sessionId: `session_${Date.now()}`,
        networkIds: [position.chainId],
        timestamp: Date.now(),
        gasPrice: BigNumber.from('20000000000'),
        nonce: 0
      };

      const migrationAnalysis = await migrateAssets(
        [position],
        this.providers.protocol,
        context,
        {
          targetProtocols: this.config.supportedProtocols.filter(p => p !== currentProtocol),
          minImprovementThreshold: 1.0, // 1% minimum improvement
          includeGasCosts: true
        }
      );

      if (migrationAnalysis.recommendedMigrations.length > 0) {
        const bestMigration = migrationAnalysis.recommendedMigrations[0];
        
        return {
          shouldMigrate: bestMigration.expectedImprovement > 1.0,
          targetProtocol: bestMigration.targetProtocol,
          expectedImprovement: bestMigration.expectedImprovement
        };
      }

      return {
        shouldMigrate: false,
        targetProtocol: null,
        expectedImprovement: 0
      };

    } catch (error) {
      logger.error('Failed to analyze migration opportunity', {
        positionId: position.id,
        currentProtocol,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        shouldMigrate: false,
        targetProtocol: null,
        expectedImprovement: 0
      };
    }
  }

  private async initiateMigration(
    position: YieldOpportunity,
    fromProtocol: string,
    toProtocol: string
  ): Promise<void> {
    const executionId = `migrate_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    logger.info('Initiating asset migration', {
      executionId,
      positionId: position.id,
      fromProtocol,
      toProtocol
    });

    // Implementation would handle the migration process
    // This involves withdrawing from source protocol and depositing to target protocol
  }

  private async implementOptimization(optimizationResult: any): Promise<void> {
    logger.info('Implementing yield optimization', {
      agentId: this.agentId,
      optimizedPositions: optimizationResult.optimizedPositions.length,
      expectedImprovement: optimizationResult.expectedImprovement
    });

    // Implementation would apply the optimization recommendations
  }

  private async updateYieldPositions(protocol: string): Promise<void> {
    try {
      const positions = await this.providers.protocol.getUserPositions(protocol);
      this.yieldPositions.set(protocol, positions);

    } catch (error) {
      logger.error('Failed to update yield positions', {
        protocol,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private shouldSearchForNewOpportunities(): boolean {
    // Search for new opportunities every 6 hours or when positions are below target
    const lastSearch = this.state.memory.shortTerm.lastOpportunitySearch || 0;
    const searchInterval = 6 * 60 * 60 * 1000; // 6 hours
    
    const timeSinceLastSearch = Date.now() - lastSearch;
    const hasLowPositionCount = this.getTotalPositionsCount() < this.config.diversificationMinimum;
    
    return timeSinceLastSearch >= searchInterval || hasLowPositionCount;
  }

  private getTotalPositionsCount(): number {
    return Array.from(this.yieldPositions.values()).reduce((sum, positions) => sum + positions.length, 0);
  }

  private updatePerformanceMetrics(execution: AgentExecution, success: boolean): void {
    const metrics = this.state.performance.metrics;
    
    metrics.totalExecutions++;
    
    if (success) {
      metrics.successfulExecutions++;
      if (execution.actualOutcome?.totalValueUsd) {
        const profit = ethers.utils.parseEther(execution.actualOutcome.totalValueUsd.toString());
        metrics.totalProfit = metrics.totalProfit.add(profit);
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
    logger.info('YieldAgent paused', { agentId: this.agentId });
  }

  async resumeAgent(): Promise<void> {
    if (this.state.status === 'paused') {
      this.state.status = 'idle';
      logger.info('YieldAgent resumed', { agentId: this.agentId });
    }
  }

  getState(): AgentState {
    return { ...this.state };
  }

  getActiveExecutions(): AgentExecution[] {
    return Array.from(this.activeExecutions.values());
  }

  getYieldPositions(): Map<string, YieldOpportunity[]> {
    return new Map(this.yieldPositions);
  }

  async updateConfiguration(newConfig: Partial<YieldAgentConfig>): Promise<void> {
    this.config = { ...this.config, ...newConfig };
    
    logger.info('YieldAgent configuration updated', {
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
    
    // Check success rate
    if (this.state.performance.metrics.successRate < 80) {
      issues.push('Low success rate for yield operations');
      recommendations.push('Review yield strategies and protocol selection');
    }

    // Check position diversification
    const totalPositions = this.getTotalPositionsCount();
    if (totalPositions < this.config.diversificationMinimum) {
      issues.push('Insufficient position diversification');
      recommendations.push('Increase number of yield positions across protocols');
    }

    // Check gas allowance
    if (this.state.resources.gasAllowanceRemaining.lt(ethers.utils.parseEther('0.5'))) {
      issues.push('Low gas allowance');
      recommendations.push('Refill gas allowance for continued operation');
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

export default YieldAgent;
