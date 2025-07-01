import { ethers, BigNumber } from 'ethers';
import { logger } from '../../shared/utils/logger';
import { getAgentConfig } from '../../config/agentConfig';
import { getElizaConfig, invokeBedrockModel } from '../../config/elizaConfig';
import { renderPromptTemplate } from '../../config/modelConfig';
import { AgentContext, AgentDecision, AgentExecution, AgentState } from '../../shared/types/agent';

// Import actions
import { coordinateAgents } from './actions/coordinateAgents';
import { manageResources } from './actions/manageResources';
import { handleConflicts } from './actions/handleConflicts';

// Import evaluators
import { SystemEvaluator } from './evaluators/systemEvaluator';

// Import providers
import { OrchestrationProvider } from './providers/orchestrationProvider';

// Import other agents
import ArbitrageAgent from '../ArbitrageAgent';
import PortfolioAgent from '../PortfolioAgent';
import YieldAgent from '../YieldAgent';
import SecurityAgent from '../SecurityAgent';

export interface OrchestratorConfig {
  maxConcurrentAgents: number;
  resourceAllocationStrategy: 'balanced' | 'priority_based' | 'performance_based';
  conflictResolutionStrategy: 'consensus' | 'priority' | 'ai_mediated';
  systemMonitoringInterval: number;
  autoScaling: boolean;
  emergencyProtocols: {
    enabled: boolean;
    triggers: string[];
    actions: string[];
  };
  coordinationMode: 'centralized' | 'distributed' | 'hybrid';
  performanceThresholds: {
    responseTime: number;
    errorRate: number;
    resourceUtilization: number;
  };
}

export interface AgentRegistry {
  arbitrage: ArbitrageAgent | null;
  portfolio: PortfolioAgent | null;
  yield: YieldAgent | null;
  security: SecurityAgent | null;
}

export interface SystemMetrics {
  totalExecutions: number;
  activeAgents: number;
  averageResponseTime: number;
  systemLoad: number;
  errorRate: number;
  resourceUtilization: {
    cpu: number;
    memory: number;
    network: number;
    gas: BigNumber;
  };
  agentPerformance: Record<string, {
    executionCount: number;
    successRate: number;
    averageExecutionTime: number;
    lastExecution: number;
  }>;
}

export interface CoordinationDecision {
  primaryAgent: string;
  supportingAgents: string[];
  executionOrder: string[];
  resourceAllocation: Record<string, number>;
  expectedDuration: number;
  riskAssessment: {
    overall: number;
    conflicts: string[];
    mitigations: string[];
  };
  reasoning: string;
}

export class OrchestratorAgent {
  private agentId: string;
  private config: OrchestratorConfig;
  private state: AgentState;
  private agentRegistry: AgentRegistry;
  private providers: {
    orchestration: OrchestrationProvider;
  };
  private evaluators: {
    system: SystemEvaluator;
  };
  private systemMonitoringInterval: NodeJS.Timeout | null = null;
  private activeExecutions: Map<string, AgentExecution> = new Map();
  private systemMetrics: SystemMetrics;
  private coordinationQueue: Array<{
    id: string;
    request: any;
    priority: number;
    timestamp: number;
    requester: string;
  }> = [];

  constructor(agentId: string, config: OrchestratorConfig) {
    this.agentId = agentId;
    this.config = config;

    // Initialize providers
    this.providers = {
      orchestration: new OrchestrationProvider()
    };

    // Initialize evaluators
    this.evaluators = {
      system: new SystemEvaluator()
    };

    // Initialize agent registry
    this.agentRegistry = {
      arbitrage: null,
      portfolio: null,
      yield: null,
      security: null
    };

    // Initialize state
    this.state = this.initializeState();

    // Initialize system metrics
    this.systemMetrics = this.initializeSystemMetrics();

    logger.agentStarted(this.agentId, 'orchestrator', {
      config: this.config
    });
  }

  private initializeState(): AgentState {
    const agentConfig = getAgentConfig('orchestrator');
    
    return {
      agentId: this.agentId,
      status: 'idle',
      healthScore: 100,
      errorCount: 0,
      warningCount: 0,
      memory: {
        shortTerm: {},
        longTerm: {
          coordinationHistory: [],
          systemPatterns: [],
          performanceBaselines: {}
        },
        episodic: [],
        semantic: {
          agentCapabilities: {},
          systemTopology: {},
          decisionPatterns: {}
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
        apiCallsRemaining: 10000,
        gasAllowanceRemaining: BigNumber.from('10000000000000000000') // 10 ETH
      }
    };
  }

  private initializeSystemMetrics(): SystemMetrics {
    return {
      totalExecutions: 0,
      activeAgents: 0,
      averageResponseTime: 0,
      systemLoad: 0,
      errorRate: 0,
      resourceUtilization: {
        cpu: 0,
        memory: 0,
        network: 0,
        gas: BigNumber.from(0)
      },
      agentPerformance: {}
    };
  }

  async start(): Promise<void> {
    if (this.state.status !== 'idle') {
      throw new Error('Orchestrator agent is already running');
    }

    this.state.status = 'initializing';

    // Initialize orchestration provider
    await this.providers.orchestration.initialize();

    // Start specialized agents
    await this.initializeSpecializedAgents();

    // Start system monitoring
    if (this.config.systemMonitoringInterval > 0) {
      this.systemMonitoringInterval = setInterval(async () => {
        await this.monitorSystem();
      }, this.config.systemMonitoringInterval);
    }

    // Start coordination processing
    this.startCoordinationProcessing();

    this.state.status = 'active';

    logger.info('OrchestratorAgent started', {
      agentId: this.agentId,
      config: this.config,
      registeredAgents: Object.keys(this.agentRegistry).filter(
        key => this.agentRegistry[key as keyof AgentRegistry] !== null
      )
    });
  }

  async stop(): Promise<void> {
    this.state.status = 'stopping';

    // Stop system monitoring
    if (this.systemMonitoringInterval) {
      clearInterval(this.systemMonitoringInterval);
      this.systemMonitoringInterval = null;
    }

    // Stop all specialized agents
    await this.stopSpecializedAgents();

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

    this.state.status = 'stopped';

    logger.agentStopped(this.agentId, 'orchestrator');
  }

  private async initializeSpecializedAgents(): Promise<void> {
    try {
      // Initialize ArbitrageAgent
      const arbitrageConfig = getAgentConfig('arbitrage');
      this.agentRegistry.arbitrage = new ArbitrageAgent(
        `${this.agentId}_arbitrage`,
        {
          supportedDEXs: arbitrageConfig.supportedDEXs || ['uniswap', 'sushiswap'],
          supportedChains: arbitrageConfig.supportedChains || [1, 137, 42161],
          minProfitThreshold: arbitrageConfig.minProfitThreshold || 0.5,
          maxPositionSize: ethers.utils.parseEther('10'),
          slippageTolerance: arbitrageConfig.slippageTolerance || 1.0,
          gasOptimization: true,
          flashLoanProviders: ['aave', 'compound'],
          realTimeExecution: true
        }
      );

      // Initialize PortfolioAgent
      const portfolioConfig = getAgentConfig('portfolio');
      this.agentRegistry.portfolio = new PortfolioAgent(
        `${this.agentId}_portfolio`,
        {
          rebalancingStrategy: portfolioConfig.rebalancingStrategy || 'dynamic',
          riskTolerance: portfolioConfig.riskTolerance || 'moderate',
          diversificationTarget: portfolioConfig.diversificationTarget || 5,
          rebalanceThreshold: portfolioConfig.rebalanceThreshold || 5.0,
          maxAllocationPerAsset: portfolioConfig.maxAllocationPerAsset || 25.0,
          supportedAssets: portfolioConfig.supportedAssets || ['ETH', 'BTC', 'stablecoins'],
          supportedChains: portfolioConfig.supportedChains || [1, 137, 42161],
          autoRebalancing: true,
          performanceTracking: true
        }
      );

      // Initialize YieldAgent
      const yieldConfig = getAgentConfig('yield');
      this.agentRegistry.yield = new YieldAgent(
        `${this.agentId}_yield`,
        {
          supportedProtocols: yieldConfig.supportedProtocols || ['aave', 'compound', 'yearn'],
          supportedChains: yieldConfig.supportedChains || [1, 137, 42161],
          minAPY: yieldConfig.minAPY || 2.0,
          maxRiskScore: yieldConfig.maxRiskScore || 70,
          autoCompoundPreference: true,
          yieldOptimization: true,
          riskManagement: true
        }
      );

      // Initialize SecurityAgent
      const securityConfig = getAgentConfig('security');
      this.agentRegistry.security = new SecurityAgent(
        `${this.agentId}_security`,
        {
          monitoringEnabled: true,
          alertThresholds: securityConfig.alertThresholds || {
            suspiciousTransaction: 70,
            volumeAnomaly: 3.0,
            priceManipulation: 20,
            rugPull: 80,
            phishing: 90
          },
          monitoredAddresses: securityConfig.monitoredAddresses || [],
          monitoredContracts: securityConfig.monitoredContracts || [],
          supportedChains: securityConfig.supportedChains || [1, 137, 42161],
          realTimeMonitoring: true,
          emergencyResponse: true,
          notificationChannels: ['webhook', 'email']
        }
      );

      // Start all agents
      await Promise.all([
        this.agentRegistry.arbitrage?.start(),
        this.agentRegistry.portfolio?.start(),
        this.agentRegistry.yield?.start(),
        this.agentRegistry.security?.start()
      ]);

      // Update metrics
      this.systemMetrics.activeAgents = Object.values(this.agentRegistry).filter(agent => agent !== null).length;

      logger.info('Specialized agents initialized', {
        activeAgents: this.systemMetrics.activeAgents,
        agentTypes: Object.keys(this.agentRegistry).filter(
          key => this.agentRegistry[key as keyof AgentRegistry] !== null
        )
      });

    } catch (error) {
      logger.error('Failed to initialize specialized agents', {
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  private async stopSpecializedAgents(): Promise<void> {
    const stopPromises = Object.values(this.agentRegistry)
      .filter(agent => agent !== null)
      .map(agent => agent!.stop());

    await Promise.all(stopPromises);

    // Clear registry
    Object.keys(this.agentRegistry).forEach(key => {
      this.agentRegistry[key as keyof AgentRegistry] = null;
    });

    this.systemMetrics.activeAgents = 0;
  }

  private async monitorSystem(): Promise<void> {
    try {
      this.state.status = 'monitoring';

      // Update system metrics
      await this.updateSystemMetrics();

      // Evaluate system health
      const systemEvaluation = await this.evaluators.system.evaluateSystemHealth(
        this.systemMetrics,
        this.agentRegistry
      );

      // Check for performance issues
      if (systemEvaluation.healthScore < 70) {
        await this.handleSystemDegradation(systemEvaluation);
      }

      // Check for resource constraints
      if (this.systemMetrics.resourceUtilization.cpu > 80 || 
          this.systemMetrics.resourceUtilization.memory > 80) {
        await this.handleResourceConstraints();
      }

      // Process coordination queue
      await this.processCoordinationQueue();

      this.state.status = 'active';

    } catch (error) {
      this.state.status = 'error';
      this.state.errorCount++;
      
      logger.agentError(this.agentId, 'orchestrator', error as Error, {
        systemMonitoring: true
      });
    }
  }

  private async updateSystemMetrics(): Promise<void> {
    try {
      // Collect metrics from all agents
      const agentMetrics = await this.collectAgentMetrics();
      
      // Update system-wide metrics
      this.systemMetrics.totalExecutions = agentMetrics.reduce(
        (sum, metrics) => sum + metrics.totalExecutions, 0
      );

      this.systemMetrics.averageResponseTime = agentMetrics.length > 0
        ? agentMetrics.reduce((sum, metrics) => sum + metrics.averageResponseTime, 0) / agentMetrics.length
        : 0;

      this.systemMetrics.errorRate = agentMetrics.length > 0
        ? agentMetrics.reduce((sum, metrics) => sum + metrics.errorRate, 0) / agentMetrics.length
        : 0;

      // Update resource utilization
      await this.updateResourceUtilization();

      // Update agent performance tracking
      this.updateAgentPerformanceMetrics(agentMetrics);

    } catch (error) {
      logger.error('Failed to update system metrics', {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async collectAgentMetrics(): Promise<any[]> {
    const metrics: any[] = [];

    for (const [agentType, agent] of Object.entries(this.agentRegistry)) {
      if (agent) {
        try {
          const agentState = agent.getState();
          metrics.push({
            agentType,
            totalExecutions: agentState.performance.metrics.totalExecutions,
            successRate: agentState.performance.metrics.successRate,
            averageResponseTime: agentState.performance.metrics.averageExecutionTime,
            errorRate: agentState.errorCount / (agentState.performance.metrics.totalExecutions || 1),
            healthScore: agentState.healthScore
          });
        } catch (error) {
          logger.debug('Failed to collect metrics from agent', {
            agentType,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }
    }

    return metrics;
  }

  private async updateResourceUtilization(): Promise<void> {
    // Monitor system resources
    this.systemMetrics.resourceUtilization.cpu = await this.getCPUUsage();
    this.systemMetrics.resourceUtilization.memory = await this.getMemoryUsage();
    this.systemMetrics.resourceUtilization.network = await this.getNetworkUsage();

    // Update gas usage tracking
    const totalGasUsed = Object.values(this.agentRegistry)
      .filter(agent => agent !== null)
      .reduce((sum, agent) => {
        const state = agent!.getState();
        return sum.add(state.performance.metrics.totalGasUsed);
      }, BigNumber.from(0));

    this.systemMetrics.resourceUtilization.gas = totalGasUsed;
  }

  private updateAgentPerformanceMetrics(agentMetrics: any[]): void {
    agentMetrics.forEach(metrics => {
      this.systemMetrics.agentPerformance[metrics.agentType] = {
        executionCount: metrics.totalExecutions,
        successRate: metrics.successRate,
        averageExecutionTime: metrics.averageResponseTime,
        lastExecution: Date.now()
      };
    });
  }

  private async handleSystemDegradation(evaluation: any): Promise<void> {
    logger.warn('System degradation detected', {
      healthScore: evaluation.healthScore,
      issues: evaluation.issues
    });

    // Implement degradation response
    if (evaluation.healthScore < 50) {
      // Critical degradation
      await this.emergencySystemRecovery(evaluation);
    } else {
      // Moderate degradation
      await this.systemOptimization(evaluation);
    }
  }

  private async handleResourceConstraints(): Promise<void> {
    logger.warn('Resource constraints detected', {
      cpu: this.systemMetrics.resourceUtilization.cpu,
      memory: this.systemMetrics.resourceUtilization.memory
    });

    if (this.config.autoScaling) {
      await this.scaleSystemResources();
    } else {
      await this.optimizeResourceUsage();
    }
  }

  private startCoordinationProcessing(): void {
    // Process coordination requests every second
    setInterval(async () => {
      await this.processCoordinationQueue();
    }, 1000);
  }

  private async processCoordinationQueue(): Promise<void> {
    if (this.coordinationQueue.length === 0) return;

    // Sort by priority and timestamp
    this.coordinationQueue.sort((a, b) => {
      if (a.priority !== b.priority) return b.priority - a.priority;
      return a.timestamp - b.timestamp;
    });

    const request = this.coordinationQueue.shift();
    if (!request) return;

    try {
      await this.processCoordinationRequest(request);
    } catch (error) {
      logger.error('Failed to process coordination request', {
        requestId: request.id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async processCoordinationRequest(request: any): Promise<void> {
    const executionId = `orchestration_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const execution: AgentExecution = {
      id: executionId,
      agentId: this.agentId,
      decision: {
        action: 'coordinate_agents',
        confidence: 0.8,
        reasoning: `Processing coordination request from ${request.requester}`,
        parameters: request,
        riskScore: 30,
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

    logger.executionStarted(this.agentId, executionId, 'coordinate_agents', {
      requestId: request.id,
      requester: request.requester
    });

    try {
      execution.status = 'executing';

      const context: AgentContext = {
        agentId: this.agentId,
        agentType: 'orchestrator',
        userId: request.requester,
        sessionId: `session_${Date.now()}`,
        networkIds: [1, 137, 42161], // Multi-chain
        timestamp: Date.now(),
        gasPrice: BigNumber.from('20000000000'),
        nonce: 0
      };

      const coordinationResult = await coordinateAgents(
        this.agentRegistry,
        this.providers.orchestration,
        context,
        {
          requestType: request.request.type,
          parameters: request.request.parameters,
          constraints: request.request.constraints,
          priority: request.priority,
          coordinationStrategy: this.config.coordinationMode
        }
      );

      execution.status = 'completed';
      execution.endTime = Date.now();
      execution.actualOutcome = coordinationResult;

      logger.executionCompleted(this.agentId, executionId, true, {
        coordinatedAgents: coordinationResult.coordinatedAgents,
        duration: execution.endTime - execution.startTime
      });

    } catch (error) {
      execution.status = 'failed';
      execution.endTime = Date.now();
      execution.errors.push(error instanceof Error ? error.message : String(error));

      logger.executionCompleted(this.agentId, executionId, false, {
        error: error instanceof Error ? error.message : String(error),
        duration: execution.endTime! - execution.startTime
      });

    } finally {
      this.activeExecutions.delete(executionId);
    }
  }

  // Public coordination methods
  async requestCoordination(
    requestType: string,
    parameters: any,
    priority: number = 1,
    requester: string = 'system'
  ): Promise<string> {
    const requestId = `coord_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    this.coordinationQueue.push({
      id: requestId,
      request: {
        type: requestType,
        parameters,
        constraints: {}
      },
      priority,
      timestamp: Date.now(),
      requester
    });

    logger.info('Coordination request queued', {
      requestId,
      requestType,
      priority,
      requester,
      queueLength: this.coordinationQueue.length
    });

    return requestId;
  }

  async executeMultiAgentOperation(
    operation: string,
    participants: string[],
    parameters: any
  ): Promise<any> {
    const context: AgentContext = {
      agentId: this.agentId,
      agentType: 'orchestrator',
      userId: 'system',
      sessionId: `session_${Date.now()}`,
      networkIds: [1, 137, 42161],
      timestamp: Date.now(),
      gasPrice: BigNumber.from('20000000000'),
      nonce: 0
    };

    return await coordinateAgents(
      this.agentRegistry,
      this.providers.orchestration,
      context,
      {
        requestType: operation,
        parameters: {
          ...parameters,
          participants
        },
        constraints: {},
        priority: 5,
        coordinationStrategy: this.config.coordinationMode
      }
    );
  }

  async manageSystemResources(
    operation: 'allocate' | 'deallocate' | 'optimize',
    resourceType: 'cpu' | 'memory' | 'network' | 'gas',
    amount?: number
  ): Promise<any> {
    const context: AgentContext = {
      agentId: this.agentId,
      agentType: 'orchestrator',
      userId: 'system',
      sessionId: `session_${Date.now()}`,
      networkIds: [1],
      timestamp: Date.now(),
      gasPrice: BigNumber.from('20000000000'),
      nonce: 0
    };

    return await manageResources(
      this.systemMetrics,
      this.agentRegistry,
      this.providers.orchestration,
      context,
      {
        operation,
        resourceType,
        amount: amount || 0,
        strategy: this.config.resourceAllocationStrategy,
        autoScaling: this.config.autoScaling,
        thresholds: this.config.performanceThresholds
      }
    );
  }

  async resolveAgentConflict(
    conflictingAgents: string[],
    conflictType: string,
    conflictData: any
  ): Promise<any> {
    const context: AgentContext = {
      agentId: this.agentId,
      agentType: 'orchestrator',
      userId: 'system',
      sessionId: `session_${Date.now()}`,
      networkIds: [1],
      timestamp: Date.now(),
      gasPrice: BigNumber.from('20000000000'),
      nonce: 0
    };

    return await handleConflicts(
      this.agentRegistry,
      this.providers.orchestration,
      context,
      {
        conflictingAgents,
        conflictType,
        conflictData,
        resolutionStrategy: this.config.conflictResolutionStrategy,
        priority: 10 // High priority for conflicts
      }
    );
  }

  // System management methods
  async emergencySystemRecovery(evaluation: any): Promise<void> {
    logger.warn('Initiating emergency system recovery', {
      healthScore: evaluation.healthScore,
      issues: evaluation.issues
    });

    // Stop non-critical agents
    const nonCriticalAgents = ['arbitrage', 'yield'];
    for (const agentType of nonCriticalAgents) {
      const agent = this.agentRegistry[agentType as keyof AgentRegistry];
      if (agent) {
        await agent.pauseAgent();
      }
    }

    // Reduce system load
    this.coordinationQueue = this.coordinationQueue.filter(req => req.priority >= 8);

    // Trigger emergency protocols
    if (this.config.emergencyProtocols.enabled) {
      await this.executeEmergencyProtocols(evaluation);
    }
  }

  private async systemOptimization(evaluation: any): Promise<void> {
    logger.info('Initiating system optimization', {
      healthScore: evaluation.healthScore
    });

    // Optimize resource allocation
    await this.optimizeResourceUsage();

    // Adjust agent priorities
    await this.adjustAgentPriorities(evaluation);

    // Clear low-priority requests from queue
    this.coordinationQueue = this.coordinationQueue.filter(req => req.priority >= 3);
  }

  private async scaleSystemResources(): Promise<void> {
    logger.info('Scaling system resources');

    // Implement auto-scaling logic
    // This would integrate with cloud infrastructure APIs
    
    // For now, adjust internal resource allocation
    await this.optimizeResourceUsage();
  }

  private async optimizeResourceUsage(): Promise<void> {
    logger.info('Optimizing resource usage');

    // Reduce resource consumption
    for (const [agentType, agent] of Object.entries(this.agentRegistry)) {
      if (agent) {
        try {
          // Reduce agent execution frequency temporarily
          // Implementation would depend on agent-specific methods
        } catch (error) {
          logger.debug('Failed to optimize agent resources', {
            agentType,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }
    }
  }

  private async executeEmergencyProtocols(evaluation: any): Promise<void> {
    for (const action of this.config.emergencyProtocols.actions) {
      try {
        switch (action) {
          case 'pause_non_critical':
            await this.pauseNonCriticalAgents();
            break;
          case 'reduce_load':
            this.coordinationQueue = [];
            break;
          case 'notify_operators':
            await this.notifySystemOperators(evaluation);
            break;
        }
      } catch (error) {
        logger.error('Emergency protocol action failed', {
          action,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }
  }

  private async pauseNonCriticalAgents(): Promise<void> {
    const nonCritical = ['arbitrage', 'yield'];
    for (const agentType of nonCritical) {
      const agent = this.agentRegistry[agentType as keyof AgentRegistry];
      if (agent) {
        await agent.pauseAgent();
      }
    }
  }

  private async adjustAgentPriorities(evaluation: any): Promise<void> {
    // Adjust agent execution priorities based on system evaluation
    // Implementation would modify agent scheduling and resource allocation
  }

  private async notifySystemOperators(evaluation: any): Promise<void> {
    // Send notifications to system operators about critical issues
    logger.warn('System operators notified of critical issues', {
      healthScore: evaluation.healthScore,
      issues: evaluation.issues
    });
  }

  // Utility methods for system metrics
  private async getCPUUsage(): Promise<number> {
    // In production, would integrate with system monitoring
    return Math.random() * 100; // Mock implementation
  }

  private async getMemoryUsage(): Promise<number> {
    // In production, would integrate with system monitoring
    return Math.random() * 100; // Mock implementation
  }

  private async getNetworkUsage(): Promise<number> {
    // In production, would integrate with system monitoring
    return Math.random() * 100; // Mock implementation
  }

  // Public getters
  getState(): AgentState {
    return { ...this.state };
  }

  getSystemMetrics(): SystemMetrics {
    return { ...this.systemMetrics };
  }

  getActiveExecutions(): AgentExecution[] {
    return Array.from(this.activeExecutions.values());
  }

  getAgentRegistry(): AgentRegistry {
    return { ...this.agentRegistry };
  }

  getCoordinationQueue(): Array<{
    id: string;
    request: any;
    priority: number;
    timestamp: number;
    requester: string;
  }> {
    return [...this.coordinationQueue];
  }

  async getSystemHealth(): Promise<{
    overallHealth: number;
    agentHealth: Record<string, number>;
    systemLoad: number;
    activeExecutions: number;
    queueLength: number;
    issues: string[];
    recommendations: string[];
  }> {
    const agentHealth: Record<string, number> = {};
    const issues: string[] = [];
    const recommendations: string[] = [];

    // Collect agent health scores
    for (const [agentType, agent] of Object.entries(this.agentRegistry)) {
      if (agent) {
        const state = agent.getState();
        agentHealth[agentType] = state.healthScore;
        
        if (state.healthScore < 70) {
          issues.push(`${agentType} agent health below optimal`);
          recommendations.push(`Review ${agentType} agent performance and logs`);
        }
      }
    }

    // Calculate overall health
    const healthScores = Object.values(agentHealth);
    const overallHealth = healthScores.length > 0 
      ? healthScores.reduce((sum, score) => sum + score, 0) / healthScores.length
      : 100;

    // Check system load
    if (this.systemMetrics.resourceUtilization.cpu > 80) {
      issues.push('High CPU utilization');
      recommendations.push('Consider scaling resources or optimizing agent workloads');
    }

    if (this.coordinationQueue.length > 50) {
      issues.push('Large coordination queue');
      recommendations.push('Review coordination request patterns and optimize processing');
    }

    return {
      overallHealth,
      agentHealth,
      systemLoad: this.systemMetrics.systemLoad,
      activeExecutions: this.activeExecutions.size,
      queueLength: this.coordinationQueue.length,
      issues,
      recommendations
    };
  }

  async updateConfiguration(newConfig: Partial<OrchestratorConfig>): Promise<void> {
    this.config = { ...this.config, ...newConfig };
    
    logger.info('OrchestratorAgent configuration updated', {
      agentId: this.agentId,
      newConfig
    });

    // Apply configuration changes
    if (newConfig.systemMonitoringInterval && this.systemMonitoringInterval) {
      clearInterval(this.systemMonitoringInterval);
      this.systemMonitoringInterval = setInterval(async () => {
        await this.monitorSystem();
      }, newConfig.systemMonitoringInterval);
    }
  }
}

export default OrchestratorAgent;
