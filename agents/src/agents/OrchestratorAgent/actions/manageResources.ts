import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { AgentContext } from '../../../shared/types/agent';
import { getElizaConfig, invokeBedrockModel } from '../../../config/elizaConfig';
import { renderPromptTemplate } from '../../../config/modelConfig';
import { OrchestrationProvider } from '../providers/orchestrationProvider';

export interface ResourceManagementConfig {
  operation: 'allocate' | 'deallocate' | 'optimize' | 'scale' | 'monitor';
  resourceType: 'cpu' | 'memory' | 'network' | 'gas' | 'storage' | 'all';
  amount?: number;
  strategy: 'balanced' | 'priority_based' | 'performance_based' | 'cost_optimized';
  autoScaling: boolean;
  thresholds: {
    responseTime: number;
    errorRate: number;
    resourceUtilization: number;
  };
  constraints?: {
    maxCost?: BigNumber;
    maxLatency?: number;
    minReliability?: number;
    preferredProviders?: string[];
  };
}

export interface ResourceManagementResult {
  success: boolean;
  operation: string;
  resourcesAffected: ResourceChange[];
  newAllocation: ResourceAllocation;
  optimizationGains: OptimizationMetrics;
  recommendations: ResourceRecommendation[];
  costs: ResourceCosts;
  duration: number;
  errors: string[];
}

export interface ResourceChange {
  resourceType: string;
  agentId: string;
  previousAllocation: number;
  newAllocation: number;
  changePercent: number;
  reason: string;
  timestamp: number;
}

export interface ResourceAllocation {
  cpu: {
    total: number;
    allocated: number;
    available: number;
    reserved: number;
    agentAllocations: Record<string, number>;
  };
  memory: {
    total: number;
    allocated: number;
    available: number;
    reserved: number;
    agentAllocations: Record<string, number>;
  };
  network: {
    bandwidth: number;
    allocated: number;
    available: number;
    latency: number;
    agentAllocations: Record<string, number>;
  };
  gas: {
    total: BigNumber;
    allocated: BigNumber;
    available: BigNumber;
    reserved: BigNumber;
    agentAllocations: Record<string, BigNumber>;
    priceGwei: BigNumber;
  };
  storage: {
    total: number;
    allocated: number;
    available: number;
    agentAllocations: Record<string, number>;
  };
}

export interface OptimizationMetrics {
  performanceGain: number;
  costReduction: number;
  efficiencyImprovement: number;
  responseTimeImprovement: number;
  resourceUtilizationImprovement: number;
  reliabilityImprovement: number;
}

export interface ResourceRecommendation {
  type: 'scale_up' | 'scale_down' | 'reallocate' | 'optimize' | 'monitor';
  priority: 'immediate' | 'urgent' | 'normal' | 'low';
  resourceType: string;
  agentId?: string;
  description: string;
  expectedBenefit: string;
  implementationSteps: string[];
  estimatedCost: BigNumber;
  riskLevel: 'low' | 'medium' | 'high';
}

export interface ResourceCosts {
  computational: {
    cpuCost: BigNumber;
    memoryCost: BigNumber;
    storageCost: BigNumber;
  };
  network: {
    bandwidthCost: BigNumber;
    latencyCost: BigNumber;
  };
  blockchain: {
    gasCost: BigNumber;
    transactionFees: BigNumber;
  };
  external: {
    apiCosts: BigNumber;
    dataFeedCosts: BigNumber;
    chainlinkCosts: BigNumber;
  };
  total: BigNumber;
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

export async function manageResources(
  systemMetrics: SystemMetrics,
  agentRegistry: any,
  orchestrationProvider: OrchestrationProvider,
  context: AgentContext,
  config: ResourceManagementConfig
): Promise<ResourceManagementResult> {
  const startTime = Date.now();

  logger.info('Starting resource management', {
    agentId: context.agentId,
    operation: config.operation,
    resourceType: config.resourceType,
    strategy: config.strategy
  });

  try {
    // Step 1: Analyze current resource state
    const currentAllocation = await analyzeCurrentResourceState(
      systemMetrics,
      agentRegistry,
      orchestrationProvider
    );

    // Step 2: Determine resource requirements
    const resourceRequirements = await determineResourceRequirements(
      config,
      systemMetrics,
      agentRegistry,
      orchestrationProvider
    );

    // Step 3: Generate optimization plan
    const optimizationPlan = await generateOptimizationPlan(
      config,
      currentAllocation,
      resourceRequirements,
      orchestrationProvider
    );

    // Step 4: Execute resource management operation
    const executionResult = await executeResourceManagement(
      config,
      optimizationPlan,
      agentRegistry,
      orchestrationProvider
    );

    // Step 5: Calculate optimization gains
    const optimizationGains = await calculateOptimizationGains(
      currentAllocation,
      executionResult.newAllocation,
      systemMetrics
    );

    // Step 6: Generate recommendations
    const recommendations = await generateResourceRecommendations(
      executionResult.newAllocation,
      systemMetrics,
      config,
      orchestrationProvider
    );

    // Step 7: Calculate costs
    const costs = await calculateResourceCosts(
      executionResult.resourcesAffected,
      executionResult.newAllocation,
      config
    );

    const result: ResourceManagementResult = {
      success: executionResult.success,
      operation: config.operation,
      resourcesAffected: executionResult.resourcesAffected,
      newAllocation: executionResult.newAllocation,
      optimizationGains,
      recommendations,
      costs,
      duration: Date.now() - startTime,
      errors: executionResult.errors
    };

    logger.info('Resource management completed', {
      agentId: context.agentId,
      operation: config.operation,
      success: result.success,
      optimizationGains: result.optimizationGains,
      duration: result.duration
    });

    return result;

  } catch (error) {
    logger.error('Resource management failed', {
      agentId: context.agentId,
      operation: config.operation,
      error: error instanceof Error ? error.message : String(error),
      duration: Date.now() - startTime
    });

    return {
      success: false,
      operation: config.operation,
      resourcesAffected: [],
      newAllocation: await getDefaultResourceAllocation(),
      optimizationGains: getZeroOptimizationMetrics(),
      recommendations: [],
      costs: getZeroCosts(),
      duration: Date.now() - startTime,
      errors: [error instanceof Error ? error.message : String(error)]
    };
  }
}

async function analyzeCurrentResourceState(
  systemMetrics: SystemMetrics,
  agentRegistry: any,
  orchestrationProvider: OrchestrationProvider
): Promise<ResourceAllocation> {
  try {
    // Collect current resource allocation from all agents
    const agentAllocations = {
      cpu: {} as Record<string, number>,
      memory: {} as Record<string, number>,
      network: {} as Record<string, number>,
      gas: {} as Record<string, BigNumber>,
      storage: {} as Record<string, number>
    };

    let totalCpuAllocated = 0;
    let totalMemoryAllocated = 0;
    let totalNetworkAllocated = 0;
    let totalGasAllocated = BigNumber.from(0);
    let totalStorageAllocated = 0;

    // Analyze each agent's resource usage
    for (const [agentType, agent] of Object.entries(agentRegistry)) {
      if (agent) {
        try {
          const agentState = agent.getState();
          const resourceUsage = await getAgentResourceUsage(agent, agentType);

          agentAllocations.cpu[agentType] = resourceUsage.cpu;
          agentAllocations.memory[agentType] = resourceUsage.memory;
          agentAllocations.network[agentType] = resourceUsage.network;
          agentAllocations.gas[agentType] = resourceUsage.gas;
          agentAllocations.storage[agentType] = resourceUsage.storage;

          totalCpuAllocated += resourceUsage.cpu;
          totalMemoryAllocated += resourceUsage.memory;
          totalNetworkAllocated += resourceUsage.network;
          totalGasAllocated = totalGasAllocated.add(resourceUsage.gas);
          totalStorageAllocated += resourceUsage.storage;

        } catch (error) {
          logger.warn('Failed to get resource usage for agent', {
            agentType,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }
    }

    // Get system-wide resource limits
    const systemLimits = await getSystemResourceLimits(orchestrationProvider);

    const currentAllocation: ResourceAllocation = {
      cpu: {
        total: systemLimits.cpu,
        allocated: totalCpuAllocated,
        available: systemLimits.cpu - totalCpuAllocated,
        reserved: systemLimits.cpu * 0.1, // 10% reserved
        agentAllocations: agentAllocations.cpu
      },
      memory: {
        total: systemLimits.memory,
        allocated: totalMemoryAllocated,
        available: systemLimits.memory - totalMemoryAllocated,
        reserved: systemLimits.memory * 0.1,
        agentAllocations: agentAllocations.memory
      },
      network: {
        bandwidth: systemLimits.network.bandwidth,
        allocated: totalNetworkAllocated,
        available: systemLimits.network.bandwidth - totalNetworkAllocated,
        latency: systemLimits.network.latency,
        agentAllocations: agentAllocations.network
      },
      gas: {
        total: systemLimits.gas.total,
        allocated: totalGasAllocated,
        available: systemLimits.gas.total.sub(totalGasAllocated),
        reserved: systemLimits.gas.total.mul(10).div(100), // 10% reserved
        agentAllocations: agentAllocations.gas,
        priceGwei: systemLimits.gas.priceGwei
      },
      storage: {
        total: systemLimits.storage,
        allocated: totalStorageAllocated,
        available: systemLimits.storage - totalStorageAllocated,
        agentAllocations: agentAllocations.storage
      }
    };

    return currentAllocation;

  } catch (error) {
    logger.error('Failed to analyze current resource state', {
      error: error instanceof Error ? error.message : String(error)
    });

    return await getDefaultResourceAllocation();
  }
}

async function determineResourceRequirements(
  config: ResourceManagementConfig,
  systemMetrics: SystemMetrics,
  agentRegistry: any,
  orchestrationProvider: OrchestrationProvider
): Promise<{
  immediate: Record<string, number>;
  projected: Record<string, number>;
  peak: Record<string, number>;
  reasoning: string[];
}> {
  try {
    const immediate: Record<string, number> = {};
    const projected: Record<string, number> = {};
    const peak: Record<string, number> = {};
    const reasoning: string[] = [];

    // Analyze current load and trends
    const loadAnalysis = await analyzeSystemLoad(systemMetrics, agentRegistry);
    reasoning.push(`Current system load: ${loadAnalysis.currentLoad}%`);

    // Determine immediate requirements based on current performance
    if (systemMetrics.averageResponseTime > config.thresholds.responseTime) {
      const cpuIncrease = Math.ceil((systemMetrics.averageResponseTime / config.thresholds.responseTime - 1) * 50);
      immediate.cpu = cpuIncrease;
      reasoning.push(`Response time above threshold, increasing CPU by ${cpuIncrease}%`);
    }

    if (systemMetrics.errorRate > config.thresholds.errorRate) {
      const memoryIncrease = Math.ceil((systemMetrics.errorRate / config.thresholds.errorRate - 1) * 30);
      immediate.memory = memoryIncrease;
      reasoning.push(`Error rate above threshold, increasing memory by ${memoryIncrease}%`);
    }

    // Project future requirements based on trends
    const growthTrends = await analyzeGrowthTrends(systemMetrics, orchestrationProvider);
    projected.cpu = growthTrends.cpu;
    projected.memory = growthTrends.memory;
    projected.network = growthTrends.network;
    reasoning.push(`Projected growth: CPU ${growthTrends.cpu}%, Memory ${growthTrends.memory}%, Network ${growthTrends.network}%`);

    // Calculate peak requirements for worst-case scenarios
    peak.cpu = Math.max(immediate.cpu || 0, projected.cpu || 0) * 1.5;
    peak.memory = Math.max(immediate.memory || 0, projected.memory || 0) * 1.5;
    peak.network = Math.max(immediate.network || 0, projected.network || 0) * 1.5;
    reasoning.push(`Peak requirements calculated with 50% safety margin`);

    // Use AI for intelligent requirement prediction
    try {
      const aiRequirements = await predictResourceRequirementsWithAI(
        systemMetrics,
        loadAnalysis,
        growthTrends,
        config
      );
      
      if (aiRequirements.success) {
        Object.assign(projected, aiRequirements.projected);
        reasoning.push('AI-enhanced projection incorporated');
      }
    } catch (error) {
      logger.debug('AI requirement prediction failed', {
        error: error instanceof Error ? error.message : String(error)
      });
    }

    return { immediate, projected, peak, reasoning };

  } catch (error) {
    logger.error('Failed to determine resource requirements', {
      error: error instanceof Error ? error.message : String(error)
    });

    return {
      immediate: {},
      projected: {},
      peak: {},
      reasoning: ['Failed to analyze requirements - using conservative estimates']
    };
  }
}

async function generateOptimizationPlan(
  config: ResourceManagementConfig,
  currentAllocation: ResourceAllocation,
  requirements: any,
  orchestrationProvider: OrchestrationProvider
): Promise<{
  steps: OptimizationStep[];
  expectedGains: OptimizationMetrics;
  riskAssessment: RiskAssessment;
}> {
  const steps: OptimizationStep[] = [];
  
  try {
    // Generate optimization steps based on strategy
    switch (config.strategy) {
      case 'balanced':
        steps.push(...await generateBalancedOptimizationSteps(currentAllocation, requirements));
        break;
      
      case 'priority_based':
        steps.push(...await generatePriorityBasedOptimizationSteps(currentAllocation, requirements, config));
        break;
      
      case 'performance_based':
        steps.push(...await generatePerformanceBasedOptimizationSteps(currentAllocation, requirements));
        break;
      
      case 'cost_optimized':
        steps.push(...await generateCostOptimizedOptimizationSteps(currentAllocation, requirements, config));
        break;
    }

    // Calculate expected gains
    const expectedGains = await calculateExpectedOptimizationGains(steps, currentAllocation);

    // Assess risks
    const riskAssessment = await assessOptimizationRisks(steps, currentAllocation, config);

    return { steps, expectedGains, riskAssessment };

  } catch (error) {
    logger.error('Failed to generate optimization plan', {
      error: error instanceof Error ? error.message : String(error)
    });

    return {
      steps: [],
      expectedGains: getZeroOptimizationMetrics(),
      riskAssessment: {
        overallRisk: 'high',
        riskFactors: ['Plan generation failed'],
        mitigations: ['Manual intervention required']
      }
    };
  }
}

async function executeResourceManagement(
  config: ResourceManagementConfig,
  optimizationPlan: any,
  agentRegistry: any,
  orchestrationProvider: OrchestrationProvider
): Promise<{
  success: boolean;
  resourcesAffected: ResourceChange[];
  newAllocation: ResourceAllocation;
  errors: string[];
}> {
  const resourcesAffected: ResourceChange[] = [];
  const errors: string[] = [];
  let success = true;

  try {
    // Execute optimization steps
    for (const step of optimizationPlan.steps) {
      try {
        const stepResult = await executeOptimizationStep(step, agentRegistry, orchestrationProvider);
        
        if (stepResult.success) {
          resourcesAffected.push(...stepResult.changes);
        } else {
          errors.push(`Step ${step.id} failed: ${stepResult.error}`);
          success = false;
        }

      } catch (error) {
        const errorMessage = `Failed to execute step ${step.id}: ${error instanceof Error ? error.message : String(error)}`;
        errors.push(errorMessage);
        success = false;
      }
    }

    // Get updated resource allocation
    const newAllocation = await analyzeCurrentResourceState(
      await getCurrentSystemMetrics(agentRegistry),
      agentRegistry,
      orchestrationProvider
    );

    return {
      success,
      resourcesAffected,
      newAllocation,
      errors
    };

  } catch (error) {
    return {
      success: false,
      resourcesAffected,
      newAllocation: await getDefaultResourceAllocation(),
      errors: [error instanceof Error ? error.message : String(error)]
    };
  }
}

async function calculateOptimizationGains(
  beforeAllocation: ResourceAllocation,
  afterAllocation: ResourceAllocation,
  systemMetrics: SystemMetrics
): Promise<OptimizationMetrics> {
  try {
    // Calculate performance improvements
    const cpuUtilizationBefore = (beforeAllocation.cpu.allocated / beforeAllocation.cpu.total) * 100;
    const cpuUtilizationAfter = (afterAllocation.cpu.allocated / afterAllocation.cpu.total) * 100;
    const cpuEfficiencyGain = cpuUtilizationBefore - cpuUtilizationAfter;

    const memoryUtilizationBefore = (beforeAllocation.memory.allocated / beforeAllocation.memory.total) * 100;
    const memoryUtilizationAfter = (afterAllocation.memory.allocated / afterAllocation.memory.total) * 100;
    const memoryEfficiencyGain = memoryUtilizationBefore - memoryUtilizationAfter;

    // Calculate cost savings
    const gasCostBefore = beforeAllocation.gas.allocated.mul(beforeAllocation.gas.priceGwei);
    const gasCostAfter = afterAllocation.gas.allocated.mul(afterAllocation.gas.priceGwei);
    const gasCostReduction = parseFloat(ethers.utils.formatEther(gasCostBefore.sub(gasCostAfter)));

    const performanceGain = (cpuEfficiencyGain + memoryEfficiencyGain) / 2;
    const costReduction = gasCostReduction;
    const efficiencyImprovement = performanceGain;

    return {
      performanceGain: Math.max(0, performanceGain),
      costReduction: Math.max(0, costReduction),
      efficiencyImprovement: Math.max(0, efficiencyImprovement),
      responseTimeImprovement: calculateResponseTimeImprovement(beforeAllocation, afterAllocation),
      resourceUtilizationImprovement: performanceGain,
      reliabilityImprovement: calculateReliabilityImprovement(beforeAllocation, afterAllocation)
    };

  } catch (error) {
    logger.error('Failed to calculate optimization gains', {
      error: error instanceof Error ? error.message : String(error)
    });

    return getZeroOptimizationMetrics();
  }
}

async function generateResourceRecommendations(
  currentAllocation: ResourceAllocation,
  systemMetrics: SystemMetrics,
  config: ResourceManagementConfig,
  orchestrationProvider: OrchestrationProvider
): Promise<ResourceRecommendation[]> {
  const recommendations: ResourceRecommendation[] = [];

  try {
    // CPU recommendations
    const cpuUtilization = (currentAllocation.cpu.allocated / currentAllocation.cpu.total) * 100;
    if (cpuUtilization > 80) {
      recommendations.push({
        type: 'scale_up',
        priority: 'urgent',
        resourceType: 'cpu',
        description: `CPU utilization at ${cpuUtilization.toFixed(1)}% - scale up recommended`,
        expectedBenefit: 'Improved response times and system stability',
        implementationSteps: [
          'Provision additional CPU resources',
          'Redistribute workload across agents',
          'Monitor performance improvements'
        ],
        estimatedCost: ethers.utils.parseEther('0.1'),
        riskLevel: 'low'
      });
    } else if (cpuUtilization < 30) {
      recommendations.push({
        type: 'scale_down',
        priority: 'normal',
        resourceType: 'cpu',
        description: `CPU utilization at ${cpuUtilization.toFixed(1)}% - scale down possible`,
        expectedBenefit: 'Cost savings without performance impact',
        implementationSteps: [
          'Analyze usage patterns over time',
          'Gradually reduce CPU allocation',
          'Monitor for performance degradation'
        ],
        estimatedCost: ethers.utils.parseEther('-0.05'), // Negative cost = savings
        riskLevel: 'medium'
      });
    }

    // Memory recommendations
    const memoryUtilization = (currentAllocation.memory.allocated / currentAllocation.memory.total) * 100;
    if (memoryUtilization > 85) {
      recommendations.push({
        type: 'scale_up',
        priority: 'urgent',
        resourceType: 'memory',
        description: `Memory utilization at ${memoryUtilization.toFixed(1)}% - immediate scaling needed`,
        expectedBenefit: 'Prevent out-of-memory errors and crashes',
        implementationSteps: [
          'Immediately provision additional memory',
          'Optimize memory usage in agents',
          'Implement memory monitoring alerts'
        ],
        estimatedCost: ethers.utils.parseEther('0.08'),
        riskLevel: 'low'
      });
    }

    // Gas optimization recommendations
    const gasUtilization = parseFloat(ethers.utils.formatEther(
      currentAllocation.gas.allocated.mul(100).div(currentAllocation.gas.total)
    ));
    
    if (gasUtilization > 70) {
      recommendations.push({
        type: 'optimize',
        priority: 'normal',
        resourceType: 'gas',
        description: `Gas utilization at ${gasUtilization.toFixed(1)}% - optimization recommended`,
        expectedBenefit: 'Reduced transaction costs and improved efficiency',
        implementationSteps: [
          'Implement gas optimization strategies',
          'Batch transactions where possible',
          'Use layer 2 solutions for high-frequency operations'
        ],
        estimatedCost: ethers.utils.parseEther('0.02'),
        riskLevel: 'low'
      });
    }

    // Agent-specific recommendations
    for (const [agentType, agentAllocation] of Object.entries(currentAllocation.cpu.agentAllocations)) {
      const agentCpuPercent = (agentAllocation / currentAllocation.cpu.total) * 100;
      
      if (agentCpuPercent > 40) {
        recommendations.push({
          type: 'reallocate',
          priority: 'normal',
          resourceType: 'cpu',
          agentId: agentType,
          description: `${agentType} agent using ${agentCpuPercent.toFixed(1)}% of total CPU`,
          expectedBenefit: 'Better resource distribution across agents',
          implementationSteps: [
            `Analyze ${agentType} agent workload`,
            'Optimize agent algorithms',
            'Consider splitting agent responsibilities'
          ],
          estimatedCost: ethers.utils.parseEther('0.01'),
          riskLevel: 'medium'
        });
      }
    }

    // System-wide recommendations based on performance
    if (systemMetrics.errorRate > config.thresholds.errorRate) {
      recommendations.push({
        type: 'monitor',
        priority: 'urgent',
        resourceType: 'all',
        description: `System error rate at ${systemMetrics.errorRate.toFixed(2)}% - monitoring required`,
        expectedBenefit: 'Identify and resolve system issues',
        implementationSteps: [
          'Increase monitoring frequency',
          'Analyze error patterns',
          'Implement preventive measures'
        ],
        estimatedCost: ethers.utils.parseEther('0.005'),
        riskLevel: 'low'
      });
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { immediate: 4, urgent: 3, normal: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });

  } catch (error) {
    logger.error('Failed to generate resource recommendations', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function calculateResourceCosts(
  resourcesAffected: ResourceChange[],
  newAllocation: ResourceAllocation,
  config: ResourceManagementConfig
): Promise<ResourceCosts> {
  try {
    const costs: ResourceCosts = {
      computational: {
        cpuCost: BigNumber.from(0),
        memoryCost: BigNumber.from(0),
        storageCost: BigNumber.from(0)
      },
      network: {
        bandwidthCost: BigNumber.from(0),
        latencyCost: BigNumber.from(0)
      },
      blockchain: {
        gasCost: newAllocation.gas.allocated.mul(newAllocation.gas.priceGwei),
        transactionFees: BigNumber.from(0)
      },
      external: {
        apiCosts: BigNumber.from(0),
        dataFeedCosts: BigNumber.from(0),
        chainlinkCosts: BigNumber.from(0)
      },
      total: BigNumber.from(0)
    };

    // Calculate computational costs
    const cpuHourCost = ethers.utils.parseUnits('0.1', 'gwei'); // 0.1 gwei per CPU hour
    const memoryHourCost = ethers.utils.parseUnits('0.05', 'gwei'); // 0.05 gwei per GB hour
    const storageHourCost = ethers.utils.parseUnits('0.01', 'gwei'); // 0.01 gwei per GB hour

    costs.computational.cpuCost = cpuHourCost.mul(newAllocation.cpu.allocated);
    costs.computational.memoryCost = memoryHourCost.mul(newAllocation.memory.allocated);
    costs.computational.storageCost = storageHourCost.mul(newAllocation.storage.allocated);

    // Calculate network costs
    const bandwidthCost = ethers.utils.parseUnits('0.02', 'gwei'); // 0.02 gwei per Mbps
    costs.network.bandwidthCost = bandwidthCost.mul(newAllocation.network.allocated);

    // Estimate external service costs
    costs.external.chainlinkCosts = ethers.utils.parseUnits('0.1', 'ether'); // Estimated Chainlink costs

    // Calculate total cost
    costs.total = costs.computational.cpuCost
      .add(costs.computational.memoryCost)
      .add(costs.computational.storageCost)
      .add(costs.network.bandwidthCost)
      .add(costs.network.latencyCost)
      .add(costs.blockchain.gasCost)
      .add(costs.blockchain.transactionFees)
      .add(costs.external.apiCosts)
      .add(costs.external.dataFeedCosts)
      .add(costs.external.chainlinkCosts);

    return costs;

  } catch (error) {
    logger.error('Failed to calculate resource costs', {
      error: error instanceof Error ? error.message : String(error)
    });

    return getZeroCosts();
  }
}

// Helper functions
async function getAgentResourceUsage(agent: any, agentType: string): Promise<{
  cpu: number;
  memory: number;
  network: number;
  gas: BigNumber;
  storage: number;
}> {
  try {
    const state = agent.getState();
    
    // Base resource usage by agent type
    const baseUsage = {
      arbitrage: { cpu: 25, memory: 20, network: 30, gas: '0.3', storage: 10 },
      portfolio: { cpu: 15, memory: 25, network: 15, gas: '0.1', storage: 20 },
      yield: { cpu: 20, memory: 20, network: 25, gas: '0.2', storage: 15 },
      security: { cpu: 30, memory: 15, network: 40, gas: '0.05', storage: 5 }
    };

    const base = baseUsage[agentType as keyof typeof baseUsage] || baseUsage.portfolio;
    
    // Adjust based on actual usage
    const loadFactor = Math.min(state.resources.cpuUsage / 100, 2.0);
    
    return {
      cpu: base.cpu * loadFactor,
      memory: base.memory * loadFactor,
      network: base.network * loadFactor,
      gas: ethers.utils.parseEther(base.gas).mul(Math.floor(loadFactor * 100)).div(100),
      storage: base.storage * loadFactor
    };

  } catch (error) {
    // Return default values if agent state unavailable
    return {
      cpu: 20,
      memory: 20,
      network: 20,
      gas: ethers.utils.parseEther('0.1'),
      storage: 10
    };
  }
}

async function getSystemResourceLimits(orchestrationProvider: OrchestrationProvider): Promise<{
  cpu: number;
  memory: number;
  network: { bandwidth: number; latency: number };
  gas: { total: BigNumber; priceGwei: BigNumber };
  storage: number;
}> {
  try {
    // Get limits from orchestration provider
    const limits = await orchestrationProvider.getSystemLimits();
    return limits;
  } catch (error) {
    // Return default limits
    return {
      cpu: 100,
      memory: 100,
      network: { bandwidth: 1000, latency: 100 },
      gas: { 
        total: ethers.utils.parseEther('10'), 
        priceGwei: ethers.utils.parseUnits('20', 'gwei') 
      },
      storage: 1000
    };
  }
}

async function analyzeSystemLoad(systemMetrics: SystemMetrics, agentRegistry: any): Promise<{
  currentLoad: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  bottlenecks: string[];
}> {
  const currentLoad = (
    systemMetrics.resourceUtilization.cpu +
    systemMetrics.resourceUtilization.memory +
    systemMetrics.resourceUtilization.network
  ) / 3;

  // Simple trend analysis (would be more sophisticated in production)
  const trend: 'increasing' | 'decreasing' | 'stable' = 
    currentLoad > 70 ? 'increasing' : 
    currentLoad < 30 ? 'decreasing' : 'stable';

  const bottlenecks: string[] = [];
  if (systemMetrics.resourceUtilization.cpu > 80) bottlenecks.push('CPU');
  if (systemMetrics.resourceUtilization.memory > 80) bottlenecks.push('Memory');
  if (systemMetrics.resourceUtilization.network > 80) bottlenecks.push('Network');

  return { currentLoad, trend, bottlenecks };
}

async function analyzeGrowthTrends(
  systemMetrics: SystemMetrics,
  orchestrationProvider: OrchestrationProvider
): Promise<{ cpu: number; memory: number; network: number }> {
  // Simple growth projection based on current metrics
  const baseGrowth = systemMetrics.totalExecutions > 1000 ? 10 : 5;
  
  return {
    cpu: baseGrowth,
    memory: baseGrowth,
    network: baseGrowth * 1.5 // Network typically grows faster
  };
}

async function predictResourceRequirementsWithAI(
  systemMetrics: SystemMetrics,
  loadAnalysis: any,
  growthTrends: any,
  config: ResourceManagementConfig
): Promise<{ success: boolean; projected: Record<string, number> }> {
  try {
    const elizaConfig = getElizaConfig('orchestrator');
    
    const prompt = renderPromptTemplate('resource_prediction', {
      systemMetrics: JSON.stringify(systemMetrics),
      loadAnalysis: JSON.stringify(loadAnalysis),
      growthTrends: JSON.stringify(growthTrends),
      strategy: config.strategy,
      thresholds: JSON.stringify(config.thresholds)
    });

    const aiResponse = await invokeBedrockModel({
      modelId: elizaConfig.modelId,
      prompt,
      maxTokens: elizaConfig.maxTokens,
      temperature: 0.3
    });

    const prediction = JSON.parse(aiResponse);
    
    return {
      success: true,
      projected: prediction.projected || {}
    };

  } catch (error) {
    return { success: false, projected: {} };
  }
}

// Additional helper functions would continue here...

async function getDefaultResourceAllocation(): Promise<ResourceAllocation> {
  return {
    cpu: {
      total: 100,
      allocated: 50,
      available: 45,
      reserved: 5,
      agentAllocations: {}
    },
    memory: {
      total: 100,
      allocated: 40,
      available: 55,
      reserved: 5,
      agentAllocations: {}
    },
    network: {
      bandwidth: 1000,
      allocated: 300,
      available: 700,
      latency: 100,
      agentAllocations: {}
    },
    gas: {
      total: ethers.utils.parseEther('10'),
      allocated: ethers.utils.parseEther('2'),
      available: ethers.utils.parseEther('8'),
      reserved: ethers.utils.parseEther('0.5'),
      agentAllocations: {},
      priceGwei: ethers.utils.parseUnits('20', 'gwei')
    },
    storage: {
      total: 1000,
      allocated: 200,
      available: 800,
      agentAllocations: {}
    }
  };
}

function getZeroOptimizationMetrics(): OptimizationMetrics {
  return {
    performanceGain: 0,
    costReduction: 0,
    efficiencyImprovement: 0,
    responseTimeImprovement: 0,
    resourceUtilizationImprovement: 0,
    reliabilityImprovement: 0
  };
}

function getZeroCosts(): ResourceCosts {
  return {
    computational: {
      cpuCost: BigNumber.from(0),
      memoryCost: BigNumber.from(0),
      storageCost: BigNumber.from(0)
    },
    network: {
      bandwidthCost: BigNumber.from(0),
      latencyCost: BigNumber.from(0)
    },
    blockchain: {
      gasCost: BigNumber.from(0),
      transactionFees: BigNumber.from(0)
    },
    external: {
      apiCosts: BigNumber.from(0),
      dataFeedCosts: BigNumber.from(0),
      chainlinkCosts: BigNumber.from(0)
    },
    total: BigNumber.from(0)
  };
}