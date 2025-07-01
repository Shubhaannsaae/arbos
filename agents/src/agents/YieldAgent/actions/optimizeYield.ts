import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { AgentContext } from '../../../shared/types/agent';
import { YieldOpportunity, YieldStrategy } from '../../../shared/types/market';
import { YIELD_THRESHOLDS } from '../../../shared/constants/thresholds';
import { ProtocolProvider } from '../providers/protocolProvider';

export interface YieldOptimizationConfig {
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  diversificationTarget: number;
  maxPositionSize: BigNumber;
  rebalanceThreshold: number;
  includeNewProtocols: boolean;
  autoCompoundPreference: boolean;
  liquidityRequirement: BigNumber;
  maxGasUsage: BigNumber;
}

export interface OptimizationResult {
  success: boolean;
  currentPortfolio: PortfolioAnalysis;
  optimizedPositions: OptimizedPosition[];
  expectedImprovement: number;
  riskReduction: number;
  implementation: ImplementationPlan;
  alternatives: AlternativeStrategy[];
  reasoning: string;
}

export interface PortfolioAnalysis {
  totalValue: BigNumber;
  weightedAPY: number;
  riskScore: number;
  diversificationScore: number;
  liquidityScore: number;
  protocolDistribution: Record<string, number>;
  strategyDistribution: Record<YieldStrategy, number>;
  concentrationRisks: ConcentrationRisk[];
}

export interface OptimizedPosition {
  protocol: string;
  poolId: string;
  strategy: YieldStrategy;
  allocatedAmount: BigNumber;
  allocatedPercentage: number;
  expectedAPY: number;
  riskScore: number;
  reasoning: string;
  comparedTo?: {
    currentProtocol?: string;
    improvementType: 'new_position' | 'reallocation' | 'migration';
    expectedBenefit: number;
  };
}

export interface ImplementationPlan {
  totalSteps: number;
  estimatedGas: BigNumber;
  estimatedTime: number;
  phaseExecution: boolean;
  steps: OptimizationStep[];
  riskMitigation: string[];
  successProbability: number;
}

export interface OptimizationStep {
  stepNumber: number;
  action: 'withdraw' | 'deposit' | 'rebalance' | 'harvest' | 'migrate';
  protocol: string;
  poolId: string;
  amount: BigNumber;
  description: string;
  estimatedGas: BigNumber;
  dependencies: number[];
  risks: string[];
  priority: 'high' | 'medium' | 'low';
}

export interface AlternativeStrategy {
  name: string;
  description: string;
  expectedAPY: number;
  riskScore: number;
  implementation: string[];
  tradeOffs: string[];
}

export interface ConcentrationRisk {
  type: 'protocol' | 'strategy' | 'chain' | 'token';
  entity: string;
  concentration: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  recommendation: string;
}

export async function optimizeYield(
  currentPositions: YieldOpportunity[],
  protocolProvider: ProtocolProvider,
  context: AgentContext,
  config: YieldOptimizationConfig
): Promise<OptimizationResult> {
  const startTime = Date.now();

  logger.info('Starting yield optimization', {
    agentId: context.agentId,
    currentPositions: currentPositions.length,
    riskTolerance: config.riskTolerance,
    diversificationTarget: config.diversificationTarget
  });

  try {
    // Step 1: Analyze current portfolio
    const currentPortfolio = await analyzeCurrentPortfolio(currentPositions, protocolProvider);

    // Step 2: Identify optimization opportunities
    const optimizationOpportunities = await identifyOptimizationOpportunities(
      currentPositions,
      currentPortfolio,
      protocolProvider,
      context,
      config
    );

    // Step 3: Generate optimal allocation
    const optimizedAllocation = await generateOptimalAllocation(
      optimizationOpportunities,
      currentPortfolio,
      config,
      protocolProvider
    );

    // Step 4: Calculate expected improvements
    const improvements = calculateExpectedImprovements(currentPortfolio, optimizedAllocation);

    // Step 5: Create implementation plan
    const implementationPlan = await createImplementationPlan(
      currentPositions,
      optimizedAllocation,
      protocolProvider,
      context,
      config
    );

    // Step 6: Generate alternative strategies
    const alternatives = await generateAlternativeStrategies(
      currentPortfolio,
      optimizationOpportunities,
      config
    );

    // Step 7: Generate optimization reasoning
    const reasoning = generateOptimizationReasoning(
      currentPortfolio,
      optimizedAllocation,
      improvements,
      config
    );

    const result: OptimizationResult = {
      success: true,
      currentPortfolio,
      optimizedPositions: optimizedAllocation,
      expectedImprovement: improvements.yieldImprovement,
      riskReduction: improvements.riskReduction,
      implementation: implementationPlan,
      alternatives,
      reasoning
    };

    logger.info('Yield optimization completed', {
      agentId: context.agentId,
      expectedImprovement: improvements.yieldImprovement.toFixed(2),
      riskReduction: improvements.riskReduction.toFixed(2),
      optimizedPositions: optimizedAllocation.length,
      duration: Date.now() - startTime
    });

    return result;

  } catch (error) {
    logger.error('Yield optimization failed', {
      agentId: context.agentId,
      error: error instanceof Error ? error.message : String(error),
      duration: Date.now() - startTime
    });

    return {
      success: false,
      currentPortfolio: await analyzeCurrentPortfolio(currentPositions, protocolProvider),
      optimizedPositions: [],
      expectedImprovement: 0,
      riskReduction: 0,
      implementation: {
        totalSteps: 0,
        estimatedGas: BigNumber.from(0),
        estimatedTime: 0,
        phaseExecution: false,
        steps: [],
        riskMitigation: [],
        successProbability: 0
      },
      alternatives: [],
      reasoning: `Optimization failed: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}

async function analyzeCurrentPortfolio(
  positions: YieldOpportunity[],
  protocolProvider: ProtocolProvider
): Promise<PortfolioAnalysis> {
  try {
    // Calculate total value
    const totalValue = positions.reduce((sum, pos) => sum.add(pos.position.value), BigNumber.from(0));

    // Calculate weighted APY
    const weightedAPY = positions.reduce((sum, pos) => {
      const weight = parseFloat(ethers.utils.formatEther(pos.position.value)) / 
                    parseFloat(ethers.utils.formatEther(totalValue));
      return sum + (pos.apy * weight);
    }, 0);

    // Calculate weighted risk score
    const weightedRiskScore = positions.reduce((sum, pos) => {
      const weight = parseFloat(ethers.utils.formatEther(pos.position.value)) / 
                    parseFloat(ethers.utils.formatEther(totalValue));
      return sum + (pos.riskScore * weight);
    }, 0);

    // Calculate diversification score
    const diversificationScore = calculateDiversificationScore(positions);

    // Calculate liquidity score
    const liquidityScore = calculateLiquidityScore(positions);

    // Analyze protocol distribution
    const protocolDistribution = calculateProtocolDistribution(positions);

    // Analyze strategy distribution
    const strategyDistribution = calculateStrategyDistribution(positions);

    // Identify concentration risks
    const concentrationRisks = identifyConcentrationRisks(positions, protocolDistribution, strategyDistribution);

    return {
      totalValue,
      weightedAPY,
      riskScore: weightedRiskScore,
      diversificationScore,
      liquidityScore,
      protocolDistribution,
      strategyDistribution,
      concentrationRisks
    };

  } catch (error) {
    logger.error('Failed to analyze current portfolio', {
      error: error instanceof Error ? error.message : String(error)
    });

    return {
      totalValue: BigNumber.from(0),
      weightedAPY: 0,
      riskScore: 0,
      diversificationScore: 0,
      liquidityScore: 0,
      protocolDistribution: {},
      strategyDistribution: {},
      concentrationRisks: []
    };
  }
}

async function identifyOptimizationOpportunities(
  currentPositions: YieldOpportunity[],
  currentPortfolio: PortfolioAnalysis,
  protocolProvider: ProtocolProvider,
  context: AgentContext,
  config: YieldOptimizationConfig
): Promise<YieldOpportunity[]> {
  const opportunities: YieldOpportunity[] = [];

  try {
    // Get available opportunities across all supported protocols
    const allProtocols = await protocolProvider.getSupportedProtocols();
    
    for (const protocol of allProtocols) {
      for (const chainId of context.networkIds) {
        try {
          const protocolOpportunities = await protocolProvider.getAvailablePools(protocol, chainId);
          
          for (const pool of protocolOpportunities) {
            const poolDetails = await protocolProvider.getPoolDetails(protocol, pool.id, chainId);
            
            if (poolDetails && isOpportunityViable(poolDetails, config, currentPortfolio)) {
              const yieldOpportunity = await createYieldOpportunityFromPool(
                poolDetails,
                protocol,
                chainId,
                protocolProvider
              );
              
              opportunities.push(yieldOpportunity);
            }
          }

        } catch (error) {
          logger.debug('Failed to get opportunities from protocol', {
            protocol,
            chainId,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }
    }

    // Filter and rank opportunities
    const filteredOpportunities = filterOpportunitiesByConfig(opportunities, config);
    const rankedOpportunities = rankOpportunitiesByScore(filteredOpportunities, config, currentPortfolio);

    logger.debug('Optimization opportunities identified', {
      totalFound: opportunities.length,
      afterFiltering: filteredOpportunities.length,
      topRanked: rankedOpportunities.slice(0, 10).length
    });

    return rankedOpportunities;

  } catch (error) {
    logger.error('Failed to identify optimization opportunities', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function generateOptimalAllocation(
  opportunities: YieldOpportunity[],
  currentPortfolio: PortfolioAnalysis,
  config: YieldOptimizationConfig,
  protocolProvider: ProtocolProvider
): Promise<OptimizedPosition[]> {
  try {
    const totalValue = currentPortfolio.totalValue;
    const optimizedPositions: OptimizedPosition[] = [];

    // Apply optimization algorithm based on risk tolerance
    switch (config.riskTolerance) {
      case 'conservative':
        return await generateConservativeAllocation(opportunities, totalValue, config);
      
      case 'moderate':
        return await generateModerateAllocation(opportunities, totalValue, config, currentPortfolio);
      
      case 'aggressive':
        return await generateAggressiveAllocation(opportunities, totalValue, config);
      
      default:
        return await generateModerateAllocation(opportunities, totalValue, config, currentPortfolio);
    }

  } catch (error) {
    logger.error('Failed to generate optimal allocation', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function generateConservativeAllocation(
  opportunities: YieldOpportunity[],
  totalValue: BigNumber,
  config: YieldOptimizationConfig
): Promise<OptimizedPosition[]> {
  const positions: OptimizedPosition[] = [];
  
  // Conservative: Focus on low-risk, well-established protocols
  const conservativeOpportunities = opportunities.filter(opp => 
    opp.riskScore <= 30 && 
    opp.verified && 
    opp.auditScore && opp.auditScore >= 70
  );

  // Limit to top protocols by safety
  const safeProtocols = ['aave', 'compound', 'yearn'];
  const filteredOpportunities = conservativeOpportunities.filter(opp => 
    safeProtocols.some(protocol => opp.protocol.toLowerCase().includes(protocol))
  );

  // Equal weighting with maximum 25% per protocol
  const maxPositions = Math.min(4, filteredOpportunities.length);
  const equalWeight = 1 / maxPositions;

  for (let i = 0; i < maxPositions; i++) {
    const opportunity = filteredOpportunities[i];
    const allocatedAmount = totalValue.mul(Math.floor(equalWeight * 10000)).div(10000);
    
    positions.push({
      protocol: opportunity.protocol,
      poolId: opportunity.poolId,
      strategy: opportunity.strategy,
      allocatedAmount,
      allocatedPercentage: equalWeight * 100,
      expectedAPY: opportunity.apy,
      riskScore: opportunity.riskScore,
      reasoning: 'Conservative allocation prioritizing safety and stability',
      comparedTo: {
        improvementType: 'new_position',
        expectedBenefit: opportunity.apy
      }
    });
  }

  return positions;
}

async function generateModerateAllocation(
  opportunities: YieldOpportunity[],
  totalValue: BigNumber,
  config: YieldOptimizationConfig,
  currentPortfolio: PortfolioAnalysis
): Promise<OptimizedPosition[]> {
  const positions: OptimizedPosition[] = [];
  
  // Moderate: Balance between yield and risk
  const moderateOpportunities = opportunities.filter(opp => 
    opp.riskScore <= 60 && 
    opp.apy >= YIELD_THRESHOLDS.MIN_APY
  );

  // Use Modern Portfolio Theory principles
  const optimalWeights = await calculateOptimalWeights(
    moderateOpportunities,
    config.riskTolerance,
    config.diversificationTarget
  );

  for (const [index, opportunity] of moderateOpportunities.entries()) {
    if (index >= optimalWeights.length) break;
    
    const weight = optimalWeights[index];
    if (weight < 0.05) continue; // Skip positions under 5%

    const allocatedAmount = totalValue.mul(Math.floor(weight * 10000)).div(10000);
    
    positions.push({
      protocol: opportunity.protocol,
      poolId: opportunity.poolId,
      strategy: opportunity.strategy,
      allocatedAmount,
      allocatedPercentage: weight * 100,
      expectedAPY: opportunity.apy,
      riskScore: opportunity.riskScore,
      reasoning: 'Moderate allocation balancing yield and risk using portfolio theory',
      comparedTo: {
        improvementType: 'reallocation',
        expectedBenefit: opportunity.apy - currentPortfolio.weightedAPY
      }
    });
  }

  return positions;
}

async function generateAggressiveAllocation(
  opportunities: YieldOpportunity[],
  totalValue: BigNumber,
  config: YieldOptimizationConfig
): Promise<OptimizedPosition[]> {
  const positions: OptimizedPosition[] = [];
  
  // Aggressive: Focus on highest yields
  const aggressiveOpportunities = opportunities
    .filter(opp => opp.apy >= YIELD_THRESHOLDS.MIN_APY * 2)
    .sort((a, b) => b.apy - a.apy);

  // Concentration in top performers with risk limits
  const maxPositions = Math.min(6, aggressiveOpportunities.length);
  const weights = calculateAggressiveWeights(maxPositions);

  for (let i = 0; i < maxPositions; i++) {
    const opportunity = aggressiveOpportunities[i];
    const weight = weights[i];
    const allocatedAmount = totalValue.mul(Math.floor(weight * 10000)).div(10000);
    
    positions.push({
      protocol: opportunity.protocol,
      poolId: opportunity.poolId,
      strategy: opportunity.strategy,
      allocatedAmount,
      allocatedPercentage: weight * 100,
      expectedAPY: opportunity.apy,
      riskScore: opportunity.riskScore,
      reasoning: 'Aggressive allocation focusing on maximum yield potential',
      comparedTo: {
        improvementType: 'new_position',
        expectedBenefit: opportunity.apy
      }
    });
  }

  return positions;
}

async function calculateOptimalWeights(
  opportunities: YieldOpportunity[],
  riskTolerance: string,
  diversificationTarget: number
): Promise<number[]> {
  // Simplified Markowitz optimization
  const returns = opportunities.map(opp => opp.apy / 100);
  const risks = opportunities.map(opp => opp.riskScore / 100);
  
  // Risk tolerance parameter
  const riskParam = {
    'conservative': 0.5,
    'moderate': 1.0,
    'aggressive': 2.0
  }[riskTolerance] || 1.0;

  // Calculate utility scores: return - (risk_aversion * risk^2)
  const utilities = returns.map((ret, i) => ret - (1/riskParam) * Math.pow(risks[i], 2));
  
  // Normalize utilities to get weights
  const totalUtility = utilities.reduce((sum, util) => sum + Math.max(0, util), 0);
  
  if (totalUtility === 0) {
    // Equal weights fallback
    return new Array(opportunities.length).fill(1 / opportunities.length);
  }
  
  const weights = utilities.map(util => Math.max(0, util) / totalUtility);
  
  // Apply diversification constraints
  const maxWeight = 1 / diversificationTarget;
  return weights.map(weight => Math.min(weight, maxWeight));
}

function calculateAggressiveWeights(positions: number): number[] {
  // Aggressive weighting: higher concentration in top positions
  const weights: number[] = [];
  let remainingWeight = 1.0;
  
  for (let i = 0; i < positions; i++) {
    let weight: number;
    
    if (i === 0) {
      weight = 0.4; // 40% in top position
    } else if (i === 1) {
      weight = 0.25; // 25% in second position
    } else if (i === 2) {
      weight = 0.15; // 15% in third position
    } else {
      // Distribute remaining weight equally
      weight = remainingWeight / (positions - i);
    }
    
    weight = Math.min(weight, remainingWeight);
    weights.push(weight);
    remainingWeight -= weight;
  }
  
  return weights;
}

function calculateExpectedImprovements(
  currentPortfolio: PortfolioAnalysis,
  optimizedPositions: OptimizedPosition[]
): {
  yieldImprovement: number;
  riskReduction: number;
  diversificationImprovement: number;
} {
  // Calculate new weighted APY
  const newWeightedAPY = optimizedPositions.reduce((sum, pos) => {
    return sum + (pos.expectedAPY * pos.allocatedPercentage / 100);
  }, 0);

  // Calculate new weighted risk
  const newWeightedRisk = optimizedPositions.reduce((sum, pos) => {
    return sum + (pos.riskScore * pos.allocatedPercentage / 100);
  }, 0);

  // Calculate new diversification score
  const newDiversificationScore = calculateNewDiversificationScore(optimizedPositions);

  return {
    yieldImprovement: newWeightedAPY - currentPortfolio.weightedAPY,
    riskReduction: currentPortfolio.riskScore - newWeightedRisk,
    diversificationImprovement: newDiversificationScore - currentPortfolio.diversificationScore
  };
}

async function createImplementationPlan(
  currentPositions: YieldOpportunity[],
  optimizedPositions: OptimizedPosition[],
  protocolProvider: ProtocolProvider,
  context: AgentContext,
  config: YieldOptimizationConfig
): Promise<ImplementationPlan> {
  const steps: OptimizationStep[] = [];
  let totalGas = BigNumber.from(0);
  let stepNumber = 1;

  try {
    // Phase 1: Harvest existing rewards
    for (const position of currentPositions) {
      if (position.claimableRewards && position.claimableRewards.length > 0) {
        const harvestGas = await protocolProvider.estimateHarvestGas(position.protocol, position.id);
        
        steps.push({
          stepNumber: stepNumber++,
          action: 'harvest',
          protocol: position.protocol,
          poolId: position.poolId,
          amount: BigNumber.from(0),
          description: `Harvest rewards from ${position.protocol}`,
          estimatedGas: harvestGas,
          dependencies: [],
          risks: ['Transaction failure', 'Gas price volatility'],
          priority: 'medium'
        });
        
        totalGas = totalGas.add(harvestGas);
      }
    }

    // Phase 2: Withdraw from positions not in optimized allocation
    const currentProtocols = new Set(currentPositions.map(p => `${p.protocol}_${p.poolId}`));
    const optimizedProtocols = new Set(optimizedPositions.map(p => `${p.protocol}_${p.poolId}`));
    
    for (const position of currentPositions) {
      const positionKey = `${position.protocol}_${position.poolId}`;
      
      if (!optimizedProtocols.has(positionKey)) {
        const withdrawGas = await protocolProvider.estimateWithdrawGas(position.protocol, position.id);
        
        steps.push({
          stepNumber: stepNumber++,
          action: 'withdraw',
          protocol: position.protocol,
          poolId: position.poolId,
          amount: position.position.amount,
          description: `Withdraw from ${position.protocol} pool ${position.poolId}`,
          estimatedGas: withdrawGas,
          dependencies: [],
          risks: ['Withdrawal fees', 'Lock-up constraints', 'Slippage'],
          priority: 'high'
        });
        
        totalGas = totalGas.add(withdrawGas);
      }
    }

    // Phase 3: Deposit to new optimized positions
    for (const position of optimizedPositions) {
      const positionKey = `${position.protocol}_${position.poolId}`;
      
      if (!currentProtocols.has(positionKey)) {
        const depositGas = await protocolProvider.estimateDepositGas(position.protocol, position.poolId);
        
        steps.push({
          stepNumber: stepNumber++,
          action: 'deposit',
          protocol: position.protocol,
          poolId: position.poolId,
          amount: position.allocatedAmount,
          description: `Deposit to ${position.protocol} pool ${position.poolId}`,
          estimatedGas: depositGas,
          dependencies: [],
          risks: ['Pool capacity limits', 'Smart contract risk', 'Slippage'],
          priority: 'high'
        });
        
        totalGas = totalGas.add(depositGas);
      }
    }

    // Phase 4: Rebalance existing positions
    for (const position of optimizedPositions) {
      const currentPosition = currentPositions.find(p => 
        p.protocol === position.protocol && p.poolId === position.poolId
      );
      
      if (currentPosition) {
        const currentAmount = currentPosition.position.amount;
        const targetAmount = position.allocatedAmount;
        
        if (!currentAmount.eq(targetAmount)) {
          const rebalanceGas = BigNumber.from('200000'); // Estimate
          
          steps.push({
            stepNumber: stepNumber++,
            action: 'rebalance',
            protocol: position.protocol,
            poolId: position.poolId,
            amount: targetAmount.sub(currentAmount),
            description: `Rebalance ${position.protocol} position`,
            estimatedGas: rebalanceGas,
            dependencies: [],
            risks: ['Rebalancing fees', 'Market timing risk'],
            priority: 'medium'
          });
          
          totalGas = totalGas.add(rebalanceGas);
        }
      }
    }

    // Calculate success probability
    const successProbability = calculateSuccessProbability(steps, totalGas, config);

    // Generate risk mitigation strategies
    const riskMitigation = generateRiskMitigation(steps, config);

    return {
      totalSteps: steps.length,
      estimatedGas: totalGas,
      estimatedTime: steps.length * 120, // 2 minutes per step
      phaseExecution: steps.length > 5,
      steps,
      riskMitigation,
      successProbability
    };

  } catch (error) {
    logger.error('Failed to create implementation plan', {
      error: error instanceof Error ? error.message : String(error)
    });

    return {
      totalSteps: 0,
      estimatedGas: BigNumber.from(0),
      estimatedTime: 0,
      phaseExecution: false,
      steps: [],
      riskMitigation: [],
      successProbability: 0
    };
  }
}

async function generateAlternativeStrategies(
  currentPortfolio: PortfolioAnalysis,
  opportunities: YieldOpportunity[],
  config: YieldOptimizationConfig
): Promise<AlternativeStrategy[]> {
  const alternatives: AlternativeStrategy[] = [];

  // Strategy 1: Conservative rebalancing
  alternatives.push({
    name: 'Conservative Rebalancing',
    description: 'Minimal changes focusing only on lowest-risk improvements',
    expectedAPY: currentPortfolio.weightedAPY + 0.5,
    riskScore: Math.max(currentPortfolio.riskScore - 5, 0),
    implementation: [
      'Harvest existing rewards',
      'Minor rebalancing within current protocols',
      'Focus on established pools only'
    ],
    tradeOffs: [
      'Lower yield improvement',
      'Minimal gas costs',
      'Lower execution risk'
    ]
  });

  // Strategy 2: Maximum yield strategy
  const highYieldOpps = opportunities.filter(opp => opp.apy > currentPortfolio.weightedAPY + 5);
  if (highYieldOpps.length > 0) {
    alternatives.push({
      name: 'Maximum Yield Strategy',
      description: 'Concentrate in highest yielding opportunities regardless of risk',
      expectedAPY: Math.max(...highYieldOpps.map(opp => opp.apy)),
      riskScore: Math.max(...highYieldOpps.map(opp => opp.riskScore)),
      implementation: [
        'Withdraw from all current positions',
        'Concentrate in top 2-3 highest yield pools',
        'Accept higher risk for higher returns'
      ],
      tradeOffs: [
        'Significantly higher risk',
        'Lower diversification',
        'Potential for higher volatility'
      ]
    });
  }

  // Strategy 3: Gradual migration
  alternatives.push({
    name: 'Gradual Migration',
    description: 'Implement changes over multiple phases to reduce risk',
    expectedAPY: currentPortfolio.weightedAPY + 2,
    riskScore: currentPortfolio.riskScore,
    implementation: [
      'Phase 1: Optimize 25% of portfolio',
      'Phase 2: Migrate another 25% after 30 days',
      'Phase 3: Complete migration over 90 days'
    ],
    tradeOffs: [
      'Slower optimization',
      'Multiple gas costs',
      'Opportunity cost of delayed optimization'
    ]
  });

  return alternatives;
}

// Helper functions
function isOpportunityViable(
  poolDetails: any,
  config: YieldOptimizationConfig,
  currentPortfolio: PortfolioAnalysis
): boolean {
  // Check minimum APY
  if (poolDetails.apy < YIELD_THRESHOLDS.MIN_APY) {
    return false;
  }

  // Check liquidity requirement
  if (poolDetails.tvl.lt(config.liquidityRequirement)) {
    return false;
  }

  // Check if it would improve the portfolio
  if (poolDetails.apy <= currentPortfolio.weightedAPY && 
      poolDetails.riskScore >= currentPortfolio.riskScore) {
    return false;
  }

  return true;
}

async function createYieldOpportunityFromPool(
  poolDetails: any,
  protocol: string,
  chainId: number,
  protocolProvider: ProtocolProvider
): Promise<YieldOpportunity> {
  // Implementation similar to findYieldOpportunities.ts
  return {
    id: `${protocol}_${poolDetails.id}_${chainId}`,
    protocol,
    chainId,
    poolId: poolDetails.id,
    tokenPair: poolDetails.tokenPair || poolDetails.token || 'Unknown',
    strategy: 'staking',
    apy: poolDetails.apy || 0,
    apyHistory: [],
    tvl: poolDetails.tvl || BigNumber.from(0),
    liquidity: poolDetails.liquidity || poolDetails.tvl || BigNumber.from(0),
    fees: poolDetails.fees || { deposit: 0, withdrawal: 0, performance: 0 },
    riskScore: poolDetails.riskScore || 50,
    auditScore: poolDetails.auditScore || 50,
    liquidityScore: 50,
    impermanentLossRisk: 0,
    lockupPeriod: poolDetails.lockupPeriod || 0,
    minimumDeposit: poolDetails.minimumDeposit || BigNumber.from(0),
    maximumDeposit: poolDetails.maximumDeposit,
    autoCompound: poolDetails.autoCompound || false,
    verified: true,
    contractAddress: poolDetails.contractAddress || '',
    depositToken: poolDetails.depositToken || {
      address: '',
      symbol: 'UNKNOWN',
      name: 'Unknown Token',
      decimals: 18,
      chainId,
      tags: [],
      isStable: false,
      isNative: false
    },
    rewardTokens: poolDetails.rewardTokens || [],
    lastUpdated: Date.now(),
    lastHarvest: 0,
    claimableRewards: [],
    totalEarned: BigNumber.from(0),
    position: {
      amount: BigNumber.from(0),
      value: BigNumber.from(0),
      shares: BigNumber.from(0),
      depositedAt: 0,
      lastAction: 0
    }
  };
}

function filterOpportunitiesByConfig(
  opportunities: YieldOpportunity[],
  config: YieldOptimizationConfig
): YieldOpportunity[] {
  return opportunities.filter(opp => {
    // Risk tolerance filter
    const maxRisk = {
      'conservative': 40,
      'moderate': 70,
      'aggressive': 90
    }[config.riskTolerance];

    if (opp.riskScore > maxRisk) return false;

    // Liquidity requirement
    if (opp.tvl.lt(config.liquidityRequirement)) return false;

    // Auto-compound preference
    if (config.autoCompoundPreference && !opp.autoCompound) return false;

    return true;
  });
}

function rankOpportunitiesByScore(
  opportunities: YieldOpportunity[],
  config: YieldOptimizationConfig,
  currentPortfolio: PortfolioAnalysis
): YieldOpportunity[] {
  return opportunities.sort((a, b) => {
    const scoreA = calculateOpportunityScore(a, config, currentPortfolio);
    const scoreB = calculateOpportunityScore(b, config, currentPortfolio);
    return scoreB - scoreA;
  });
}

function calculateOpportunityScore(
  opportunity: YieldOpportunity,
  config: YieldOptimizationConfig,
  currentPortfolio: PortfolioAnalysis
): number {
  const weights = {
    conservative: { yield: 0.3, risk: 0.4, liquidity: 0.2, audit: 0.1 },
    moderate: { yield: 0.4, risk: 0.3, liquidity: 0.2, audit: 0.1 },
    aggressive: { yield: 0.5, risk: 0.2, liquidity: 0.2, audit: 0.1 }
  }[config.riskTolerance];

  const yieldScore = Math.min(opportunity.apy / 50 * 100, 100);
  const riskScore = 100 - opportunity.riskScore;
  const liquidityScore = opportunity.liquidityScore || 50;
  const auditScore = opportunity.auditScore || 50;

  return (
    yieldScore * weights.yield +
    riskScore * weights.risk +
    liquidityScore * weights.liquidity +
    auditScore * weights.audit
  );
}

function calculateDiversificationScore(positions: YieldOpportunity[]): number {
  if (positions.length === 0) return 0;

  // Calculate Herfindahl-Hirschman Index for protocols
  const protocolWeights: Record<string, number> = {};
  const totalValue = positions.reduce((sum, pos) => sum.add(pos.position.value), BigNumber.from(0));

  positions.forEach(pos => {
    const weight = parseFloat(ethers.utils.formatEther(pos.position.value)) / 
                  parseFloat(ethers.utils.formatEther(totalValue));
    protocolWeights[pos.protocol] = (protocolWeights[pos.protocol] || 0) + weight;
  });

  const hhi = Object.values(protocolWeights).reduce((sum, weight) => sum + weight * weight, 0);
  
  // Convert HHI to diversification score (0-100, where 100 is perfectly diversified)
  return Math.max(0, (1 - hhi) * 100);
}

function calculateLiquidityScore(positions: YieldOpportunity[]): number {
  if (positions.length === 0) return 0;

  const totalValue = positions.reduce((sum, pos) => sum.add(pos.position.value), BigNumber.from(0));
  
  const weightedLiquidityScore = positions.reduce((sum, pos) => {
    const weight = parseFloat(ethers.utils.formatEther(pos.position.value)) / 
                  parseFloat(ethers.utils.formatEther(totalValue));
    return sum + (pos.liquidityScore * weight);
  }, 0);

  return weightedLiquidityScore;
}

function calculateProtocolDistribution(positions: YieldOpportunity[]): Record<string, number> {
  const distribution: Record<string, number> = {};
  const totalValue = positions.reduce((sum, pos) => sum.add(pos.position.value), BigNumber.from(0));

  positions.forEach(pos => {
    const weight = parseFloat(ethers.utils.formatEther(pos.position.value)) / 
                  parseFloat(ethers.utils.formatEther(totalValue)) * 100;
    distribution[pos.protocol] = (distribution[pos.protocol] || 0) + weight;
  });

  return distribution;
}

function calculateStrategyDistribution(positions: YieldOpportunity[]): Record<YieldStrategy, number> {
  const distribution: Record<YieldStrategy, number> = {};
  const totalValue = positions.reduce((sum, pos) => sum.add(pos.position.value), BigNumber.from(0));

  positions.forEach(pos => {
    const weight = parseFloat(ethers.utils.formatEther(pos.position.value)) / 
                  parseFloat(ethers.utils.formatEther(totalValue)) * 100;
    distribution[pos.strategy] = (distribution[pos.strategy] || 0) + weight;
  });

  return distribution;
}

function identifyConcentrationRisks(
  positions: YieldOpportunity[],
  protocolDistribution: Record<string, number>,
  strategyDistribution: Record<YieldStrategy, number>
): ConcentrationRisk[] {
  const risks: ConcentrationRisk[] = [];

  // Protocol concentration risks
  Object.entries(protocolDistribution).forEach(([protocol, percentage]) => {
    if (percentage > 50) {
      risks.push({
        type: 'protocol',
        entity: protocol,
        concentration: percentage,
        riskLevel: 'critical',
        recommendation: `Reduce ${protocol} exposure below 30%`
      });
    } else if (percentage > 30) {
      risks.push({
        type: 'protocol',
        entity: protocol,
        concentration: percentage,
        riskLevel: 'high',
        recommendation: `Consider diversifying ${protocol} exposure`
      });
    }
  });

  // Strategy concentration risks
  Object.entries(strategyDistribution).forEach(([strategy, percentage]) => {
    if (percentage > 60) {
      risks.push({
        type: 'strategy',
        entity: strategy,
        concentration: percentage,
        riskLevel: 'high',
        recommendation: `Diversify across different yield strategies`
      });
    }
  });

  return risks;
}

function calculateNewDiversificationScore(positions: OptimizedPosition[]): number {
  if (positions.length === 0) return 0;

  // Calculate HHI for optimized positions
  const hhi = positions.reduce((sum, pos) => {
    const weight = pos.allocatedPercentage / 100;
    return sum + weight * weight;
  }, 0);

  return Math.max(0, (1 - hhi) * 100);
}

function calculateSuccessProbability(
  steps: OptimizationStep[],
  totalGas: BigNumber,
  config: YieldOptimizationConfig
): number {
  let probability = 0.95; // Base success probability

  // Reduce probability based on complexity
  probability -= steps.length * 0.02; // 2% reduction per step

  // Reduce probability based on gas cost
  const gasThreshold = config.maxGasUsage;
  if (totalGas.gt(gasThreshold)) {
    probability -= 0.1; // 10% reduction for high gas
  }

  // Reduce probability for high-risk steps
  const highRiskSteps = steps.filter(step => step.risks.length > 2);
  probability -= highRiskSteps.length * 0.05;

  return Math.max(0.5, Math.min(0.95, probability));
}

function generateRiskMitigation(steps: OptimizationStep[], config: YieldOptimizationConfig): string[] {
  const mitigation: string[] = [];

  if (steps.length > 5) {
    mitigation.push('Execute optimization in phases to reduce complexity');
  }

  if (steps.some(step => step.action === 'withdraw')) {
    mitigation.push('Check lockup periods before executing withdrawals');
  }

  if (steps.some(step => step.risks.includes('Slippage'))) {
    mitigation.push('Set conservative slippage tolerances');
  }

  mitigation.push('Monitor gas prices and execute during low congestion');
  mitigation.push('Maintain emergency pause capability');

  return mitigation;
}

function generateOptimizationReasoning(
  currentPortfolio: PortfolioAnalysis,
  optimizedPositions: OptimizedPosition[],
  improvements: any,
  config: YieldOptimizationConfig
): string {
  const reasons: string[] = [];

  if (improvements.yieldImprovement > 1) {
    reasons.push(`Expected yield improvement of ${improvements.yieldImprovement.toFixed(2)}%`);
  }

  if (improvements.riskReduction > 5) {
    reasons.push(`Risk reduction of ${improvements.riskReduction.toFixed(1)} points`);
  }

  if (improvements.diversificationImprovement > 10) {
    reasons.push(`Improved diversification by ${improvements.diversificationImprovement.toFixed(1)} points`);
  }

  if (currentPortfolio.concentrationRisks.length > 0) {
    reasons.push('Addresses concentration risks in current portfolio');
  }

  const protocolCount = new Set(optimizedPositions.map(p => p.protocol)).size;
  if (protocolCount >= config.diversificationTarget) {
    reasons.push(`Achieves target diversification across ${protocolCount} protocols`);
  }

  return reasons.length > 0 
    ? `Optimization recommended: ${reasons.join(', ')}`
    : 'Standard yield optimization based on current market conditions';
}
