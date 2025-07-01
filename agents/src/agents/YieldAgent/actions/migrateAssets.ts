import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { AgentContext } from '../../../shared/types/agent';
import { YieldOpportunity } from '../../../shared/types/market';
import { YIELD_THRESHOLDS } from '../../../shared/constants/thresholds';
import { ProtocolProvider } from '../providers/protocolProvider';

export interface MigrationConfig {
  targetProtocols: string[];
  minImprovementThreshold: number; // Minimum % improvement required
  includeGasCosts: boolean;
  maxSlippage: number;
  phaseExecution?: boolean;
  emergencyOnly?: boolean;
}

export interface MigrationAnalysis {
  currentPositions: YieldOpportunity[];
  recommendedMigrations: MigrationRecommendation[];
  totalImprovementPotential: number;
  estimatedGasCosts: BigNumber;
  estimatedDuration: number;
  riskAssessment: {
    overallRisk: number;
    migrationRisks: string[];
    mitigationStrategies: string[];
  };
}

export interface MigrationRecommendation {
  fromPosition: YieldOpportunity;
  targetProtocol: string;
  targetPool: string;
  expectedImprovement: number;
  riskAdjustedImprovement: number;
  migrationSteps: MigrationStep[];
  estimatedCost: BigNumber;
  estimatedTime: number;
  priority: 'high' | 'medium' | 'low';
  reasoning: string;
}

export interface MigrationStep {
  step: number;
  action: 'withdraw' | 'swap' | 'deposit' | 'approve';
  protocol: string;
  description: string;
  estimatedGas: BigNumber;
  requiredApprovals: string[];
  risks: string[];
}

export interface MigrationResult {
  success: boolean;
  migratedPositions: Array<{
    fromPosition: YieldOpportunity;
    toPosition: YieldOpportunity;
    actualImprovement: number;
  }>;
  failedMigrations: Array<{
    position: YieldOpportunity;
    error: string;
    recoveryAction: string;
  }>;
  totalGasUsed: BigNumber;
  totalValueMigrated: BigNumber;
  actualImprovementAchieved: number;
  executionTime: number;
}

export async function migrateAssets(
  currentPositions: YieldOpportunity[],
  protocolProvider: ProtocolProvider,
  context: AgentContext,
  config: MigrationConfig
): Promise<MigrationAnalysis> {
  const startTime = Date.now();

  logger.info('Starting asset migration analysis', {
    agentId: context.agentId,
    currentPositions: currentPositions.length,
    targetProtocols: config.targetProtocols,
    minImprovement: config.minImprovementThreshold
  });

  try {
    // Step 1: Analyze current positions
    const positionAnalysis = await analyzeCurrentPositions(currentPositions, protocolProvider);

    // Step 2: Find migration opportunities
    const migrationOpportunities = await findMigrationOpportunities(
      currentPositions,
      config.targetProtocols,
      protocolProvider,
      context
    );

    // Step 3: Evaluate and rank migration opportunities
    const evaluatedMigrations = await evaluateMigrationOpportunities(
      migrationOpportunities,
      config,
      protocolProvider,
      context
    );

    // Step 4: Calculate gas costs and execution plan
    const migrationPlan = await createMigrationPlan(evaluatedMigrations, config, protocolProvider);

    // Step 5: Assess migration risks
    const riskAssessment = await assessMigrationRisks(migrationPlan, protocolProvider);

    // Step 6: Filter and finalize recommendations
    const finalRecommendations = filterRecommendations(migrationPlan, config, riskAssessment);

    const analysis: MigrationAnalysis = {
      currentPositions,
      recommendedMigrations: finalRecommendations,
      totalImprovementPotential: calculateTotalImprovement(finalRecommendations),
      estimatedGasCosts: calculateTotalGasCosts(finalRecommendations),
      estimatedDuration: calculateTotalDuration(finalRecommendations),
      riskAssessment
    };

    logger.info('Asset migration analysis completed', {
      agentId: context.agentId,
      recommendedMigrations: finalRecommendations.length,
      totalImprovement: analysis.totalImprovementPotential.toFixed(2),
      estimatedGasCosts: ethers.utils.formatEther(analysis.estimatedGasCosts),
      duration: Date.now() - startTime
    });

    return analysis;

  } catch (error) {
    logger.error('Asset migration analysis failed', {
      agentId: context.agentId,
      error: error instanceof Error ? error.message : String(error),
      duration: Date.now() - startTime
    });

    return {
      currentPositions,
      recommendedMigrations: [],
      totalImprovementPotential: 0,
      estimatedGasCosts: BigNumber.from(0),
      estimatedDuration: 0,
      riskAssessment: {
        overallRisk: 100,
        migrationRisks: ['Analysis failed'],
        mitigationStrategies: []
      }
    };
  }
}

async function analyzeCurrentPositions(
  positions: YieldOpportunity[],
  protocolProvider: ProtocolProvider
): Promise<Array<{
  position: YieldOpportunity;
  currentPerformance: number;
  healthScore: number;
  issuesIdentified: string[];
}>> {
  const analysis = [];

  for (const position of positions) {
    try {
      // Get current position performance
      const performance = await protocolProvider.getPositionPerformance(
        position.protocol,
        position.id,
        position.chainId
      );

      // Calculate health score
      const healthScore = calculatePositionHealthScore(position, performance);

      // Identify issues
      const issues = identifyPositionIssues(position, performance);

      analysis.push({
        position,
        currentPerformance: performance.realizedAPY || position.apy,
        healthScore,
        issuesIdentified: issues
      });

    } catch (error) {
      logger.warn('Failed to analyze position', {
        positionId: position.id,
        error: error instanceof Error ? error.message : String(error)
      });

      analysis.push({
        position,
        currentPerformance: position.apy,
        healthScore: 50,
        issuesIdentified: ['Analysis failed']
      });
    }
  }

  return analysis;
}

async function findMigrationOpportunities(
  currentPositions: YieldOpportunity[],
  targetProtocols: string[],
  protocolProvider: ProtocolProvider,
  context: AgentContext
): Promise<Array<{
  fromPosition: YieldOpportunity;
  targetOpportunities: YieldOpportunity[];
}>> {
  const opportunities = [];

  for (const position of currentPositions) {
    const targetOpportunities = [];

    for (const targetProtocol of targetProtocols) {
      try {
        // Skip if same protocol
        if (targetProtocol === position.protocol) continue;

        // Find compatible pools in target protocol
        const compatiblePools = await findCompatiblePools(
          position,
          targetProtocol,
          protocolProvider,
          context
        );

        targetOpportunities.push(...compatiblePools);

      } catch (error) {
        logger.debug('Failed to find opportunities in protocol', {
          targetProtocol,
          fromPosition: position.id,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    if (targetOpportunities.length > 0) {
      opportunities.push({
        fromPosition: position,
        targetOpportunities
      });
    }
  }

  return opportunities;
}

async function findCompatiblePools(
  currentPosition: YieldOpportunity,
  targetProtocol: string,
  protocolProvider: ProtocolProvider,
  context: AgentContext
): Promise<YieldOpportunity[]> {
  try {
    // Get available pools in target protocol
    const availablePools = await protocolProvider.getAvailablePools(targetProtocol, currentPosition.chainId);

    const compatiblePools = [];

    for (const pool of availablePools) {
      // Check if pool is compatible with current position
      const isCompatible = await isPoolCompatible(currentPosition, pool, protocolProvider);
      
      if (isCompatible) {
        // Get detailed pool information
        const poolDetails = await protocolProvider.getPoolDetails(targetProtocol, pool.id, currentPosition.chainId);
        
        if (poolDetails) {
          // Create yield opportunity from pool details
          const yieldOpportunity = await createYieldOpportunityFromPool(
            poolDetails,
            targetProtocol,
            currentPosition.chainId
          );
          
          compatiblePools.push(yieldOpportunity);
        }
      }
    }

    return compatiblePools;

  } catch (error) {
    logger.error('Failed to find compatible pools', {
      currentPosition: currentPosition.id,
      targetProtocol,
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function evaluateMigrationOpportunities(
  opportunities: Array<{
    fromPosition: YieldOpportunity;
    targetOpportunities: YieldOpportunity[];
  }>,
  config: MigrationConfig,
  protocolProvider: ProtocolProvider,
  context: AgentContext
): Promise<MigrationRecommendation[]> {
  const recommendations: MigrationRecommendation[] = [];

  for (const opportunity of opportunities) {
    const { fromPosition, targetOpportunities } = opportunity;

    for (const targetOpportunity of targetOpportunities) {
      try {
        // Calculate potential improvement
        const improvement = calculateImprovement(fromPosition, targetOpportunity);

        // Skip if improvement is below threshold
        if (improvement < config.minImprovementThreshold) {
          continue;
        }

        // Create migration steps
        const migrationSteps = await createMigrationSteps(
          fromPosition,
          targetOpportunity,
          protocolProvider,
          context
        );

        // Calculate migration costs
        const migrationCost = await calculateMigrationCost(migrationSteps, context);

        // Calculate risk-adjusted improvement
        const riskAdjustedImprovement = calculateRiskAdjustedImprovement(
          improvement,
          fromPosition,
          targetOpportunity,
          migrationCost
        );

        // Generate reasoning
        const reasoning = generateMigrationReasoning(
          fromPosition,
          targetOpportunity,
          improvement,
          migrationCost
        );

        const recommendation: MigrationRecommendation = {
          fromPosition,
          targetProtocol: targetOpportunity.protocol,
          targetPool: targetOpportunity.poolId,
          expectedImprovement: improvement,
          riskAdjustedImprovement,
          migrationSteps,
          estimatedCost: migrationCost,
          estimatedTime: calculateMigrationTime(migrationSteps),
          priority: determinePriority(riskAdjustedImprovement, fromPosition.riskScore),
          reasoning
        };

        recommendations.push(recommendation);

      } catch (error) {
        logger.warn('Failed to evaluate migration opportunity', {
          fromPosition: fromPosition.id,
          targetProtocol: targetOpportunity.protocol,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }
  }

  // Sort by risk-adjusted improvement
  return recommendations.sort((a, b) => b.riskAdjustedImprovement - a.riskAdjustedImprovement);
}

async function createMigrationSteps(
  fromPosition: YieldOpportunity,
  targetOpportunity: YieldOpportunity,
  protocolProvider: ProtocolProvider,
  context: AgentContext
): Promise<MigrationStep[]> {
  const steps: MigrationStep[] = [];
  let stepNumber = 1;

  try {
    // Step 1: Harvest any pending rewards
    if (fromPosition.claimableRewards && fromPosition.claimableRewards.length > 0) {
      steps.push({
        step: stepNumber++,
        action: 'withdraw',
        protocol: fromPosition.protocol,
        description: 'Harvest pending rewards',
        estimatedGas: BigNumber.from('150000'),
        requiredApprovals: [],
        risks: ['Transaction failure', 'Gas price volatility']
      });
    }

    // Step 2: Withdraw from current position
    steps.push({
      step: stepNumber++,
      action: 'withdraw',
      protocol: fromPosition.protocol,
      description: `Withdraw ${fromPosition.depositToken.symbol} from ${fromPosition.protocol}`,
      estimatedGas: await protocolProvider.estimateWithdrawGas(fromPosition.protocol, fromPosition.id),
      requiredApprovals: [],
      risks: ['Withdrawal fees', 'Slippage on withdrawal', 'Lock-up period constraints']
    });

    // Step 3: Token swap if different deposit tokens
    if (fromPosition.depositToken.symbol !== targetOpportunity.depositToken.symbol) {
      steps.push({
        step: stepNumber++,
        action: 'approve',
        protocol: 'DEX',
        description: `Approve ${fromPosition.depositToken.symbol} for swapping`,
        estimatedGas: BigNumber.from('50000'),
        requiredApprovals: [fromPosition.depositToken.address],
        risks: ['Approval transaction failure']
      });

      steps.push({
        step: stepNumber++,
        action: 'swap',
        protocol: 'DEX',
        description: `Swap ${fromPosition.depositToken.symbol} to ${targetOpportunity.depositToken.symbol}`,
        estimatedGas: BigNumber.from('200000'),
        requiredApprovals: [],
        risks: ['Slippage', 'MEV attacks', 'Price impact']
      });
    }

    // Step 4: Approve tokens for target protocol
    steps.push({
      step: stepNumber++,
      action: 'approve',
      protocol: targetOpportunity.protocol,
      description: `Approve ${targetOpportunity.depositToken.symbol} for ${targetOpportunity.protocol}`,
      estimatedGas: BigNumber.from('50000'),
      requiredApprovals: [targetOpportunity.depositToken.address],
      risks: ['Approval transaction failure']
    });

    // Step 5: Deposit to target protocol
    steps.push({
      step: stepNumber++,
      action: 'deposit',
      protocol: targetOpportunity.protocol,
      description: `Deposit ${targetOpportunity.depositToken.symbol} to ${targetOpportunity.protocol}`,
      estimatedGas: await protocolProvider.estimateDepositGas(targetOpportunity.protocol, targetOpportunity.id),
      requiredApprovals: [],
      risks: ['Deposit failure', 'Pool capacity limits', 'Smart contract risk']
    });

    return steps;

  } catch (error) {
    logger.error('Failed to create migration steps', {
      fromPosition: fromPosition.id,
      targetOpportunity: targetOpportunity.id,
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function createMigrationPlan(
  evaluatedMigrations: MigrationRecommendation[],
  config: MigrationConfig,
  protocolProvider: ProtocolProvider
): Promise<MigrationRecommendation[]> {
  // Sort migrations by priority and risk-adjusted improvement
  const sortedMigrations = evaluatedMigrations.sort((a, b) => {
    const priorityScore = { high: 3, medium: 2, low: 1 };
    const priorityDiff = priorityScore[b.priority] - priorityScore[a.priority];
    
    if (priorityDiff !== 0) return priorityDiff;
    
    return b.riskAdjustedImprovement - a.riskAdjustedImprovement;
  });

  // Filter migrations based on configuration
  const filteredMigrations = sortedMigrations.filter(migration => {
    // Emergency only filter
    if (config.emergencyOnly && migration.priority !== 'high') {
      return false;
    }

    // Minimum improvement threshold
    if (migration.riskAdjustedImprovement < config.minImprovementThreshold) {
      return false;
    }

    return true;
  });

  return filteredMigrations;
}

async function assessMigrationRisks(
  migrationPlan: MigrationRecommendation[],
  protocolProvider: ProtocolProvider
): Promise<{
  overallRisk: number;
  migrationRisks: string[];
  mitigationStrategies: string[];
}> {
  const risks: string[] = [];
  const mitigations: string[] = [];
  let overallRisk = 0;

  // Check each migration for risks
  for (const migration of migrationPlan) {
    // Protocol risks
    if (!migration.fromPosition.verified || migration.fromPosition.riskScore > 50) {
      risks.push('High risk source protocol');
      overallRisk += 15;
    }

    // Target protocol audit status
    const auditInfo = await protocolProvider.getProtocolAuditInfo(migration.targetProtocol);
    if (!auditInfo || auditInfo.auditCount === 0) {
      risks.push('Target protocol not audited');
      overallRisk += 20;
    }

    // Gas cost risks
    const gasCostPercentage = parseFloat(ethers.utils.formatEther(
      migration.estimatedCost.mul(100).div(migration.fromPosition.position.value)
    ));
    
    if (gasCostPercentage > 5) {
      risks.push('High gas costs relative to position size');
      overallRisk += 10;
    }

    // Liquidity risks
    if (migration.fromPosition.tvl.lt(ethers.utils.parseEther('1000000'))) {
      risks.push('Low liquidity in source pool');
      overallRisk += 10;
    }

    // Multiple protocol risk
    const uniqueProtocols = new Set(migrationPlan.map(m => m.targetProtocol));
    if (uniqueProtocols.size > 3) {
      risks.push('High protocol diversification complexity');
      overallRisk += 5;
    }
  }

  // Generate mitigation strategies
  if (risks.length > 0) {
    mitigations.push('Execute migrations in phases to reduce exposure');
    mitigations.push('Monitor gas prices and execute during low congestion');
    mitigations.push('Set conservative slippage tolerances');
    mitigations.push('Maintain emergency exit strategy');
    
    if (overallRisk > 50) {
      mitigations.push('Consider reducing position sizes for high-risk migrations');
    }
  }

  return {
    overallRisk: Math.min(overallRisk, 100),
    migrationRisks: risks,
    mitigationStrategies: mitigations
  };
}

// Helper functions
function calculateImprovement(fromPosition: YieldOpportunity, targetOpportunity: YieldOpportunity): number {
  const currentAPY = fromPosition.apy;
  const targetAPY = targetOpportunity.apy;
  
  return ((targetAPY - currentAPY) / currentAPY) * 100;
}

function calculateRiskAdjustedImprovement(
  improvement: number,
  fromPosition: YieldOpportunity,
  targetOpportunity: YieldOpportunity,
  migrationCost: BigNumber
): number {
  // Adjust for risk difference
  const riskAdjustment = (fromPosition.riskScore - targetOpportunity.riskScore) / 100;
  
  // Adjust for migration costs
  const positionValue = parseFloat(ethers.utils.formatEther(fromPosition.position.value));
  const costPercentage = parseFloat(ethers.utils.formatEther(migrationCost)) / positionValue * 100;
  
  return improvement + riskAdjustment - costPercentage;
}

async function calculateMigrationCost(steps: MigrationStep[], context: AgentContext): Promise<BigNumber> {
  const totalGas = steps.reduce((sum, step) => sum.add(step.estimatedGas), BigNumber.from(0));
  return totalGas.mul(context.gasPrice);
}

function calculateMigrationTime(steps: MigrationStep[]): number {
  // Estimate 2 minutes per step plus block confirmation times
  return steps.length * 120 + steps.length * 15; // seconds
}

function determinePriority(riskAdjustedImprovement: number, currentRiskScore: number): 'high' | 'medium' | 'low' {
  if (riskAdjustedImprovement > 10 || currentRiskScore > 80) return 'high';
  if (riskAdjustedImprovement > 5 || currentRiskScore > 60) return 'medium';
  return 'low';
}

function generateMigrationReasoning(
  fromPosition: YieldOpportunity,
  targetOpportunity: YieldOpportunity,
  improvement: number,
  migrationCost: BigNumber
): string {
  const reasons: string[] = [];

  if (improvement > 10) {
    reasons.push(`Significant APY improvement of ${improvement.toFixed(2)}%`);
  }

  if (fromPosition.riskScore > targetOpportunity.riskScore + 10) {
    reasons.push('Reduced risk profile in target protocol');
  }

  if (targetOpportunity.auditScore && targetOpportunity.auditScore > (fromPosition.auditScore || 0) + 20) {
    reasons.push('Better audit score and security in target protocol');
  }

  if (targetOpportunity.tvl.gt(fromPosition.tvl.mul(2))) {
    reasons.push('Higher liquidity in target protocol');
  }

  const costPercentage = parseFloat(ethers.utils.formatEther(
    migrationCost.mul(100).div(fromPosition.position.value)
  ));

  if (costPercentage < 1) {
    reasons.push('Low migration costs');
  }

  return reasons.length > 0 ? reasons.join(', ') : 'Standard yield optimization migration';
}

async function isPoolCompatible(
  currentPosition: YieldOpportunity,
  targetPool: any,
  protocolProvider: ProtocolProvider
): Promise<boolean> {
  try {
    // Check if deposit token is compatible
    const compatibleTokens = await protocolProvider.getCompatibleTokens(
      currentPosition.depositToken.symbol,
      targetPool.chainId
    );

    const isTokenCompatible = compatibleTokens.includes(targetPool.depositToken?.symbol) ||
                             targetPool.acceptedTokens?.includes(currentPosition.depositToken.symbol);

    // Check minimum deposit requirements
    const hasMinimumValue = !targetPool.minimumDeposit || 
                           currentPosition.position.value.gte(targetPool.minimumDeposit);

    // Check if pool is active and not deprecated
    const isActive = targetPool.status === 'active' && !targetPool.deprecated;

    return isTokenCompatible && hasMinimumValue && isActive;

  } catch (error) {
    logger.debug('Pool compatibility check failed', {
      currentPosition: currentPosition.id,
      targetPool: targetPool.id,
      error: error instanceof Error ? error.message : String(error)
    });

    return false;
  }
}

async function createYieldOpportunityFromPool(
  poolDetails: any,
  protocol: string,
  chainId: number
): Promise<YieldOpportunity> {
  // This would create a full YieldOpportunity object from pool details
  // Implementation similar to what's in findYieldOpportunities.ts
  
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
    riskScore: 50, // Would be calculated
    auditScore: 50,
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

function calculatePositionHealthScore(position: YieldOpportunity, performance: any): number {
  let score = 50; // Base score

  // APY performance
  if (performance.realizedAPY > position.apy * 0.9) score += 20;
  else if (performance.realizedAPY < position.apy * 0.7) score -= 20;

  // Risk score
  if (position.riskScore < 30) score += 15;
  else if (position.riskScore > 70) score -= 15;

  // Liquidity
  const tvlUsd = parseFloat(ethers.utils.formatEther(position.tvl));
  if (tvlUsd > 10000000) score += 10; // >$10M
  else if (tvlUsd < 1000000) score -= 10; // <$1M

  // Audit status
  if (position.verified && position.auditScore && position.auditScore > 70) score += 15;

  return Math.max(0, Math.min(100, score));
}

function identifyPositionIssues(position: YieldOpportunity, performance: any): string[] {
  const issues: string[] = [];

  if (performance.realizedAPY < position.apy * 0.8) {
    issues.push('Underperforming expected APY');
  }

  if (position.riskScore > 70) {
    issues.push('High risk score');
  }

  if (!position.verified) {
    issues.push('Unverified protocol');
  }

  if (position.tvl.lt(ethers.utils.parseEther('1000000'))) {
    issues.push('Low liquidity');
  }

  if (position.lockupPeriod > 30 * 24 * 60 * 60) { // 30 days
    issues.push('Long lockup period');
  }

  return issues;
}

function filterRecommendations(
  recommendations: MigrationRecommendation[],
  config: MigrationConfig,
  riskAssessment: any
): MigrationRecommendation[] {
  return recommendations.filter(recommendation => {
    // Filter by improvement threshold
    if (recommendation.riskAdjustedImprovement < config.minImprovementThreshold) {
      return false;
    }

    // Filter high-risk migrations if overall risk is high
    if (riskAssessment.overallRisk > 70 && recommendation.priority === 'low') {
      return false;
    }

    return true;
  });
}

function calculateTotalImprovement(recommendations: MigrationRecommendation[]): number {
  if (recommendations.length === 0) return 0;
  
  return recommendations.reduce((sum, rec) => {
    const positionWeight = parseFloat(ethers.utils.formatEther(rec.fromPosition.position.value));
    return sum + (rec.riskAdjustedImprovement * positionWeight);
  }, 0) / recommendations.reduce((sum, rec) => {
    return sum + parseFloat(ethers.utils.formatEther(rec.fromPosition.position.value));
  }, 0);
}

function calculateTotalGasCosts(recommendations: MigrationRecommendation[]): BigNumber {
  return recommendations.reduce((sum, rec) => sum.add(rec.estimatedCost), BigNumber.from(0));
}

function calculateTotalDuration(recommendations: MigrationRecommendation[]): number {
  // Assume migrations can be done in parallel by protocol
  const protocolGroups = recommendations.reduce((groups, rec) => {
    const protocol = rec.fromPosition.protocol;
    if (!groups[protocol]) groups[protocol] = [];
    groups[protocol].push(rec.estimatedTime);
    return groups;
  }, {} as Record<string, number[]>);

  // Return maximum time across protocol groups
  return Math.max(...Object.values(protocolGroups).map(times => 
    times.reduce((sum, time) => sum + time, 0)
  ));
}
