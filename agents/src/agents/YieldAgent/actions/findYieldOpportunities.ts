import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { AgentContext } from '../../../shared/types/agent';
import { YieldOpportunity, YieldStrategy } from '../../../shared/types/market';
import { YIELD_THRESHOLDS } from '../../../shared/constants/thresholds';
import { ProtocolProvider } from '../providers/protocolProvider';

export interface OpportunitySearchConfig {
  supportedProtocols: string[];
  supportedChains: number[];
  minAPY: number;
  maxRiskScore: number;
  maxPositionSize: BigNumber;
  includeNewProtocols?: boolean;
  tokenFilters?: string[];
  liquidityThreshold?: BigNumber;
}

export interface OpportunityMetrics {
  totalScanned: number;
  opportunitiesFound: number;
  averageAPY: number;
  highestAPY: number;
  averageRiskScore: number;
  protocolDistribution: Record<string, number>;
  chainDistribution: Record<number, number>;
}

export async function findYieldOpportunities(
  protocolProvider: ProtocolProvider,
  context: AgentContext,
  config: OpportunitySearchConfig
): Promise<YieldOpportunity[]> {
  const startTime = Date.now();
  const opportunities: YieldOpportunity[] = [];

  logger.info('Starting yield opportunity discovery', {
    agentId: context.agentId,
    supportedProtocols: config.supportedProtocols,
    supportedChains: config.supportedChains,
    minAPY: config.minAPY,
    maxRiskScore: config.maxRiskScore
  });

  try {
    // Step 1: Scan each protocol on each supported chain
    for (const protocol of config.supportedProtocols) {
      for (const chainId of config.supportedChains) {
        try {
          const protocolOpportunities = await scanProtocolOpportunities(
            protocolProvider,
            protocol,
            chainId,
            config,
            context
          );

          opportunities.push(...protocolOpportunities);

        } catch (error) {
          logger.warn('Failed to scan protocol opportunities', {
            protocol,
            chainId,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }
    }

    // Step 2: Filter and rank opportunities
    const filteredOpportunities = filterOpportunities(opportunities, config);
    const rankedOpportunities = rankOpportunities(filteredOpportunities, config);

    // Step 3: Validate opportunity data with Chainlink price feeds
    const validatedOpportunities = await validateOpportunityPrices(
      rankedOpportunities,
      protocolProvider,
      context
    );

    // Step 4: Calculate opportunity metrics
    const metrics = calculateOpportunityMetrics(validatedOpportunities);

    logger.info('Yield opportunity discovery completed', {
      agentId: context.agentId,
      totalScanned: metrics.totalScanned,
      opportunitiesFound: metrics.opportunitiesFound,
      averageAPY: metrics.averageAPY.toFixed(2),
      highestAPY: metrics.highestAPY.toFixed(2),
      duration: Date.now() - startTime
    });

    return validatedOpportunities;

  } catch (error) {
    logger.error('Yield opportunity discovery failed', {
      agentId: context.agentId,
      error: error instanceof Error ? error.message : String(error),
      duration: Date.now() - startTime
    });

    return [];
  }
}

async function scanProtocolOpportunities(
  protocolProvider: ProtocolProvider,
  protocol: string,
  chainId: number,
  config: OpportunitySearchConfig,
  context: AgentContext
): Promise<YieldOpportunity[]> {
  const opportunities: YieldOpportunity[] = [];

  try {
    logger.debug('Scanning protocol opportunities', { protocol, chainId });

    // Get available pools/vaults for the protocol
    const pools = await protocolProvider.getAvailablePools(protocol, chainId);

    for (const pool of pools) {
      try {
        // Skip if pool doesn't meet minimum criteria
        if (!meetsBasicCriteria(pool, config)) {
          continue;
        }

        // Get detailed pool information
        const poolDetails = await protocolProvider.getPoolDetails(protocol, pool.id, chainId);

        if (!poolDetails) {
          continue;
        }

        // Calculate yield metrics
        const yieldMetrics = await calculateYieldMetrics(poolDetails, protocolProvider, chainId);

        // Assess protocol and pool risks
        const riskAssessment = await assessPoolRisk(poolDetails, protocolProvider, protocol, chainId);

        // Create yield opportunity
        const opportunity: YieldOpportunity = {
          id: `${protocol}_${pool.id}_${chainId}`,
          protocol,
          chainId,
          poolId: pool.id,
          tokenPair: poolDetails.tokenPair || poolDetails.token || 'Unknown',
          strategy: determineYieldStrategy(protocol, poolDetails),
          apy: yieldMetrics.currentAPY,
          apyHistory: yieldMetrics.historicalAPY,
          tvl: poolDetails.tvl,
          liquidity: poolDetails.availableLiquidity || poolDetails.tvl,
          fees: poolDetails.fees || { deposit: 0, withdrawal: 0, performance: 0 },
          riskScore: riskAssessment.overallRisk,
          auditScore: riskAssessment.auditScore,
          liquidityScore: riskAssessment.liquidityScore,
          impermanentLossRisk: riskAssessment.impermanentLossRisk,
          lockupPeriod: poolDetails.lockupPeriod || 0,
          minimumDeposit: poolDetails.minimumDeposit || BigNumber.from(0),
          maximumDeposit: poolDetails.maximumDeposit,
          autoCompound: poolDetails.autoCompound || false,
          verified: riskAssessment.isVerified,
          contractAddress: poolDetails.contractAddress,
          depositToken: {
            address: poolDetails.depositToken.address,
            symbol: poolDetails.depositToken.symbol,
            name: poolDetails.depositToken.name,
            decimals: poolDetails.depositToken.decimals,
            chainId: chainId,
            tags: [],
            isStable: isStablecoin(poolDetails.depositToken.symbol),
            isNative: poolDetails.depositToken.address === ethers.constants.AddressZero
          },
          rewardTokens: poolDetails.rewardTokens || [],
          lastUpdated: Date.now(),
          lastHarvest: poolDetails.lastHarvest,
          claimableRewards: poolDetails.claimableRewards || [],
          totalEarned: BigNumber.from(0),
          position: poolDetails.userPosition || {
            amount: BigNumber.from(0),
            value: BigNumber.from(0),
            shares: BigNumber.from(0),
            depositedAt: 0,
            lastAction: 0
          }
        };

        // Validate opportunity meets all criteria
        if (isValidOpportunity(opportunity, config)) {
          opportunities.push(opportunity);
        }

      } catch (error) {
        logger.debug('Failed to process pool', {
          protocol,
          poolId: pool.id,
          chainId,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    logger.debug('Protocol scan completed', {
      protocol,
      chainId,
      poolsScanned: pools.length,
      opportunitiesFound: opportunities.length
    });

    return opportunities;

  } catch (error) {
    logger.error('Protocol scanning failed', {
      protocol,
      chainId,
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function calculateYieldMetrics(
  poolDetails: any,
  protocolProvider: ProtocolProvider,
  chainId: number
): Promise<{
  currentAPY: number;
  historicalAPY: number[];
  projectedAPY: number;
  volatility: number;
}> {
  try {
    // Get current APY from protocol
    const currentAPY = poolDetails.apy || 0;

    // Get historical APY data (30 days)
    const historicalAPY = await protocolProvider.getHistoricalAPY(
      poolDetails.protocol,
      poolDetails.id,
      chainId,
      30
    );

    // Calculate projected APY based on current conditions
    const projectedAPY = await calculateProjectedAPY(poolDetails, protocolProvider, chainId);

    // Calculate APY volatility
    const volatility = historicalAPY.length > 1 
      ? calculateAPYVolatility(historicalAPY)
      : 0;

    return {
      currentAPY,
      historicalAPY,
      projectedAPY,
      volatility
    };

  } catch (error) {
    logger.warn('Failed to calculate yield metrics', {
      poolId: poolDetails.id,
      error: error instanceof Error ? error.message : String(error)
    });

    return {
      currentAPY: poolDetails.apy || 0,
      historicalAPY: [],
      projectedAPY: poolDetails.apy || 0,
      volatility: 0
    };
  }
}

async function calculateProjectedAPY(
  poolDetails: any,
  protocolProvider: ProtocolProvider,
  chainId: number
): Promise<number> {
  try {
    // Get current protocol metrics
    const protocolMetrics = await protocolProvider.getProtocolMetrics(poolDetails.protocol, chainId);
    
    // Get current market conditions
    const marketConditions = await protocolProvider.getMarketConditions(chainId);

    // Calculate base yield from underlying assets
    let baseYield = 0;
    
    if (poolDetails.underlyingAssets) {
      for (const asset of poolDetails.underlyingAssets) {
        const assetYield = await getAssetBaseYield(asset, protocolProvider, chainId);
        baseYield += assetYield * (asset.weight || 1);
      }
    }

    // Add protocol-specific rewards
    const protocolRewards = calculateProtocolRewards(poolDetails, protocolMetrics);

    // Adjust for market conditions
    const marketAdjustment = calculateMarketAdjustment(marketConditions);

    // Calculate projected APY
    const projectedAPY = (baseYield + protocolRewards) * marketAdjustment;

    return Math.max(0, projectedAPY);

  } catch (error) {
    logger.warn('Failed to calculate projected APY', {
      poolId: poolDetails.id,
      error: error instanceof Error ? error.message : String(error)
    });

    return poolDetails.apy || 0;
  }
}

async function assessPoolRisk(
  poolDetails: any,
  protocolProvider: ProtocolProvider,
  protocol: string,
  chainId: number
): Promise<{
  overallRisk: number;
  auditScore: number;
  liquidityScore: number;
  impermanentLossRisk: number;
  isVerified: boolean;
}> {
  try {
    // Get protocol audit information
    const auditInfo = await protocolProvider.getProtocolAuditInfo(protocol);
    const auditScore = calculateAuditScore(auditInfo);

    // Assess liquidity risk
    const liquidityScore = calculateLiquidityScore(poolDetails);

    // Calculate impermanent loss risk for LP tokens
    const impermanentLossRisk = await calculateImpermanentLossRisk(poolDetails, protocolProvider, chainId);

    // Check if protocol is verified/whitelisted
    const isVerified = await protocolProvider.isProtocolVerified(protocol);

    // Calculate overall risk score
    const riskFactors = {
      auditRisk: (100 - auditScore) * 0.3,
      liquidityRisk: (100 - liquidityScore) * 0.25,
      impermanentLossRisk: impermanentLossRisk * 0.2,
      protocolRisk: isVerified ? 0 : 20,
      newPoolRisk: poolDetails.age < 30 ? 15 : 0, // 30 days minimum
      concentrationRisk: calculateConcentrationRisk(poolDetails) * 0.25
    };

    const overallRisk = Object.values(riskFactors).reduce((sum, risk) => sum + risk, 0);

    return {
      overallRisk: Math.min(overallRisk, 100),
      auditScore,
      liquidityScore,
      impermanentLossRisk,
      isVerified
    };

  } catch (error) {
    logger.warn('Failed to assess pool risk', {
      protocol,
      poolId: poolDetails.id,
      error: error instanceof Error ? error.message : String(error)
    });

    return {
      overallRisk: 75, // Conservative default
      auditScore: 50,
      liquidityScore: 50,
      impermanentLossRisk: 50,
      isVerified: false
    };
  }
}

function determineYieldStrategy(protocol: string, poolDetails: any): YieldStrategy {
  // Determine strategy based on protocol and pool characteristics
  if (protocol.toLowerCase().includes('compound') || protocol.toLowerCase().includes('aave')) {
    return 'lending';
  }
  
  if (protocol.toLowerCase().includes('uniswap') || protocol.toLowerCase().includes('sushiswap')) {
    return 'liquidity_provision';
  }
  
  if (protocol.toLowerCase().includes('yearn') || protocol.toLowerCase().includes('harvest')) {
    return 'yield_farming';
  }
  
  if (poolDetails.tokenPair && poolDetails.tokenPair.includes('/')) {
    return 'liquidity_provision';
  }
  
  if (poolDetails.autoCompound) {
    return 'auto_compound';
  }

  return 'staking';
}

function meetsBasicCriteria(pool: any, config: OpportunitySearchConfig): boolean {
  // Check minimum TVL
  if (config.liquidityThreshold && pool.tvl.lt(config.liquidityThreshold)) {
    return false;
  }

  // Check token filters
  if (config.tokenFilters && config.tokenFilters.length > 0) {
    const poolTokens = pool.tokens || [pool.token];
    const hasMatchingToken = poolTokens.some((token: string) => 
      config.tokenFilters!.includes(token.toUpperCase())
    );
    
    if (!hasMatchingToken) {
      return false;
    }
  }

  // Check minimum APY
  if (pool.apy < config.minAPY) {
    return false;
  }

  return true;
}

function isValidOpportunity(opportunity: YieldOpportunity, config: OpportunitySearchConfig): boolean {
  // APY check
  if (opportunity.apy < config.minAPY) {
    return false;
  }

  // Risk score check
  if (opportunity.riskScore > config.maxRiskScore) {
    return false;
  }

  // TVL check
  if (config.liquidityThreshold && opportunity.tvl.lt(config.liquidityThreshold)) {
    return false;
  }

  // Maximum deposit check
  if (opportunity.maximumDeposit && opportunity.maximumDeposit.lt(config.maxPositionSize.div(10))) {
    return false;
  }

  return true;
}

function filterOpportunities(
  opportunities: YieldOpportunity[],
  config: OpportunitySearchConfig
): YieldOpportunity[] {
  return opportunities.filter(opportunity => {
    // Remove duplicates (same pool on same chain)
    const isDuplicate = opportunities.some(other => 
      other !== opportunity &&
      other.protocol === opportunity.protocol &&
      other.poolId === opportunity.poolId &&
      other.chainId === opportunity.chainId
    );

    return !isDuplicate && isValidOpportunity(opportunity, config);
  });
}

function rankOpportunities(
  opportunities: YieldOpportunity[],
  config: OpportunitySearchConfig
): YieldOpportunity[] {
  return opportunities.sort((a, b) => {
    // Calculate risk-adjusted yield score
    const scoreA = calculateOpportunityScore(a, config);
    const scoreB = calculateOpportunityScore(b, config);
    
    return scoreB - scoreA; // Descending order
  });
}

function calculateOpportunityScore(opportunity: YieldOpportunity, config: OpportunitySearchConfig): number {
  // Multi-factor scoring system
  const weights = {
    apy: 0.4,
    risk: 0.25,
    liquidity: 0.2,
    audit: 0.15
  };

  // Normalize scores to 0-100
  const apyScore = Math.min(opportunity.apy / 50 * 100, 100); // Cap at 50% APY
  const riskScore = 100 - opportunity.riskScore; // Invert risk (lower risk = higher score)
  const liquidityScore = opportunity.liquidityScore || 50;
  const auditScore = opportunity.auditScore || 50;

  return (
    apyScore * weights.apy +
    riskScore * weights.risk +
    liquidityScore * weights.liquidity +
    auditScore * weights.audit
  );
}

async function validateOpportunityPrices(
  opportunities: YieldOpportunity[],
  protocolProvider: ProtocolProvider,
  context: AgentContext
): Promise<YieldOpportunity[]> {
  const validatedOpportunities: YieldOpportunity[] = [];

  for (const opportunity of opportunities) {
    try {
      // Validate token prices with Chainlink price feeds
      const tokenPrices = await protocolProvider.validateTokenPrices(
        opportunity.depositToken.symbol,
        opportunity.chainId
      );

      if (tokenPrices.isValid) {
        // Update opportunity with validated data
        opportunity.lastUpdated = Date.now();
        validatedOpportunities.push(opportunity);
      } else {
        logger.warn('Opportunity price validation failed', {
          opportunityId: opportunity.id,
          token: opportunity.depositToken.symbol
        });
      }

    } catch (error) {
      logger.warn('Failed to validate opportunity prices', {
        opportunityId: opportunity.id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  return validatedOpportunities;
}

function calculateOpportunityMetrics(opportunities: YieldOpportunity[]): OpportunityMetrics {
  if (opportunities.length === 0) {
    return {
      totalScanned: 0,
      opportunitiesFound: 0,
      averageAPY: 0,
      highestAPY: 0,
      averageRiskScore: 0,
      protocolDistribution: {},
      chainDistribution: {}
    };
  }

  const apys = opportunities.map(o => o.apy);
  const riskScores = opportunities.map(o => o.riskScore);

  const protocolDistribution: Record<string, number> = {};
  const chainDistribution: Record<number, number> = {};

  opportunities.forEach(opportunity => {
    protocolDistribution[opportunity.protocol] = (protocolDistribution[opportunity.protocol] || 0) + 1;
    chainDistribution[opportunity.chainId] = (chainDistribution[opportunity.chainId] || 0) + 1;
  });

  return {
    totalScanned: opportunities.length,
    opportunitiesFound: opportunities.length,
    averageAPY: apys.reduce((sum, apy) => sum + apy, 0) / apys.length,
    highestAPY: Math.max(...apys),
    averageRiskScore: riskScores.reduce((sum, risk) => sum + risk, 0) / riskScores.length,
    protocolDistribution,
    chainDistribution
  };
}

// Helper functions
function calculateAPYVolatility(historicalAPY: number[]): number {
  if (historicalAPY.length < 2) return 0;

  const mean = historicalAPY.reduce((sum, apy) => sum + apy, 0) / historicalAPY.length;
  const variance = historicalAPY.reduce((sum, apy) => sum + Math.pow(apy - mean, 2), 0) / historicalAPY.length;
  
  return Math.sqrt(variance);
}

async function getAssetBaseYield(
  asset: any,
  protocolProvider: ProtocolProvider,
  chainId: number
): Promise<number> {
  try {
    // Get base yield for individual assets (e.g., staking rewards, lending rates)
    return await protocolProvider.getAssetBaseYield(asset.symbol, chainId);
  } catch {
    return 0;
  }
}

function calculateProtocolRewards(poolDetails: any, protocolMetrics: any): number {
  // Calculate additional protocol-specific rewards
  const baseRewardAPY = protocolMetrics.averageRewardAPY || 0;
  const poolMultiplier = poolDetails.rewardMultiplier || 1;
  
  return baseRewardAPY * poolMultiplier;
}

function calculateMarketAdjustment(marketConditions: any): number {
  // Adjust yield projections based on market conditions
  const volatility = marketConditions.volatility || 0.5;
  const trend = marketConditions.trend || 'neutral';
  
  let adjustment = 1.0;
  
  if (trend === 'bullish') {
    adjustment += 0.1;
  } else if (trend === 'bearish') {
    adjustment -= 0.1;
  }
  
  // Reduce adjustment in high volatility
  adjustment *= (1 - volatility * 0.2);
  
  return Math.max(0.5, Math.min(1.5, adjustment));
}

function calculateAuditScore(auditInfo: any): number {
  if (!auditInfo) return 25; // Low score for unaudited

  let score = 50; // Base score for having audits

  // High-tier audit firms
  const topAuditFirms = ['trail-of-bits', 'consensys', 'openzeppelin', 'quantstamp', 'certik'];
  if (auditInfo.firms?.some((firm: string) => topAuditFirms.includes(firm.toLowerCase()))) {
    score += 30;
  }

  // Multiple audits
  if (auditInfo.auditCount > 1) {
    score += 15;
  }

  // Recent audits
  const daysSinceLastAudit = (Date.now() - auditInfo.lastAuditDate) / (1000 * 60 * 60 * 24);
  if (daysSinceLastAudit < 180) { // 6 months
    score += 5;
  }

  return Math.min(score, 100);
}

function calculateLiquidityScore(poolDetails: any): number {
  const tvl = parseFloat(ethers.utils.formatEther(poolDetails.tvl));
  
  let score = 0;
  
  if (tvl > 100000000) score = 100; // >$100M
  else if (tvl > 50000000) score = 90; // >$50M
  else if (tvl > 10000000) score = 80; // >$10M
  else if (tvl > 5000000) score = 70; // >$5M
  else if (tvl > 1000000) score = 60; // >$1M
  else if (tvl > 500000) score = 50; // >$500k
  else if (tvl > 100000) score = 40; // >$100k
  else score = 20; // <$100k
  
  return score;
}

async function calculateImpermanentLossRisk(
  poolDetails: any,
  protocolProvider: ProtocolProvider,
  chainId: number
): Promise<number> {
  // Only relevant for liquidity provision strategies
  if (!poolDetails.tokenPair || !poolDetails.tokenPair.includes('/')) {
    return 0;
  }

  try {
    const [token0, token1] = poolDetails.tokenPair.split('/');
    
    // Get correlation between tokens
    const correlation = await protocolProvider.getTokenCorrelation(token0, token1, chainId);
    
    // Get historical volatility
    const volatility0 = await protocolProvider.getTokenVolatility(token0, chainId);
    const volatility1 = await protocolProvider.getTokenVolatility(token1, chainId);
    
    // Calculate impermanent loss risk based on correlation and volatility
    const avgVolatility = (volatility0 + volatility1) / 2;
    const ilRisk = avgVolatility * (1 - Math.abs(correlation)) * 100;
    
    return Math.min(ilRisk, 100);

  } catch (error) {
    // Default risk for unknown token pairs
    return 30;
  }
}

function calculateConcentrationRisk(poolDetails: any): number {
  // Check if pool is overly concentrated in specific assets or strategies
  if (poolDetails.assets && poolDetails.assets.length > 0) {
    const maxWeight = Math.max(...poolDetails.assets.map((asset: any) => asset.weight || 0));
    
    if (maxWeight > 0.8) return 80; // Very concentrated
    if (maxWeight > 0.6) return 60; // Highly concentrated
    if (maxWeight > 0.4) return 40; // Moderately concentrated
    
    return 20; // Well diversified
  }
  
  return 30; // Unknown concentration
}

function isStablecoin(symbol: string): boolean {
  const stablecoins = ['USDC', 'USDT', 'DAI', 'BUSD', 'FRAX', 'LUSD', 'MIM', 'TUSD'];
  return stablecoins.includes(symbol.toUpperCase());
}
