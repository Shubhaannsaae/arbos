import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { Portfolio, AllocationTarget } from '../../../shared/types/market';
import { MarketDataProvider } from '../providers/marketDataProvider';

export interface OptimizationConfig {
  optimizationObjective: 'sharpe_ratio' | 'return' | 'risk' | 'sortino_ratio';
  constraints: {
    maxPositions: number;
    minPositionSize: BigNumber;
    maxPositionSize: BigNumber;
    maxSectorExposure?: number;
    maxSingleAssetExposure?: number;
  };
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  timeHorizon: 'short' | 'medium' | 'long';
  includeAlternatives?: boolean;
}

export interface OptimizationResult {
  success: boolean;
  optimizedAllocation: AllocationTarget[];
  expectedMetrics: {
    expectedReturn: number;
    expectedVolatility: number;
    expectedSharpeRatio: number;
    expectedSortinoRatio: number;
    maxDrawdown: number;
    valueAtRisk: number;
  };
  comparisonMetrics: {
    currentReturn: number;
    currentVolatility: number;
    currentSharpeRatio: number;
    improvement: {
      returnImprovement: number;
      riskReduction: number;
      sharpeImprovement: number;
    };
  };
  expectedImprovement: number;
  confidenceLevel: number;
  reasoning: string;
}

export async function optimizeAllocation(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider,
  config: OptimizationConfig
): Promise<OptimizationResult> {
  const startTime = Date.now();
  
  logger.info('Starting portfolio optimization', {
    portfolioId: portfolio.id,
    objective: config.optimizationObjective,
    riskTolerance: config.riskTolerance,
    constraints: config.constraints
  });

  try {
    // Step 1: Gather market data and asset universe
    const assetUniverse = await buildAssetUniverse(portfolio, marketDataProvider, config);
    
    // Step 2: Calculate historical returns and correlations
    const historicalData = await calculateHistoricalMetrics(assetUniverse, marketDataProvider);
    
    // Step 3: Run modern portfolio theory optimization
    const optimizedWeights = await runMPTOptimization(
      historicalData,
      config.optimizationObjective,
      config.constraints,
      config.riskTolerance
    );

    // Step 4: Convert weights to allocation targets
    const optimizedAllocation = await convertWeightsToAllocation(
      optimizedWeights,
      assetUniverse,
      portfolio.totalValue
    );

    // Step 5: Calculate expected performance metrics
    const expectedMetrics = await calculateExpectedMetrics(
      optimizedAllocation,
      historicalData,
      marketDataProvider
    );

    // Step 6: Compare with current portfolio
    const comparisonMetrics = await compareWithCurrentPortfolio(
      portfolio,
      expectedMetrics,
      historicalData
    );

    // Step 7: Generate optimization reasoning
    const reasoning = generateOptimizationReasoning(
      config,
      optimizedAllocation,
      expectedMetrics,
      comparisonMetrics
    );

    // Step 8: Calculate confidence level based on data quality and market conditions
    const confidenceLevel = calculateOptimizationConfidence(
      historicalData,
      assetUniverse,
      marketDataProvider
    );

    const expectedImprovement = (
      comparisonMetrics.improvement.returnImprovement + 
      comparisonMetrics.improvement.sharpeImprovement
    ) / 2;

    logger.info('Portfolio optimization completed', {
      portfolioId: portfolio.id,
      expectedReturn: expectedMetrics.expectedReturn.toFixed(2),
      expectedVolatility: expectedMetrics.expectedVolatility.toFixed(2),
      expectedSharpeRatio: expectedMetrics.expectedSharpeRatio.toFixed(3),
      confidenceLevel: confidenceLevel.toFixed(2),
      duration: Date.now() - startTime
    });

    return {
      success: true,
      optimizedAllocation,
      expectedMetrics,
      comparisonMetrics,
      expectedImprovement,
      confidenceLevel,
      reasoning
    };

  } catch (error) {
    logger.error('Portfolio optimization failed', {
      portfolioId: portfolio.id,
      error: error instanceof Error ? error.message : String(error),
      duration: Date.now() - startTime
    });

    throw error;
  }
}

async function buildAssetUniverse(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider,
  config: OptimizationConfig
): Promise<Array<{
  symbol: string;
  name: string;
  sector: string;
  marketCap: BigNumber;
  liquidity: BigNumber;
  eligible: boolean;
}>> {
  const universe: any[] = [];

  try {
    // Start with current portfolio assets
    for (const position of portfolio.positions) {
      const assetData = await marketDataProvider.getAssetMetadata(position.token.symbol);
      universe.push({
        symbol: position.token.symbol,
        name: position.token.name,
        sector: assetData.sector || 'Unknown',
        marketCap: assetData.marketCap || BigNumber.from(0),
        liquidity: assetData.averageDailyVolume || BigNumber.from(0),
        eligible: true
      });
    }

    // Add additional assets based on criteria
    const additionalAssets = await marketDataProvider.getTopAssetsByMarketCap(50);
    
    for (const asset of additionalAssets) {
      // Skip if already in universe
      if (universe.find(u => u.symbol === asset.symbol)) continue;

      // Apply eligibility filters
      const eligible = await isAssetEligible(asset, config.constraints, marketDataProvider);
      
      if (eligible) {
        universe.push({
          symbol: asset.symbol,
          name: asset.name,
          sector: asset.sector || 'Unknown',
          marketCap: asset.marketCap,
          liquidity: asset.averageDailyVolume,
          eligible: true
        });
      }
    }

    // Add alternative assets if enabled
    if (config.includeAlternatives) {
      const alternatives = await getAlternativeAssets(marketDataProvider);
      universe.push(...alternatives.filter(alt => 
        isAssetEligible(alt, config.constraints, marketDataProvider)
      ));
    }

    logger.debug('Asset universe built', {
      totalAssets: universe.length,
      eligibleAssets: universe.filter(a => a.eligible).length
    });

    return universe;

  } catch (error) {
    logger.error('Failed to build asset universe', {
      error: error instanceof Error ? error.message : String(error)
    });

    // Return current portfolio assets as fallback
    return portfolio.positions.map(position => ({
      symbol: position.token.symbol,
      name: position.token.name,
      sector: 'Unknown',
      marketCap: BigNumber.from(0),
      liquidity: BigNumber.from(0),
      eligible: true
    }));
  }
}

async function calculateHistoricalMetrics(
  assetUniverse: any[],
  marketDataProvider: MarketDataProvider
): Promise<{
  returns: Record<string, number[]>;
  correlationMatrix: Record<string, Record<string, number>>;
  volatilities: Record<string, number>;
  expectedReturns: Record<string, number>;
  sharpeRatios: Record<string, number>;
}> {
  const returns: Record<string, number[]> = {};
  const volatilities: Record<string, number> = {};
  const expectedReturns: Record<string, number> = {};
  const sharpeRatios: Record<string, number> = {};

  // Get historical price data (252 trading days = 1 year)
  const lookbackPeriod = 252;
  
  for (const asset of assetUniverse.filter(a => a.eligible)) {
    try {
      const priceHistory = await marketDataProvider.getHistoricalPrices(
        asset.symbol,
        lookbackPeriod
      );

      // Calculate daily returns
      const dailyReturns = [];
      for (let i = 1; i < priceHistory.length; i++) {
        const prevPrice = parseFloat(ethers.utils.formatEther(priceHistory[i - 1].price));
        const currentPrice = parseFloat(ethers.utils.formatEther(priceHistory[i].price));
        const dailyReturn = (currentPrice - prevPrice) / prevPrice;
        dailyReturns.push(dailyReturn);
      }

      returns[asset.symbol] = dailyReturns;

      // Calculate annualized metrics
      const meanReturn = dailyReturns.reduce((sum, r) => sum + r, 0) / dailyReturns.length;
      const variance = dailyReturns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / dailyReturns.length;
      
      expectedReturns[asset.symbol] = meanReturn * 252; // Annualized
      volatilities[asset.symbol] = Math.sqrt(variance * 252); // Annualized
      
      // Assuming risk-free rate of 3%
      const riskFreeRate = 0.03;
      sharpeRatios[asset.symbol] = (expectedReturns[asset.symbol] - riskFreeRate) / volatilities[asset.symbol];

    } catch (error) {
      logger.warn('Failed to calculate metrics for asset', {
        asset: asset.symbol,
        error: error instanceof Error ? error.message : String(error)
      });

      // Use default values
      returns[asset.symbol] = [];
      expectedReturns[asset.symbol] = 0.08; // 8% default
      volatilities[asset.symbol] = 0.15; // 15% default
      sharpeRatios[asset.symbol] = 0.33; // Default Sharpe
    }
  }

  // Calculate correlation matrix
  const correlationMatrix = calculateCorrelationMatrix(returns);

  return {
    returns,
    correlationMatrix,
    volatilities,
    expectedReturns,
    sharpeRatios
  };
}

async function runMPTOptimization(
  historicalData: any,
  objective: string,
  constraints: any,
  riskTolerance: string
): Promise<Record<string, number>> {
  // Simplified Mean-Variance Optimization
  // In production, this would use a proper optimization library like scipy.optimize
  
  const assets = Object.keys(historicalData.expectedReturns);
  const weights: Record<string, number> = {};

  try {
    switch (objective) {
      case 'sharpe_ratio':
        return optimizeForSharpeRatio(historicalData, assets, constraints, riskTolerance);
      
      case 'return':
        return optimizeForReturn(historicalData, assets, constraints, riskTolerance);
      
      case 'risk':
        return optimizeForMinimumRisk(historicalData, assets, constraints, riskTolerance);
      
      case 'sortino_ratio':
        return optimizeForSortinoRatio(historicalData, assets, constraints, riskTolerance);
      
      default:
        return optimizeForSharpeRatio(historicalData, assets, constraints, riskTolerance);
    }

  } catch (error) {
    logger.error('MPT optimization failed', {
      objective,
      error: error instanceof Error ? error.message : String(error)
    });

    // Return equal weights as fallback
    const equalWeight = 1 / assets.length;
    assets.forEach(asset => {
      weights[asset] = equalWeight;
    });

    return weights;
  }
}

function optimizeForSharpeRatio(
  data: any,
  assets: string[],
  constraints: any,
  riskTolerance: string
): Record<string, number> {
  const weights: Record<string, number> = {};
  
  // Simple optimization: weight by Sharpe ratio with adjustments
  const totalSharpe = assets.reduce((sum, asset) => sum + Math.max(0, data.sharpeRatios[asset]), 0);
  
  assets.forEach(asset => {
    const sharpeRatio = Math.max(0, data.sharpeRatios[asset]);
    let weight = totalSharpe > 0 ? sharpeRatio / totalSharpe : 1 / assets.length;
    
    // Apply risk tolerance adjustments
    if (riskTolerance === 'conservative') {
      // Prefer lower volatility assets
      const volAdjustment = 1 / (1 + data.volatilities[asset]);
      weight = weight * volAdjustment;
    } else if (riskTolerance === 'aggressive') {
      // Prefer higher return assets
      const returnAdjustment = 1 + Math.max(0, data.expectedReturns[asset]);
      weight = weight * returnAdjustment;
    }

    weights[asset] = weight;
  });

  // Normalize weights
  const totalWeight = Object.values(weights).reduce((sum, w) => sum + w, 0);
  Object.keys(weights).forEach(asset => {
    weights[asset] = weights[asset] / totalWeight;
  });

  // Apply constraints
  return applyConstraints(weights, constraints);
}

function optimizeForReturn(
  data: any,
  assets: string[],
  constraints: any,
  riskTolerance: string
): Record<string, number> {
  const weights: Record<string, number> = {};
  
  // Weight by expected returns
  const totalReturn = assets.reduce((sum, asset) => sum + Math.max(0, data.expectedReturns[asset]), 0);
  
  assets.forEach(asset => {
    const expectedReturn = Math.max(0, data.expectedReturns[asset]);
    weights[asset] = totalReturn > 0 ? expectedReturn / totalReturn : 1 / assets.length;
  });

  return applyConstraints(weights, constraints);
}

function optimizeForMinimumRisk(
  data: any,
  assets: string[],
  constraints: any,
  riskTolerance: string
): Record<string, number> {
  const weights: Record<string, number> = {};
  
  // Weight inversely by volatility
  const totalInverseVol = assets.reduce((sum, asset) => sum + (1 / (1 + data.volatilities[asset])), 0);
  
  assets.forEach(asset => {
    const inverseVol = 1 / (1 + data.volatilities[asset]);
    weights[asset] = inverseVol / totalInverseVol;
  });

  return applyConstraints(weights, constraints);
}

function optimizeForSortinoRatio(
  data: any,
  assets: string[],
  constraints: any,
  riskTolerance: string
): Record<string, number> {
  // Simplified Sortino calculation
  const sortinoRatios: Record<string, number> = {};
  
  assets.forEach(asset => {
    const returns = data.returns[asset] || [];
    const targetReturn = 0; // Use 0 as minimum acceptable return
    
    // Calculate downside deviation
    const downsideReturns = returns.filter(r => r < targetReturn);
    const downsideVariance = downsideReturns.length > 0 
      ? downsideReturns.reduce((sum, r) => sum + Math.pow(r - targetReturn, 2), 0) / downsideReturns.length
      : 0;
    
    const downsideDeviation = Math.sqrt(downsideVariance * 252); // Annualized
    
    sortinoRatios[asset] = downsideDeviation > 0 
      ? (data.expectedReturns[asset] - 0.03) / downsideDeviation // Risk-free rate = 3%
      : 0;
  });

  // Weight by Sortino ratios
  const weights: Record<string, number> = {};
  const totalSortino = assets.reduce((sum, asset) => sum + Math.max(0, sortinoRatios[asset]), 0);
  
  assets.forEach(asset => {
    weights[asset] = totalSortino > 0 ? Math.max(0, sortinoRatios[asset]) / totalSortino : 1 / assets.length;
  });

  return applyConstraints(weights, constraints);
}

function applyConstraints(
  weights: Record<string, number>,
  constraints: any
): Record<string, number> {
  const constrainedWeights = { ...weights };
  const assets = Object.keys(weights);

  // Apply maximum single asset exposure
  if (constraints.maxSingleAssetExposure) {
    const maxWeight = constraints.maxSingleAssetExposure / 100;
    Object.keys(constrainedWeights).forEach(asset => {
      if (constrainedWeights[asset] > maxWeight) {
        constrainedWeights[asset] = maxWeight;
      }
    });
  }

  // Apply minimum position size (remove positions below threshold)
  Object.keys(constrainedWeights).forEach(asset => {
    if (constrainedWeights[asset] < 0.01) { // 1% minimum
      delete constrainedWeights[asset];
    }
  });

  // Apply maximum number of positions
  if (Object.keys(constrainedWeights).length > constraints.maxPositions) {
    // Keep top positions by weight
    const sortedAssets = Object.entries(constrainedWeights)
      .sort(([,a], [,b]) => b - a)
      .slice(0, constraints.maxPositions);
    
    const newWeights: Record<string, number> = {};
    sortedAssets.forEach(([asset, weight]) => {
      newWeights[asset] = weight;
    });
    
    Object.assign(constrainedWeights, newWeights);
  }

  // Renormalize weights
  const totalWeight = Object.values(constrainedWeights).reduce((sum, w) => sum + w, 0);
  if (totalWeight > 0) {
    Object.keys(constrainedWeights).forEach(asset => {
      constrainedWeights[asset] = constrainedWeights[asset] / totalWeight;
    });
  }

  return constrainedWeights;
}

async function convertWeightsToAllocation(
  weights: Record<string, number>,
  assetUniverse: any[],
  totalValue: BigNumber
): Promise<AllocationTarget[]> {
  const allocation: AllocationTarget[] = [];

  for (const [symbol, weight] of Object.entries(weights)) {
    const asset = assetUniverse.find(a => a.symbol === symbol);
    if (asset && weight > 0) {
      allocation.push({
        token: {
          address: '',
          symbol: asset.symbol,
          name: asset.name,
          decimals: 18,
          chainId: 1,
          tags: [],
          isStable: false,
          isNative: false
        },
        targetPercentage: weight * 100,
        minPercentage: Math.max(0, weight * 100 - 5), // 5% tolerance
        maxPercentage: weight * 100 + 5,
        rebalanceThreshold: 2.5 // 2.5% rebalance threshold
      });
    }
  }

  return allocation;
}

async function calculateExpectedMetrics(
  allocation: AllocationTarget[],
  historicalData: any,
  marketDataProvider: MarketDataProvider
): Promise<{
  expectedReturn: number;
  expectedVolatility: number;
  expectedSharpeRatio: number;
  expectedSortinoRatio: number;
  maxDrawdown: number;
  valueAtRisk: number;
}> {
  // Calculate portfolio expected return
  const expectedReturn = allocation.reduce((sum, target) => {
    const assetReturn = historicalData.expectedReturns[target.token.symbol] || 0.08;
    return sum + (target.targetPercentage / 100) * assetReturn;
  }, 0);

  // Calculate portfolio volatility (simplified)
  const expectedVolatility = allocation.reduce((sum, target) => {
    const assetVol = historicalData.volatilities[target.token.symbol] || 0.15;
    return sum + Math.pow(target.targetPercentage / 100, 2) * Math.pow(assetVol, 2);
  }, 0);

  const portfolioVolatility = Math.sqrt(expectedVolatility);

  // Calculate Sharpe ratio
  const riskFreeRate = 0.03;
  const expectedSharpeRatio = (expectedReturn - riskFreeRate) / portfolioVolatility;

  // Estimate other metrics (simplified)
  const expectedSortinoRatio = expectedSharpeRatio * 1.2; // Approximation
  const maxDrawdown = portfolioVolatility * 2; // Rough estimate
  const valueAtRisk = portfolioVolatility * 1.645; // 95% VaR approximation

  return {
    expectedReturn,
    expectedVolatility: portfolioVolatility,
    expectedSharpeRatio,
    expectedSortinoRatio,
    maxDrawdown,
    valueAtRisk
  };
}

async function compareWithCurrentPortfolio(
  portfolio: Portfolio,
  expectedMetrics: any,
  historicalData: any
): Promise<{
  currentReturn: number;
  currentVolatility: number;
  currentSharpeRatio: number;
  improvement: {
    returnImprovement: number;
    riskReduction: number;
    sharpeImprovement: number;
  };
}> {
  // Calculate current portfolio metrics
  const currentReturn = portfolio.positions.reduce((sum, position) => {
    const assetReturn = historicalData.expectedReturns[position.token.symbol] || 0.08;
    return sum + (position.percentage / 100) * assetReturn;
  }, 0);

  const currentVolatility = Math.sqrt(portfolio.positions.reduce((sum, position) => {
    const assetVol = historicalData.volatilities[position.token.symbol] || 0.15;
    return sum + Math.pow(position.percentage / 100, 2) * Math.pow(assetVol, 2);
  }, 0));

  const riskFreeRate = 0.03;
  const currentSharpeRatio = (currentReturn - riskFreeRate) / currentVolatility;

  // Calculate improvements
  const returnImprovement = ((expectedMetrics.expectedReturn - currentReturn) / currentReturn) * 100;
  const riskReduction = ((currentVolatility - expectedMetrics.expectedVolatility) / currentVolatility) * 100;
  const sharpeImprovement = ((expectedMetrics.expectedSharpeRatio - currentSharpeRatio) / Math.abs(currentSharpeRatio)) * 100;

  return {
    currentReturn,
    currentVolatility,
    currentSharpeRatio,
    improvement: {
      returnImprovement,
      riskReduction,
      sharpeImprovement
    }
  };
}

// Helper functions
function calculateCorrelationMatrix(returns: Record<string, number[]>): Record<string, Record<string, number>> {
  const assets = Object.keys(returns);
  const correlationMatrix: Record<string, Record<string, number>> = {};

  assets.forEach(asset1 => {
    correlationMatrix[asset1] = {};
    assets.forEach(asset2 => {
      if (asset1 === asset2) {
        correlationMatrix[asset1][asset2] = 1;
      } else {
        const correlation = calculateCorrelation(returns[asset1], returns[asset2]);
        correlationMatrix[asset1][asset2] = correlation;
      }
    });
  });

  return correlationMatrix;
}

function calculateCorrelation(returns1: number[], returns2: number[]): number {
  const n = Math.min(returns1.length, returns2.length);
  if (n < 2) return 0;

  const mean1 = returns1.slice(0, n).reduce((sum, r) => sum + r, 0) / n;
  const mean2 = returns2.slice(0, n).reduce((sum, r) => sum + r, 0) / n;

  let numerator = 0;
  let sum1Sq = 0;
  let sum2Sq = 0;

  for (let i = 0; i < n; i++) {
    const diff1 = returns1[i] - mean1;
    const diff2 = returns2[i] - mean2;
    
    numerator += diff1 * diff2;
    sum1Sq += diff1 * diff1;
    sum2Sq += diff2 * diff2;
  }

  const denominator = Math.sqrt(sum1Sq * sum2Sq);
  return denominator === 0 ? 0 : numerator / denominator;
}

async function isAssetEligible(asset: any, constraints: any, marketDataProvider: MarketDataProvider): Promise<boolean> {
  // Check basic eligibility criteria
  if (!asset.marketCap || asset.marketCap.lt(ethers.utils.parseEther('1000000'))) {
    return false; // Minimum $1M market cap
  }

  if (!asset.liquidity || asset.liquidity.lt(ethers.utils.parseEther('10000'))) {
    return false; // Minimum $10k daily volume
  }

  // Additional checks could include:
  // - Regulatory compliance
  // - Technical analysis indicators
  // - Fundamental analysis scores
  
  return true;
}

async function getAlternativeAssets(marketDataProvider: MarketDataProvider): Promise<any[]> {
  // Return alternative assets like REITs, commodities, etc.
  // This would be implemented based on available data sources
  return [];
}

function generateOptimizationReasoning(
  config: OptimizationConfig,
  allocation: AllocationTarget[],
  expectedMetrics: any,
  comparisonMetrics: any
): string {
  const improvements = [];
  
  if (comparisonMetrics.improvement.returnImprovement > 1) {
    improvements.push(`${comparisonMetrics.improvement.returnImprovement.toFixed(1)}% higher expected return`);
  }
  
  if (comparisonMetrics.improvement.riskReduction > 1) {
    improvements.push(`${comparisonMetrics.improvement.riskReduction.toFixed(1)}% risk reduction`);
  }
  
  if (comparisonMetrics.improvement.sharpeImprovement > 5) {
    improvements.push(`${comparisonMetrics.improvement.sharpeImprovement.toFixed(1)}% better risk-adjusted returns`);
  }

  const topAllocations = allocation
    .sort((a, b) => b.targetPercentage - a.targetPercentage)
    .slice(0, 3)
    .map(a => `${a.token.symbol} (${a.targetPercentage.toFixed(1)}%)`)
    .join(', ');

  return `Optimized for ${config.optimizationObjective.replace('_', ' ')} with ${config.riskTolerance} risk tolerance. ` +
    `Top allocations: ${topAllocations}. ` +
    `Expected improvements: ${improvements.join(', ') || 'minimal changes recommended'}.`;
}

function calculateOptimizationConfidence(
  historicalData: any,
  assetUniverse: any[],
  marketDataProvider: MarketDataProvider
): number {
  let confidence = 100;

  // Reduce confidence based on data quality
  const assetsWithData = Object.keys(historicalData.returns).length;
  const totalAssets = assetUniverse.filter(a => a.eligible).length;
  
  if (assetsWithData < totalAssets * 0.8) {
    confidence -= 20; // Reduce confidence if missing data
  }

  // Reduce confidence for high market volatility periods
  const avgVolatility = Object.values(historicalData.volatilities).reduce((sum: number, vol: any) => sum + vol, 0) / Object.keys(historicalData.volatilities).length;
  
  if (avgVolatility > 0.3) {
    confidence -= 15; // High volatility environment
  }

  // Reduce confidence if limited historical data
  const avgDataPoints = Object.values(historicalData.returns).reduce((sum: number, returns: any) => sum + returns.length, 0) / Object.keys(historicalData.returns).length;
  
  if (avgDataPoints < 100) {
    confidence -= 25; // Limited historical data
  }

  return Math.max(confidence, 30) / 100; // Convert to decimal, minimum 30%
}
