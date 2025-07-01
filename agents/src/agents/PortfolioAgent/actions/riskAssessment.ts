import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { Portfolio, PortfolioRiskMetrics } from '../../../shared/types/market';
import { PORTFOLIO_THRESHOLDS, RISK_THRESHOLDS } from '../../../shared/constants/thresholds';
import { MarketDataProvider } from '../providers/marketDataProvider';

export interface RiskAssessmentConfig {
  riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  timeHorizon: 'short' | 'medium' | 'long';
  confidenceLevel: number;
  includeStressTests: boolean;
  marketRegime: 'bull' | 'bear' | 'neutral' | 'volatile';
}

export interface RiskAssessmentResult {
  overallRisk: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  riskMetrics: PortfolioRiskMetrics;
  riskContributions: Array<{
    asset: string;
    riskContribution: number;
    concentration: number;
    volatilityContribution: number;
  }>;
  stressTesting: {
    scenarios: Array<{
      name: string;
      description: string;
      portfolioImpact: number;
      probability: number;
    }>;
    worstCase: {
      scenario: string;
      loss: BigNumber;
      lossPercentage: number;
    };
  };
  recommendations: Array<{
    priority: 'high' | 'medium' | 'low';
    action: string;
    description: string;
    expectedImpact: number;
  }>;
  monitoring: {
    keyIndicators: Array<{
      metric: string;
      currentValue: number;
      threshold: number;
      status: 'normal' | 'warning' | 'critical';
    }>;
    alerts: string[];
  };
}

export async function assessRisk(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider,
  config: RiskAssessmentConfig
): Promise<RiskAssessmentResult> {
  const startTime = Date.now();

  logger.info('Starting portfolio risk assessment', {
    portfolioId: portfolio.id,
    totalValue: ethers.utils.formatEther(portfolio.totalValue),
    positionCount: portfolio.positions.length,
    config
  });

  try {
    // Step 1: Calculate basic risk metrics
    const basicRiskMetrics = await calculateBasicRiskMetrics(portfolio, marketDataProvider, config);

    // Step 2: Assess concentration risk
    const concentrationRisk = calculateConcentrationRisk(portfolio);

    // Step 3: Calculate liquidity risk
    const liquidityRisk = await calculateLiquidityRisk(portfolio, marketDataProvider);

    // Step 4: Assess volatility risk
    const volatilityRisk = await calculateVolatilityRisk(portfolio, marketDataProvider, config);

    // Step 5: Calculate correlation risk
    const correlationRisk = await calculateCorrelationRisk(portfolio, marketDataProvider);

    // Step 6: Perform stress testing
    const stressTesting = config.includeStressTests 
      ? await performStressTesting(portfolio, marketDataProvider, config)
      : getDefaultStressTesting();

    // Step 7: Calculate risk contributions by asset
    const riskContributions = await calculateRiskContributions(portfolio, marketDataProvider);

    // Step 8: Compile comprehensive risk metrics
    const riskMetrics: PortfolioRiskMetrics = {
      overallRiskScore: 0, // Will be calculated below
      concentrationRisk,
      liquidityRisk,
      volatilityRisk,
      correlationRisk,
      valueAtRisk: basicRiskMetrics.valueAtRisk,
      expectedShortfall: basicRiskMetrics.expectedShortfall,
      riskContributions: riskContributions.map(rc => ({
        token: rc.token,
        contribution: rc.riskContribution
      }))
    };

    // Step 9: Calculate overall risk score
    const overallRisk = calculateOverallRiskScore(riskMetrics, config);
    riskMetrics.overallRiskScore = overallRisk;

    // Step 10: Determine risk level
    const riskLevel = determineRiskLevel(overallRisk);

    // Step 11: Generate recommendations
    const recommendations = generateRiskRecommendations(riskMetrics, stressTesting, config);

    // Step 12: Set up monitoring framework
    const monitoring = setupRiskMonitoring(portfolio, riskMetrics, config);

    logger.info('Portfolio risk assessment completed', {
      portfolioId: portfolio.id,
      overallRisk,
      riskLevel,
      duration: Date.now() - startTime
    });

    return {
      overallRisk,
      riskLevel,
      riskMetrics,
      riskContributions,
      stressTesting,
      recommendations,
      monitoring
    };

  } catch (error) {
    logger.error('Portfolio risk assessment failed', {
      portfolioId: portfolio.id,
      error: error instanceof Error ? error.message : String(error),
      duration: Date.now() - startTime
    });

    throw error;
  }
}

async function calculateBasicRiskMetrics(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider,
  config: RiskAssessmentConfig
): Promise<{
  valueAtRisk: {
    day1: BigNumber;
    week1: BigNumber;
    month1: BigNumber;
  };
  expectedShortfall: {
    day1: BigNumber;
    week1: BigNumber;
    month1: BigNumber;
  };
}> {
  try {
    // Get historical data for VaR calculation
    const historicalReturns = await getPortfolioHistoricalReturns(portfolio, marketDataProvider, 252); // 1 year

    if (historicalReturns.length < 30) {
      // Insufficient data, use parametric approach
      return calculateParametricVaR(portfolio, marketDataProvider, config);
    }

    // Calculate historical VaR and ES at different confidence levels
    const confidenceLevel = config.confidenceLevel || 0.95;
    const portfolioValue = portfolio.totalValue;

    // Sort returns for percentile calculation
    const sortedReturns = historicalReturns.sort((a, b) => a - b);
    const varIndex = Math.floor((1 - confidenceLevel) * sortedReturns.length);

    // 1-day VaR
    const dailyVaR = Math.abs(sortedReturns[varIndex]);
    const day1VaR = portfolioValue.mul(Math.floor(dailyVaR * 10000)).div(10000);

    // Scale to different time horizons (assuming normal distribution)
    const week1VaR = day1VaR.mul(Math.floor(Math.sqrt(7) * 100)).div(100);
    const month1VaR = day1VaR.mul(Math.floor(Math.sqrt(30) * 100)).div(100);

    // Expected Shortfall (average of losses beyond VaR)
    const tailReturns = sortedReturns.slice(0, varIndex);
    const avgTailReturn = tailReturns.length > 0 
      ? tailReturns.reduce((sum, ret) => sum + ret, 0) / tailReturns.length
      : dailyVaR;

    const day1ES = portfolioValue.mul(Math.floor(Math.abs(avgTailReturn) * 10000)).div(10000);
    const week1ES = day1ES.mul(Math.floor(Math.sqrt(7) * 100)).div(100);
    const month1ES = day1ES.mul(Math.floor(Math.sqrt(30) * 100)).div(100);

    return {
      valueAtRisk: {
        day1: day1VaR,
        week1: week1VaR,
        month1: month1VaR
      },
      expectedShortfall: {
        day1: day1ES,
        week1: week1ES,
        month1: month1ES
      }
    };

  } catch (error) {
    logger.warn('Failed to calculate historical VaR, using parametric method', {
      portfolioId: portfolio.id,
      error: error instanceof Error ? error.message : String(error)
    });

    return calculateParametricVaR(portfolio, marketDataProvider, config);
  }
}

async function calculateParametricVaR(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider,
  config: RiskAssessmentConfig
): Promise<{
  valueAtRisk: any;
  expectedShortfall: any;
}> {
  // Parametric VaR calculation using portfolio volatility
  const portfolioVolatility = await calculatePortfolioVolatility(portfolio, marketDataProvider);
  const confidenceLevel = config.confidenceLevel || 0.95;
  
  // Z-score for confidence level (95% = 1.645, 99% = 2.326)
  const zScore = confidenceLevel === 0.99 ? 2.326 : 1.645;
  const esMultiplier = confidenceLevel === 0.99 ? 2.665 : 2.063;

  const portfolioValue = portfolio.totalValue;
  const dailyVaR = portfolioValue.mul(Math.floor(portfolioVolatility * zScore * 100)).div(10000);
  
  return {
    valueAtRisk: {
      day1: dailyVaR,
      week1: dailyVaR.mul(Math.floor(Math.sqrt(7) * 100)).div(100),
      month1: dailyVaR.mul(Math.floor(Math.sqrt(30) * 100)).div(100)
    },
    expectedShortfall: {
      day1: dailyVaR.mul(Math.floor(esMultiplier * 100)).div(100),
      week1: dailyVaR.mul(Math.floor(esMultiplier * Math.sqrt(7) * 100)).div(100),
      month1: dailyVaR.mul(Math.floor(esMultiplier * Math.sqrt(30) * 100)).div(100)
    }
  };
}

function calculateConcentrationRisk(portfolio: Portfolio): number {
  // Calculate Herfindahl-Hirschman Index for concentration
  let hhi = 0;
  let maxConcentration = 0;

  for (const position of portfolio.positions) {
    const weight = position.percentage / 100;
    hhi += weight * weight;
    maxConcentration = Math.max(maxConcentration, position.percentage);
  }

  // Normalize HHI to 0-100 scale
  const normalizedHHI = (hhi - (1 / portfolio.positions.length)) / (1 - (1 / portfolio.positions.length)) * 100;
  
  // Concentration risk score (higher = more concentrated = more risky)
  let concentrationRisk = normalizedHHI;
  
  // Penalty for single position dominance
  if (maxConcentration > 50) {
    concentrationRisk += 30;
  } else if (maxConcentration > 30) {
    concentrationRisk += 15;
  }

  // Penalty for too few positions
  if (portfolio.positions.length < 3) {
    concentrationRisk += 40;
  } else if (portfolio.positions.length < 5) {
    concentrationRisk += 20;
  }

  return Math.min(concentrationRisk, 100);
}

async function calculateLiquidityRisk(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider
): Promise<number> {
  let weightedLiquidityRisk = 0;

  for (const position of portfolio.positions) {
    try {
      const liquidityData = await marketDataProvider.getLiquidityMetrics(position.token.symbol);
      
      // Calculate liquidity score based on multiple factors
      let liquidityScore = 0;
      
      // Daily volume factor
      const positionValue = parseFloat(ethers.utils.formatEther(position.value));
      const dailyVolume = parseFloat(ethers.utils.formatEther(liquidityData.averageDailyVolume || BigNumber.from(0)));
      
      if (dailyVolume > 0) {
        const volumeRatio = positionValue / dailyVolume;
        
        if (volumeRatio > 0.1) {
          liquidityScore += 50; // Position is >10% of daily volume
        } else if (volumeRatio > 0.05) {
          liquidityScore += 30; // Position is >5% of daily volume
        } else if (volumeRatio > 0.01) {
          liquidityScore += 15; // Position is >1% of daily volume
        }
      } else {
        liquidityScore += 80; // No volume data available
      }

      // Bid-ask spread factor
      const spread = liquidityData.bidAskSpread || 0.5; // Default 0.5%
      liquidityScore += Math.min(spread * 20, 30); // Max 30 points for spread

      // Market depth factor
      const marketDepth = parseFloat(ethers.utils.formatEther(liquidityData.marketDepth || BigNumber.from(0)));
      if (marketDepth < positionValue * 2) {
        liquidityScore += 25; // Insufficient market depth
      }

      const positionWeight = position.percentage / 100;
      weightedLiquidityRisk += liquidityScore * positionWeight;

    } catch (error) {
      logger.warn('Failed to get liquidity data for asset', {
        asset: position.token.symbol,
        error: error instanceof Error ? error.message : String(error)
      });

      // Use high risk score for unknown liquidity
      const positionWeight = position.percentage / 100;
      weightedLiquidityRisk += 70 * positionWeight;
    }
  }

  return Math.min(weightedLiquidityRisk, 100);
}

async function calculateVolatilityRisk(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider,
  config: RiskAssessmentConfig
): Promise<number> {
  try {
    const portfolioVolatility = await calculatePortfolioVolatility(portfolio, marketDataProvider);
    
    // Volatility risk thresholds based on risk tolerance
    const thresholds = {
      'conservative': { low: 0.1, high: 0.2 },
      'moderate': { low: 0.15, high: 0.3 },
      'aggressive': { low: 0.2, high: 0.5 }
    };

    const threshold = thresholds[config.riskTolerance];
    
    let volatilityRisk = 0;
    
    if (portfolioVolatility > threshold.high) {
      volatilityRisk = 80 + ((portfolioVolatility - threshold.high) / threshold.high) * 20;
    } else if (portfolioVolatility > threshold.low) {
      volatilityRisk = 40 + ((portfolioVolatility - threshold.low) / (threshold.high - threshold.low)) * 40;
    } else {
      volatilityRisk = (portfolioVolatility / threshold.low) * 40;
    }

    return Math.min(volatilityRisk, 100);

  } catch (error) {
    logger.error('Failed to calculate volatility risk', {
      portfolioId: portfolio.id,
      error: error instanceof Error ? error.message : String(error)
    });

    return 50; // Default moderate risk
  }
}

async function calculateCorrelationRisk(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider
): Promise<number> {
  try {
    // Get correlation matrix for portfolio assets
    const correlationMatrix = await marketDataProvider.getCorrelationMatrix(
      portfolio.positions.map(p => p.token.symbol)
    );

    if (!correlationMatrix || Object.keys(correlationMatrix).length < 2) {
      return 30; // Default low-medium correlation risk
    }

    // Calculate average correlation
    let totalCorrelation = 0;
    let pairCount = 0;
    const assets = Object.keys(correlationMatrix);

    for (let i = 0; i < assets.length; i++) {
      for (let j = i + 1; j < assets.length; j++) {
        const correlation = correlationMatrix[assets[i]][assets[j]] || 0;
        totalCorrelation += Math.abs(correlation);
        pairCount++;
      }
    }

    const avgCorrelation = pairCount > 0 ? totalCorrelation / pairCount : 0;

    // High correlation = high risk (limited diversification)
    let correlationRisk = 0;
    
    if (avgCorrelation > 0.8) {
      correlationRisk = 90; // Very high correlation
    } else if (avgCorrelation > 0.6) {
      correlationRisk = 70; // High correlation
    } else if (avgCorrelation > 0.4) {
      correlationRisk = 50; // Medium correlation
    } else if (avgCorrelation > 0.2) {
      correlationRisk = 30; // Low correlation
    } else {
      correlationRisk = 10; // Very low correlation
    }

    // Adjust for concentration - fewer assets with high correlation is worse
    if (portfolio.positions.length <= 3 && avgCorrelation > 0.6) {
      correlationRisk += 20;
    }

    return Math.min(correlationRisk, 100);

  } catch (error) {
    logger.error('Failed to calculate correlation risk', {
      portfolioId: portfolio.id,
      error: error instanceof Error ? error.message : String(error)
    });

    return 40; // Default medium correlation risk
  }
}

async function performStressTesting(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider,
  config: RiskAssessmentConfig
): Promise<{
  scenarios: Array<{
    name: string;
    description: string;
    portfolioImpact: number;
    probability: number;
  }>;
  worstCase: {
    scenario: string;
    loss: BigNumber;
    lossPercentage: number;
  };
}> {
  const scenarios = [
    {
      name: 'Market Crash',
      description: '30% overall market decline',
      marketShock: -0.30,
      probability: 0.05
    },
    {
      name: 'Sector Rotation',
      description: 'Major sector-specific decline',
      marketShock: -0.20,
      probability: 0.15
    },
    {
      name: 'Liquidity Crisis',
      description: 'Severe liquidity constraints',
      marketShock: -0.15,
      probability: 0.10
    },
    {
      name: 'Interest Rate Shock',
      description: 'Rapid interest rate changes',
      marketShock: -0.12,
      probability: 0.20
    },
    {
      name: 'Crypto Winter',
      description: 'Extended bear market in crypto',
      marketShock: -0.50,
      probability: 0.08
    }
  ];

  const stressResults = [];
  let worstCaseScenario = { scenario: '', loss: BigNumber.from(0), lossPercentage: 0 };

  for (const scenario of scenarios) {
    try {
      // Calculate portfolio impact under stress scenario
      let portfolioImpact = 0;
      
      for (const position of portfolio.positions) {
        // Get asset-specific stress response
        const assetBeta = await getAssetBeta(position.token.symbol, marketDataProvider);
        const assetStressImpact = scenario.marketShock * assetBeta;
        
        portfolioImpact += assetStressImpact * (position.percentage / 100);
      }

      // Add liquidity impact in stress scenarios
      if (scenario.name === 'Liquidity Crisis') {
        const liquidityRisk = await calculateLiquidityRisk(portfolio, marketDataProvider);
        portfolioImpact -= (liquidityRisk / 100) * 0.1; // Additional 10% impact for illiquid assets
      }

      const portfolioImpactPercent = portfolioImpact * 100;
      const loss = portfolio.totalValue.mul(Math.floor(Math.abs(portfolioImpact) * 10000)).div(10000);

      stressResults.push({
        name: scenario.name,
        description: scenario.description,
        portfolioImpact: portfolioImpactPercent,
        probability: scenario.probability
      });

      // Track worst case
      if (Math.abs(portfolioImpactPercent) > Math.abs(worstCaseScenario.lossPercentage)) {
        worstCaseScenario = {
          scenario: scenario.name,
          loss,
          lossPercentage: portfolioImpactPercent
        };
      }

    } catch (error) {
      logger.warn('Stress test scenario failed', {
        scenario: scenario.name,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  return {
    scenarios: stressResults,
    worstCase: worstCaseScenario
  };
}

function getDefaultStressTesting() {
  return {
    scenarios: [
      {
        name: 'No stress testing',
        description: 'Stress testing disabled',
        portfolioImpact: 0,
        probability: 0
      }
    ],
    worstCase: {
      scenario: 'Unknown',
      loss: BigNumber.from(0),
      lossPercentage: 0
    }
  };
}

async function calculateRiskContributions(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider
): Promise<Array<{
  token: any;
  riskContribution: number;
  concentration: number;
  volatilityContribution: number;
}>> {
  const contributions = [];

  for (const position of portfolio.positions) {
    try {
      // Calculate individual asset volatility
      const assetVolatility = await getAssetVolatility(position.token.symbol, marketDataProvider);
      
      // Position's contribution to portfolio volatility
      const positionWeight = position.percentage / 100;
      const volatilityContribution = positionWeight * assetVolatility;
      
      // Concentration contribution
      const concentration = position.percentage;
      
      // Overall risk contribution (simplified)
      const riskContribution = (volatilityContribution * 0.6) + (concentration * 0.4);

      contributions.push({
        token: position.token,
        riskContribution,
        concentration,
        volatilityContribution: volatilityContribution * 100
      });

    } catch (error) {
      logger.warn('Failed to calculate risk contribution for asset', {
        asset: position.token.symbol,
        error: error instanceof Error ? error.message : String(error)
      });

      contributions.push({
        token: position.token,
        riskContribution: position.percentage / 2, // Default estimate
        concentration: position.percentage,
        volatilityContribution: 15 // Default 15%
      });
    }
  }

  return contributions.sort((a, b) => b.riskContribution - a.riskContribution);
}

function calculateOverallRiskScore(metrics: PortfolioRiskMetrics, config: RiskAssessmentConfig): number {
  // Weighted combination of risk factors
  const weights = {
    concentration: 0.25,
    liquidity: 0.20,
    volatility: 0.30,
    correlation: 0.25
  };

  // Adjust weights based on risk tolerance
  if (config.riskTolerance === 'conservative') {
    weights.volatility += 0.1;
    weights.correlation += 0.1;
    weights.concentration -= 0.1;
    weights.liquidity -= 0.1;
  } else if (config.riskTolerance === 'aggressive') {
    weights.concentration += 0.1;
    weights.liquidity += 0.1;
    weights.volatility -= 0.1;
    weights.correlation -= 0.1;
  }

  const overallRisk = 
    metrics.concentrationRisk * weights.concentration +
    metrics.liquidityRisk * weights.liquidity +
    metrics.volatilityRisk * weights.volatility +
    metrics.correlationRisk * weights.correlation;

  return Math.min(Math.max(overallRisk, 0), 100);
}

function determineRiskLevel(overallRisk: number): 'low' | 'medium' | 'high' | 'critical' {
  if (overallRisk >= RISK_THRESHOLDS.CRITICAL_RISK) return 'critical';
  if (overallRisk >= RISK_THRESHOLDS.HIGH_RISK) return 'high';
  if (overallRisk >= RISK_THRESHOLDS.MEDIUM_RISK) return 'medium';
  return 'low';
}

function generateRiskRecommendations(
  riskMetrics: PortfolioRiskMetrics,
  stressTesting: any,
  config: RiskAssessmentConfig
): Array<{
  priority: 'high' | 'medium' | 'low';
  action: string;
  description: string;
  expectedImpact: number;
}> {
  const recommendations = [];

  // Concentration risk recommendations
  if (riskMetrics.concentrationRisk > 70) {
    recommendations.push({
      priority: 'high' as const,
      action: 'Reduce concentration',
      description: 'Diversify holdings to reduce single-asset risk',
      expectedImpact: 25
    });
  }

  // Liquidity risk recommendations
  if (riskMetrics.liquidityRisk > 60) {
    recommendations.push({
      priority: 'medium' as const,
      action: 'Improve liquidity',
      description: 'Add more liquid assets or reduce position sizes in illiquid assets',
      expectedImpact: 15
    });
  }

  // Volatility risk recommendations
  if (riskMetrics.volatilityRisk > 80) {
    recommendations.push({
      priority: 'high' as const,
      action: 'Reduce volatility',
      description: 'Add stable assets or reduce exposure to high-volatility positions',
      expectedImpact: 30
    });
  }

  // Correlation risk recommendations
  if (riskMetrics.correlationRisk > 70) {
    recommendations.push({
      priority: 'medium' as const,
      action: 'Improve diversification',
      description: 'Add uncorrelated assets to improve portfolio diversification',
      expectedImpact: 20
    });
  }

  // Stress testing recommendations
  if (stressTesting.worstCase.lossPercentage < -40) {
    recommendations.push({
      priority: 'high' as const,
      action: 'Add hedging',
      description: 'Consider protective positions or hedging strategies',
      expectedImpact: 35
    });
  }

  return recommendations.sort((a, b) => {
    const priorityOrder = { high: 3, medium: 2, low: 1 };
    return priorityOrder[b.priority] - priorityOrder[a.priority];
  });
}

function setupRiskMonitoring(
  portfolio: Portfolio,
  riskMetrics: PortfolioRiskMetrics,
  config: RiskAssessmentConfig
): {
  keyIndicators: Array<{
    metric: string;
    currentValue: number;
    threshold: number;
    status: 'normal' | 'warning' | 'critical';
  }>;
  alerts: string[];
} {
  const indicators = [
    {
      metric: 'Overall Risk Score',
      currentValue: riskMetrics.overallRiskScore,
      threshold: config.riskTolerance === 'conservative' ? 50 : 
                config.riskTolerance === 'moderate' ? 70 : 85,
      status: 'normal' as 'normal' | 'warning' | 'critical'
    },
    {
      metric: 'Concentration Risk',
      currentValue: riskMetrics.concentrationRisk,
      threshold: 60,
      status: 'normal' as 'normal' | 'warning' | 'critical'
    },
    {
      metric: 'Liquidity Risk',
      currentValue: riskMetrics.liquidityRisk,
      threshold: 50,
      status: 'normal' as 'normal' | 'warning' | 'critical'
    },
    {
      metric: 'Volatility Risk',
      currentValue: riskMetrics.volatilityRisk,
      threshold: config.riskTolerance === 'conservative' ? 40 : 
                config.riskTolerance === 'moderate' ? 60 : 80,
      status: 'normal' as 'normal' | 'warning' | 'critical'
    }
  ];

  const alerts: string[] = [];

  // Update status and generate alerts
  indicators.forEach(indicator => {
    if (indicator.currentValue >= indicator.threshold * 1.2) {
      indicator.status = 'critical';
      alerts.push(`CRITICAL: ${indicator.metric} is ${indicator.currentValue.toFixed(1)} (threshold: ${indicator.threshold})`);
    } else if (indicator.currentValue >= indicator.threshold) {
      indicator.status = 'warning';
      alerts.push(`WARNING: ${indicator.metric} is ${indicator.currentValue.toFixed(1)} (threshold: ${indicator.threshold})`);
    }
  });

  return {
    keyIndicators: indicators,
    alerts
  };
}

// Helper functions
async function getPortfolioHistoricalReturns(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider,
  days: number
): Promise<number[]> {
  try {
    // This would calculate portfolio returns based on historical position data
    // For now, we'll estimate based on individual asset returns
    const assetReturns: Record<string, number[]> = {};
    
    for (const position of portfolio.positions) {
      const returns = await marketDataProvider.getHistoricalReturns(position.token.symbol, days);
      assetReturns[position.token.symbol] = returns;
    }

    // Calculate portfolio returns
    const portfolioReturns: number[] = [];
    const maxLength = Math.max(...Object.values(assetReturns).map(r => r.length));

    for (let i = 0; i < maxLength; i++) {
      let portfolioReturn = 0;
      
      for (const position of portfolio.positions) {
        const assetReturn = assetReturns[position.token.symbol]?.[i] || 0;
        const weight = position.percentage / 100;
        portfolioReturn += assetReturn * weight;
      }
      
      portfolioReturns.push(portfolioReturn);
    }

    return portfolioReturns;

  } catch (error) {
    logger.error('Failed to get portfolio historical returns', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function calculatePortfolioVolatility(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider
): Promise<number> {
  try {
    let portfolioVariance = 0;

    // Calculate portfolio variance using position weights and correlations
    for (let i = 0; i < portfolio.positions.length; i++) {
      for (let j = 0; j < portfolio.positions.length; j++) {
        const position_i = portfolio.positions[i];
        const position_j = portfolio.positions[j];
        
        const weight_i = position_i.percentage / 100;
        const weight_j = position_j.percentage / 100;
        
        const vol_i = await getAssetVolatility(position_i.token.symbol, marketDataProvider);
        const vol_j = await getAssetVolatility(position_j.token.symbol, marketDataProvider);
        
        let correlation = 0;
        if (i === j) {
          correlation = 1;
        } else {
          const correlationMatrix = await marketDataProvider.getCorrelationMatrix([
            position_i.token.symbol,
            position_j.token.symbol
          ]);
          correlation = correlationMatrix?.[position_i.token.symbol]?.[position_j.token.symbol] || 0.3;
        }
        
        portfolioVariance += weight_i * weight_j * vol_i * vol_j * correlation;
      }
    }

    return Math.sqrt(portfolioVariance);

  } catch (error) {
    logger.error('Failed to calculate portfolio volatility', {
      error: error instanceof Error ? error.message : String(error)
    });

    // Return weighted average volatility as fallback
    let weightedVol = 0;
    for (const position of portfolio.positions) {
      const weight = position.percentage / 100;
      const vol = 0.15; // Default 15% volatility
      weightedVol += weight * vol;
    }
    
    return weightedVol;
  }
}

async function getAssetVolatility(symbol: string, marketDataProvider: MarketDataProvider): Promise<number> {
  try {
    const volatilityData = await marketDataProvider.getAssetVolatility(symbol);
    return volatilityData.annualizedVolatility || 0.15; // Default 15%
  } catch {
    return 0.15; // Default volatility
  }
}

async function getAssetBeta(symbol: string, marketDataProvider: MarketDataProvider): Promise<number> {
  try {
    const betaData = await marketDataProvider.getAssetBeta(symbol);
    return betaData.beta || 1.0; // Default beta of 1
  } catch {
    return 1.0; // Default beta
  }
}
