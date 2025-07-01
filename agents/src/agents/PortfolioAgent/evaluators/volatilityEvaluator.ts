import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { Portfolio, PortfolioPosition } from '../../../shared/types/market';

export interface VolatilityMetrics {
  portfolioVolatility: number;
  volatilityContributions: Array<{
    asset: string;
    contribution: number;
    weight: number;
    assetVolatility: number;
    marginalContribution: number;
  }>;
  riskDecomposition: {
    assetSpecificRisk: number;
    systematicRisk: number;
    correlationEffect: number;
  };
  volatilityForecasts: {
    nextDay: number;
    nextWeek: number;
    nextMonth: number;
    confidence: number;
  };
  regimeAnalysis: {
    currentRegime: 'low' | 'medium' | 'high' | 'extreme';
    regimeProbabilities: Record<string, number>;
    expectedDuration: number;
  };
}

export interface GARCHModel {
  omega: number;
  alpha: number;
  beta: number;
  forecast: number;
  persistence: number;
}

export interface VolatilityCluster {
  startDate: number;
  endDate: number;
  avgVolatility: number;
  duration: number;
  type: 'low' | 'medium' | 'high' | 'extreme';
}

export class VolatilityEvaluator {
  private riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  private volatilityCache: Map<string, any> = new Map();
  private garchModels: Map<string, GARCHModel> = new Map();

  constructor(riskTolerance: 'conservative' | 'moderate' | 'aggressive') {
    this.riskTolerance = riskTolerance;
  }

  async calculateVolatility(portfolio: Portfolio): Promise<VolatilityMetrics> {
    const startTime = Date.now();

    logger.info('Starting portfolio volatility analysis', {
      portfolioId: portfolio.id,
      riskTolerance: this.riskTolerance,
      positionCount: portfolio.positions.length
    });

    try {
      // Step 1: Get historical data for all assets
      const historicalData = await this.getHistoricalVolatilityData(portfolio);

      // Step 2: Calculate individual asset volatilities
      const assetVolatilities = await this.calculateAssetVolatilities(portfolio.positions, historicalData);

      // Step 3: Calculate correlation matrix
      const correlationMatrix = await this.calculateCorrelationMatrix(portfolio.positions, historicalData);

      // Step 4: Calculate portfolio volatility using matrix algebra
      const portfolioVolatility = this.calculatePortfolioVolatility(
        portfolio.positions,
        assetVolatilities,
        correlationMatrix
      );

      // Step 5: Calculate volatility contributions
      const volatilityContributions = this.calculateVolatilityContributions(
        portfolio.positions,
        assetVolatilities,
        correlationMatrix,
        portfolioVolatility
      );

      // Step 6: Decompose risk sources
      const riskDecomposition = this.decomposeRiskSources(
        portfolio.positions,
        assetVolatilities,
        correlationMatrix
      );

      // Step 7: Generate volatility forecasts using GARCH models
      const volatilityForecasts = await this.generateVolatilityForecasts(portfolio, historicalData);

      // Step 8: Perform volatility regime analysis
      const regimeAnalysis = await this.analyzeVolatilityRegimes(portfolio, historicalData);

      const metrics: VolatilityMetrics = {
        portfolioVolatility: portfolioVolatility * 100, // Convert to percentage
        volatilityContributions,
        riskDecomposition,
        volatilityForecasts,
        regimeAnalysis
      };

      // Cache results
      this.volatilityCache.set(portfolio.id, {
        metrics,
        timestamp: Date.now()
      });

      logger.info('Portfolio volatility analysis completed', {
        portfolioId: portfolio.id,
        portfolioVolatility: (portfolioVolatility * 100).toFixed(2),
        currentRegime: regimeAnalysis.currentRegime,
        duration: Date.now() - startTime
      });

      return metrics;

    } catch (error) {
      logger.error('Portfolio volatility analysis failed', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime
      });

      throw error;
    }
  }

  private async getHistoricalVolatilityData(portfolio: Portfolio): Promise<Record<string, number[]>> {
    const historicalData: Record<string, number[]> = {};
    const lookbackPeriods = 252; // 1 year of daily data

    for (const position of portfolio.positions) {
      try {
        // In production, this would fetch real historical price data
        // For now, we'll generate realistic historical returns
        const returns = this.generateHistoricalReturns(position.token.symbol, lookbackPeriods);
        historicalData[position.token.symbol] = returns;

      } catch (error) {
        logger.warn('Failed to get historical data for asset', {
          asset: position.token.symbol,
          error: error instanceof Error ? error.message : String(error)
        });

        // Use default returns if data unavailable
        historicalData[position.token.symbol] = this.generateDefaultReturns(lookbackPeriods);
      }
    }

    return historicalData;
  }

  private generateHistoricalReturns(symbol: string, periods: number): number[] {
    const returns: number[] = [];
    
    // Asset-specific volatility parameters
    const volatilityParams: Record<string, { vol: number; drift: number; clustering: number }> = {
      'ETH': { vol: 0.055, drift: 0.0003, clustering: 0.1 },
      'BTC': { vol: 0.065, drift: 0.0004, clustering: 0.12 },
      'LINK': { vol: 0.075, drift: 0.0005, clustering: 0.15 },
      'USDC': { vol: 0.001, drift: 0.00008, clustering: 0.01 },
      'USDT': { vol: 0.001, drift: 0.00007, clustering: 0.01 },
      'MATIC': { vol: 0.085, drift: 0.0006, clustering: 0.18 },
      'AVAX': { vol: 0.095, drift: 0.0007, clustering: 0.2 },
      'UNI': { vol: 0.09, drift: 0.0006, clustering: 0.17 },
      'AAVE': { vol: 0.1, drift: 0.0008, clustering: 0.22 },
      'COMP': { vol: 0.08, drift: 0.0005, clustering: 0.16 }
    };

    const params = volatilityParams[symbol] || { vol: 0.06, drift: 0.0003, clustering: 0.1 };
    
    let currentVolatility = params.vol;
    
    for (let i = 0; i < periods; i++) {
      // GARCH-like volatility clustering
      const randomShock = (Math.random() - 0.5) * 2;
      const volatilityShock = Math.abs(randomShock) > 1.5 ? params.clustering : 0;
      currentVolatility = params.vol * (1 + volatilityShock * Math.sign(randomShock));
      
      // Generate return with current volatility
      const return_ = params.drift + currentVolatility * (Math.random() - 0.5) * 2;
      returns.push(return_);
    }

    return returns;
  }

  private generateDefaultReturns(periods: number): number[] {
    const returns: number[] = [];
    const defaultVol = 0.05; // 5% default volatility
    const defaultDrift = 0.0003; // ~8% annual return

    for (let i = 0; i < periods; i++) {
      const return_ = defaultDrift + defaultVol * (Math.random() - 0.5) * 2;
      returns.push(return_);
    }

    return returns;
  }

  private async calculateAssetVolatilities(
    positions: PortfolioPosition[],
    historicalData: Record<string, number[]>
  ): Promise<Record<string, number>> {
    const volatilities: Record<string, number> = {};

    for (const position of positions.values()) {
      const returns = historicalData[position.token.symbol] || [];
      
      if (returns.length > 0) {
        // Calculate standard deviation of returns
        const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
        const volatility = Math.sqrt(variance * 252); // Annualized
        
        volatilities[position.token.symbol] = volatility;
      } else {
        volatilities[position.token.symbol] = 0.15; // Default 15% volatility
      }
    }

    return volatilities;
  }

  private async calculateCorrelationMatrix(
    positions: PortfolioPosition[],
    historicalData: Record<string, number[]>
  ): Promise<Record<string, Record<string, number>>> {
    const correlationMatrix: Record<string, Record<string, number>> = {};
    const assets = positions.map(p => p.token.symbol);

    for (const asset1 of assets) {
      correlationMatrix[asset1] = {};
      
      for (const asset2 of assets) {
        if (asset1 === asset2) {
          correlationMatrix[asset1][asset2] = 1.0;
        } else {
          const returns1 = historicalData[asset1] || [];
          const returns2 = historicalData[asset2] || [];
          
          const correlation = this.calculateCorrelation(returns1, returns2);
          correlationMatrix[asset1][asset2] = correlation;
        }
      }
    }

    return correlationMatrix;
  }

  private calculateCorrelation(returns1: number[], returns2: number[]): number {
    const n = Math.min(returns1.length, returns2.length);
    if (n < 10) return 0.3; // Default correlation for insufficient data

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

  private calculatePortfolioVolatility(
    positions: PortfolioPosition[],
    assetVolatilities: Record<string, number>,
    correlationMatrix: Record<string, Record<string, number>>
  ): number {
    let portfolioVariance = 0;

    // Portfolio variance = w'Σw where w is weights vector and Σ is covariance matrix
    for (let i = 0; i < positions.length; i++) {
      for (let j = 0; j < positions.length; j++) {
        const asset_i = positions[i].token.symbol;
        const asset_j = positions[j].token.symbol;
        
        const weight_i = positions[i].percentage / 100;
        const weight_j = positions[j].percentage / 100;
        
        const vol_i = assetVolatilities[asset_i] || 0.15;
        const vol_j = assetVolatilities[asset_j] || 0.15;
        
        const correlation = correlationMatrix[asset_i]?.[asset_j] || 0;
        const covariance = vol_i * vol_j * correlation;
        
        portfolioVariance += weight_i * weight_j * covariance;
      }
    }

    return Math.sqrt(portfolioVariance);
  }

  private calculateVolatilityContributions(
    positions: PortfolioPosition[],
    assetVolatilities: Record<string, number>,
    correlationMatrix: Record<string, Record<string, number>>,
    portfolioVolatility: number
  ): Array<{
    asset: string;
    contribution: number;
    weight: number;
    assetVolatility: number;
    marginalContribution: number;
  }> {
    const contributions: any[] = [];

    for (const position of positions) {
      const asset = position.token.symbol;
      const weight = position.percentage / 100;
      const assetVol = assetVolatilities[asset] || 0.15;

      // Calculate marginal contribution to risk (MCTR)
      let marginalContribution = 0;
      
      for (const otherPosition of positions) {
        const otherAsset = otherPosition.token.symbol;
        const otherWeight = otherPosition.percentage / 100;
        const otherVol = assetVolatilities[otherAsset] || 0.15;
        const correlation = correlationMatrix[asset]?.[otherAsset] || 0;
        
        marginalContribution += otherWeight * assetVol * otherVol * correlation;
      }

      marginalContribution = portfolioVolatility > 0 ? marginalContribution / portfolioVolatility : 0;

      // Component contribution to risk (CTR)
      const contribution = weight * marginalContribution;

      contributions.push({
        asset,
        contribution: contribution * 100, // Convert to percentage
        weight: weight * 100,
        assetVolatility: assetVol * 100,
        marginalContribution: marginalContribution * 100
      });
    }

    return contributions.sort((a, b) => b.contribution - a.contribution);
  }

  private decomposeRiskSources(
    positions: PortfolioPosition[],
    assetVolatilities: Record<string, number>,
    correlationMatrix: Record<string, Record<string, number>>
  ): {
    assetSpecificRisk: number;
    systematicRisk: number;
    correlationEffect: number;
  } {
    // Calculate weighted average of individual asset risks
    const assetSpecificRisk = positions.reduce((sum, position) => {
      const weight = position.percentage / 100;
      const assetVol = assetVolatilities[position.token.symbol] || 0.15;
      return sum + Math.pow(weight * assetVol, 2);
    }, 0);

    // Calculate correlation effects
    let correlationEffect = 0;
    for (let i = 0; i < positions.length; i++) {
      for (let j = i + 1; j < positions.length; j++) {
        const asset_i = positions[i].token.symbol;
        const asset_j = positions[j].token.symbol;
        
        const weight_i = positions[i].percentage / 100;
        const weight_j = positions[j].percentage / 100;
        
        const vol_i = assetVolatilities[asset_i] || 0.15;
        const vol_j = assetVolatilities[asset_j] || 0.15;
        
        const correlation = correlationMatrix[asset_i]?.[asset_j] || 0;
        
        correlationEffect += 2 * weight_i * weight_j * vol_i * vol_j * correlation;
      }
    }

    // Systematic risk approximation (correlation with market)
    const avgCorrelation = this.calculateAverageCorrelation(correlationMatrix);
    const systematicRisk = avgCorrelation * Math.sqrt(assetSpecificRisk);

    return {
      assetSpecificRisk: Math.sqrt(assetSpecificRisk) * 100,
      systematicRisk: systematicRisk * 100,
      correlationEffect: Math.sqrt(Math.abs(correlationEffect)) * 100
    };
  }

  private calculateAverageCorrelation(correlationMatrix: Record<string, Record<string, number>>): number {
    const assets = Object.keys(correlationMatrix);
    let totalCorrelation = 0;
    let pairCount = 0;

    for (let i = 0; i < assets.length; i++) {
      for (let j = i + 1; j < assets.length; j++) {
        totalCorrelation += Math.abs(correlationMatrix[assets[i]][assets[j]] || 0);
        pairCount++;
      }
    }

    return pairCount > 0 ? totalCorrelation / pairCount : 0;
  }

  private async generateVolatilityForecasts(
    portfolio: Portfolio,
    historicalData: Record<string, number[]>
  ): Promise<{
    nextDay: number;
    nextWeek: number;
    nextMonth: number;
    confidence: number;
  }> {
    try {
      // Fit GARCH models for each asset
      const garchForecasts: Record<string, GARCHModel> = {};
      
      for (const position of portfolio.positions) {
        const returns = historicalData[position.token.symbol] || [];
        if (returns.length >= 50) {
          garchForecasts[position.token.symbol] = this.fitGARCHModel(returns);
        }
      }

      // Calculate portfolio volatility forecasts
      let portfolioForecast = 0;
      let confidenceScore = 0;

      for (const position of portfolio.positions) {
        const weight = position.percentage / 100;
        const garchModel = garchForecasts[position.token.symbol];
        
        if (garchModel) {
          portfolioForecast += Math.pow(weight * garchModel.forecast, 2);
          confidenceScore += weight * (1 - Math.abs(garchModel.persistence - 0.9)); // Confidence based on model stability
        } else {
          // Use historical volatility for assets without GARCH model
          const returns = historicalData[position.token.symbol] || [];
          const historicalVol = returns.length > 0 
            ? Math.sqrt(returns.reduce((sum, r) => sum + r * r, 0) / returns.length * 252)
            : 0.15;
          
          portfolioForecast += Math.pow(weight * historicalVol, 2);
          confidenceScore += weight * 0.5; // Lower confidence for non-GARCH forecasts
        }
      }

      const nextDay = Math.sqrt(portfolioForecast);
      const nextWeek = nextDay * Math.sqrt(7);   // Scale for weekly
      const nextMonth = nextDay * Math.sqrt(30); // Scale for monthly

      return {
        nextDay: nextDay * 100,
        nextWeek: nextWeek * 100,
        nextMonth: nextMonth * 100,
        confidence: Math.min(confidenceScore, 1) * 100
      };

    } catch (error) {
      logger.error('Volatility forecasting failed', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return conservative estimates
      return {
        nextDay: 2.5,
        nextWeek: 6.6,
        nextMonth: 13.6,
        confidence: 50
      };
    }
  }

  private fitGARCHModel(returns: number[]): GARCHModel {
    // Simplified GARCH(1,1) model fitting
    // In production, this would use proper maximum likelihood estimation
    
    const variance = returns.reduce((sum, r) => sum + r * r, 0) / returns.length;
    const longTermVariance = variance;
    
    // Default GARCH parameters for crypto assets
    const omega = longTermVariance * 0.05;  // Long-term variance weight
    const alpha = 0.1;  // Lagged return impact
    const beta = 0.85;  // Lagged variance impact
    
    // Calculate forecast (simplified)
    const lastReturn = returns[returns.length - 1] || 0;
    const lastVariance = variance;
    
    const forecast = Math.sqrt(omega + alpha * Math.pow(lastReturn, 2) + beta * lastVariance);
    const persistence = alpha + beta;

    return {
      omega,
      alpha,
      beta,
      forecast,
      persistence
    };
  }

  private async analyzeVolatilityRegimes(
    portfolio: Portfolio,
    historicalData: Record<string, number[]>
  ): Promise<{
    currentRegime: 'low' | 'medium' | 'high' | 'extreme';
    regimeProbabilities: Record<string, number>;
    expectedDuration: number;
  }> {
    try {
      // Calculate portfolio returns from historical data
      const portfolioReturns = this.calculatePortfolioReturns(portfolio, historicalData);
      
      if (portfolioReturns.length < 50) {
        return {
          currentRegime: 'medium',
          regimeProbabilities: { low: 0.25, medium: 0.5, high: 0.2, extreme: 0.05 },
          expectedDuration: 30
        };
      }

      // Identify volatility clusters
      const volatilityClusters = this.identifyVolatilityClusters(portfolioReturns);
      
      // Classify current regime
      const recentVolatility = this.calculateRollingVolatility(portfolioReturns, 20); // 20-day rolling
      const currentRegime = this.classifyVolatilityRegime(recentVolatility);
      
      // Calculate regime probabilities using Markov chain approach
      const regimeProbabilities = this.calculateRegimeProbabilities(volatilityClusters, currentRegime);
      
      // Estimate expected duration in current regime
      const expectedDuration = this.estimateRegimeDuration(volatilityClusters, currentRegime);

      return {
        currentRegime,
        regimeProbabilities,
        expectedDuration
      };

    } catch (error) {
      logger.error('Volatility regime analysis failed', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        currentRegime: 'medium',
        regimeProbabilities: { low: 0.25, medium: 0.5, high: 0.2, extreme: 0.05 },
        expectedDuration: 30
      };
    }
  }

  private calculatePortfolioReturns(
    portfolio: Portfolio,
    historicalData: Record<string, number[]>
  ): number[] {
    const maxLength = Math.max(...Object.values(historicalData).map(data => data.length));
    const portfolioReturns: number[] = [];

    for (let i = 0; i < maxLength; i++) {
      let portfolioReturn = 0;
      
      for (const position of portfolio.positions) {
        const assetReturns = historicalData[position.token.symbol] || [];
        const assetReturn = assetReturns[i] || 0;
        const weight = position.percentage / 100;
        
        portfolioReturn += weight * assetReturn;
      }
      
      portfolioReturns.push(portfolioReturn);
    }

    return portfolioReturns;
  }

  private identifyVolatilityClusters(returns: number[]): VolatilityCluster[] {
    const clusters: VolatilityCluster[] = [];
    const windowSize = 20; // 20-day rolling window
    
    const rollingVolatilities = [];
    for (let i = windowSize; i < returns.length; i++) {
      const window = returns.slice(i - windowSize, i);
      const vol = this.calculateVolatilityFromReturns(window);
      rollingVolatilities.push(vol);
    }

    // Identify clusters of similar volatility
    let clusterStart = 0;
    let currentRegime = this.classifyVolatilityRegime(rollingVolatilities[0]);

    for (let i = 1; i < rollingVolatilities.length; i++) {
      const regime = this.classifyVolatilityRegime(rollingVolatilities[i]);
      
      if (regime !== currentRegime) {
        // End current cluster and start new one
        clusters.push({
          startDate: Date.now() - (rollingVolatilities.length - clusterStart) * 24 * 60 * 60 * 1000,
          endDate: Date.now() - (rollingVolatilities.length - i) * 24 * 60 * 60 * 1000,
          avgVolatility: rollingVolatilities.slice(clusterStart, i).reduce((sum, vol) => sum + vol, 0) / (i - clusterStart),
          duration: i - clusterStart,
          type: currentRegime
        });
        
        clusterStart = i;
        currentRegime = regime;
      }
    }

    // Add final cluster
    if (clusterStart < rollingVolatilities.length) {
      clusters.push({
        startDate: Date.now() - (rollingVolatilities.length - clusterStart) * 24 * 60 * 60 * 1000,
        endDate: Date.now(),
        avgVolatility: rollingVolatilities.slice(clusterStart).reduce((sum, vol) => sum + vol, 0) / (rollingVolatilities.length - clusterStart),
        duration: rollingVolatilities.length - clusterStart,
        type: currentRegime
      });
    }

    return clusters;
  }

  private calculateRollingVolatility(returns: number[], window: number): number {
    if (returns.length < window) return 0;
    
    const recentReturns = returns.slice(-window);
    return this.calculateVolatilityFromReturns(recentReturns);
  }

  private calculateVolatilityFromReturns(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    return Math.sqrt(variance * 252); // Annualized
  }

  private classifyVolatilityRegime(volatility: number): 'low' | 'medium' | 'high' | 'extreme' {
    // Volatility regime thresholds (annualized)
    if (volatility < 0.15) return 'low';       // < 15%
    if (volatility < 0.30) return 'medium';    // 15-30%
    if (volatility < 0.60) return 'high';      // 30-60%
    return 'extreme';                          // > 60%
  }

  private calculateRegimeProbabilities(
    clusters: VolatilityCluster[],
    currentRegime: string
  ): Record<string, number> {
    if (clusters.length === 0) {
      return { low: 0.25, medium: 0.5, high: 0.2, extreme: 0.05 };
    }

    // Calculate transition probabilities
    const regimeCounts = { low: 0, medium: 0, high: 0, extreme: 0 };
    const transitions: Record<string, Record<string, number>> = {
      low: { low: 0, medium: 0, high: 0, extreme: 0 },
      medium: { low: 0, medium: 0, high: 0, extreme: 0 },
      high: { low: 0, medium: 0, high: 0, extreme: 0 },
      extreme: { low: 0, medium: 0, high: 0, extreme: 0 }
    };

    clusters.forEach(cluster => {
      regimeCounts[cluster.type]++;
    });

    for (let i = 1; i < clusters.length; i++) {
      const fromRegime = clusters[i - 1].type;
      const toRegime = clusters[i].type;
      transitions[fromRegime][toRegime]++;
    }

    // Calculate next-period probabilities
    const totalTransitions = Object.values(transitions[currentRegime]).reduce((sum, count) => sum + count, 0);
    
    if (totalTransitions === 0) {
      // No historical transitions, use long-term frequencies
      const totalClusters = clusters.length;
      return {
        low: regimeCounts.low / totalClusters,
        medium: regimeCounts.medium / totalClusters,
        high: regimeCounts.high / totalClusters,
        extreme: regimeCounts.extreme / totalClusters
      };
    }

    return {
      low: transitions[currentRegime].low / totalTransitions,
      medium: transitions[currentRegime].medium / totalTransitions,
      high: transitions[currentRegime].high / totalTransitions,
      extreme: transitions[currentRegime].extreme / totalTransitions
    };
  }

  private estimateRegimeDuration(
    clusters: VolatilityCluster[],
    currentRegime: string
  ): number {
    const regimeClusters = clusters.filter(cluster => cluster.type === currentRegime);
    
    if (regimeClusters.length === 0) return 30; // Default 30 days
    
    const avgDuration = regimeClusters.reduce((sum, cluster) => sum + cluster.duration, 0) / regimeClusters.length;
    return Math.round(avgDuration);
  }

  // Public utility methods
  async getVolatilityBreakdown(portfolio: Portfolio): Promise<{
    assetContributions: Array<{ asset: string; contribution: number }>;
    riskSources: { assetSpecific: number; systematic: number; correlation: number };
    regimeInfo: { current: string; probability: number };
  }> {
    const cached = this.volatilityCache.get(portfolio.id);
    
    if (cached && Date.now() - cached.timestamp < 300000) { // 5 minutes cache
      return {
        assetContributions: cached.metrics.volatilityContributions.map((vc: any) => ({
          asset: vc.asset,
          contribution: vc.contribution
        })),
        riskSources: {
          assetSpecific: cached.metrics.riskDecomposition.assetSpecificRisk,
          systematic: cached.metrics.riskDecomposition.systematicRisk,
          correlation: cached.metrics.riskDecomposition.correlationEffect
        },
        regimeInfo: {
          current: cached.metrics.regimeAnalysis.currentRegime,
          probability: cached.metrics.regimeAnalysis.regimeProbabilities[cached.metrics.regimeAnalysis.currentRegime] || 0
        }
      };
    }

    // Recalculate if cache is stale
    const metrics = await this.calculateVolatility(portfolio);
    return this.getVolatilityBreakdown(portfolio);
  }

  setRiskTolerance(riskTolerance: 'conservative' | 'moderate' | 'aggressive'): void {
    this.riskTolerance = riskTolerance;
    this.volatilityCache.clear(); // Clear cache when risk tolerance changes
  }

  clearCache(): void {
    this.volatilityCache.clear();
    this.garchModels.clear();
  }
}
