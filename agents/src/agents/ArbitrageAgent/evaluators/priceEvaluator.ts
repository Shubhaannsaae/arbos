import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { ArbitrageOpportunity, PriceData } from '../../../shared/types/market';
import { CHAINLINK_THRESHOLDS } from '../../../shared/constants/thresholds';
import { ChainlinkProvider } from '../providers/chainlinkProvider';

export interface PriceEvaluationResult {
  isValid: boolean;
  confidence: number;
  freshness: number;
  deviation: number;
  recommendation: 'proceed' | 'monitor' | 'reject';
  reasons: string[];
  chainlinkValidation: {
    sourceValid: boolean;
    targetValid: boolean;
    crossValidation: boolean;
    ageSeconds: number;
  };
}

export class PriceEvaluator {
  private chainlinkProvider: ChainlinkProvider;
  private priceHistory: Map<string, PriceData[]> = new Map();

  constructor(chainlinkProvider: ChainlinkProvider) {
    this.chainlinkProvider = chainlinkProvider;
  }

  async evaluatePriceDiscrepancy(opportunity: ArbitrageOpportunity): Promise<PriceEvaluationResult> {
    const reasons: string[] = [];
    let confidence = 1.0;
    let recommendation: 'proceed' | 'monitor' | 'reject' = 'proceed';

    logger.debug('Evaluating price discrepancy', {
      opportunityId: opportunity.id,
      tokenPair: opportunity.tokenPair,
      priceDifference: opportunity.priceDifferencePercentage
    });

    try {
      // 1. Validate price freshness using Chainlink data
      const freshnessResult = await this.validatePriceFreshness(opportunity);
      
      if (!freshnessResult.valid) {
        reasons.push(`Stale price data: ${freshnessResult.ageSeconds}s old`);
        confidence *= 0.5;
        recommendation = 'monitor';
      }

      // 2. Cross-validate with Chainlink price feeds
      const chainlinkValidation = await this.crossValidateWithChainlink(opportunity);
      
      if (!chainlinkValidation.crossValidation) {
        reasons.push('Price discrepancy not confirmed by Chainlink feeds');
        confidence *= 0.3;
        
        if (confidence < 0.5) {
          recommendation = 'reject';
        }
      }

      // 3. Analyze price volatility and stability
      const volatilityAnalysis = await this.analyzeVolatility(opportunity);
      
      if (volatilityAnalysis.isHighVolatility) {
        reasons.push(`High volatility detected: ${volatilityAnalysis.volatilityPercent}%`);
        confidence *= 0.7;
      }

      // 4. Check for potential price manipulation
      const manipulationRisk = await this.detectPriceManipulation(opportunity);
      
      if (manipulationRisk.riskLevel > 0.7) {
        reasons.push(`Potential price manipulation detected: ${manipulationRisk.indicators.join(', ')}`);
        confidence *= 0.2;
        recommendation = 'reject';
      }

      // 5. Validate price impact expectations
      const priceImpactValidation = await this.validatePriceImpact(opportunity);
      
      if (!priceImpactValidation.reasonable) {
        reasons.push(`Unreasonable price impact: ${priceImpactValidation.impactPercent}%`);
        confidence *= 0.6;
      }

      // 6. Calculate overall deviation from expected ranges
      const deviation = this.calculatePriceDeviation(opportunity, chainlinkValidation);

      // Final confidence adjustment based on deviation
      if (deviation > 10) {
        confidence *= 0.4;
        recommendation = 'reject';
      } else if (deviation > 5) {
        confidence *= 0.7;
        recommendation = 'monitor';
      }

      const result: PriceEvaluationResult = {
        isValid: confidence > 0.5 && recommendation !== 'reject',
        confidence,
        freshness: freshnessResult.ageSeconds,
        deviation,
        recommendation,
        reasons,
        chainlinkValidation
      };

      logger.debug('Price evaluation completed', {
        opportunityId: opportunity.id,
        result: {
          isValid: result.isValid,
          confidence: result.confidence,
          recommendation: result.recommendation,
          reasonsCount: result.reasons.length
        }
      });

      return result;

    } catch (error) {
      logger.error('Price evaluation failed', {
        opportunityId: opportunity.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        isValid: false,
        confidence: 0,
        freshness: 0,
        deviation: 100,
        recommendation: 'reject',
        reasons: ['Price evaluation failed: ' + (error instanceof Error ? error.message : String(error))],
        chainlinkValidation: {
          sourceValid: false,
          targetValid: false,
          crossValidation: false,
          ageSeconds: 0
        }
      };
    }
  }

  private async validatePriceFreshness(opportunity: ArbitrageOpportunity): Promise<{
    valid: boolean;
    ageSeconds: number;
  }> {
    const maxAgeSeconds = CHAINLINK_THRESHOLDS.MAX_PRICE_DEVIATION;
    const now = Date.now() / 1000;
    
    const sourceAge = now - (opportunity.chainlinkData.priceAge || now);
    const targetAge = opportunity.targetExchange.chainId !== opportunity.sourceExchange.chainId 
      ? await this.getChainlinkPriceAge(opportunity.tokenPair, opportunity.targetExchange.chainId)
      : sourceAge;

    const maxAge = Math.max(sourceAge, targetAge);
    
    return {
      valid: maxAge < maxAgeSeconds,
      ageSeconds: maxAge
    };
  }

  private async crossValidateWithChainlink(opportunity: ArbitrageOpportunity): Promise<{
    sourceValid: boolean;
    targetValid: boolean;
    crossValidation: boolean;
    ageSeconds: number;
  }> {
    try {
      // Get current Chainlink prices for both chains
      const [sourcePrice, targetPrice] = await Promise.all([
        this.chainlinkProvider.getLatestPrice(opportunity.tokenPair, opportunity.sourceExchange.chainId),
        this.chainlinkProvider.getLatestPrice(opportunity.tokenPair, opportunity.targetExchange.chainId)
      ]);

      const sourceValid = sourcePrice !== null;
      const targetValid = targetPrice !== null;

      let crossValidation = false;
      let ageSeconds = 0;

      if (sourceValid && targetValid) {
        // Compare DEX prices with Chainlink prices
        const sourceDexPrice = parseFloat(ethers.utils.formatEther(opportunity.sourceExchange.price));
        const targetDexPrice = parseFloat(ethers.utils.formatEther(opportunity.targetExchange.price));
        
        const sourceChainlinkPrice = parseFloat(ethers.utils.formatUnits(sourcePrice.answer, sourcePrice.decimals));
        const targetChainlinkPrice = parseFloat(ethers.utils.formatUnits(targetPrice.answer, targetPrice.decimals));

        // Check if DEX prices are within reasonable range of Chainlink prices
        const sourceDeviation = Math.abs(sourceDexPrice - sourceChainlinkPrice) / sourceChainlinkPrice;
        const targetDeviation = Math.abs(targetDexPrice - targetChainlinkPrice) / targetChainlinkPrice;

        // Allow 5% deviation from Chainlink prices
        const deviationThreshold = 0.05;
        
        if (sourceDeviation < deviationThreshold && targetDeviation < deviationThreshold) {
          // Check if the arbitrage opportunity is confirmed by Chainlink price difference
          const chainlinkPriceDiff = Math.abs(targetChainlinkPrice - sourceChainlinkPrice);
          const dexPriceDiff = Math.abs(targetDexPrice - sourceDexPrice);
          
          // Arbitrage should be in the same direction and similar magnitude
          const directionMatch = (targetDexPrice > sourceDexPrice) === (targetChainlinkPrice > sourceChainlinkPrice);
          const magnitudeMatch = Math.abs(chainlinkPriceDiff - dexPriceDiff) / Math.max(chainlinkPriceDiff, dexPriceDiff) < 0.2;
          
          crossValidation = directionMatch && magnitudeMatch;
        }

        ageSeconds = Math.max(
          Date.now() / 1000 - sourcePrice.updatedAt,
          Date.now() / 1000 - targetPrice.updatedAt
        );
      }

      return {
        sourceValid,
        targetValid,
        crossValidation,
        ageSeconds
      };

    } catch (error) {
      logger.warn('Chainlink cross-validation failed', {
        opportunityId: opportunity.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        sourceValid: false,
        targetValid: false,
        crossValidation: false,
        ageSeconds: 0
      };
    }
  }

  private async analyzeVolatility(opportunity: ArbitrageOpportunity): Promise<{
    isHighVolatility: boolean;
    volatilityPercent: number;
  }> {
    try {
      // Get recent price history for volatility calculation
      const priceHistory = await this.getPriceHistory(opportunity.tokenPair, 24); // 24 hours
      
      if (priceHistory.length < 10) {
        // Not enough data, assume moderate volatility
        return { isHighVolatility: false, volatilityPercent: 5 };
      }

      // Calculate standard deviation of price changes
      const priceChanges = [];
      for (let i = 1; i < priceHistory.length; i++) {
        const prevPrice = parseFloat(ethers.utils.formatEther(priceHistory[i - 1].price));
        const currentPrice = parseFloat(ethers.utils.formatEther(priceHistory[i].price));
        const change = (currentPrice - prevPrice) / prevPrice * 100;
        priceChanges.push(change);
      }

      const mean = priceChanges.reduce((sum, change) => sum + change, 0) / priceChanges.length;
      const variance = priceChanges.reduce((sum, change) => sum + Math.pow(change - mean, 2), 0) / priceChanges.length;
      const volatilityPercent = Math.sqrt(variance);

      // Consider high volatility if standard deviation > 3%
      const isHighVolatility = volatilityPercent > 3;

      return { isHighVolatility, volatilityPercent };

    } catch (error) {
      logger.warn('Volatility analysis failed', {
        opportunityId: opportunity.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return { isHighVolatility: false, volatilityPercent: 0 };
    }
  }

  private async detectPriceManipulation(opportunity: ArbitrageOpportunity): Promise<{
    riskLevel: number;
    indicators: string[];
  }> {
    const indicators: string[] = [];
    let riskLevel = 0;

    // 1. Check for unrealistic price differences
    if (opportunity.priceDifferencePercentage > 10) {
      indicators.push('Unrealistic price difference');
      riskLevel += 0.4;
    }

    // 2. Check liquidity to price difference ratio
    const minLiquidity = Math.min(
      parseFloat(ethers.utils.formatEther(opportunity.sourceExchange.liquidity)),
      parseFloat(ethers.utils.formatEther(opportunity.targetExchange.liquidity))
    );

    if (minLiquidity < 100000 && opportunity.priceDifferencePercentage > 2) {
      indicators.push('Low liquidity with high price difference');
      riskLevel += 0.3;
    }

    // 3. Check for flash loan style patterns
    const recentHistory = await this.getPriceHistory(opportunity.tokenPair, 1); // Last hour
    if (recentHistory.length > 0) {
      const recentVolatility = this.calculateRecentVolatility(recentHistory);
      if (recentVolatility > 5) {
        indicators.push('Recent high volatility suggesting manipulation');
        riskLevel += 0.2;
      }
    }

    // 4. Check for exchange-specific anomalies
    if (this.isUnknownExchange(opportunity.sourceExchange.name) || 
        this.isUnknownExchange(opportunity.targetExchange.name)) {
      indicators.push('Unknown or high-risk exchange involved');
      riskLevel += 0.3;
    }

    return {
      riskLevel: Math.min(riskLevel, 1.0),
      indicators
    };
  }

  private async validatePriceImpact(opportunity: ArbitrageOpportunity): Promise<{
    reasonable: boolean;
    impactPercent: number;
  }> {
    // Estimate price impact based on trade size vs liquidity
    const sourceImpact = parseFloat(ethers.utils.formatEther(opportunity.maxTradeSize)) / 
                        parseFloat(ethers.utils.formatEther(opportunity.sourceExchange.liquidity)) * 100;
    
    const targetImpact = parseFloat(ethers.utils.formatEther(opportunity.maxTradeSize)) / 
                        parseFloat(ethers.utils.formatEther(opportunity.targetExchange.liquidity)) * 100;

    const maxImpact = Math.max(sourceImpact, targetImpact);

    // Price impact should generally be less than 2% for reasonable arbitrage
    const reasonable = maxImpact < 2.0;

    return {
      reasonable,
      impactPercent: maxImpact
    };
  }

  private calculatePriceDeviation(
    opportunity: ArbitrageOpportunity, 
    chainlinkValidation: any
  ): number {
    if (!chainlinkValidation.crossValidation) {
      return 15; // High deviation if no Chainlink validation
    }

    // Calculate deviation from expected price based on historical patterns
    const historicalAverage = this.getHistoricalAverageSpread(opportunity.tokenPair);
    const currentSpread = opportunity.priceDifferencePercentage;

    return Math.abs(currentSpread - historicalAverage);
  }

  private async getChainlinkPriceAge(tokenPair: string, chainId: number): Promise<number> {
    try {
      const priceData = await this.chainlinkProvider.getLatestPrice(tokenPair, chainId);
      return priceData ? Date.now() / 1000 - priceData.updatedAt : 0;
    } catch {
      return 0;
    }
  }

  private async getPriceHistory(tokenPair: string, hoursBack: number): Promise<PriceData[]> {
    const key = `${tokenPair}_${hoursBack}h`;
    
    if (this.priceHistory.has(key)) {
      const cached = this.priceHistory.get(key)!;
      // Return cached data if less than 5 minutes old
      if (cached.length > 0 && Date.now() - cached[0].timestamp < 300000) {
        return cached;
      }
    }

    try {
      // In production, this would fetch from a price history API or database
      // For now, return empty array
      const history: PriceData[] = [];
      this.priceHistory.set(key, history);
      return history;
    } catch {
      return [];
    }
  }

  private calculateRecentVolatility(priceHistory: PriceData[]): number {
    if (priceHistory.length < 2) return 0;

    const changes = [];
    for (let i = 1; i < priceHistory.length; i++) {
      const prev = parseFloat(ethers.utils.formatEther(priceHistory[i - 1].price));
      const current = parseFloat(ethers.utils.formatEther(priceHistory[i].price));
      changes.push(Math.abs((current - prev) / prev * 100));
    }

    return changes.reduce((sum, change) => sum + change, 0) / changes.length;
  }

  private isUnknownExchange(exchangeName: string): boolean {
    const knownExchanges = [
      'uniswap_v3', 'uniswap_v2', 'sushiswap', 'curve', 'balancer', 
      'pancakeswap', 'trader_joe', 'quickswap'
    ];
    
    return !knownExchanges.includes(exchangeName.toLowerCase());
  }

  private getHistoricalAverageSpread(tokenPair: string): number {
    // In production, this would calculate from historical data
    // For now, return reasonable defaults based on token pair
    const spreads: Record<string, number> = {
      'ETH/USD': 0.1,
      'BTC/USD': 0.15,
      'USDC/USDT': 0.02,
      'ETH/USDC': 0.05,
      'BTC/ETH': 0.08
    };

    return spreads[tokenPair] || 0.2; // Default 0.2% spread
  }

  // Public method to update price history (called by external price feeds)
  public updatePriceHistory(tokenPair: string, priceData: PriceData): void {
    const keys = [`${tokenPair}_1h`, `${tokenPair}_24h`];
    
    keys.forEach(key => {
      const history = this.priceHistory.get(key) || [];
      history.unshift(priceData);
      
      // Keep only relevant data (1 hour = 60 points, 24 hours = 1440 points)
      const maxPoints = key.includes('1h') ? 60 : 1440;
      if (history.length > maxPoints) {
        history.splice(maxPoints);
      }
      
      this.priceHistory.set(key, history);
    });
  }
}
