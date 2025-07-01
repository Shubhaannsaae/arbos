import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { ArbitrageOpportunity } from '../../../shared/types/market';
import { RISK_THRESHOLDS, ARBITRAGE_THRESHOLDS } from '../../../shared/constants/thresholds';

export interface RiskAssessment {
  overallRisk: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  liquidityRisk: number;
  slippageRisk: number;
  crossChainRisk: number;
  marketRisk: number;
  executionRisk: number;
  timeRisk: number;
  gasRisk: number;
  concentrationRisk: number;
  recommendation: 'proceed' | 'caution' | 'avoid';
  riskFactors: RiskFactor[];
  mitigation: MitigationStrategy[];
}

export interface RiskFactor {
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  impact: number;
  description: string;
  probability: number;
}

export interface MitigationStrategy {
  risk: string;
  strategy: string;
  effectiveness: number;
  cost: number;
}

export class RiskEvaluator {
  private riskTolerance: 'low' | 'medium' | 'high';
  private riskWeights: Record<string, number>;

  constructor(riskTolerance: 'low' | 'medium' | 'high') {
    this.riskTolerance = riskTolerance;
    
    // Risk weights based on tolerance level
    this.riskWeights = this.getRiskWeights(riskTolerance);
  }

  async evaluateArbitrageRisk(
    opportunity: ArbitrageOpportunity,
    profitabilityAnalysis?: any
  ): Promise<RiskAssessment> {
    
    logger.debug('Starting risk assessment', {
      opportunityId: opportunity.id,
      riskTolerance: this.riskTolerance
    });

    const riskFactors: RiskFactor[] = [];

    // 1. Assess liquidity risk
    const liquidityRisk = this.assessLiquidityRisk(opportunity, riskFactors);

    // 2. Assess slippage risk
    const slippageRisk = this.assessSlippageRisk(opportunity, profitabilityAnalysis, riskFactors);

    // 3. Assess cross-chain execution risk
    const crossChainRisk = this.assessCrossChainRisk(opportunity, riskFactors);

    // 4. Assess market and volatility risk
    const marketRisk = this.assessMarketRisk(opportunity, riskFactors);

    // 5. Assess execution complexity risk
    const executionRisk = this.assessExecutionRisk(opportunity, riskFactors);

    // 6. Assess timing and latency risk
    const timeRisk = this.assessTimeRisk(opportunity, riskFactors);

    // 7. Assess gas price risk
    const gasRisk = this.assessGasRisk(opportunity, riskFactors);

    // 8. Assess concentration risk
    const concentrationRisk = this.assessConcentrationRisk(opportunity, riskFactors);

    // Calculate weighted overall risk
    const overallRisk = this.calculateOverallRisk({
      liquidityRisk,
      slippageRisk,
      crossChainRisk,
      marketRisk,
      executionRisk,
      timeRisk,
      gasRisk,
      concentrationRisk
    });

    // Determine risk level and recommendation
    const riskLevel = this.determineRiskLevel(overallRisk);
    const recommendation = this.getRecommendation(overallRisk, riskLevel);

    // Generate mitigation strategies
    const mitigation = this.generateMitigationStrategies(riskFactors);

    const assessment: RiskAssessment = {
      overallRisk,
      riskLevel,
      liquidityRisk,
      slippageRisk,
      crossChainRisk,
      marketRisk,
      executionRisk,
      timeRisk,
      gasRisk,
      concentrationRisk,
      recommendation,
      riskFactors,
      mitigation
    };

    logger.debug('Risk assessment completed', {
      opportunityId: opportunity.id,
      overallRisk,
      riskLevel,
      recommendation,
      factorCount: riskFactors.length
    });

    return assessment;
  }

  private assessLiquidityRisk(opportunity: ArbitrageOpportunity, riskFactors: RiskFactor[]): number {
    let risk = 0;

    const sourceDepth = parseFloat(ethers.utils.formatEther(opportunity.sourceExchange.liquidity));
    const targetDepth = parseFloat(ethers.utils.formatEther(opportunity.targetExchange.liquidity));
    const tradeSize = parseFloat(ethers.utils.formatEther(opportunity.maxTradeSize));

    // Assess source liquidity
    const sourceUtilization = tradeSize / sourceDepth * 100;
    if (sourceUtilization > 20) {
      risk += 40;
      riskFactors.push({
        type: 'liquidity',
        severity: 'high',
        impact: 40,
        description: `High source liquidity utilization: ${sourceUtilization.toFixed(1)}%`,
        probability: 0.8
      });
    } else if (sourceUtilization > 10) {
      risk += 20;
      riskFactors.push({
        type: 'liquidity',
        severity: 'medium',
        impact: 20,
        description: `Medium source liquidity utilization: ${sourceUtilization.toFixed(1)}%`,
        probability: 0.6
      });
    }

    // Assess target liquidity
    const targetUtilization = tradeSize / targetDepth * 100;
    if (targetUtilization > 20) {
      risk += 40;
      riskFactors.push({
        type: 'liquidity',
        severity: 'high',
        impact: 40,
        description: `High target liquidity utilization: ${targetUtilization.toFixed(1)}%`,
        probability: 0.8
      });
    } else if (targetUtilization > 10) {
      risk += 20;
      riskFactors.push({
        type: 'liquidity',
        severity: 'medium',
        impact: 20,
        description: `Medium target liquidity utilization: ${targetUtilization.toFixed(1)}%`,
        probability: 0.6
      });
    }

    // Assess absolute liquidity levels
    const minLiquidity = Math.min(sourceDepth, targetDepth);
    if (minLiquidity < 100000) {
      risk += 30;
      riskFactors.push({
        type: 'liquidity',
        severity: 'high',
        impact: 30,
        description: `Low absolute liquidity: $${minLiquidity.toLocaleString()}`,
        probability: 0.9
      });
    } else if (minLiquidity < 500000) {
      risk += 15;
      riskFactors.push({
        type: 'liquidity',
        severity: 'medium',
        impact: 15,
        description: `Medium absolute liquidity: $${minLiquidity.toLocaleString()}`,
        probability: 0.7
      });
    }

    return Math.min(risk, 100);
  }

  private assessSlippageRisk(
    opportunity: ArbitrageOpportunity, 
    profitabilityAnalysis: any, 
    riskFactors: RiskFactor[]
  ): number {
    let risk = 0;

    // Use profitability analysis if available, otherwise estimate
    const estimatedSlippage = profitabilityAnalysis?.slippageAnalysis?.estimated || 
                             this.estimateSlippage(opportunity);

    if (estimatedSlippage > 2.0) {
      risk += 50;
      riskFactors.push({
        type: 'slippage',
        severity: 'critical',
        impact: 50,
        description: `Very high estimated slippage: ${estimatedSlippage.toFixed(2)}%`,
        probability: 0.9
      });
    } else if (estimatedSlippage > 1.0) {
      risk += 30;
      riskFactors.push({
        type: 'slippage',
        severity: 'high',
        impact: 30,
        description: `High estimated slippage: ${estimatedSlippage.toFixed(2)}%`,
        probability: 0.8
      });
    } else if (estimatedSlippage > 0.5) {
      risk += 15;
      riskFactors.push({
        type: 'slippage',
        severity: 'medium',
        impact: 15,
        description: `Medium estimated slippage: ${estimatedSlippage.toFixed(2)}%`,
        probability: 0.7
      });
    }

    // Factor in price volatility
    if (opportunity.priceDifferencePercentage > 5) {
      risk += 20;
      riskFactors.push({
        type: 'slippage',
        severity: 'high',
        impact: 20,
        description: 'High price volatility increases slippage risk',
        probability: 0.8
      });
    }

    return Math.min(risk, 100);
  }

  private assessCrossChainRisk(opportunity: ArbitrageOpportunity, riskFactors: RiskFactor[]): number {
    if (opportunity.sourceExchange.chainId === opportunity.targetExchange.chainId) {
      return 0; // No cross-chain risk for same-chain arbitrage
    }

    let risk = 20; // Base cross-chain risk

    // Assess chain reliability
    const chainRiskScores = this.getChainRiskScores();
    const sourceChainRisk = chainRiskScores[opportunity.sourceExchange.chainId] || 50;
    const targetChainRisk = chainRiskScores[opportunity.targetExchange.chainId] || 50;

    risk += (sourceChainRisk + targetChainRisk) / 4; // Average and scale

    if (sourceChainRisk > 30 || targetChainRisk > 30) {
      riskFactors.push({
        type: 'cross_chain',
        severity: 'high',
        impact: 25,
        description: 'High-risk blockchain involved in cross-chain execution',
        probability: 0.3
      });
    }

    // Assess bridge reliability and time
    const estimatedBridgeTime = this.estimateBridgeTime(
      opportunity.sourceExchange.chainId,
      opportunity.targetExchange.chainId
    );

    if (estimatedBridgeTime > 1800) { // 30 minutes
      risk += 25;
      riskFactors.push({
        type: 'cross_chain',
        severity: 'high',
        impact: 25,
        description: `Long bridge time increases execution risk: ${estimatedBridgeTime / 60} minutes`,
        probability: 0.7
      });
    } else if (estimatedBridgeTime > 600) { // 10 minutes
      risk += 15;
      riskFactors.push({
        type: 'cross_chain',
        severity: 'medium',
        impact: 15,
        description: `Medium bridge time: ${estimatedBridgeTime / 60} minutes`,
        probability: 0.5
      });
    }

    // CCIP reliability factor
    if (opportunity.chainlinkData.feedReliability < 0.9) {
      risk += 20;
      riskFactors.push({
        type: 'cross_chain',
        severity: 'high',
        impact: 20,
        description: 'Low Chainlink feed reliability for cross-chain validation',
        probability: 0.6
      });
    }

    return Math.min(risk, 100);
  }

  private assessMarketRisk(opportunity: ArbitrageOpportunity, riskFactors: RiskFactor[]): number {
    let risk = 10; // Base market risk

    // Assess market volatility based on price difference magnitude
    if (opportunity.priceDifferencePercentage > 10) {
      risk += 40;
      riskFactors.push({
        type: 'market',
        severity: 'critical',
        impact: 40,
        description: 'Extremely high price difference suggests market instability',
        probability: 0.8
      });
    } else if (opportunity.priceDifferencePercentage > 5) {
      risk += 25;
      riskFactors.push({
        type: 'market',
        severity: 'high',
        impact: 25,
        description: 'High price difference suggests market volatility',
        probability: 0.7
      });
    } else if (opportunity.priceDifferencePercentage > 2) {
      risk += 10;
      riskFactors.push({
        type: 'market',
        severity: 'medium',
        impact: 10,
        description: 'Medium price difference indicates some market stress',
        probability: 0.5
      });
    }

    // Assess confidence level
    if (opportunity.confidence < 0.7) {
      risk += 30;
      riskFactors.push({
        type: 'market',
        severity: 'high',
        impact: 30,
        description: `Low confidence in opportunity: ${(opportunity.confidence * 100).toFixed(1)}%`,
        probability: 0.8
      });
    } else if (opportunity.confidence < 0.8) {
      risk += 15;
      riskFactors.push({
        type: 'market',
        severity: 'medium',
        impact: 15,
        description: `Medium confidence in opportunity: ${(opportunity.confidence * 100).toFixed(1)}%`,
        probability: 0.6
      });
    }

    return Math.min(risk, 100);
  }

  private assessExecutionRisk(opportunity: ArbitrageOpportunity, riskFactors: RiskFactor[]): number {
    let risk = 0;

    // Complexity-based risk
    switch (opportunity.executionComplexity) {
      case 'complex':
        risk += 30;
        riskFactors.push({
          type: 'execution',
          severity: 'high',
          impact: 30,
          description: 'Complex execution increases failure probability',
          probability: 0.6
        });
        break;
      case 'medium':
        risk += 15;
        riskFactors.push({
          type: 'execution',
          severity: 'medium',
          impact: 15,
          description: 'Medium complexity execution',
          probability: 0.4
        });
        break;
    }

    // MEV and front-running risk
    if (opportunity.potentialProfitPercentage > 3) {
      risk += 25;
      riskFactors.push({
        type: 'execution',
        severity: 'high',
        impact: 25,
        description: 'High profit opportunity attracts MEV competition',
        probability: 0.7
      });
    } else if (opportunity.potentialProfitPercentage > 1) {
      risk += 10;
      riskFactors.push({
        type: 'execution',
        severity: 'medium',
        impact: 10,
        description: 'Medium profit opportunity may attract competition',
        probability: 0.4
      });
    }

    return Math.min(risk, 100);
  }

  private assessTimeRisk(opportunity: ArbitrageOpportunity, riskFactors: RiskFactor[]): number {
    let risk = 0;

    const now = Date.now();
    const timeToExpiry = opportunity.expiresAt - now;
    const executionTime = opportunity.estimatedExecutionTime * 1000; // Convert to ms

    // Risk based on time pressure
    if (timeToExpiry < executionTime * 2) {
      risk += 40;
      riskFactors.push({
        type: 'time',
        severity: 'critical',
        impact: 40,
        description: 'Very tight timing - opportunity may expire during execution',
        probability: 0.8
      });
    } else if (timeToExpiry < executionTime * 5) {
      risk += 20;
      riskFactors.push({
        type: 'time',
        severity: 'high',
        impact: 20,
        description: 'Limited time window for execution',
        probability: 0.6
      });
    }

    // Risk based on estimated execution time
    if (executionTime > 300000) { // 5 minutes
      risk += 25;
      riskFactors.push({
        type: 'time',
        severity: 'high',
        impact: 25,
        description: 'Long execution time increases market movement risk',
        probability: 0.7
      });
    } else if (executionTime > 120000) { // 2 minutes
      risk += 10;
      riskFactors.push({
        type: 'time',
        severity: 'medium',
        impact: 10,
        description: 'Medium execution time',
        probability: 0.5
      });
    }

    return Math.min(risk, 100);
  }

  private assessGasRisk(opportunity: ArbitrageOpportunity, riskFactors: RiskFactor[]): number {
    let risk = 0;

    // Estimate current gas costs as percentage of profit
    const estimatedGasCost = parseFloat(ethers.utils.formatEther(opportunity.estimatedGasCost));
    const potentialProfit = parseFloat(ethers.utils.formatEther(opportunity.potentialProfit));
    
    if (potentialProfit > 0) {
      const gasCostPercentage = (estimatedGasCost / potentialProfit) * 100;

      if (gasCostPercentage > 50) {
        risk += 40;
        riskFactors.push({
          type: 'gas',
          severity: 'critical',
          impact: 40,
          description: `Gas costs are ${gasCostPercentage.toFixed(1)}% of potential profit`,
          probability: 0.9
        });
      } else if (gasCostPercentage > 25) {
        risk += 25;
        riskFactors.push({
          type: 'gas',
          severity: 'high',
          impact: 25,
          description: `Gas costs are ${gasCostPercentage.toFixed(1)}% of potential profit`,
          probability: 0.8
        });
      } else if (gasCostPercentage > 10) {
        risk += 10;
        riskFactors.push({
          type: 'gas',
          severity: 'medium',
          impact: 10,
          description: `Gas costs are ${gasCostPercentage.toFixed(1)}% of potential profit`,
          probability: 0.6
        });
      }
    }

    return Math.min(risk, 100);
  }

  private assessConcentrationRisk(opportunity: ArbitrageOpportunity, riskFactors: RiskFactor[]): number {
    let risk = 0;

    // Risk from using unknown or centralized exchanges
    const exchangeRiskScores = this.getExchangeRiskScores();
    const sourceRisk = exchangeRiskScores[opportunity.sourceExchange.name] || 30;
    const targetRisk = exchangeRiskScores[opportunity.targetExchange.name] || 30;

    if (sourceRisk > 40 || targetRisk > 40) {
      risk += 25;
      riskFactors.push({
        type: 'concentration',
        severity: 'high',
        impact: 25,
        description: 'High-risk exchange involved',
        probability: 0.5
      });
    } else if (sourceRisk > 25 || targetRisk > 25) {
      risk += 15;
      riskFactors.push({
        type: 'concentration',
        severity: 'medium',
        impact: 15,
        description: 'Medium-risk exchange involved',
        probability: 0.3
      });
    }

    return Math.min(risk, 100);
  }

  private calculateOverallRisk(risks: Record<string, number>): number {
    let weightedRisk = 0;
    let totalWeight = 0;

    Object.entries(risks).forEach(([riskType, riskValue]) => {
      const weight = this.riskWeights[riskType] || 1;
      weightedRisk += riskValue * weight;
      totalWeight += weight;
    });

    return Math.min(weightedRisk / totalWeight, 100);
  }

  private determineRiskLevel(overallRisk: number): 'low' | 'medium' | 'high' | 'critical' {
    if (overallRisk >= RISK_THRESHOLDS.CRITICAL_RISK) return 'critical';
    if (overallRisk >= RISK_THRESHOLDS.HIGH_RISK) return 'high';
    if (overallRisk >= RISK_THRESHOLDS.MEDIUM_RISK) return 'medium';
    return 'low';
  }

  private getRecommendation(overallRisk: number, riskLevel: string): 'proceed' | 'caution' | 'avoid' {
    // Adjust recommendation based on risk tolerance
    const toleranceMultiplier = {
      'low': 0.8,
      'medium': 1.0,
      'high': 1.2
    }[this.riskTolerance];

    const adjustedRisk = overallRisk / toleranceMultiplier;

    if (adjustedRisk >= 80) return 'avoid';
    if (adjustedRisk >= 50) return 'caution';
    return 'proceed';
  }

  private generateMitigationStrategies(riskFactors: RiskFactor[]): MitigationStrategy[] {
    const strategies: MitigationStrategy[] = [];

    // Generate strategies for each significant risk factor
    riskFactors.forEach(factor => {
      switch (factor.type) {
        case 'liquidity':
          strategies.push({
            risk: 'liquidity',
            strategy: 'Reduce trade size or split into multiple smaller trades',
            effectiveness: 0.7,
            cost: 0.1
          });
          break;

        case 'slippage':
          strategies.push({
            risk: 'slippage',
            strategy: 'Use limit orders and increase slippage tolerance',
            effectiveness: 0.6,
            cost: 0.2
          });
          break;

        case 'cross_chain':
          strategies.push({
            risk: 'cross_chain',
            strategy: 'Use multiple bridge providers and implement fallback options',
            effectiveness: 0.8,
            cost: 0.3
          });
          break;

        case 'gas':
          strategies.push({
            risk: 'gas',
            strategy: 'Use gas optimization and priority fee management',
            effectiveness: 0.5,
            cost: 0.1
          });
          break;

        case 'execution':
          strategies.push({
            risk: 'execution',
            strategy: 'Implement MEV protection and use private mempools',
            effectiveness: 0.7,
            cost: 0.2
          });
          break;
      }
    });

    return strategies;
  }

  private getRiskWeights(riskTolerance: 'low' | 'medium' | 'high'): Record<string, number> {
    const baseWeights = {
      liquidityRisk: 1.5,
      slippageRisk: 1.2,
      crossChainRisk: 1.3,
      marketRisk: 1.0,
      executionRisk: 1.1,
      timeRisk: 0.9,
      gasRisk: 0.8,
      concentrationRisk: 0.7
    };

    // Adjust weights based on risk tolerance
    const adjustments = {
      'low': { multiplier: 1.2, focus: ['liquidityRisk', 'crossChainRisk'] },
      'medium': { multiplier: 1.0, focus: [] },
      'high': { multiplier: 0.8, focus: ['marketRisk', 'executionRisk'] }
    }[riskTolerance];

    const adjustedWeights = { ...baseWeights };
    
    // Apply general multiplier
    Object.keys(adjustedWeights).forEach(key => {
      adjustedWeights[key] *= adjustments.multiplier;
    });

    // Increase weights for focus areas
    adjustments.focus.forEach(key => {
      adjustedWeights[key] *= 1.3;
    });

    return adjustedWeights;
  }

  private estimateSlippage(opportunity: ArbitrageOpportunity): number {
    const tradeSize = parseFloat(ethers.utils.formatEther(opportunity.maxTradeSize));
    const sourceLiquidity = parseFloat(ethers.utils.formatEther(opportunity.sourceExchange.liquidity));
    const targetLiquidity = parseFloat(ethers.utils.formatEther(opportunity.targetExchange.liquidity));

    // Simple slippage estimation based on trade size vs liquidity
    const sourceSlippage = (tradeSize / sourceLiquidity) * 100 * 0.5; // Assume 0.5% per 1% of liquidity
    const targetSlippage = (tradeSize / targetLiquidity) * 100 * 0.5;

    return sourceSlippage + targetSlippage;
  }

  private getChainRiskScores(): Record<number, number> {
    return {
      1: 10,     // Ethereum - lowest risk
      137: 15,   // Polygon - low risk
      42161: 12, // Arbitrum - low risk
      43114: 18, // Avalanche - low-medium risk
      56: 25,    // BSC - medium risk
      250: 30    // Fantom - medium-high risk
    };
  }

  private getExchangeRiskScores(): Record<string, number> {
    return {
      'uniswap_v3': 10,
      'uniswap_v2': 15,
      'sushiswap': 20,
      'curve': 15,
      'balancer': 20,
      'pancakeswap': 25,
      'trader_joe': 25,
      'quickswap': 30
    };
  }

  private estimateBridgeTime(sourceChain: number, targetChain: number): number {
    // Estimated bridge times in seconds
    const bridgeTimes: Record<string, number> = {
      '1-137': 1200,    // Ethereum to Polygon - 20 minutes
      '1-42161': 600,   // Ethereum to Arbitrum - 10 minutes  
      '1-43114': 900,   // Ethereum to Avalanche - 15 minutes
      '137-42161': 1800, // Polygon to Arbitrum - 30 minutes
      '137-43114': 1200, // Polygon to Avalanche - 20 minutes
      '42161-43114': 1500 // Arbitrum to Avalanche - 25 minutes
    };

    const key = `${Math.min(sourceChain, targetChain)}-${Math.max(sourceChain, targetChain)}`;
    return bridgeTimes[key] || 1800; // Default 30 minutes for unknown combinations
  }
}
