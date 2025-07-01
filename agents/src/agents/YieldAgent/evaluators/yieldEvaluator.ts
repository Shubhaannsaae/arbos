import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { YieldOpportunity } from '../../../shared/types/market';
import { YIELD_THRESHOLDS } from '../../../shared/constants/thresholds';

export interface YieldEvaluation {
  opportunityId: string;
  overallScore: number;
  riskScore: number;
  yieldScore: number;
  liquidityScore: number;
  auditScore: number;
  sustainabilityScore: number;
  recommendation: 'strong_buy' | 'buy' | 'hold' | 'avoid';
  confidence: number;
  keyFactors: YieldFactor[];
  risks: YieldRisk[];
  projectedReturns: ProjectedReturns;
  competitiveAnalysis: CompetitiveAnalysis;
}

export interface YieldFactor {
  factor: string;
  impact: 'positive' | 'negative' | 'neutral';
  weight: number;
  description: string;
  score: number;
}

export interface YieldRisk {
  type: 'smart_contract' | 'liquidity' | 'market' | 'regulatory' | 'operational';
  level: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  probability: number;
  impact: number;
  mitigation: string[];
}

export interface ProjectedReturns {
  conservative: {
    apy: number;
    timeToBreakeven: number;
    confidenceInterval: [number, number];
  };
  realistic: {
    apy: number;
    timeToBreakeven: number;
    confidenceInterval: [number, number];
  };
  optimistic: {
    apy: number;
    timeToBreakeven: number;
    confidenceInterval: [number, number];
  };
}

export interface CompetitiveAnalysis {
  ranking: number;
  totalOpportunities: number;
  betterAlternatives: Array<{
    protocol: string;
    apy: number;
    riskScore: number;
    advantage: string;
  }>;
  marketPosition: 'top_tier' | 'competitive' | 'average' | 'below_average';
  uniqueSellingPoints: string[];
}

export class YieldEvaluator {
  private riskTolerance: 'conservative' | 'moderate' | 'aggressive';
  private evaluationCache: Map<string, YieldEvaluation> = new Map();

  constructor(riskTolerance: 'conservative' | 'moderate' | 'aggressive') {
    this.riskTolerance = riskTolerance;
  }

  async evaluateOpportunity(opportunity: YieldOpportunity): Promise<YieldEvaluation> {
    const startTime = Date.now();

    logger.debug('Starting yield opportunity evaluation', {
      opportunityId: opportunity.id,
      protocol: opportunity.protocol,
      apy: opportunity.apy,
      riskTolerance: this.riskTolerance
    });

    try {
      // Check cache first
      const cacheKey = `${opportunity.id}_${this.riskTolerance}`;
      const cached = this.evaluationCache.get(cacheKey);
      
      if (cached && Date.now() - opportunity.lastUpdated < 300000) { // 5 minutes cache
        return cached;
      }

      // Step 1: Calculate individual scores
      const yieldScore = this.calculateYieldScore(opportunity);
      const riskScore = this.calculateRiskScore(opportunity);
      const liquidityScore = this.calculateLiquidityScore(opportunity);
      const auditScore = this.calculateAuditScore(opportunity);
      const sustainabilityScore = await this.calculateSustainabilityScore(opportunity);

      // Step 2: Calculate overall score
      const overallScore = this.calculateOverallScore({
        yieldScore,
        riskScore,
        liquidityScore,
        auditScore,
        sustainabilityScore
      });

      // Step 3: Identify key factors
      const keyFactors = this.identifyKeyFactors(opportunity, {
        yieldScore,
        riskScore,
        liquidityScore,
        auditScore,
        sustainabilityScore
      });

      // Step 4: Assess risks
      const risks = await this.assessRisks(opportunity);

      // Step 5: Project returns
      const projectedReturns = this.projectReturns(opportunity);

      // Step 6: Competitive analysis
      const competitiveAnalysis = await this.performCompetitiveAnalysis(opportunity);

      // Step 7: Generate recommendation
      const { recommendation, confidence } = this.generateRecommendation(
        overallScore,
        risks,
        opportunity
      );

      const evaluation: YieldEvaluation = {
        opportunityId: opportunity.id,
        overallScore,
        riskScore,
        yieldScore,
        liquidityScore,
        auditScore,
        sustainabilityScore,
        recommendation,
        confidence,
        keyFactors,
        risks,
        projectedReturns,
        competitiveAnalysis
      };

      // Cache the evaluation
      this.evaluationCache.set(cacheKey, evaluation);

      logger.debug('Yield opportunity evaluation completed', {
        opportunityId: opportunity.id,
        overallScore,
        recommendation,
        confidence,
        duration: Date.now() - startTime
      });

      return evaluation;

    } catch (error) {
      logger.error('Yield opportunity evaluation failed', {
        opportunityId: opportunity.id,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime
      });

      throw error;
    }
  }

  private calculateYieldScore(opportunity: YieldOpportunity): number {
    let score = 0;
    
    // Base APY score (0-40 points)
    const normalizedAPY = Math.min(opportunity.apy / 50, 1); // Cap at 50% APY
    score += normalizedAPY * 40;

    // APY consistency bonus (0-15 points)
    if (opportunity.apyHistory && opportunity.apyHistory.length >= 30) {
      const avgHistoricalAPY = opportunity.apyHistory.reduce((sum, apy) => sum + apy, 0) / opportunity.apyHistory.length;
      const consistency = 1 - Math.abs(opportunity.apy - avgHistoricalAPY) / avgHistoricalAPY;
      score += Math.max(0, consistency) * 15;
    }

    // Auto-compound bonus (0-10 points)
    if (opportunity.autoCompound) {
      score += 10;
    }

    // Reward token diversity bonus (0-10 points)
    if (opportunity.rewardTokens && opportunity.rewardTokens.length > 1) {
      score += Math.min(opportunity.rewardTokens.length * 2, 10);
    }

    // Fee structure penalty (0 to -15 points)
    const totalFees = (opportunity.fees.deposit || 0) + 
                     (opportunity.fees.withdrawal || 0) + 
                     (opportunity.fees.performance || 0);
    
    if (totalFees > 5) {
      score -= Math.min(totalFees, 15);
    }

    // Strategy-specific adjustments
    score += this.getStrategyAdjustment(opportunity);

    return Math.max(0, Math.min(100, score));
  }

  private calculateRiskScore(opportunity: YieldOpportunity): number {
    let riskScore = opportunity.riskScore || 50; // Base risk score
    
    // Smart contract risk adjustments
    if (!opportunity.verified) {
      riskScore += 20;
    }

    // Lockup period risk
    if (opportunity.lockupPeriod > 0) {
      const lockupDays = opportunity.lockupPeriod / (24 * 60 * 60);
      if (lockupDays > 365) riskScore += 15; // >1 year
      else if (lockupDays > 90) riskScore += 10; // >3 months
      else if (lockupDays > 30) riskScore += 5; // >1 month
    }

    // Impermanent loss risk (for LP strategies)
    if (opportunity.strategy === 'liquidity_provision') {
      riskScore += opportunity.impermanentLossRisk || 15;
    }

    // New protocol penalty
    const protocolAge = Date.now() - (opportunity.lastUpdated - 365 * 24 * 60 * 60 * 1000); // Estimate
    if (protocolAge < 90 * 24 * 60 * 60 * 1000) { // Less than 90 days
      riskScore += 25;
    }

    // High yield risk (yields too good to be true)
    if (opportunity.apy > 100) {
      riskScore += 30;
    } else if (opportunity.apy > 50) {
      riskScore += 15;
    }

    return Math.max(0, Math.min(100, riskScore));
  }

  private calculateLiquidityScore(opportunity: YieldOpportunity): number {
    let score = opportunity.liquidityScore || 50;

    // TVL-based scoring
    const tvlUsd = parseFloat(ethers.utils.formatEther(opportunity.tvl));
    
    if (tvlUsd > 1000000000) score = Math.max(score, 95); // >$1B
    else if (tvlUsd > 500000000) score = Math.max(score, 90); // >$500M
    else if (tvlUsd > 100000000) score = Math.max(score, 85); // >$100M
    else if (tvlUsd > 50000000) score = Math.max(score, 80); // >$50M
    else if (tvlUsd > 10000000) score = Math.max(score, 70); // >$10M
    else if (tvlUsd > 1000000) score = Math.max(score, 60); // >$1M
    else if (tvlUsd > 100000) score = Math.max(score, 40); // >$100K
    else score = Math.min(score, 20); // <$100K

    // Available liquidity vs position size
    const availableLiquidity = parseFloat(ethers.utils.formatEther(opportunity.liquidity));
    const maxPositionValue = parseFloat(ethers.utils.formatEther(
      opportunity.maximumDeposit || ethers.constants.MaxUint256.div(1000)
    ));

    const liquidityRatio = maxPositionValue / availableLiquidity;
    if (liquidityRatio > 0.1) score -= 20; // Position would be >10% of liquidity
    else if (liquidityRatio > 0.05) score -= 10; // Position would be >5% of liquidity

    return Math.max(0, Math.min(100, score));
  }

  private calculateAuditScore(opportunity: YieldOpportunity): number {
    let score = opportunity.auditScore || 0;

    // Base audit score from opportunity
    if (score === 0) {
      // Calculate based on available information
      if (opportunity.verified) {
        score = 50; // Base score for verified protocols
        
        // Well-known protocols get higher scores
        const wellKnownProtocols = ['aave', 'compound', 'uniswap', 'yearn', 'curve'];
        if (wellKnownProtocols.some(protocol => 
          opportunity.protocol.toLowerCase().includes(protocol)
        )) {
          score = 85;
        }
      } else {
        score = 25; // Low score for unverified
      }
    }

    // Contract address verification bonus
    if (opportunity.contractAddress && 
        ethers.utils.isAddress(opportunity.contractAddress)) {
      score += 5;
    }

    return Math.max(0, Math.min(100, score));
  }

  private async calculateSustainabilityScore(opportunity: YieldOpportunity): Promise<number> {
    let score = 50; // Base sustainability score

    try {
      // Revenue model sustainability
      score += this.assessRevenueModel(opportunity);

      // Token economics
      score += this.assessTokenEconomics(opportunity);

      // Protocol fundamentals
      score += this.assessProtocolFundamentals(opportunity);

      // Market demand sustainability
      score += await this.assessMarketDemand(opportunity);

      return Math.max(0, Math.min(100, score));

    } catch (error) {
      logger.warn('Failed to calculate sustainability score', {
        opportunityId: opportunity.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return 50; // Default sustainability score
    }
  }

  private calculateOverallScore(scores: {
    yieldScore: number;
    riskScore: number;
    liquidityScore: number;
    auditScore: number;
    sustainabilityScore: number;
  }): number {
    // Risk tolerance-based weighting
    const weights = this.getScoreWeights();

    const invertedRiskScore = 100 - scores.riskScore; // Invert risk (lower risk = better)

    const weightedScore = 
      scores.yieldScore * weights.yield +
      invertedRiskScore * weights.risk +
      scores.liquidityScore * weights.liquidity +
      scores.auditScore * weights.audit +
      scores.sustainabilityScore * weights.sustainability;

    return Math.max(0, Math.min(100, weightedScore));
  }

  private getScoreWeights(): {
    yield: number;
    risk: number;
    liquidity: number;
    audit: number;
    sustainability: number;
  } {
    switch (this.riskTolerance) {
      case 'conservative':
        return {
          yield: 0.15,
          risk: 0.35,
          liquidity: 0.25,
          audit: 0.20,
          sustainability: 0.05
        };
      
      case 'moderate':
        return {
          yield: 0.25,
          risk: 0.30,
          liquidity: 0.20,
          audit: 0.15,
          sustainability: 0.10
        };
      
      case 'aggressive':
        return {
          yield: 0.40,
          risk: 0.20,
          liquidity: 0.15,
          audit: 0.10,
          sustainability: 0.15
        };
      
      default:
        return {
          yield: 0.25,
          risk: 0.30,
          liquidity: 0.20,
          audit: 0.15,
          sustainability: 0.10
        };
    }
  }

  private identifyKeyFactors(
    opportunity: YieldOpportunity,
    scores: any
  ): YieldFactor[] {
    const factors: YieldFactor[] = [];

    // APY factor
    if (opportunity.apy > YIELD_THRESHOLDS.HIGH_APY) {
      factors.push({
        factor: 'High APY',
        impact: 'positive',
        weight: 0.25,
        description: `Exceptional APY of ${opportunity.apy.toFixed(2)}%`,
        score: scores.yieldScore
      });
    } else if (opportunity.apy < YIELD_THRESHOLDS.MIN_APY) {
      factors.push({
        factor: 'Low APY',
        impact: 'negative',
        weight: 0.25,
        description: `Below minimum APY threshold at ${opportunity.apy.toFixed(2)}%`,
        score: scores.yieldScore
      });
    }

    // Risk factor
    if (scores.riskScore > 70) {
      factors.push({
        factor: 'High Risk',
        impact: 'negative',
        weight: 0.30,
        description: `High risk score of ${scores.riskScore.toFixed(1)}`,
        score: scores.riskScore
      });
    } else if (scores.riskScore < 30) {
      factors.push({
        factor: 'Low Risk',
        impact: 'positive',
        weight: 0.30,
        description: `Low risk score of ${scores.riskScore.toFixed(1)}`,
        score: scores.riskScore
      });
    }

    // Liquidity factor
    const tvlUsd = parseFloat(ethers.utils.formatEther(opportunity.tvl));
    if (tvlUsd > 100000000) {
      factors.push({
        factor: 'High Liquidity',
        impact: 'positive',
        weight: 0.20,
        description: `Strong liquidity with $${(tvlUsd / 1000000).toFixed(0)}M TVL`,
        score: scores.liquidityScore
      });
    } else if (tvlUsd < 1000000) {
      factors.push({
        factor: 'Low Liquidity',
        impact: 'negative',
        weight: 0.20,
        description: `Limited liquidity with $${(tvlUsd / 1000).toFixed(0)}K TVL`,
        score: scores.liquidityScore
      });
    }

    // Audit factor
    if (scores.auditScore > 80) {
      factors.push({
        factor: 'Well Audited',
        impact: 'positive',
        weight: 0.15,
        description: 'Strong audit score indicating security',
        score: scores.auditScore
      });
    } else if (scores.auditScore < 40) {
      factors.push({
        factor: 'Poor Audit Coverage',
        impact: 'negative',
        weight: 0.15,
        description: 'Limited audit coverage raises security concerns',
        score: scores.auditScore
      });
    }

    // Auto-compound factor
    if (opportunity.autoCompound) {
      factors.push({
        factor: 'Auto-Compounding',
        impact: 'positive',
        weight: 0.10,
        description: 'Automatic reward compounding maximizes returns',
        score: 80
      });
    }

    return factors.sort((a, b) => b.weight - a.weight);
  }

  private async assessRisks(opportunity: YieldOpportunity): Promise<YieldRisk[]> {
    const risks: YieldRisk[] = [];

    // Smart contract risk
    if (opportunity.riskScore > 60 || !opportunity.verified) {
      risks.push({
        type: 'smart_contract',
        level: opportunity.riskScore > 80 ? 'critical' : 
               opportunity.riskScore > 60 ? 'high' : 'medium',
        description: 'Smart contract vulnerabilities could lead to loss of funds',
        probability: opportunity.riskScore / 100,
        impact: 90,
        mitigation: [
          'Verify contract audits',
          'Start with small position',
          'Monitor protocol updates'
        ]
      });
    }

    // Liquidity risk
    const tvlUsd = parseFloat(ethers.utils.formatEther(opportunity.tvl));
    if (tvlUsd < 10000000) {
      risks.push({
        type: 'liquidity',
        level: tvlUsd < 1000000 ? 'high' : 'medium',
        description: 'Low liquidity may impact ability to exit position',
        probability: 0.3,
        impact: 60,
        mitigation: [
          'Monitor TVL changes',
          'Limit position size',
          'Have exit strategy ready'
        ]
      });
    }

    // Market risk
    if (opportunity.apy > 50) {
      risks.push({
        type: 'market',
        level: 'high',
        description: 'Unsustainably high yields may not be maintained',
        probability: 0.7,
        impact: 70,
        mitigation: [
          'Harvest rewards frequently',
          'Monitor yield sustainability',
          'Diversify across protocols'
        ]
      });
    }

    // Regulatory risk
    if (opportunity.strategy === 'yield_farming' && !opportunity.verified) {
      risks.push({
        type: 'regulatory',
        level: 'medium',
        description: 'Unregulated yield farming may face regulatory scrutiny',
        probability: 0.2,
        impact: 50,
        mitigation: [
          'Stay informed on regulations',
          'Consider geographic restrictions',
          'Maintain compliance records'
        ]
      });
    }

    // Operational risk
    if (opportunity.lockupPeriod > 90 * 24 * 60 * 60) { // 90 days
      risks.push({
        type: 'operational',
        level: 'medium',
        description: 'Long lockup period reduces flexibility',
        probability: 1.0,
        impact: 40,
        mitigation: [
          'Understand lockup terms',
          'Plan position timing',
          'Consider opportunity cost'
        ]
      });
    }

    return risks.sort((a, b) => (b.probability * b.impact) - (a.probability * a.impact));
  }

  private projectReturns(opportunity: YieldOpportunity): ProjectedReturns {
    const baseAPY = opportunity.apy;
    const volatility = this.estimateAPYVolatility(opportunity);

    // Conservative projection (worst case)
    const conservativeAPY = Math.max(0, baseAPY - volatility * 2);
    
    // Realistic projection (expected case)
    const realisticAPY = baseAPY - volatility * 0.5;
    
    // Optimistic projection (best case)
    const optimisticAPY = baseAPY + volatility;

    return {
      conservative: {
        apy: conservativeAPY,
        timeToBreakeven: this.calculateBreakevenTime(conservativeAPY, opportunity),
        confidenceInterval: [conservativeAPY * 0.8, conservativeAPY * 1.2]
      },
      realistic: {
        apy: realisticAPY,
        timeToBreakeven: this.calculateBreakevenTime(realisticAPY, opportunity),
        confidenceInterval: [realisticAPY * 0.9, realisticAPY * 1.1]
      },
      optimistic: {
        apy: optimisticAPY,
        timeToBreakeven: this.calculateBreakevenTime(optimisticAPY, opportunity),
        confidenceInterval: [optimisticAPY * 0.95, optimisticAPY * 1.15]
      }
    };
  }

  private async performCompetitiveAnalysis(opportunity: YieldOpportunity): Promise<CompetitiveAnalysis> {
    // In a real implementation, this would compare against other opportunities
    // For now, we'll provide a simplified analysis
    
    const marketPosition = this.determineMarketPosition(opportunity);
    const uniqueSellingPoints = this.identifyUniqueSellingPoints(opportunity);

    return {
      ranking: 1, // Would be calculated against other opportunities
      totalOpportunities: 1,
      betterAlternatives: [], // Would be populated with actual alternatives
      marketPosition,
      uniqueSellingPoints
    };
  }

  private generateRecommendation(
    overallScore: number,
    risks: YieldRisk[],
    opportunity: YieldOpportunity
  ): { recommendation: 'strong_buy' | 'buy' | 'hold' | 'avoid'; confidence: number } {
    let recommendation: 'strong_buy' | 'buy' | 'hold' | 'avoid';
    let confidence = 0.7; // Base confidence

    // Critical risks = avoid
    const criticalRisks = risks.filter(r => r.level === 'critical');
    if (criticalRisks.length > 0) {
      return { recommendation: 'avoid', confidence: 0.9 };
    }

    // Risk tolerance adjustments
    const riskThresholds = {
      conservative: { buy: 75, strongBuy: 85 },
      moderate: { buy: 65, strongBuy: 80 },
      aggressive: { buy: 55, strongBuy: 75 }
    }[this.riskTolerance];

    if (overallScore >= riskThresholds.strongBuy) {
      recommendation = 'strong_buy';
      confidence = 0.85;
    } else if (overallScore >= riskThresholds.buy) {
      recommendation = 'buy';
      confidence = 0.75;
    } else if (overallScore >= 40) {
      recommendation = 'hold';
      confidence = 0.6;
    } else {
      recommendation = 'avoid';
      confidence = 0.8;
    }

    // Adjust confidence based on risk count
    const highRisks = risks.filter(r => r.level === 'high').length;
    confidence -= highRisks * 0.1;

    // Adjust confidence based on data quality
    if (!opportunity.verified) confidence -= 0.15;
    if (!opportunity.auditScore || opportunity.auditScore < 50) confidence -= 0.1;

    return {
      recommendation,
      confidence: Math.max(0.3, Math.min(0.95, confidence))
    };
  }

  // Helper methods
  private getStrategyAdjustment(opportunity: YieldOpportunity): number {
    switch (opportunity.strategy) {
      case 'lending':
        return 5; // Generally stable
      case 'staking':
        return 3; // Moderate stability
      case 'liquidity_provision':
        return -5; // Impermanent loss risk
      case 'yield_farming':
        return -10; // Higher risk
      case 'auto_compound':
        return 8; // Efficiency bonus
      default:
        return 0;
    }
  }

  private assessRevenueModel(opportunity: YieldOpportunity): number {
    // Assess how sustainable the protocol's revenue model is
    let score = 0;

    if (opportunity.strategy === 'lending') {
      score += 15; // Lending has proven revenue model
    } else if (opportunity.strategy === 'liquidity_provision') {
      score += 10; // DEX fees are sustainable
    } else if (opportunity.strategy === 'staking') {
      score += 8; // Staking rewards from inflation
    } else if (opportunity.strategy === 'yield_farming') {
      score -= 5; // Often unsustainable
    }

    return score;
  }

  private assessTokenEconomics(opportunity: YieldOpportunity): number {
    let score = 0;

    // Multiple reward tokens can be good for diversification
    if (opportunity.rewardTokens && opportunity.rewardTokens.length > 1) {
      score += 5;
    }

    // Auto-compounding improves token economics
    if (opportunity.autoCompound) {
      score += 8;
    }

    // Low fees improve economics
    const totalFees = (opportunity.fees.deposit || 0) + 
                     (opportunity.fees.withdrawal || 0) + 
                     (opportunity.fees.performance || 0);
    
    if (totalFees < 2) score += 5;
    else if (totalFees > 5) score -= 8;

    return score;
  }

  private assessProtocolFundamentals(opportunity: YieldOpportunity): number {
    let score = 0;

    // TVL indicates protocol health
    const tvlUsd = parseFloat(ethers.utils.formatEther(opportunity.tvl));
    if (tvlUsd > 1000000000) score += 15; // >$1B
    else if (tvlUsd > 100000000) score += 10; // >$100M
    else if (tvlUsd > 10000000) score += 5; // >$10M

    // Verification and audits
    if (opportunity.verified) score += 10;
    if (opportunity.auditScore && opportunity.auditScore > 70) score += 8;

    return score;
  }

  private async assessMarketDemand(opportunity: YieldOpportunity): Promise<number> {
    // This would assess market demand indicators
    // For now, return a base score
    return 0;
  }

  private estimateAPYVolatility(opportunity: YieldOpportunity): number {
    if (opportunity.apyHistory && opportunity.apyHistory.length > 7) {
      const mean = opportunity.apyHistory.reduce((sum, apy) => sum + apy, 0) / opportunity.apyHistory.length;
      const variance = opportunity.apyHistory.reduce((sum, apy) => sum + Math.pow(apy - mean, 2), 0) / opportunity.apyHistory.length;
      return Math.sqrt(variance);
    }

    // Default volatility estimates by strategy
    const defaultVolatility = {
      'lending': 2,
      'staking': 5,
      'liquidity_provision': 10,
      'yield_farming': 20,
      'auto_compound': 3
    };

    return defaultVolatility[opportunity.strategy] || 10;
  }

  private calculateBreakevenTime(apy: number, opportunity: YieldOpportunity): number {
    if (apy <= 0) return Infinity;

    const totalFees = (opportunity.fees.deposit || 0) + (opportunity.fees.withdrawal || 0);
    
    // Time to recover fees through yield (in days)
    return (totalFees / apy) * 365;
  }

  private determineMarketPosition(opportunity: YieldOpportunity): 'top_tier' | 'competitive' | 'average' | 'below_average' {
    const score = (opportunity.apy / 20) + ((100 - opportunity.riskScore) / 100) * 50;
    
    if (score > 75) return 'top_tier';
    if (score > 60) return 'competitive';
    if (score > 40) return 'average';
    return 'below_average';
  }

  private identifyUniqueSellingPoints(opportunity: YieldOpportunity): string[] {
    const points: string[] = [];

    if (opportunity.autoCompound) {
      points.push('Automatic reward compounding');
    }

    if (opportunity.lockupPeriod === 0) {
      points.push('No lockup period');
    }

    if (opportunity.fees.deposit === 0 && opportunity.fees.withdrawal === 0) {
      points.push('No deposit or withdrawal fees');
    }

    if (opportunity.apy > 50) {
      points.push('Exceptionally high APY');
    }

    if (opportunity.verified && opportunity.auditScore && opportunity.auditScore > 80) {
      points.push('Highly audited and secure');
    }

    const tvlUsd = parseFloat(ethers.utils.formatEther(opportunity.tvl));
    if (tvlUsd > 500000000) {
      points.push('Large and established protocol');
    }

    return points;
  }

  // Public utility methods
  setRiskTolerance(riskTolerance: 'conservative' | 'moderate' | 'aggressive'): void {
    this.riskTolerance = riskTolerance;
    this.evaluationCache.clear(); // Clear cache when risk tolerance changes
  }

  clearCache(): void {
    this.evaluationCache.clear();
  }

  async batchEvaluate(opportunities: YieldOpportunity[]): Promise<YieldEvaluation[]> {
    const evaluationPromises = opportunities.map(opportunity => 
      this.evaluateOpportunity(opportunity)
    );

    return Promise.all(evaluationPromises);
  }

  async compareOpportunities(opportunities: YieldOpportunity[]): Promise<{
    rankings: Array<{ opportunity: YieldOpportunity; evaluation: YieldEvaluation; rank: number }>;
    summary: {
      bestOverall: YieldOpportunity;
      safest: YieldOpportunity;
      highestYield: YieldOpportunity;
      mostLiquid: YieldOpportunity;
    };
  }> {
    const evaluations = await this.batchEvaluate(opportunities);
    
    const rankings = evaluations
      .map((evaluation, index) => ({
        opportunity: opportunities[index],
        evaluation,
        rank: 0
      }))
      .sort((a, b) => b.evaluation.overallScore - a.evaluation.overallScore)
      .map((item, index) => ({ ...item, rank: index + 1 }));

    // Find best in each category
    const bestOverall = rankings[0]?.opportunity;
    const safest = rankings.reduce((safest, current) => 
      current.evaluation.riskScore < safest.evaluation.riskScore ? current : safest
    ).opportunity;
    const highestYield = opportunities.reduce((highest, current) => 
      current.apy > highest.apy ? current : highest
    );
    const mostLiquid = rankings.reduce((mostLiquid, current) => 
      current.evaluation.liquidityScore > mostLiquid.evaluation.liquidityScore ? current : mostLiquid
    ).opportunity;

    return {
      rankings,
      summary: {
        bestOverall,
        safest,
        highestYield,
        mostLiquid
      }
    };
  }
}
