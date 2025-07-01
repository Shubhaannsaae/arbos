import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { Portfolio, PortfolioPerformance } from '../../../shared/types/market';

export interface PerformanceMetrics {
  totalReturn: BigNumber;
  totalReturnPercentage: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  maxDrawdownDuration: number;
  winRate: number;
  profitFactor: number;
  beta: number;
  alpha: number;
  informationRatio: number;
  trackingError: number;
  downsidevDeviation: number;
  valueAtRisk: BigNumber;
  conditionalValueAtRisk: BigNumber;
}

export interface BenchmarkComparison {
  benchmarkReturn: number;
  outperformance: number;
  correlation: number;
  upCapture: number;
  downCapture: number;
  battingAverage: number;
}

export interface AttributionAnalysis {
  securitySelection: number;
  assetAllocation: number;
  interaction: number;
  totalAttribution: number;
  topContributors: Array<{
    asset: string;
    contribution: number;
    weight: number;
    return: number;
  }>;
  topDetractors: Array<{
    asset: string;
    contribution: number;
    weight: number;
    return: number;
  }>;
}

export class PerformanceEvaluator {
  private benchmarkIndex: string;
  private riskFreeRate: number = 0.03; // 3% annual risk-free rate
  private performanceCache: Map<string, any> = new Map();

  constructor(benchmarkIndex: string = 'ETH') {
    this.benchmarkIndex = benchmarkIndex;
  }

  async evaluatePortfolio(portfolio: Portfolio): Promise<PortfolioPerformance> {
    const startTime = Date.now();
    
    logger.info('Starting portfolio performance evaluation', {
      portfolioId: portfolio.id,
      benchmarkIndex: this.benchmarkIndex,
      totalValue: ethers.utils.formatEther(portfolio.totalValue)
    });

    try {
      // Get historical portfolio data
      const historicalData = await this.getPortfolioHistoricalData(portfolio);
      
      if (historicalData.length < 30) {
        logger.warn('Insufficient historical data for comprehensive analysis', {
          portfolioId: portfolio.id,
          dataPoints: historicalData.length
        });
      }

      // Calculate core performance metrics
      const metrics = await this.calculatePerformanceMetrics(historicalData, portfolio);
      
      // Get benchmark comparison
      const benchmarkComparison = await this.compareToBenchmark(historicalData, portfolio);
      
      // Calculate risk-adjusted metrics
      const riskAdjustedMetrics = this.calculateRiskAdjustedMetrics(metrics, benchmarkComparison);

      // Perform attribution analysis
      const attribution = await this.performAttributionAnalysis(portfolio, historicalData);

      const performance: PortfolioPerformance = {
        totalReturn: metrics.totalReturn,
        totalReturnPercentage: metrics.totalReturnPercentage,
        annualizedReturn: metrics.annualizedReturn,
        volatility: metrics.volatility,
        sharpeRatio: riskAdjustedMetrics.sharpeRatio,
        calmarRatio: riskAdjustedMetrics.calmarRatio,
        maxDrawdown: metrics.maxDrawdown,
        winRate: metrics.winRate,
        profitFactor: metrics.profitFactor,
        beta: benchmarkComparison.beta,
        alpha: riskAdjustedMetrics.alpha,
        informationRatio: riskAdjustedMetrics.informationRatio,
        trackingError: benchmarkComparison.trackingError
      };

      // Cache results for performance
      this.performanceCache.set(portfolio.id, {
        performance,
        timestamp: Date.now(),
        attribution,
        benchmarkComparison
      });

      logger.info('Portfolio performance evaluation completed', {
        portfolioId: portfolio.id,
        annualizedReturn: metrics.annualizedReturn.toFixed(2),
        sharpeRatio: riskAdjustedMetrics.sharpeRatio.toFixed(3),
        maxDrawdown: metrics.maxDrawdown.toFixed(2),
        duration: Date.now() - startTime
      });

      return performance;

    } catch (error) {
      logger.error('Portfolio performance evaluation failed', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime
      });

      throw error;
    }
  }

  private async getPortfolioHistoricalData(portfolio: Portfolio): Promise<Array<{
    date: number;
    value: BigNumber;
    returns: number;
    positions: Array<{
      asset: string;
      weight: number;
      price: BigNumber;
      return: number;
    }>;
  }>> {
    try {
      // In production, this would fetch from a database or external API
      // For now, we'll simulate historical data based on current portfolio
      const historicalData: any[] = [];
      const days = 252; // 1 year of trading days
      const startDate = Date.now() - (days * 24 * 60 * 60 * 1000);

      // Generate historical portfolio values
      for (let i = 0; i < days; i++) {
        const date = startDate + (i * 24 * 60 * 60 * 1000);
        
        // Simulate portfolio performance with some randomness
        const baseReturn = 0.0003; // ~8% annual base return
        const volatility = 0.015; // ~1.5% daily volatility
        const randomReturn = (Math.random() - 0.5) * 2 * volatility + baseReturn;
        
        // Calculate portfolio value for this date
        const portfolioReturn = this.calculatePortfolioReturn(portfolio, randomReturn, i);
        const portfolioValue = i === 0 
          ? portfolio.totalValue.mul(90).div(100) // Start 10% lower
          : historicalData[i - 1].value.mul(Math.floor((1 + portfolioReturn) * 10000)).div(10000);

        // Generate position data
        const positions = portfolio.positions.map(position => ({
          asset: position.token.symbol,
          weight: position.percentage / 100,
          price: position.value.div(position.amount), // Simplified price calculation
          return: portfolioReturn + (Math.random() - 0.5) * 0.01 // Add some individual asset variation
        }));

        historicalData.push({
          date,
          value: portfolioValue,
          returns: portfolioReturn,
          positions
        });
      }

      return historicalData;

    } catch (error) {
      logger.error('Failed to get portfolio historical data', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return [];
    }
  }

  private calculatePortfolioReturn(portfolio: Portfolio, baseReturn: number, dayIndex: number): number {
    // Add some portfolio-specific characteristics
    let portfolioReturn = baseReturn;

    // Add momentum effect
    if (dayIndex > 20) {
      const recentPerformance = baseReturn * 20; // Last 20 days average
      portfolioReturn += recentPerformance * 0.1; // 10% momentum factor
    }

    // Add diversification benefit
    const diversificationBenefit = Math.min(portfolio.positions.length / 10, 0.5) * 0.002;
    portfolioReturn += diversificationBenefit;

    // Add concentration penalty
    const maxWeight = Math.max(...portfolio.positions.map(p => p.percentage / 100));
    if (maxWeight > 0.3) {
      portfolioReturn -= (maxWeight - 0.3) * 0.01; // Concentration penalty
    }

    return portfolioReturn;
  }

  private async calculatePerformanceMetrics(
    historicalData: any[],
    portfolio: Portfolio
  ): Promise<PerformanceMetrics> {
    if (historicalData.length < 2) {
      throw new Error('Insufficient data for performance calculation');
    }

    const returns = historicalData.map(d => d.returns);
    const values = historicalData.map(d => d.value);

    // Total return calculation
    const initialValue = values[0];
    const finalValue = values[values.length - 1];
    const totalReturn = finalValue.sub(initialValue);
    const totalReturnPercentage = parseFloat(ethers.utils.formatEther(
      totalReturn.mul(10000).div(initialValue)
    )) / 100;

    // Annualized return
    const periods = historicalData.length / 252; // Convert days to years
    const annualizedReturn = Math.pow(1 + totalReturnPercentage / 100, 1 / periods) - 1;

    // Volatility (annualized)
    const meanReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance * 252); // Annualized

    // Maximum drawdown calculation
    const { maxDrawdown, maxDrawdownDuration } = this.calculateMaxDrawdown(values);

    // Win rate
    const positiveReturns = returns.filter(r => r > 0).length;
    const winRate = (positiveReturns / returns.length) * 100;

    // Profit factor
    const gains = returns.filter(r => r > 0).reduce((sum, r) => sum + r, 0);
    const losses = Math.abs(returns.filter(r => r < 0).reduce((sum, r) => sum + r, 0));
    const profitFactor = losses > 0 ? gains / losses : gains > 0 ? 10 : 1;

    // Downside deviation
    const targetReturn = 0; // Use 0 as minimum acceptable return
    const downsideReturns = returns.filter(r => r < targetReturn).map(r => r - targetReturn);
    const downsideVariance = downsideReturns.length > 0
      ? downsideReturns.reduce((sum, r) => sum + r * r, 0) / downsideReturns.length
      : 0;
    const downsidevDeviation = Math.sqrt(downsideVariance * 252);

    // Value at Risk (95% confidence)
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const varIndex = Math.floor(0.05 * sortedReturns.length);
    const dailyVaR = Math.abs(sortedReturns[varIndex]);
    const valueAtRisk = portfolio.totalValue.mul(Math.floor(dailyVaR * 10000)).div(10000);

    // Conditional VaR (Expected Shortfall)
    const tailReturns = sortedReturns.slice(0, varIndex);
    const avgTailReturn = tailReturns.length > 0 
      ? tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length
      : dailyVaR;
    const conditionalValueAtRisk = portfolio.totalValue.mul(Math.floor(Math.abs(avgTailReturn) * 10000)).div(10000);

    return {
      totalReturn,
      totalReturnPercentage,
      annualizedReturn: annualizedReturn * 100,
      volatility: volatility * 100,
      sharpeRatio: 0, // Will be calculated in risk-adjusted metrics
      sortinoRatio: downsidevDeviation > 0 ? (annualizedReturn - this.riskFreeRate) / downsidevDeviation : 0,
      calmarRatio: 0, // Will be calculated in risk-adjusted metrics
      maxDrawdown: maxDrawdown * 100,
      maxDrawdownDuration,
      winRate,
      profitFactor,
      beta: 0, // Will be calculated in benchmark comparison
      alpha: 0, // Will be calculated in risk-adjusted metrics
      informationRatio: 0, // Will be calculated in risk-adjusted metrics
      trackingError: 0, // Will be calculated in benchmark comparison
      downsidevDeviation: downsidevDeviation * 100,
      valueAtRisk,
      conditionalValueAtRisk
    };
  }

  private calculateMaxDrawdown(values: BigNumber[]): { maxDrawdown: number; maxDrawdownDuration: number } {
    let maxDrawdown = 0;
    let maxDrawdownDuration = 0;
    let currentDrawdownDuration = 0;
    let peak = values[0];

    for (let i = 1; i < values.length; i++) {
      const currentValue = values[i];
      
      if (currentValue.gt(peak)) {
        peak = currentValue;
        currentDrawdownDuration = 0;
      } else {
        const drawdown = parseFloat(ethers.utils.formatEther(
          peak.sub(currentValue).mul(ethers.utils.parseEther('1')).div(peak)
        ));
        
        maxDrawdown = Math.max(maxDrawdown, drawdown);
        currentDrawdownDuration++;
        maxDrawdownDuration = Math.max(maxDrawdownDuration, currentDrawdownDuration);
      }
    }

    return { maxDrawdown, maxDrawdownDuration };
  }

  private async compareToBenchmark(
    historicalData: any[],
    portfolio: Portfolio
  ): Promise<BenchmarkComparison & { beta: number; trackingError: number }> {
    try {
      // Get benchmark returns (simplified - in production would fetch real benchmark data)
      const benchmarkReturns = this.generateBenchmarkReturns(historicalData.length);
      const portfolioReturns = historicalData.map(d => d.returns);

      if (benchmarkReturns.length !== portfolioReturns.length) {
        throw new Error('Portfolio and benchmark data length mismatch');
      }

      // Calculate benchmark metrics
      const benchmarkTotalReturn = benchmarkReturns.reduce((prod, r) => prod * (1 + r), 1) - 1;
      const portfolioTotalReturn = portfolioReturns.reduce((prod, r) => prod * (1 + r), 1) - 1;
      const outperformance = (portfolioTotalReturn - benchmarkTotalReturn) * 100;

      // Calculate correlation
      const correlation = this.calculateCorrelation(portfolioReturns, benchmarkReturns);

      // Calculate beta
      const beta = this.calculateBeta(portfolioReturns, benchmarkReturns);

      // Calculate tracking error
      const excessReturns = portfolioReturns.map((pr, i) => pr - benchmarkReturns[i]);
      const trackingErrorDaily = Math.sqrt(
        excessReturns.reduce((sum, er) => sum + er * er, 0) / excessReturns.length
      );
      const trackingError = trackingErrorDaily * Math.sqrt(252) * 100; // Annualized

      // Calculate up/down capture ratios
      const { upCapture, downCapture } = this.calculateCaptureRatios(portfolioReturns, benchmarkReturns);

      // Batting average (percentage of periods outperforming benchmark)
      const outperformingPeriods = excessReturns.filter(er => er > 0).length;
      const battingAverage = (outperformingPeriods / excessReturns.length) * 100;

      return {
        benchmarkReturn: benchmarkTotalReturn * 100,
        outperformance,
        correlation,
        upCapture,
        downCapture,
        battingAverage,
        beta,
        trackingError
      };

    } catch (error) {
      logger.error('Benchmark comparison failed', {
        portfolioId: portfolio.id,
        benchmark: this.benchmarkIndex,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        benchmarkReturn: 8, // Default 8% benchmark return
        outperformance: 0,
        correlation: 0.8,
        upCapture: 100,
        downCapture: 100,
        battingAverage: 50,
        beta: 1.0,
        trackingError: 5
      };
    }
  }

  private generateBenchmarkReturns(length: number): number[] {
    // Generate benchmark returns (e.g., ETH performance)
    const returns: number[] = [];
    const annualReturn = 0.12; // 12% annual return for ETH
    const volatility = 0.06; // 6% daily volatility
    const dailyReturn = annualReturn / 252;

    for (let i = 0; i < length; i++) {
      const randomReturn = (Math.random() - 0.5) * 2 * volatility + dailyReturn;
      returns.push(randomReturn);
    }

    return returns;
  }

  private calculateCorrelation(returns1: number[], returns2: number[]): number {
    const n = Math.min(returns1.length, returns2.length);
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

  private calculateBeta(portfolioReturns: number[], benchmarkReturns: number[]): number {
    const n = Math.min(portfolioReturns.length, benchmarkReturns.length);
    const portfolioMean = portfolioReturns.slice(0, n).reduce((sum, r) => sum + r, 0) / n;
    const benchmarkMean = benchmarkReturns.slice(0, n).reduce((sum, r) => sum + r, 0) / n;

    let covariance = 0;
    let benchmarkVariance = 0;

    for (let i = 0; i < n; i++) {
      const portfolioDiff = portfolioReturns[i] - portfolioMean;
      const benchmarkDiff = benchmarkReturns[i] - benchmarkMean;
      
      covariance += portfolioDiff * benchmarkDiff;
      benchmarkVariance += benchmarkDiff * benchmarkDiff;
    }

    return benchmarkVariance === 0 ? 1 : (covariance / n) / (benchmarkVariance / n);
  }

  private calculateCaptureRatios(
    portfolioReturns: number[],
    benchmarkReturns: number[]
  ): { upCapture: number; downCapture: number } {
    const upPeriods: { portfolio: number; benchmark: number }[] = [];
    const downPeriods: { portfolio: number; benchmark: number }[] = [];

    for (let i = 0; i < Math.min(portfolioReturns.length, benchmarkReturns.length); i++) {
      if (benchmarkReturns[i] > 0) {
        upPeriods.push({ portfolio: portfolioReturns[i], benchmark: benchmarkReturns[i] });
      } else if (benchmarkReturns[i] < 0) {
        downPeriods.push({ portfolio: portfolioReturns[i], benchmark: benchmarkReturns[i] });
      }
    }

    const upCapture = upPeriods.length > 0
      ? (upPeriods.reduce((sum, p) => sum + p.portfolio, 0) / upPeriods.length) /
        (upPeriods.reduce((sum, p) => sum + p.benchmark, 0) / upPeriods.length) * 100
      : 100;

    const downCapture = downPeriods.length > 0
      ? (downPeriods.reduce((sum, p) => sum + p.portfolio, 0) / downPeriods.length) /
        (downPeriods.reduce((sum, p) => sum + p.benchmark, 0) / downPeriods.length) * 100
      : 100;

    return { upCapture, downCapture };
  }

  private calculateRiskAdjustedMetrics(
    metrics: PerformanceMetrics,
    benchmarkComparison: any
  ): { sharpeRatio: number; calmarRatio: number; alpha: number; informationRatio: number } {
    // Sharpe Ratio
    const sharpeRatio = metrics.volatility > 0 
      ? (metrics.annualizedReturn - this.riskFreeRate * 100) / metrics.volatility
      : 0;

    // Calmar Ratio
    const calmarRatio = metrics.maxDrawdown > 0 
      ? metrics.annualizedReturn / metrics.maxDrawdown
      : 0;

    // Alpha (CAPM)
    const expectedReturn = this.riskFreeRate * 100 + benchmarkComparison.beta * 
      (benchmarkComparison.benchmarkReturn - this.riskFreeRate * 100);
    const alpha = metrics.annualizedReturn - expectedReturn;

    // Information Ratio
    const informationRatio = benchmarkComparison.trackingError > 0 
      ? benchmarkComparison.outperformance / benchmarkComparison.trackingError
      : 0;

    return { sharpeRatio, calmarRatio, alpha, informationRatio };
  }

  private async performAttributionAnalysis(
    portfolio: Portfolio,
    historicalData: any[]
  ): Promise<AttributionAnalysis> {
    try {
      // Simplified attribution analysis
      const contributions = portfolio.positions.map(position => {
        // Calculate contribution based on weight and simulated return
        const weight = position.percentage / 100;
        const assetReturn = this.estimateAssetReturn(position.token.symbol, historicalData.length);
        const contribution = weight * assetReturn;

        return {
          asset: position.token.symbol,
          contribution,
          weight,
          return: assetReturn
        };
      });

      // Sort by contribution
      contributions.sort((a, b) => b.contribution - a.contribution);

      const topContributors = contributions.filter(c => c.contribution > 0).slice(0, 5);
      const topDetractors = contributions.filter(c => c.contribution < 0).slice(0, 5);

      // Simplified attribution components
      const totalContribution = contributions.reduce((sum, c) => sum + c.contribution, 0);
      
      return {
        securitySelection: totalContribution * 0.6, // 60% attributed to security selection
        assetAllocation: totalContribution * 0.35,  // 35% attributed to allocation
        interaction: totalContribution * 0.05,      // 5% interaction effect
        totalAttribution: totalContribution,
        topContributors,
        topDetractors
      };

    } catch (error) {
      logger.error('Attribution analysis failed', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        securitySelection: 0,
        assetAllocation: 0,
        interaction: 0,
        totalAttribution: 0,
        topContributors: [],
        topDetractors: []
      };
    }
  }

  private estimateAssetReturn(symbol: string, periods: number): number {
    // Estimate asset returns based on historical patterns
    const assetReturns: Record<string, number> = {
      'ETH': 12,
      'BTC': 15,
      'LINK': 18,
      'USDC': 3,
      'USDT': 2.5,
      'MATIC': 20,
      'AVAX': 25,
      'UNI': 22,
      'AAVE': 28,
      'COMP': 16
    };

    const baseReturn = assetReturns[symbol] || 10; // Default 10% annual return
    
    // Add some randomness based on market conditions
    const volatilityAdjustment = (Math.random() - 0.5) * 10; // ±5% adjustment
    
    return baseReturn + volatilityAdjustment;
  }

  // Public methods for external access
  async getDetailedPerformanceReport(portfolio: Portfolio): Promise<{
    metrics: PerformanceMetrics;
    benchmarkComparison: BenchmarkComparison;
    attribution: AttributionAnalysis;
    recommendations: string[];
  }> {
    const cached = this.performanceCache.get(portfolio.id);
    
    if (cached && Date.now() - cached.timestamp < 300000) { // 5 minutes cache
      return {
        metrics: cached.performance,
        benchmarkComparison: cached.benchmarkComparison,
        attribution: cached.attribution,
        recommendations: this.generatePerformanceRecommendations(cached.performance, cached.benchmarkComparison)
      };
    }

    // Recalculate if cache is stale
    await this.evaluatePortfolio(portfolio);
    return this.getDetailedPerformanceReport(portfolio);
  }

  private generatePerformanceRecommendations(
    performance: PortfolioPerformance,
    benchmarkComparison: any
  ): string[] {
    const recommendations: string[] = [];

    if (performance.sharpeRatio < 0.5) {
      recommendations.push('Consider improving risk-adjusted returns by optimizing asset allocation');
    }

    if (performance.maxDrawdown > 20) {
      recommendations.push('Implement risk management strategies to reduce maximum drawdown');
    }

    if (benchmarkComparison.outperformance < 0) {
      recommendations.push('Portfolio is underperforming benchmark - review investment strategy');
    }

    if (performance.winRate < 40) {
      recommendations.push('Low win rate suggests need for better timing or asset selection');
    }

    if (benchmarkComparison.trackingError > 10) {
      recommendations.push('High tracking error - consider closer benchmark alignment if appropriate');
    }

    return recommendations;
  }

  setBenchmark(newBenchmark: string): void {
    this.benchmarkIndex = newBenchmark;
    this.performanceCache.clear(); // Clear cache when benchmark changes
  }

  setRiskFreeRate(rate: number): void {
    this.riskFreeRate = rate;
    this.performanceCache.clear(); // Clear cache when risk-free rate changes
  }

  clearCache(): void {
    this.performanceCache.clear();
  }
}
