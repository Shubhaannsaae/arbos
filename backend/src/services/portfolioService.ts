import { ethers } from 'ethers';
import { logger } from '../utils/logger';
import { db } from '../config/database';
import { chainlinkService } from './chainlinkService';
import { web3Service } from '../config/web3';
import { mlService } from './mlService';
import {
  Portfolio,
  CreatePortfolioDto,
  UpdatePortfolioDto,
  TokenAllocation,
  PerformanceMetrics,
  RebalanceSettings,
  RebalanceTrigger
} from '../models/Portfolio';

interface RebalanceAnalysis {
  needsRebalancing: boolean;
  currentAllocation: TokenAllocation[];
  targetAllocation: TokenAllocation[];
  maxDrift: number;
  recommendedActions: RebalanceAction[];
}

interface RebalanceAction {
  type: 'buy' | 'sell';
  tokenAddress: string;
  amount: number;
  priority: number;
}

interface RebalanceExecution {
  portfolioId: string;
  userId: string;
  strategy: string;
  maxSlippage: number;
  gasOptimization: boolean;
  analysis: RebalanceAnalysis;
}

class PortfolioService {
  private automationUpkeeps: Map<string, string> = new Map(); // portfolioId -> upkeepId
  private performanceCache: Map<string, PerformanceMetrics> = new Map();

  /**
   * Create a new portfolio with Chainlink price integration
   */
  public async createPortfolio(userId: string, createPortfolioDto: CreatePortfolioDto): Promise<Portfolio> {
    try {
      // Generate unique portfolio ID
      const portfolioId = ethers.utils.keccak256(
        ethers.utils.toUtf8Bytes(`${userId}-${createPortfolioDto.name}-${Date.now()}`)
      );

      // Validate and enrich token allocations with Chainlink price data
      const enrichedAllocations = await this.enrichTokenAllocations(
        createPortfolioDto.targetAllocation,
        createPortfolioDto.chainId
      );

      // Calculate initial portfolio metrics
      const initialMetrics = await this.calculateInitialMetrics(enrichedAllocations);

      // Create portfolio object
      const portfolio: Portfolio = {
        id: portfolioId,
        userId,
        name: createPortfolioDto.name,
        description: createPortfolioDto.description,
        targetAllocation: enrichedAllocations,
        currentAllocation: enrichedAllocations.map(allocation => ({
          ...allocation,
          amount: '0', // No initial holdings
          valueUSD: 0
        })),
        totalValueUSD: 0,
        totalValueETH: 0,
        performanceMetrics: initialMetrics,
        riskScore: await this.calculateRiskScore(enrichedAllocations),
        rebalanceSettings: createPortfolioDto.rebalanceSettings,
        restrictions: createPortfolioDto.restrictions || {
          maxTokens: 20,
          minTokenPercentage: 1,
          maxTokenPercentage: 50,
          allowedTokens: [],
          blockedTokens: [],
          allowedProtocols: [],
          blockedProtocols: []
        },
        createdAt: new Date(),
        updatedAt: new Date(),
        lastRebalanced: new Date(),
        isActive: true,
        chainId: createPortfolioDto.chainId
      };

      // Save to database
      await this.savePortfolioToDatabase(portfolio);

      // Setup Chainlink Automation if auto-rebalancing is enabled
      if (portfolio.rebalanceSettings.enabled) {
        await this.setupAutomatedRebalancing(portfolio);
      }

      logger.info(`Portfolio created successfully`, {
        portfolioId: portfolio.id,
        userId,
        totalTokens: portfolio.targetAllocation.length,
        chainId: portfolio.chainId
      });

      return portfolio;
    } catch (error) {
      logger.error('Error creating portfolio:', error);
      throw error;
    }
  }

  /**
   * Enrich token allocations with real-time Chainlink price data
   */
  private async enrichTokenAllocations(
    allocations: TokenAllocation[],
    chainId: number
  ): Promise<TokenAllocation[]> {
    const enrichedAllocations = await Promise.all(
      allocations.map(async (allocation) => {
        try {
          // Get token price from Chainlink if available
          const priceData = await chainlinkService.getTokenPrice(allocation.tokenAddress, chainId);
          
          // Get token metadata
          const tokenMetadata = await this.getTokenMetadata(allocation.tokenAddress, chainId);

          return {
            ...allocation,
            priceUSD: priceData?.price || 0,
            decimals: tokenMetadata.decimals,
            name: tokenMetadata.name || allocation.name,
            amount: '0', // Will be set when tokens are deposited
            valueUSD: 0,
            chainId
          };
        } catch (error) {
          logger.warn(`Failed to enrich token ${allocation.tokenAddress}:`, error);
          return {
            ...allocation,
            priceUSD: 0,
            decimals: 18,
            amount: '0',
            valueUSD: 0,
            chainId
          };
        }
      })
    );

    return enrichedAllocations;
  }

  /**
   * Get token metadata from blockchain
   */
  private async getTokenMetadata(tokenAddress: string, chainId: number): Promise<any> {
    try {
      const provider = web3Service.getProvider(chainId);
      if (!provider) {
        throw new Error(`No provider available for chain ${chainId}`);
      }

      const tokenContract = new ethers.Contract(
        tokenAddress,
        [
          'function name() view returns (string)',
          'function symbol() view returns (string)',
          'function decimals() view returns (uint8)'
        ],
        provider
      );

      const [name, symbol, decimals] = await Promise.all([
        tokenContract.name(),
        tokenContract.symbol(),
        tokenContract.decimals()
      ]);

      return { name, symbol, decimals };
    } catch (error) {
      logger.warn(`Failed to get token metadata for ${tokenAddress}:`, error);
      return { name: 'Unknown', symbol: 'UNK', decimals: 18 };
    }
  }

  /**
   * Calculate initial performance metrics
   */
  private async calculateInitialMetrics(allocations: TokenAllocation[]): Promise<PerformanceMetrics> {
    return {
      totalReturn: 0,
      totalReturnPercentage: 0,
      dailyReturn: 0,
      weeklyReturn: 0,
      monthlyReturn: 0,
      yearlyReturn: 0,
      volatility: await this.calculatePortfolioVolatility(allocations),
      sharpeRatio: 0,
      maxDrawdown: 0,
      beta: 0,
      alpha: 0,
      winRate: 0,
      profitFactor: 0
    };
  }

  /**
   * Calculate portfolio volatility using historical price data
   */
  private async calculatePortfolioVolatility(allocations: TokenAllocation[]): Promise<number> {
    try {
      const volatilities = await Promise.all(
        allocations.map(async (allocation) => {
          const historicalData = await chainlinkService.getHistoricalPrices(
            `${allocation.symbol}/USD`,
            { days: 30 }
          );
          
          if (!historicalData || historicalData.length < 2) return 0.2; // Default 20%
          
          // Calculate daily returns
          const returns = [];
          for (let i = 1; i < historicalData.length; i++) {
            const dailyReturn = (historicalData[i].price - historicalData[i-1].price) / historicalData[i-1].price;
            returns.push(dailyReturn);
          }
          
          // Calculate standard deviation
          const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
          const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
          
          return Math.sqrt(variance) * Math.sqrt(365); // Annualized volatility
        })
      );

      // Weight-average portfolio volatility
      const totalWeight = allocations.reduce((sum, allocation) => sum + allocation.percentage, 0);
      const weightedVolatility = allocations.reduce((sum, allocation, index) => {
        const weight = allocation.percentage / totalWeight;
        return sum + (weight * volatilities[index]);
      }, 0);

      return weightedVolatility;
    } catch (error) {
      logger.error('Error calculating portfolio volatility:', error);
      return 0.2; // Default 20% volatility
    }
  }

  /**
   * Calculate risk score using ML models and market data
   */
  private async calculateRiskScore(allocations: TokenAllocation[]): Promise<number> {
    try {
      const riskFactors = await Promise.all(
        allocations.map(async (allocation) => {
          // Get token-specific risk metrics
          const tokenRisk = await mlService.calculateTokenRisk({
            tokenAddress: allocation.tokenAddress,
            chainId: allocation.chainId,
            allocation: allocation.percentage
          });

          return tokenRisk;
        })
      );

      // Aggregate risk scores
      const weightedRisk = allocations.reduce((sum, allocation, index) => {
        const weight = allocation.percentage / 100;
        return sum + (weight * riskFactors[index]);
      }, 0);

      return Math.min(100, Math.max(0, weightedRisk));
    } catch (error) {
      logger.error('Error calculating risk score:', error);
      return 50; // Default medium risk
    }
  }

  /**
   * Analyze portfolio rebalancing needs
   */
  public async analyzeRebalancingNeeds(portfolioId: string): Promise<RebalanceAnalysis> {
    try {
      const portfolio = await this.getPortfolioFromDatabase(portfolioId);
      if (!portfolio) {
        throw new Error('Portfolio not found');
      }

      // Get current token balances and prices
      const currentBalances = await this.getCurrentPortfolioBalances(portfolio);
      const currentPrices = await this.getCurrentTokenPrices(portfolio.targetAllocation, portfolio.chainId);

      // Calculate current allocation percentages
      const totalValue = currentBalances.reduce((sum, balance) => sum + balance.valueUSD, 0);
      const currentAllocation = currentBalances.map(balance => ({
        ...balance,
        percentage: totalValue > 0 ? (balance.valueUSD / totalValue) * 100 : 0
      }));

      // Find maximum drift from target allocation
      let maxDrift = 0;
      const recommendedActions: RebalanceAction[] = [];

      for (const target of portfolio.targetAllocation) {
        const current = currentAllocation.find(c => c.tokenAddress === target.tokenAddress);
        const currentPercentage = current?.percentage || 0;
        const drift = Math.abs(currentPercentage - target.percentage);
        
        if (drift > maxDrift) {
          maxDrift = drift;
        }

        // Determine if action is needed
        if (drift > portfolio.rebalanceSettings.threshold) {
          const action: RebalanceAction = {
            type: currentPercentage > target.percentage ? 'sell' : 'buy',
            tokenAddress: target.tokenAddress,
            amount: Math.abs(currentPercentage - target.percentage) * totalValue / 100,
            priority: drift / target.percentage // Higher drift = higher priority
          };
          recommendedActions.push(action);
        }
      }

      // Sort actions by priority
      recommendedActions.sort((a, b) => b.priority - a.priority);

      const needsRebalancing = maxDrift > portfolio.rebalanceSettings.threshold;

      return {
        needsRebalancing,
        currentAllocation,
        targetAllocation: portfolio.targetAllocation,
        maxDrift,
        recommendedActions
      };
    } catch (error) {
      logger.error('Error analyzing rebalancing needs:', error);
      throw error;
    }
  }

  /**
   * Execute portfolio rebalancing with Chainlink Automation
   */
  public async executeRebalancing(params: RebalanceExecution): Promise<any> {
    try {
      const { portfolioId, userId, strategy, maxSlippage, gasOptimization, analysis } = params;

      if (!analysis.needsRebalancing) {
        throw new Error('Portfolio does not need rebalancing');
      }

      const portfolio = await this.getPortfolioFromDatabase(portfolioId);
      if (!portfolio || portfolio.userId !== userId) {
        throw new Error('Portfolio not found or access denied');
      }

      // Calculate optimal execution order using ML
      const executionPlan = await mlService.optimizeRebalancingExecution({
        actions: analysis.recommendedActions,
        strategy,
        maxSlippage,
        gasOptimization,
        marketConditions: await this.getMarketConditions()
      });

      const transactions = [];
      let totalGasCost = 0;

      // Execute rebalancing transactions
      for (const action of executionPlan.actions) {
        try {
          const txResult = await this.executeRebalanceAction({
            portfolioId,
            action,
            maxSlippage,
            chainId: portfolio.chainId
          });

          transactions.push(txResult);
          totalGasCost += txResult.gasCost;

          // Add delay between transactions if needed
          if (executionPlan.delayBetweenTx > 0) {
            await new Promise(resolve => setTimeout(resolve, executionPlan.delayBetweenTx));
          }
        } catch (error) {
          logger.error(`Failed to execute rebalance action:`, error);
          // Continue with other actions
        }
      }

      // Update portfolio last rebalanced timestamp
      await this.updatePortfolioRebalanceTimestamp(portfolioId);

      // Log rebalancing event
      await this.logRebalancingEvent({
        portfolioId,
        userId,
        analysis,
        transactions,
        totalGasCost,
        strategy
      });

      logger.info(`Portfolio rebalancing completed`, {
        portfolioId,
        actionsExecuted: transactions.length,
        totalGasCost
      });

      return {
        success: true,
        transactions,
        totalGasCost,
        estimatedGasCost: executionPlan.estimatedGas,
        executionTime: Date.now() - params.startTime
      };
    } catch (error) {
      logger.error('Error executing portfolio rebalancing:', error);
      throw error;
    }
  }

  /**
   * Execute individual rebalance action
   */
  private async executeRebalanceAction(params: any): Promise<any> {
    const { portfolioId, action, maxSlippage, chainId } = params;

    try {
      const signer = web3Service.getSignerForChain('portfolio', chainId);
      if (!signer) {
        throw new Error('No signer available for portfolio operations');
      }

      let txResult: any;

      if (action.type === 'sell') {
        txResult = await this.executeSellTransaction({
          signer,
          tokenAddress: action.tokenAddress,
          amount: action.amount,
          maxSlippage,
          chainId
        });
      } else {
        txResult = await this.executeBuyTransaction({
          signer,
          tokenAddress: action.tokenAddress,
          amount: action.amount,
          maxSlippage,
          chainId
        });
      }

      return {
        hash: txResult.hash,
        type: action.type,
        tokenAddress: action.tokenAddress,
        amount: action.amount,
        gasCost: txResult.gasCost,
        timestamp: new Date()
      };
    } catch (error) {
      logger.error('Error executing rebalance action:', error);
      throw error;
    }
  }

  /**
   * Setup automated rebalancing using Chainlink Automation
   */
  private async setupAutomatedRebalancing(portfolio: Portfolio): Promise<void> {
    try {
      // Create automation upkeep
      const upkeepId = await chainlinkService.createAutomationUpkeep({
        name: `Portfolio Rebalancing - ${portfolio.name}`,
        encryptedEmail: '', // Optional
        upkeepContract: await this.getRebalanceContractAddress(portfolio.chainId),
        gasLimit: 500000,
        adminAddress: await this.getPortfolioManagerAddress(portfolio.userId),
        checkData: ethers.utils.defaultAbiCoder.encode(
          ['string', 'uint256'],
          [portfolio.id, portfolio.rebalanceSettings.threshold * 100]
        ),
        amount: ethers.utils.parseEther('1'), // 1 LINK initial funding
        source: 110, // Portfolio rebalancing source
        sender: await this.getPortfolioManagerAddress(portfolio.userId)
      });

      // Store upkeep ID for future reference
      this.automationUpkeeps.set(portfolio.id, upkeepId);

      // Save upkeep ID to database
      await this.saveAutomationUpkeepId(portfolio.id, upkeepId);

      logger.info(`Automated rebalancing setup`, {
        portfolioId: portfolio.id,
        upkeepId,
        threshold: portfolio.rebalanceSettings.threshold
      });
    } catch (error) {
      logger.error('Error setting up automated rebalancing:', error);
      throw error;
    }
  }

  /**
   * Get portfolio performance metrics
   */
  public async getPortfolioPerformance(
    portfolioId: string,
    userId: string,
    timeRange: { start: Date; end: Date }
  ): Promise<PerformanceMetrics> {
    try {
      const portfolio = await this.getPortfolioFromDatabase(portfolioId);
      if (!portfolio || portfolio.userId !== userId) {
        throw new Error('Portfolio not found or access denied');
      }

      // Check cache first
      const cacheKey = `${portfolioId}-${timeRange.start.getTime()}-${timeRange.end.getTime()}`;
      const cachedMetrics = this.performanceCache.get(cacheKey);
      if (cachedMetrics) {
        return cachedMetrics;
      }

      // Calculate performance metrics
      const metrics = await this.calculatePerformanceMetrics(portfolio, timeRange);

      // Cache for 5 minutes
      this.performanceCache.set(cacheKey, metrics);
      setTimeout(() => this.performanceCache.delete(cacheKey), 300000);

      return metrics;
    } catch (error) {
      logger.error('Error getting portfolio performance:', error);
      throw error;
    }
  }

  /**
   * Calculate comprehensive performance metrics
   */
  private async calculatePerformanceMetrics(
    portfolio: Portfolio,
    timeRange: { start: Date; end: Date }
  ): Promise<PerformanceMetrics> {
    try {
      // Get historical portfolio values
      const historicalValues = await this.getHistoricalPortfolioValues(portfolio.id, timeRange);
      
      if (historicalValues.length < 2) {
        return portfolio.performanceMetrics; // Return cached metrics if no history
      }

      // Calculate returns
      const initialValue = historicalValues[0].value;
      const finalValue = historicalValues[historicalValues.length - 1].value;
      const totalReturn = finalValue - initialValue;
      const totalReturnPercentage = initialValue > 0 ? (totalReturn / initialValue) * 100 : 0;

      // Calculate period-specific returns
      const dailyReturns = this.calculateDailyReturns(historicalValues);
      const weeklyReturn = this.calculatePeriodReturn(historicalValues, 7);
      const monthlyReturn = this.calculatePeriodReturn(historicalValues, 30);
      const yearlyReturn = this.calculateAnnualizedReturn(dailyReturns);

      // Calculate volatility (standard deviation of daily returns)
      const volatility = this.calculateVolatility(dailyReturns);

      // Calculate Sharpe ratio (assuming 3% risk-free rate)
      const riskFreeRate = 0.03;
      const sharpeRatio = volatility > 0 ? (yearlyReturn - riskFreeRate) / volatility : 0;

      // Calculate maximum drawdown
      const maxDrawdown = this.calculateMaxDrawdown(historicalValues);

      // Calculate alpha and beta (vs market benchmark)
      const { alpha, beta } = await this.calculateAlphaBeta(historicalValues, timeRange);

      // Calculate win rate and profit factor
      const { winRate, profitFactor } = this.calculateWinMetrics(dailyReturns);

      const metrics: PerformanceMetrics = {
        totalReturn,
        totalReturnPercentage,
        dailyReturn: dailyReturns.length > 0 ? dailyReturns[dailyReturns.length - 1] : 0,
        weeklyReturn,
        monthlyReturn,
        yearlyReturn,
        volatility,
        sharpeRatio,
        maxDrawdown,
        beta,
        alpha,
        winRate,
        profitFactor
      };

      return metrics;
    } catch (error) {
      logger.error('Error calculating performance metrics:', error);
      throw error;
    }
  }

  /**
   * Simulate portfolio rebalancing
   */
  public async simulateRebalancing(params: any): Promise<any> {
    try {
      const { portfolioId, userId, newAllocation, strategy } = params;

      const portfolio = await this.getPortfolioFromDatabase(portfolioId);
      if (!portfolio || portfolio.userId !== userId) {
        throw new Error('Portfolio not found or access denied');
      }

      // Get current portfolio state
      const currentState = await this.getCurrentPortfolioBalances(portfolio);
      const totalValue = currentState.reduce((sum, balance) => sum + balance.valueUSD, 0);

      // Calculate required trades for new allocation
      const requiredTrades = newAllocation.map((target: TokenAllocation) => {
        const current = currentState.find(c => c.tokenAddress === target.tokenAddress);
        const currentValue = current?.valueUSD || 0;
        const targetValue = (target.percentage / 100) * totalValue;
        const difference = targetValue - currentValue;

        return {
          tokenAddress: target.tokenAddress,
          symbol: target.symbol,
          type: difference > 0 ? 'buy' : 'sell',
          amount: Math.abs(difference),
          percentage: target.percentage,
          impact: Math.abs(difference) / totalValue * 100
        };
      });

      // Estimate costs and slippage
      const estimatedCosts = await Promise.all(
        requiredTrades.map(async (trade) => {
          const gasEstimate = await this.estimateRebalanceGas(trade, portfolio.chainId);
          const slippageEstimate = await this.estimateSlippage(trade, portfolio.chainId);
          
          return {
            ...trade,
            estimatedGas: gasEstimate,
            estimatedSlippage: slippageEstimate,
            estimatedCost: gasEstimate + (trade.amount * slippageEstimate / 100)
          };
        })
      );

      const totalEstimatedCost = estimatedCosts.reduce((sum, cost) => sum + cost.estimatedCost, 0);
      const totalEstimatedGas = estimatedCosts.reduce((sum, cost) => sum + cost.estimatedGas, 0);

      // Calculate expected outcome
      const expectedOutcome = {
        newAllocation,
        requiredTrades: estimatedCosts,
        totalCost: totalEstimatedCost,
        totalGas: totalEstimatedGas,
        costPercentage: (totalEstimatedCost / totalValue) * 100,
        expectedImprovement: await this.calculateExpectedImprovement(portfolio, newAllocation)
      };

      return expectedOutcome;
    } catch (error) {
      logger.error('Error simulating portfolio rebalancing:', error);
      throw error;
    }
  }

  /**
   * Update rebalance settings
   */
  public async updateRebalanceSettings(
    portfolioId: string,
    userId: string,
    settings: RebalanceSettings
  ): Promise<Portfolio> {
    try {
      const portfolio = await this.getPortfolioFromDatabase(portfolioId);
      if (!portfolio || portfolio.userId !== userId) {
        throw new Error('Portfolio not found or access denied');
      }

      // Update rebalance settings
      portfolio.rebalanceSettings = settings;
      portfolio.updatedAt = new Date();

      // Save to database
      await this.savePortfolioToDatabase(portfolio);

      // Update automation if settings changed
      if (settings.enabled && !this.automationUpkeeps.has(portfolioId)) {
        await this.setupAutomatedRebalancing(portfolio);
      } else if (!settings.enabled && this.automationUpkeeps.has(portfolioId)) {
        await this.cancelAutomatedRebalancing(portfolioId);
      }

      return portfolio;
    } catch (error) {
      logger.error('Error updating rebalance settings:', error);
      throw error;
    }
  }

  // Helper methods
  private async getCurrentPortfolioBalances(portfolio: Portfolio): Promise<any[]> {
    const balances = await Promise.all(
      portfolio.targetAllocation.map(async (allocation) => {
        const balance = await web3Service.getBalance(
          'portfolio',
          portfolio.chainId,
          allocation.tokenAddress
        );

        const price = await chainlinkService.getTokenPrice(allocation.tokenAddress, portfolio.chainId);
        const tokenAmount = parseFloat(ethers.utils.formatUnits(balance, allocation.decimals));
        const valueUSD = tokenAmount * (price?.price || 0);

        return {
          tokenAddress: allocation.tokenAddress,
          symbol: allocation.symbol,
          balance: tokenAmount,
          valueUSD,
          priceUSD: price?.price || 0
        };
      })
    );

    return balances;
  }

  private async getCurrentTokenPrices(allocations: TokenAllocation[], chainId: number): Promise<Map<string, number>> {
    const prices = new Map<string, number>();

    await Promise.all(
      allocations.map(async (allocation) => {
        const priceData = await chainlinkService.getTokenPrice(allocation.tokenAddress, chainId);
        prices.set(allocation.tokenAddress, priceData?.price || 0);
      })
    );

    return prices;
  }

  private async getMarketConditions(): Promise<any> {
    // Get overall market conditions for optimization
    return {
      volatility: 0.2,
      volume: 1000000,
      gasPrice: 30,
      congestion: 'low'
    };
  }

  private async executeSellTransaction(params: any): Promise<any> {
    // Implementation would execute sell transaction on DEX
    return {
      hash: '0x...',
      gasCost: 0.01 // ETH
    };
  }

  private async executeBuyTransaction(params: any): Promise<any> {
    // Implementation would execute buy transaction on DEX
    return {
      hash: '0x...',
      gasCost: 0.01 // ETH
    };
  }

  private calculateDailyReturns(historicalValues: any[]): number[] {
    const returns = [];
    for (let i = 1; i < historicalValues.length; i++) {
      const dailyReturn = (historicalValues[i].value - historicalValues[i-1].value) / historicalValues[i-1].value;
      returns.push(dailyReturn);
    }
    return returns;
  }

  private calculatePeriodReturn(historicalValues: any[], days: number): number {
    if (historicalValues.length < days) return 0;
    
    const start = historicalValues[historicalValues.length - days].value;
    const end = historicalValues[historicalValues.length - 1].value;
    
    return start > 0 ? ((end - start) / start) * 100 : 0;
  }

  private calculateAnnualizedReturn(dailyReturns: number[]): number {
    if (dailyReturns.length === 0) return 0;
    
    const averageDailyReturn = dailyReturns.reduce((sum, ret) => sum + ret, 0) / dailyReturns.length;
    return averageDailyReturn * 365 * 100; // Annualized percentage
  }

  private calculateVolatility(dailyReturns: number[]): number {
    if (dailyReturns.length < 2) return 0;
    
    const mean = dailyReturns.reduce((sum, ret) => sum + ret, 0) / dailyReturns.length;
    const variance = dailyReturns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / dailyReturns.length;
    
    return Math.sqrt(variance) * Math.sqrt(365); // Annualized volatility
  }

  private calculateMaxDrawdown(historicalValues: any[]): number {
    let maxDrawdown = 0;
    let peak = historicalValues[0]?.value || 0;
    
    for (const point of historicalValues) {
      if (point.value > peak) {
        peak = point.value;
      }
      
      const drawdown = (peak - point.value) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }
    
    return maxDrawdown * 100; // Percentage
  }

  private async calculateAlphaBeta(
    historicalValues: any[],
    timeRange: { start: Date; end: Date }
  ): Promise<{ alpha: number; beta: number }> {
    try {
      // Get benchmark data (e.g., ETH/USD)
      const benchmarkData = await chainlinkService.getHistoricalPrices('ETH/USD', {
        start: timeRange.start,
        end: timeRange.end
      });

      if (!benchmarkData || benchmarkData.length !== historicalValues.length) {
        return { alpha: 0, beta: 1 };
      }

      // Calculate returns for both portfolio and benchmark
      const portfolioReturns = this.calculateDailyReturns(historicalValues);
      const benchmarkReturns = this.calculateDailyReturns(benchmarkData);

      // Calculate beta using covariance and variance
      const covariance = this.calculateCovariance(portfolioReturns, benchmarkReturns);
      const benchmarkVariance = this.calculateVariance(benchmarkReturns);
      
      const beta = benchmarkVariance > 0 ? covariance / benchmarkVariance : 1;

      // Calculate alpha
      const portfolioMean = portfolioReturns.reduce((sum, ret) => sum + ret, 0) / portfolioReturns.length;
      const benchmarkMean = benchmarkReturns.reduce((sum, ret) => sum + ret, 0) / benchmarkReturns.length;
      const alpha = portfolioMean - (beta * benchmarkMean);

      return { alpha: alpha * 365 * 100, beta }; // Annualized alpha as percentage
    } catch (error) {
      logger.error('Error calculating alpha/beta:', error);
      return { alpha: 0, beta: 1 };
    }
  }

  private calculateCovariance(returns1: number[], returns2: number[]): number {
    if (returns1.length !== returns2.length || returns1.length === 0) return 0;
    
    const mean1 = returns1.reduce((sum, ret) => sum + ret, 0) / returns1.length;
    const mean2 = returns2.reduce((sum, ret) => sum + ret, 0) / returns2.length;
    
    const covariance = returns1.reduce((sum, ret1, index) => {
      return sum + ((ret1 - mean1) * (returns2[index] - mean2));
    }, 0) / returns1.length;
    
    return covariance;
  }

  private calculateVariance(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    return returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
  }

  private calculateWinMetrics(dailyReturns: number[]): { winRate: number; profitFactor: number } {
    const wins = dailyReturns.filter(ret => ret > 0);
    const losses = dailyReturns.filter(ret => ret < 0);
    
    const winRate = dailyReturns.length > 0 ? (wins.length / dailyReturns.length) * 100 : 0;
    
    const totalWins = wins.reduce((sum, win) => sum + win, 0);
    const totalLosses = Math.abs(losses.reduce((sum, loss) => sum + loss, 0));
    
    const profitFactor = totalLosses > 0 ? totalWins / totalLosses : 0;
    
    return { winRate, profitFactor };
  }

  // Database and infrastructure methods
  private async savePortfolioToDatabase(portfolio: Portfolio): Promise<void> {
    // Implementation would save to database
    logger.debug(`Saving portfolio to database`, { portfolioId: portfolio.id });
  }

  private async getPortfolioFromDatabase(portfolioId: string): Promise<Portfolio | null> {
    // Implementation would query database
    return null;
  }

  private async saveAutomationUpkeepId(portfolioId: string, upkeepId: string): Promise<void> {
    // Implementation would save upkeep ID to database
    logger.debug(`Saving automation upkeep ID`, { portfolioId, upkeepId });
  }

  private async updatePortfolioRebalanceTimestamp(portfolioId: string): Promise<void> {
    // Implementation would update database
    logger.debug(`Updating portfolio rebalance timestamp`, { portfolioId });
  }

  private async logRebalancingEvent(event: any): Promise<void> {
    // Implementation would log to database
    logger.info(`Rebalancing event logged`, event);
  }

  private async getRebalanceContractAddress(chainId: number): Promise<string> {
    // Return the deployed rebalance contract address for the chain
    return web3Service.getContractAddress('portfolioManager', chainId) || '';
  }

  private async getPortfolioManagerAddress(userId: string): Promise<string> {
    // Get user's portfolio manager address
    return '0x...'; // Implementation would get from user data
  }

  private async cancelAutomatedRebalancing(portfolioId: string): Promise<void> {
    const upkeepId = this.automationUpkeeps.get(portfolioId);
    if (upkeepId) {
      await chainlinkService.cancelAutomationUpkeep(upkeepId);
      this.automationUpkeeps.delete(portfolioId);
    }
  }

  private async getHistoricalPortfolioValues(portfolioId: string, timeRange: any): Promise<any[]> {
    // Implementation would query historical values from database
    return [];
  }

  private async estimateRebalanceGas(trade: any, chainId: number): Promise<number> {
    // Estimate gas cost for rebalance trade
    const gasPrice = await web3Service.getGasPrice(chainId);
    return gasPrice.toNumber() * 200000; // Estimated gas for swap
  }

  private async estimateSlippage(trade: any, chainId: number): Promise<number> {
    // Estimate slippage based on liquidity and trade size
    return 0.5; // 0.5% default slippage
  }

  private async calculateExpectedImprovement(portfolio: Portfolio, newAllocation: TokenAllocation[]): Promise<number> {
    // Calculate expected improvement from rebalancing
    return 2.5; // 2.5% expected improvement
  }

  // Public interface methods
  public async getPortfolioById(portfolioId: string, userId?: string): Promise<Portfolio | null> {
    const portfolio = await this.getPortfolioFromDatabase(portfolioId);
    if (portfolio && (!userId || portfolio.userId === userId)) {
      return portfolio;
    }
    return null;
  }

  public async getUserPortfolios(userId: string, filters: any, pagination: any): Promise<any> {
    // Implementation would query user portfolios with filters
    return { portfolios: [], total: 0 };
  }

  public async updatePortfolio(portfolioId: string, userId: string, updateData: UpdatePortfolioDto): Promise<Portfolio> {
    // Implementation would update portfolio
    const portfolio = await this.getPortfolioFromDatabase(portfolioId);
    if (!portfolio || portfolio.userId !== userId) {
      throw new Error('Portfolio not found or access denied');
    }

    const updatedPortfolio = { ...portfolio, ...updateData, updatedAt: new Date() };
    await this.savePortfolioToDatabase(updatedPortfolio);
    return updatedPortfolio;
  }

  public async deletePortfolio(portfolioId: string, userId: string): Promise<void> {
    const portfolio = await this.getPortfolioFromDatabase(portfolioId);
    if (!portfolio || portfolio.userId !== userId) {
      throw new Error('Portfolio not found or access denied');
    }

    // Cancel automation if active
    if (this.automationUpkeeps.has(portfolioId)) {
      await this.cancelAutomatedRebalancing(portfolioId);
    }

    // Delete from database
    // Implementation would delete from database
    logger.info(`Portfolio deleted`, { portfolioId, userId });
  }

  public async getRebalancingHistory(portfolioId: string, userId: string, pagination: any): Promise<any> {
    // Implementation would query rebalancing history
    return { records: [], total: 0 };
  }
}

export const portfolioService = new PortfolioService();
export default portfolioService;
