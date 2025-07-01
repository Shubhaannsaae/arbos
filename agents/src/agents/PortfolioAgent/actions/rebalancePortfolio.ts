import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { Portfolio, PortfolioPosition } from '../../../shared/types/market';
import { PORTFOLIO_THRESHOLDS } from '../../../shared/constants/thresholds';
import { PortfolioProvider } from '../providers/portfolioProvider';
import { MarketDataProvider } from '../providers/marketDataProvider';

export interface RebalanceConfig {
  maxSlippage: number;
  gasOptimization: boolean;
  useChainlinkAutomation: boolean;
  emergencyStopLoss?: number;
  maxRebalanceSize?: BigNumber;
  phaseRebalancing?: boolean;
}

export interface RebalanceResult {
  success: boolean;
  transactions: string[];
  totalGasUsed: BigNumber;
  newPositions: PortfolioPosition[];
  rebalanceMetrics: {
    totalValueBefore: BigNumber;
    totalValueAfter: BigNumber;
    costBasis: BigNumber;
    slippageExperienced: number;
    deviationReduced: number;
  };
  errors: string[];
}

export interface RebalanceStep {
  step: string;
  fromToken: string;
  toToken: string;
  amountIn: BigNumber;
  expectedAmountOut: BigNumber;
  priority: number;
  estimatedGas: BigNumber;
}

export async function rebalancePortfolio(
  portfolio: Portfolio,
  portfolioProvider: PortfolioProvider,
  marketDataProvider: MarketDataProvider,
  config: RebalanceConfig
): Promise<RebalanceResult> {
  const startTime = Date.now();
  const transactions: string[] = [];
  const errors: string[] = [];
  let totalGasUsed = BigNumber.from(0);

  logger.info('Starting portfolio rebalancing', {
    portfolioId: portfolio.id,
    currentValue: ethers.utils.formatEther(portfolio.totalValue),
    positionCount: portfolio.positions.length,
    config
  });

  try {
    // Step 1: Calculate required trades
    const rebalanceSteps = await calculateRebalanceSteps(
      portfolio,
      marketDataProvider,
      config
    );

    if (rebalanceSteps.length === 0) {
      logger.info('No rebalancing needed', { portfolioId: portfolio.id });
      
      return {
        success: true,
        transactions: [],
        totalGasUsed: BigNumber.from(0),
        newPositions: portfolio.positions,
        rebalanceMetrics: {
          totalValueBefore: portfolio.totalValue,
          totalValueAfter: portfolio.totalValue,
          costBasis: BigNumber.from(0),
          slippageExperienced: 0,
          deviationReduced: 0
        },
        errors: []
      };
    }

    // Step 2: Execute rebalancing strategy
    const executionResult = config.phaseRebalancing 
      ? await executePhaseRebalancing(rebalanceSteps, portfolioProvider, config)
      : await executeAtomicRebalancing(rebalanceSteps, portfolioProvider, config);

    transactions.push(...executionResult.transactions);
    totalGasUsed = totalGasUsed.add(executionResult.gasUsed);

    // Step 3: Update portfolio state
    const updatedPositions = await updatePortfolioPositions(
      portfolio,
      executionResult.trades,
      marketDataProvider
    );

    // Step 4: Calculate rebalancing metrics
    const rebalanceMetrics = await calculateRebalanceMetrics(
      portfolio,
      updatedPositions,
      executionResult,
      marketDataProvider
    );

    // Step 5: Set up Chainlink Automation for future rebalancing
    if (config.useChainlinkAutomation) {
      await setupAutomatedRebalancing(portfolio, portfolioProvider, config);
    }

    logger.info('Portfolio rebalancing completed', {
      portfolioId: portfolio.id,
      transactionCount: transactions.length,
      totalGasUsed: totalGasUsed.toString(),
      valueChange: ethers.utils.formatEther(
        rebalanceMetrics.totalValueAfter.sub(rebalanceMetrics.totalValueBefore)
      ),
      duration: Date.now() - startTime
    });

    return {
      success: true,
      transactions,
      totalGasUsed,
      newPositions: updatedPositions,
      rebalanceMetrics,
      errors
    };

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    errors.push(errorMessage);

    logger.error('Portfolio rebalancing failed', {
      portfolioId: portfolio.id,
      error: errorMessage,
      duration: Date.now() - startTime
    });

    return {
      success: false,
      transactions,
      totalGasUsed,
      newPositions: portfolio.positions,
      rebalanceMetrics: {
        totalValueBefore: portfolio.totalValue,
        totalValueAfter: portfolio.totalValue,
        costBasis: BigNumber.from(0),
        slippageExperienced: 0,
        deviationReduced: 0
      },
      errors
    };
  }
}

async function calculateRebalanceSteps(
  portfolio: Portfolio,
  marketDataProvider: MarketDataProvider,
  config: RebalanceConfig
): Promise<RebalanceStep[]> {
  const steps: RebalanceStep[] = [];
  
  try {
    // Get current prices for all positions
    const currentPrices = await marketDataProvider.getCurrentPrices(
      portfolio.positions.map(p => p.token.symbol)
    );

    // Calculate current allocation vs target
    const totalValue = portfolio.totalValue;
    const rebalanceSteps: Array<{
      token: string;
      currentValue: BigNumber;
      targetValue: BigNumber;
      difference: BigNumber;
      action: 'buy' | 'sell' | 'hold';
    }> = [];

    // Calculate what needs to be bought/sold
    for (const targetAllocation of portfolio.allocation) {
      const currentPosition = portfolio.positions.find(
        p => p.token.symbol === targetAllocation.token.symbol
      );

      const targetValue = totalValue.mul(Math.floor(targetAllocation.targetPercentage * 100)).div(10000);
      const currentValue = currentPosition?.value || BigNumber.from(0);
      const difference = targetValue.sub(currentValue);

      // Only rebalance if difference exceeds threshold
      const thresholdValue = totalValue.mul(Math.floor(targetAllocation.rebalanceThreshold * 100)).div(10000);
      
      if (difference.abs().gt(thresholdValue)) {
        rebalanceSteps.push({
          token: targetAllocation.token.symbol,
          currentValue,
          targetValue,
          difference,
          action: difference.gt(0) ? 'buy' : 'sell'
        });
      }
    }

    // Convert to executable trading steps
    const sellSteps = rebalanceSteps.filter(s => s.action === 'sell');
    const buySteps = rebalanceSteps.filter(s => s.action === 'buy');

    // Create sell orders first
    for (const sellStep of sellSteps) {
      const sellAmount = sellStep.difference.abs();
      
      for (const buyStep of buySteps) {
        if (sellAmount.isZero()) break;
        
        const buyAmount = buyStep.difference;
        const tradeAmount = sellAmount.lt(buyAmount) ? sellAmount : buyAmount;
        
        const expectedOutput = await marketDataProvider.getSwapQuote(
          sellStep.token,
          buyStep.token,
          tradeAmount
        );

        steps.push({
          step: `${sellStep.token}_to_${buyStep.token}`,
          fromToken: sellStep.token,
          toToken: buyStep.token,
          amountIn: tradeAmount,
          expectedAmountOut: expectedOutput,
          priority: sellSteps.indexOf(sellStep) + buySteps.indexOf(buyStep),
          estimatedGas: await estimateTradeGas(sellStep.token, buyStep.token, tradeAmount)
        });

        // Update remaining amounts
        buyStep.difference = buyStep.difference.sub(expectedOutput);
        sellStep.difference = sellStep.difference.sub(tradeAmount);
      }
    }

    // Sort steps by priority and gas efficiency
    steps.sort((a, b) => {
      const priorityDiff = a.priority - b.priority;
      if (priorityDiff !== 0) return priorityDiff;
      
      // Prefer lower gas cost trades
      return a.estimatedGas.lt(b.estimatedGas) ? -1 : 1;
    });

    logger.debug('Rebalance steps calculated', {
      portfolioId: portfolio.id,
      stepCount: steps.length,
      totalEstimatedGas: steps.reduce((sum, step) => sum.add(step.estimatedGas), BigNumber.from(0)).toString()
    });

    return steps;

  } catch (error) {
    logger.error('Failed to calculate rebalance steps', {
      portfolioId: portfolio.id,
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function executeAtomicRebalancing(
  steps: RebalanceStep[],
  portfolioProvider: PortfolioProvider,
  config: RebalanceConfig
): Promise<{
  transactions: string[];
  gasUsed: BigNumber;
  trades: Array<{
    fromToken: string;
    toToken: string;
    amountIn: BigNumber;
    amountOut: BigNumber;
    slippage: number;
  }>;
}> {
  const transactions: string[] = [];
  const trades: any[] = [];
  let totalGasUsed = BigNumber.from(0);

  logger.info('Executing atomic rebalancing', {
    stepCount: steps.length,
    gasOptimization: config.gasOptimization
  });

  for (const step of steps) {
    try {
      // Execute individual trade
      const tradeResult = await portfolioProvider.executeTrade(
        step.fromToken,
        step.toToken,
        step.amountIn,
        step.expectedAmountOut,
        config.maxSlippage,
        config.gasOptimization
      );

      transactions.push(tradeResult.transactionHash);
      totalGasUsed = totalGasUsed.add(tradeResult.gasUsed);

      // Calculate actual slippage
      const expectedValue = step.expectedAmountOut;
      const actualValue = tradeResult.amountOut;
      const slippage = Math.abs(
        parseFloat(ethers.utils.formatEther(expectedValue.sub(actualValue))) /
        parseFloat(ethers.utils.formatEther(expectedValue))
      ) * 100;

      trades.push({
        fromToken: step.fromToken,
        toToken: step.toToken,
        amountIn: step.amountIn,
        amountOut: tradeResult.amountOut,
        slippage
      });

      logger.debug('Trade executed', {
        fromToken: step.fromToken,
        toToken: step.toToken,
        amountIn: ethers.utils.formatEther(step.amountIn),
        amountOut: ethers.utils.formatEther(tradeResult.amountOut),
        slippage: slippage.toFixed(3),
        gasUsed: tradeResult.gasUsed.toString()
      });

      // Check if slippage exceeds emergency threshold
      if (config.emergencyStopLoss && slippage > config.emergencyStopLoss) {
        logger.warn('Emergency stop-loss triggered due to high slippage', {
          slippage,
          threshold: config.emergencyStopLoss
        });
        break;
      }

    } catch (error) {
      logger.error('Trade execution failed', {
        step: step.step,
        error: error instanceof Error ? error.message : String(error)
      });

      // Continue with other trades unless it's a critical error
      if (error instanceof Error && error.message.includes('insufficient')) {
        throw error; // Stop execution for insufficient funds
      }
    }
  }

  return {
    transactions,
    gasUsed: totalGasUsed,
    trades
  };
}

async function executePhaseRebalancing(
  steps: RebalanceStep[],
  portfolioProvider: PortfolioProvider,
  config: RebalanceConfig
): Promise<{
  transactions: string[];
  gasUsed: BigNumber;
  trades: Array<{
    fromToken: string;
    toToken: string;
    amountIn: BigNumber;
    amountOut: BigNumber;
    slippage: number;
  }>;
}> {
  const transactions: string[] = [];
  const trades: any[] = [];
  let totalGasUsed = BigNumber.from(0);

  logger.info('Executing phased rebalancing', {
    stepCount: steps.length,
    phases: Math.ceil(steps.length / 3) // Execute in batches of 3
  });

  // Execute in phases to reduce market impact
  const phaseSize = 3;
  for (let i = 0; i < steps.length; i += phaseSize) {
    const phase = steps.slice(i, i + phaseSize);
    
    logger.debug(`Executing rebalancing phase ${Math.floor(i / phaseSize) + 1}`, {
      stepCount: phase.length
    });

    // Execute phase with small delay between trades
    for (const [index, step] of phase.entries()) {
      try {
        // Add small delay between trades to reduce MEV risk
        if (index > 0) {
          await new Promise(resolve => setTimeout(resolve, 2000)); // 2 second delay
        }

        const tradeResult = await portfolioProvider.executeTrade(
          step.fromToken,
          step.toToken,
          step.amountIn,
          step.expectedAmountOut,
          config.maxSlippage,
          config.gasOptimization
        );

        transactions.push(tradeResult.transactionHash);
        totalGasUsed = totalGasUsed.add(tradeResult.gasUsed);

        const slippage = Math.abs(
          parseFloat(ethers.utils.formatEther(step.expectedAmountOut.sub(tradeResult.amountOut))) /
          parseFloat(ethers.utils.formatEther(step.expectedAmountOut))
        ) * 100;

        trades.push({
          fromToken: step.fromToken,
          toToken: step.toToken,
          amountIn: step.amountIn,
          amountOut: tradeResult.amountOut,
          slippage
        });

      } catch (error) {
        logger.error('Phase trade execution failed', {
          phase: Math.floor(i / phaseSize) + 1,
          step: step.step,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Longer delay between phases
    if (i + phaseSize < steps.length) {
      await new Promise(resolve => setTimeout(resolve, 10000)); // 10 second delay
    }
  }

  return {
    transactions,
    gasUsed: totalGasUsed,
    trades
  };
}

async function updatePortfolioPositions(
  portfolio: Portfolio,
  trades: Array<{
    fromToken: string;
    toToken: string;
    amountIn: BigNumber;
    amountOut: BigNumber;
    slippage: number;
  }>,
  marketDataProvider: MarketDataProvider
): Promise<PortfolioPosition[]> {
  const updatedPositions = [...portfolio.positions];

  // Apply trade results to positions
  for (const trade of trades) {
    // Reduce position for token sold
    const fromIndex = updatedPositions.findIndex(p => p.token.symbol === trade.fromToken);
    if (fromIndex !== -1) {
      const currentAmount = updatedPositions[fromIndex].amount;
      const newAmount = currentAmount.sub(trade.amountIn);
      
      if (newAmount.gt(0)) {
        updatedPositions[fromIndex] = {
          ...updatedPositions[fromIndex],
          amount: newAmount
        };
      } else {
        updatedPositions.splice(fromIndex, 1);
      }
    }

    // Increase position for token bought
    const toIndex = updatedPositions.findIndex(p => p.token.symbol === trade.toToken);
    if (toIndex !== -1) {
      updatedPositions[toIndex] = {
        ...updatedPositions[toIndex],
        amount: updatedPositions[toIndex].amount.add(trade.amountOut)
      };
    } else {
      // Create new position
      const tokenInfo = await marketDataProvider.getTokenInfo(trade.toToken);
      if (tokenInfo) {
        updatedPositions.push({
          token: tokenInfo,
          amount: trade.amountOut,
          value: trade.amountOut, // Will be updated with current price
          valueUsd: 0,
          percentage: 0,
          averageCost: trade.amountOut, // Simplified cost basis
          unrealizedPnl: BigNumber.from(0),
          unrealizedPnlPercentage: 0,
          lastUpdated: Date.now()
        });
      }
    }
  }

  // Update values and percentages
  const currentPrices = await marketDataProvider.getCurrentPrices(
    updatedPositions.map(p => p.token.symbol)
  );

  let totalValue = BigNumber.from(0);

  // Calculate new values
  for (const position of updatedPositions) {
    const price = currentPrices[position.token.symbol];
    if (price) {
      position.value = position.amount.mul(price).div(ethers.utils.parseUnits('1', position.token.decimals));
      totalValue = totalValue.add(position.value);
    }
  }

  // Calculate percentages
  for (const position of updatedPositions) {
    position.percentage = totalValue.gt(0) 
      ? parseFloat(ethers.utils.formatEther(position.value.mul(10000).div(totalValue))) / 100
      : 0;
  }

  return updatedPositions;
}

async function calculateRebalanceMetrics(
  originalPortfolio: Portfolio,
  newPositions: PortfolioPosition[],
  executionResult: any,
  marketDataProvider: MarketDataProvider
): Promise<{
  totalValueBefore: BigNumber;
  totalValueAfter: BigNumber;
  costBasis: BigNumber;
  slippageExperienced: number;
  deviationReduced: number;
}> {
  const totalValueBefore = originalPortfolio.totalValue;
  const totalValueAfter = newPositions.reduce((sum, pos) => sum.add(pos.value), BigNumber.from(0));
  
  // Calculate transaction costs
  const costBasis = executionResult.gasUsed.mul(ethers.utils.parseUnits('20', 'gwei')); // Estimate with 20 gwei

  // Calculate average slippage
  const avgSlippage = executionResult.trades.length > 0
    ? executionResult.trades.reduce((sum: number, trade: any) => sum + trade.slippage, 0) / executionResult.trades.length
    : 0;

  // Calculate deviation reduction
  const originalDeviation = calculateAllocationDeviation(originalPortfolio);
  const newDeviation = calculateAllocationDeviationFromPositions(newPositions, originalPortfolio.allocation);
  const deviationReduced = Math.max(0, originalDeviation - newDeviation);

  return {
    totalValueBefore,
    totalValueAfter,
    costBasis,
    slippageExperienced: avgSlippage,
    deviationReduced
  };
}

async function setupAutomatedRebalancing(
  portfolio: Portfolio,
  portfolioProvider: PortfolioProvider,
  config: RebalanceConfig
): Promise<void> {
  try {
    // Register Chainlink Automation upkeep for periodic rebalancing
    const upkeepId = await portfolioProvider.registerRebalanceUpkeep(
      portfolio.id,
      PORTFOLIO_THRESHOLDS.AUTOMATION_CHECK_FREQUENCY,
      PORTFOLIO_THRESHOLDS.AUTOMATION_GAS_LIMIT
    );

    logger.info('Automated rebalancing setup complete', {
      portfolioId: portfolio.id,
      upkeepId,
      checkFrequency: PORTFOLIO_THRESHOLDS.AUTOMATION_CHECK_FREQUENCY
    });

  } catch (error) {
    logger.error('Failed to setup automated rebalancing', {
      portfolioId: portfolio.id,
      error: error instanceof Error ? error.message : String(error)
    });
  }
}

// Helper functions
async function estimateTradeGas(fromToken: string, toToken: string, amount: BigNumber): Promise<BigNumber> {
  // Estimate gas based on trade complexity
  const baseGas = 150000; // Base DEX swap gas
  const complexityMultiplier = (fromToken === 'ETH' || toToken === 'ETH') ? 1.0 : 1.2;
  
  return BigNumber.from(Math.floor(baseGas * complexityMultiplier));
}

function calculateAllocationDeviation(portfolio: Portfolio): number {
  let totalDeviation = 0;
  
  for (const target of portfolio.allocation) {
    const currentPosition = portfolio.positions.find(p => p.token.symbol === target.token.symbol);
    const currentPercentage = currentPosition?.percentage || 0;
    const deviation = Math.abs(currentPercentage - target.targetPercentage);
    totalDeviation += deviation;
  }

  return totalDeviation / portfolio.allocation.length;
}

function calculateAllocationDeviationFromPositions(
  positions: PortfolioPosition[],
  targets: any[]
): number {
  let totalDeviation = 0;
  
  for (const target of targets) {
    const currentPosition = positions.find(p => p.token.symbol === target.token.symbol);
    const currentPercentage = currentPosition?.percentage || 0;
    const deviation = Math.abs(currentPercentage - target.targetPercentage);
    totalDeviation += deviation;
  }

  return totalDeviation / targets.length;
}
