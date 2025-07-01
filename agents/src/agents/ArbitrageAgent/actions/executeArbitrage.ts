import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { ArbitrageOpportunity } from '../../../shared/types/market';
import { ARBITRAGE_THRESHOLDS } from '../../../shared/constants/thresholds';
import { DexProvider } from '../providers/dexProvider';
import { ChainlinkProvider } from '../providers/chainlinkProvider';

export interface ExecutionConfig {
  maxSlippage: number;
  gasOptimization: boolean;
  mevProtection: boolean;
  maxRetries?: number;
  retryDelayMs?: number;
  emergencyStopLoss?: number;
  frontrunningProtection?: boolean;
}

export interface ExecutionResult {
  success: boolean;
  transactions: string[];
  totalGasUsed: BigNumber;
  actualProfit: BigNumber;
  actualSlippage: number;
  executionTime: number;
  ccipMessageId?: string;
  errors: string[];
  stages: ExecutionStage[];
}

export interface ExecutionStage {
  stage: string;
  status: 'pending' | 'executing' | 'completed' | 'failed';
  transactionHash?: string;
  gasUsed?: BigNumber;
  error?: string;
  timestamp: number;
}

export async function executeArbitrage(
  opportunity: ArbitrageOpportunity,
  dexProvider: DexProvider,
  chainlinkProvider: ChainlinkProvider,
  config: ExecutionConfig
): Promise<ExecutionResult> {
  const startTime = Date.now();
  const stages: ExecutionStage[] = [];
  const transactions: string[] = [];
  let totalGasUsed = BigNumber.from(0);
  let actualProfit = BigNumber.from(0);
  let actualSlippage = 0;
  const errors: string[] = [];
  const ccipMessageId: string | undefined = undefined;

  logger.info('Starting arbitrage execution', {
    opportunityId: opportunity.id,
    tokenPair: opportunity.tokenPair,
    sourceChain: opportunity.sourceExchange.chainId,
    targetChain: opportunity.targetExchange.chainId,
    config
  });

  try {
    // Stage 1: Pre-execution validation
    await addStage(stages, 'validation', 'executing');
    
    const validationResult = await validateExecution(opportunity, dexProvider, chainlinkProvider);
    if (!validationResult.valid) {
      throw new Error(`Validation failed: ${validationResult.reason}`);
    }
    
    await completeStage(stages, 'validation', 'completed');

    // Stage 2: Token approvals
    await addStage(stages, 'approvals', 'executing');
    
    const approvalTxs = await handleTokenApprovals(opportunity, dexProvider, config);
    transactions.push(...approvalTxs.map(tx => tx.hash));
    totalGasUsed = totalGasUsed.add(approvalTxs.reduce((sum, tx) => sum.add(tx.gasUsed), BigNumber.from(0)));
    
    await completeStage(stages, 'approvals', 'completed');

    // Stage 3: Execute arbitrage based on complexity
    if (opportunity.executionComplexity === 'simple') {
      const result = await executeSimpleArbitrage(opportunity, dexProvider, config, stages);
      transactions.push(...result.transactions);
      totalGasUsed = totalGasUsed.add(result.gasUsed);
      actualProfit = result.profit;
      actualSlippage = result.slippage;
    } else {
      const result = await executeCrossChainArbitrage(opportunity, dexProvider, chainlinkProvider, config, stages);
      transactions.push(...result.transactions);
      totalGasUsed = totalGasUsed.add(result.gasUsed);
      actualProfit = result.profit;
      actualSlippage = result.slippage;
    }

    // Stage 4: Post-execution validation
    await addStage(stages, 'post_validation', 'executing');
    
    const finalValidation = await validateExecutionResults(
      opportunity, 
      actualProfit, 
      actualSlippage, 
      config
    );
    
    if (!finalValidation.valid) {
      errors.push(`Post-execution validation failed: ${finalValidation.reason}`);
    }
    
    await completeStage(stages, 'post_validation', 'completed');

    const executionTime = Date.now() - startTime;

    logger.info('Arbitrage execution completed successfully', {
      opportunityId: opportunity.id,
      actualProfit: ethers.utils.formatEther(actualProfit),
      totalGasUsed: totalGasUsed.toString(),
      actualSlippage,
      executionTime,
      transactions: transactions.length
    });

    return {
      success: true,
      transactions,
      totalGasUsed,
      actualProfit,
      actualSlippage,
      executionTime,
      ccipMessageId,
      errors,
      stages
    };

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    errors.push(errorMessage);

    // Mark current stage as failed
    if (stages.length > 0) {
      const lastStage = stages[stages.length - 1];
      if (lastStage.status === 'executing') {
        lastStage.status = 'failed';
        lastStage.error = errorMessage;
      }
    }

    logger.error('Arbitrage execution failed', {
      opportunityId: opportunity.id,
      error: errorMessage,
      executionTime: Date.now() - startTime,
      completedStages: stages.filter(s => s.status === 'completed').length,
      totalStages: stages.length
    });

    return {
      success: false,
      transactions,
      totalGasUsed,
      actualProfit: BigNumber.from(0),
      actualSlippage: 0,
      executionTime: Date.now() - startTime,
      errors,
      stages
    };
  }
}

async function validateExecution(
  opportunity: ArbitrageOpportunity,
  dexProvider: DexProvider,
  chainlinkProvider: ChainlinkProvider
): Promise<{ valid: boolean; reason?: string }> {
  
  // Check if opportunity is still valid (not expired)
  if (Date.now() > opportunity.expiresAt) {
    return { valid: false, reason: 'Opportunity expired' };
  }

  // Validate current prices against Chainlink feeds
  try {
    const currentSourcePrice = await chainlinkProvider.getLatestPrice(
      opportunity.tokenPair,
      opportunity.sourceExchange.chainId
    );
    
    const currentTargetPrice = await chainlinkProvider.getLatestPrice(
      opportunity.tokenPair,
      opportunity.targetExchange.chainId
    );

    if (currentSourcePrice && currentTargetPrice) {
      const priceDiff = Math.abs(
        parseFloat(ethers.utils.formatEther(currentTargetPrice.answer)) -
        parseFloat(ethers.utils.formatEther(currentSourcePrice.answer))
      );
      
      const expectedDiff = parseFloat(ethers.utils.formatEther(opportunity.priceDifference));
      
      // Check if price difference is still within 10% of expected
      if (Math.abs(priceDiff - expectedDiff) / expectedDiff > 0.1) {
        return { valid: false, reason: 'Price discrepancy changed significantly' };
      }
    }
  } catch (error) {
    logger.warn('Could not validate prices against Chainlink', {
      opportunityId: opportunity.id,
      error: error instanceof Error ? error.message : String(error)
    });
  }

  // Check gas prices haven't spiked too much
  const currentGasPrice = await dexProvider.getGasPrice(opportunity.sourceExchange.chainId);
  const gasIncrease = currentGasPrice.sub(opportunity.estimatedGasCost).mul(100).div(opportunity.estimatedGasCost);
  
  if (gasIncrease.gt(50)) { // 50% increase
    return { valid: false, reason: 'Gas prices increased significantly' };
  }

  // Check liquidity is still available
  const sourceBook = await dexProvider.getOrderBookDepth(
    opportunity.tokenPair,
    opportunity.sourceExchange.chainId,
    opportunity.sourceExchange.name,
    5
  );

  const targetBook = await dexProvider.getOrderBookDepth(
    opportunity.tokenPair,
    opportunity.targetExchange.chainId,
    opportunity.targetExchange.name,
    5
  );

  const availableSourceLiquidity = sourceBook.bids.reduce(
    (sum, level) => sum.add(level.amount), 
    BigNumber.from(0)
  );

  const availableTargetLiquidity = targetBook.asks.reduce(
    (sum, level) => sum.add(level.amount), 
    BigNumber.from(0)
  );

  if (availableSourceLiquidity.lt(opportunity.maxTradeSize) || 
      availableTargetLiquidity.lt(opportunity.maxTradeSize)) {
    return { valid: false, reason: 'Insufficient liquidity available' };
  }

  return { valid: true };
}

async function handleTokenApprovals(
  opportunity: ArbitrageOpportunity,
  dexProvider: DexProvider,
  config: ExecutionConfig
): Promise<Array<{ hash: string; gasUsed: BigNumber }>> {
  const approvalTxs: Array<{ hash: string; gasUsed: BigNumber }> = [];

  // Get token addresses from the pair
  const tokens = parseTokenPair(opportunity.tokenPair);

  // Check and approve tokens for source exchange
  const sourceApprovals = await dexProvider.checkAndApproveTokens(
    tokens,
    opportunity.sourceExchange.chainId,
    opportunity.sourceExchange.address,
    opportunity.maxTradeSize
  );

  approvalTxs.push(...sourceApprovals);

  // Check and approve tokens for target exchange (if different chain)
  if (opportunity.sourceExchange.chainId !== opportunity.targetExchange.chainId) {
    const targetApprovals = await dexProvider.checkAndApproveTokens(
      tokens,
      opportunity.targetExchange.chainId,
      opportunity.targetExchange.address,
      opportunity.maxTradeSize
    );

    approvalTxs.push(...targetApprovals);
  }

  return approvalTxs;
}

async function executeSimpleArbitrage(
  opportunity: ArbitrageOpportunity,
  dexProvider: DexProvider,
  config: ExecutionConfig,
  stages: ExecutionStage[]
): Promise<{
  transactions: string[];
  gasUsed: BigNumber;
  profit: BigNumber;
  slippage: number;
}> {
  
  await addStage(stages, 'source_swap', 'executing');

  // Execute buy on source exchange (lower price)
  const buyResult = await dexProvider.executeSwap(
    opportunity.tokenPair,
    opportunity.sourceExchange.chainId,
    opportunity.sourceExchange.name,
    'buy',
    opportunity.maxTradeSize,
    config.maxSlippage,
    config.mevProtection
  );

  await completeStage(stages, 'source_swap', 'completed', buyResult.hash, buyResult.gasUsed);

  await addStage(stages, 'target_swap', 'executing');

  // Execute sell on target exchange (higher price)
  const sellResult = await dexProvider.executeSwap(
    opportunity.tokenPair,
    opportunity.targetExchange.chainId,
    opportunity.targetExchange.name,
    'sell',
    buyResult.outputAmount, // Use actual output from buy
    config.maxSlippage,
    config.mevProtection
  );

  await completeStage(stages, 'target_swap', 'completed', sellResult.hash, sellResult.gasUsed);

  // Calculate actual profit and slippage
  const totalGasUsed = buyResult.gasUsed.add(sellResult.gasUsed);
  const actualProfit = sellResult.outputAmount.sub(opportunity.maxTradeSize).sub(totalGasUsed);
  const actualSlippage = calculateActualSlippage(buyResult, sellResult, opportunity);

  return {
    transactions: [buyResult.hash, sellResult.hash],
    gasUsed: totalGasUsed,
    profit: actualProfit,
    slippage: actualSlippage
  };
}

async function executeCrossChainArbitrage(
  opportunity: ArbitrageOpportunity,
  dexProvider: DexProvider,
  chainlinkProvider: ChainlinkProvider,
  config: ExecutionConfig,
  stages: ExecutionStage[]
): Promise<{
  transactions: string[];
  gasUsed: BigNumber;
  profit: BigNumber;
  slippage: number;
}> {

  await addStage(stages, 'source_swap', 'executing');

  // Execute buy on source chain
  const buyResult = await dexProvider.executeSwap(
    opportunity.tokenPair,
    opportunity.sourceExchange.chainId,
    opportunity.sourceExchange.name,
    'buy',
    opportunity.maxTradeSize,
    config.maxSlippage,
    config.mevProtection
  );

  await completeStage(stages, 'source_swap', 'completed', buyResult.hash, buyResult.gasUsed);

  await addStage(stages, 'cross_chain_bridge', 'executing');

  // Bridge tokens to target chain using Chainlink CCIP
  const bridgeResult = await chainlinkProvider.sendCCIPMessage(
    opportunity.sourceExchange.chainId,
    opportunity.targetExchange.chainId,
    buyResult.outputAmount,
    {
      gasLimit: ARBITRAGE_THRESHOLDS.MAX_BRIDGE_TIME,
      mevProtection: config.mevProtection
    }
  );

  await completeStage(stages, 'cross_chain_bridge', 'completed', bridgeResult.hash, bridgeResult.gasUsed);

  // Wait for CCIP message to be delivered
  await addStage(stages, 'ccip_delivery', 'executing');
  
  const deliveryResult = await waitForCCIPDelivery(
    bridgeResult.messageId,
    opportunity.targetExchange.chainId,
    chainlinkProvider,
    ARBITRAGE_THRESHOLDS.MAX_BRIDGE_TIME * 1000 // Convert to milliseconds
  );

  await completeStage(stages, 'ccip_delivery', 'completed');

  await addStage(stages, 'target_swap', 'executing');

  // Execute sell on target chain
  const sellResult = await dexProvider.executeSwap(
    opportunity.tokenPair,
    opportunity.targetExchange.chainId,
    opportunity.targetExchange.name,
    'sell',
    deliveryResult.deliveredAmount,
    config.maxSlippage,
    config.mevProtection
  );

  await completeStage(stages, 'target_swap', 'completed', sellResult.hash, sellResult.gasUsed);

  // Calculate total costs and profit
  const totalGasUsed = buyResult.gasUsed
    .add(bridgeResult.gasUsed)
    .add(sellResult.gasUsed);

  const bridgeFees = bridgeResult.fees || BigNumber.from(0);
  const totalCosts = totalGasUsed.add(bridgeFees);
  
  const actualProfit = sellResult.outputAmount.sub(opportunity.maxTradeSize).sub(totalCosts);
  const actualSlippage = calculateCrossChainSlippage(buyResult, sellResult, bridgeResult, opportunity);

  return {
    transactions: [buyResult.hash, bridgeResult.hash, sellResult.hash],
    gasUsed: totalGasUsed,
    profit: actualProfit,
    slippage: actualSlippage
  };
}

async function waitForCCIPDelivery(
  messageId: string,
  targetChainId: number,
  chainlinkProvider: ChainlinkProvider,
  timeoutMs: number
): Promise<{ deliveredAmount: BigNumber; deliveryTime: number }> {
  
  const startTime = Date.now();
  const pollInterval = 5000; // 5 seconds

  while (Date.now() - startTime < timeoutMs) {
    try {
      const status = await chainlinkProvider.getCCIPMessageStatus(messageId, targetChainId);
      
      if (status.status === 'delivered') {
        return {
          deliveredAmount: status.deliveredAmount,
          deliveryTime: Date.now() - startTime
        };
      }
      
      if (status.status === 'failed') {
        throw new Error(`CCIP delivery failed: ${status.errorReason}`);
      }

      // Wait before next poll
      await new Promise(resolve => setTimeout(resolve, pollInterval));

    } catch (error) {
      logger.error('Error checking CCIP delivery status', {
        messageId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  throw new Error(`CCIP delivery timeout after ${timeoutMs}ms`);
}

async function validateExecutionResults(
  opportunity: ArbitrageOpportunity,
  actualProfit: BigNumber,
  actualSlippage: number,
  config: ExecutionConfig
): Promise<{ valid: boolean; reason?: string }> {

  // Check if profit is positive
  if (actualProfit.lte(0)) {
    return { valid: false, reason: 'Execution resulted in loss' };
  }

  // Check if slippage is within acceptable limits
  if (actualSlippage > config.maxSlippage) {
    return { valid: false, reason: `Slippage exceeded limit: ${actualSlippage}% > ${config.maxSlippage}%` };
  }

  // Check if profit meets minimum threshold
  const profitPercentage = parseFloat(ethers.utils.formatEther(actualProfit)) / 
                          parseFloat(ethers.utils.formatEther(opportunity.maxTradeSize)) * 100;

  if (profitPercentage < ARBITRAGE_THRESHOLDS.MIN_PROFIT_PERCENTAGE) {
    return { valid: false, reason: `Profit below minimum threshold: ${profitPercentage}%` };
  }

  return { valid: true };
}

function parseTokenPair(tokenPair: string): { base: string; quote: string } {
  const [base, quote] = tokenPair.split('/');
  return { base, quote };
}

function calculateActualSlippage(
  buyResult: any,
  sellResult: any,
  opportunity: ArbitrageOpportunity
): number {
  const expectedBuyPrice = opportunity.sourceExchange.price;
  const actualBuyPrice = buyResult.inputAmount.mul(ethers.utils.parseEther('1')).div(buyResult.outputAmount);
  
  const expectedSellPrice = opportunity.targetExchange.price;
  const actualSellPrice = sellResult.outputAmount.mul(ethers.utils.parseEther('1')).div(sellResult.inputAmount);

  const buySlippage = Math.abs(
    parseFloat(ethers.utils.formatEther(actualBuyPrice.sub(expectedBuyPrice))) /
    parseFloat(ethers.utils.formatEther(expectedBuyPrice))
  ) * 100;

  const sellSlippage = Math.abs(
    parseFloat(ethers.utils.formatEther(actualSellPrice.sub(expectedSellPrice))) /
    parseFloat(ethers.utils.formatEther(expectedSellPrice))
  ) * 100;

  return buySlippage + sellSlippage;
}

function calculateCrossChainSlippage(
  buyResult: any,
  sellResult: any,
  bridgeResult: any,
  opportunity: ArbitrageOpportunity
): number {
  const tradingSlippage = calculateActualSlippage(buyResult, sellResult, opportunity);
  
  // Add bridge slippage
  const bridgeSlippage = bridgeResult.fees ? 
    parseFloat(ethers.utils.formatEther(bridgeResult.fees)) / 
    parseFloat(ethers.utils.formatEther(buyResult.outputAmount)) * 100 : 0;

  return tradingSlippage + bridgeSlippage;
}

async function addStage(stages: ExecutionStage[], stage: string, status: 'pending' | 'executing' | 'completed' | 'failed'): Promise<void> {
  stages.push({
    stage,
    status,
    timestamp: Date.now()
  });
}

async function completeStage(
  stages: ExecutionStage[], 
  stage: string, 
  status: 'completed' | 'failed',
  transactionHash?: string,
  gasUsed?: BigNumber,
  error?: string
): Promise<void> {
  const stageIndex = stages.findIndex(s => s.stage === stage);
  if (stageIndex !== -1) {
    stages[stageIndex].status = status;
    stages[stageIndex].transactionHash = transactionHash;
    stages[stageIndex].gasUsed = gasUsed;
    stages[stageIndex].error = error;
  }
}
