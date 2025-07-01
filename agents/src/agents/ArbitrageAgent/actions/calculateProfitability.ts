import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { ArbitrageOpportunity } from '../../../shared/types/market';
import { getNetworkFees, getBlockTime } from '../../../shared/constants/networks';
import { DexProvider } from '../providers/dexProvider';
import { ChainlinkProvider } from '../providers/chainlinkProvider';

export interface ProfitabilityConfig {
  maxSlippage: number;
  gasOptimization: boolean;
  includeMEVProtection?: boolean;
  frontrunningBuffer?: number;
  priceImpactTolerance?: number;
}

export interface ProfitabilityAnalysis {
  grossProfit: BigNumber;
  grossProfitUsd: BigNumber;
  grossProfitPercentage: number;
  totalCosts: {
    gasCosts: BigNumber;
    tradingFees: BigNumber;
    bridgeFees: BigNumber;
    slippageCosts: BigNumber;
    mevProtectionCosts: BigNumber;
    totalCost: BigNumber;
  };
  netProfit: BigNumber;
  netProfitUsd: BigNumber;
  netProfitPercentage: number;
  breakEvenSize: BigNumber;
  optimalTradeSize: BigNumber;
  priceImpact: {
    source: number;
    target: number;
    total: number;
  };
  slippageAnalysis: {
    estimated: number;
    worstCase: number;
    protected: boolean;
  };
  riskAdjustedReturn: number;
  executionProbability: number;
  timeToExecution: number;
  liquidityAnalysis: {
    sourceDepth: BigNumber;
    targetDepth: BigNumber;
    combined: BigNumber;
    utilization: number;
  };
}

export async function calculateProfitability(
  opportunity: ArbitrageOpportunity,
  dexProvider: DexProvider,
  chainlinkProvider: ChainlinkProvider,
  config: ProfitabilityConfig
): Promise<ProfitabilityAnalysis> {
  const startTime = Date.now();
  
  logger.info('Calculating arbitrage profitability', {
    opportunityId: opportunity.id,
    tokenPair: opportunity.tokenPair,
    priceDifference: opportunity.priceDifferencePercentage,
    config
  });

  try {
    // Get detailed liquidity data
    const liquidityData = await analyzeLiquidityDepth(
      opportunity,
      dexProvider
    );

    // Calculate optimal trade size
    const optimalTradeSize = calculateOptimalTradeSize(
      opportunity,
      liquidityData,
      config
    );

    // Analyze price impact for the optimal trade size
    const priceImpact = await calculatePriceImpact(
      opportunity,
      optimalTradeSize,
      dexProvider
    );

    // Calculate slippage analysis
    const slippageAnalysis = calculateSlippageAnalysis(
      priceImpact,
      config
    );

    // Calculate detailed gas costs
    const gasCosts = await calculateDetailedGasCosts(
      opportunity,
      optimalTradeSize,
      dexProvider,
      config
    );

    // Calculate trading fees
    const tradingFees = calculateTradingFees(
      optimalTradeSize,
      opportunity.sourceExchange.fee,
      opportunity.targetExchange.fee
    );

    // Calculate bridge fees (if cross-chain)
    const bridgeFees = await calculateBridgeFees(
      opportunity,
      optimalTradeSize,
      dexProvider
    );

    // Calculate slippage costs
    const slippageCosts = calculateSlippageCosts(
      optimalTradeSize,
      slippageAnalysis.estimated
    );

    // Calculate MEV protection costs
    const mevProtectionCosts = config.includeMEVProtection 
      ? calculateMEVProtectionCosts(optimalTradeSize, gasCosts)
      : BigNumber.from(0);

    // Calculate gross profit
    const grossProfit = calculateGrossProfit(
      opportunity,
      optimalTradeSize,
      priceImpact
    );

    const grossProfitUsd = await convertToUSD(
      grossProfit,
      opportunity.tokenPair,
      chainlinkProvider
    );

    // Sum total costs
    const totalCost = gasCosts
      .add(tradingFees)
      .add(bridgeFees)
      .add(slippageCosts)
      .add(mevProtectionCosts);

    // Calculate net profit
    const netProfit = grossProfit.sub(totalCost);
    const netProfitUsd = grossProfitUsd.sub(
      await convertToUSD(totalCost, 'ETH/USD', chainlinkProvider)
    );

    // Calculate percentages
    const grossProfitPercentage = parseFloat(ethers.utils.formatEther(grossProfit)) / 
                                 parseFloat(ethers.utils.formatEther(optimalTradeSize)) * 100;
    
    const netProfitPercentage = parseFloat(ethers.utils.formatEther(netProfit)) / 
                               parseFloat(ethers.utils.formatEther(optimalTradeSize)) * 100;

    // Calculate break-even size
    const breakEvenSize = calculateBreakEvenSize(
      opportunity,
      totalCost
    );

    // Calculate risk-adjusted return
    const riskAdjustedReturn = calculateRiskAdjustedReturn(
      netProfitPercentage,
      opportunity.riskScore,
      opportunity.confidence
    );

    // Estimate execution probability
    const executionProbability = estimateExecutionProbability(
      opportunity,
      liquidityData,
      slippageAnalysis,
      config
    );

    // Estimate time to execution
    const timeToExecution = estimateTimeToExecution(
      opportunity,
      optimalTradeSize,
      dexProvider
    );

    const analysis: ProfitabilityAnalysis = {
      grossProfit,
      grossProfitUsd,
      grossProfitPercentage,
      totalCosts: {
        gasCosts,
        tradingFees,
        bridgeFees,
        slippageCosts,
        mevProtectionCosts,
        totalCost
      },
      netProfit,
      netProfitUsd,
      netProfitPercentage,
      breakEvenSize,
      optimalTradeSize,
      priceImpact,
      slippageAnalysis,
      riskAdjustedReturn,
      executionProbability,
      timeToExecution,
      liquidityAnalysis: {
        sourceDepth: liquidityData.sourceDepth,
        targetDepth: liquidityData.targetDepth,
        combined: liquidityData.sourceDepth.add(liquidityData.targetDepth),
        utilization: parseFloat(ethers.utils.formatEther(optimalTradeSize)) /
                    parseFloat(ethers.utils.formatEther(liquidityData.sourceDepth.add(liquidityData.targetDepth))) * 100
      }
    };

    logger.info('Profitability calculation completed', {
      opportunityId: opportunity.id,
      optimalTradeSize: ethers.utils.formatEther(optimalTradeSize),
      netProfit: ethers.utils.formatEther(netProfit),
      netProfitPercentage,
      executionProbability,
      duration: Date.now() - startTime
    });

    return analysis;

  } catch (error) {
    logger.error('Failed to calculate profitability', {
      opportunityId: opportunity.id,
      error: error instanceof Error ? error.message : String(error),
      duration: Date.now() - startTime
    });

    throw error;
  }
}

async function analyzeLiquidityDepth(
  opportunity: ArbitrageOpportunity,
  dexProvider: DexProvider
): Promise<{
  sourceDepth: BigNumber;
  targetDepth: BigNumber;
  sourceLevels: Array<{ price: BigNumber; amount: BigNumber }>;
  targetLevels: Array<{ price: BigNumber; amount: BigNumber }>;
}> {
  const [sourceOrderBook, targetOrderBook] = await Promise.all([
    dexProvider.getOrderBookDepth(
      opportunity.tokenPair,
      opportunity.sourceExchange.chainId,
      opportunity.sourceExchange.name,
      10 // Get top 10 levels
    ),
    dexProvider.getOrderBookDepth(
      opportunity.tokenPair,
      opportunity.targetExchange.chainId,
      opportunity.targetExchange.name,
      10
    )
  ]);

  const sourceDepth = sourceOrderBook.bids.reduce(
    (sum, level) => sum.add(level.amount),
    BigNumber.from(0)
  );

  const targetDepth = targetOrderBook.asks.reduce(
    (sum, level) => sum.add(level.amount),
    BigNumber.from(0)
  );

  return {
    sourceDepth,
    targetDepth,
    sourceLevels: sourceOrderBook.bids,
    targetLevels: targetOrderBook.asks
  };
}

function calculateOptimalTradeSize(
  opportunity: ArbitrageOpportunity,
  liquidityData: any,
  config: ProfitabilityConfig
): BigNumber {
  // Start with the smaller of the two liquidity pools
  const maxLiquiditySize = liquidityData.sourceDepth.lt(liquidityData.targetDepth) 
    ? liquidityData.sourceDepth 
    : liquidityData.targetDepth;

  // Use a percentage of available liquidity to minimize price impact
  const liquidityUtilization = 0.1; // Use 10% of available liquidity
  const liquidityBasedSize = maxLiquiditySize.mul(Math.floor(liquidityUtilization * 100)).div(100);

  // Don't exceed the maximum trade size from the opportunity
  const maxTradeSize = opportunity.maxTradeSize;

  // Don't exceed configured maximum position size
  const configuredMax = ethers.utils.parseEther('10000'); // Default $10k max

  // Return the minimum of all constraints
  let optimalSize = liquidityBasedSize;
  if (optimalSize.gt(maxTradeSize)) optimalSize = maxTradeSize;
  if (optimalSize.gt(configuredMax)) optimalSize = configuredMax;

  // Ensure minimum viable trade size
  const minimumTradeSize = ethers.utils.parseEther('100'); // $100 minimum
  if (optimalSize.lt(minimumTradeSize)) {
    optimalSize = minimumTradeSize;
  }

  return optimalSize;
}

async function calculatePriceImpact(
  opportunity: ArbitrageOpportunity,
  tradeSize: BigNumber,
  dexProvider: DexProvider
): Promise<{
  source: number;
  target: number;
  total: number;
}> {
  const [sourcePriceImpact, targetPriceImpact] = await Promise.all([
    dexProvider.calculatePriceImpact(
      opportunity.tokenPair,
      tradeSize,
      opportunity.sourceExchange.chainId,
      opportunity.sourceExchange.name
    ),
    dexProvider.calculatePriceImpact(
      opportunity.tokenPair,
      tradeSize,
      opportunity.targetExchange.chainId,
      opportunity.targetExchange.name
    )
  ]);

  return {
    source: sourcePriceImpact,
    target: targetPriceImpact,
    total: sourcePriceImpact + targetPriceImpact
  };
}

function calculateSlippageAnalysis(
  priceImpact: { source: number; target: number; total: number },
  config: ProfitabilityConfig
): {
  estimated: number;
  worstCase: number;
  protected: boolean;
} {
  // Estimated slippage includes price impact plus market movement
  const marketMovementBuffer = 0.1; // 0.1% buffer for market movement
  const estimated = priceImpact.total + marketMovementBuffer;

  // Worst case includes additional buffer for volatility
  const volatilityBuffer = config.frontrunningBuffer || 0.2;
  const worstCase = estimated + volatilityBuffer;

  // Check if slippage is within acceptable limits
  const protected = worstCase <= config.maxSlippage;

  return {
    estimated,
    worstCase,
    protected
  };
}

async function calculateDetailedGasCosts(
  opportunity: ArbitrageOpportunity,
  tradeSize: BigNumber,
  dexProvider: DexProvider,
  config: ProfitabilityConfig
): Promise<BigNumber> {
  const sourceNetworkFees = getNetworkFees(opportunity.sourceExchange.chainId);
  const targetNetworkFees = getNetworkFees(opportunity.targetExchange.chainId);

  // Base gas estimates for different operations
  let totalGasUnits = 0;

  // Source chain swap
  totalGasUnits += await estimateSwapGas(
    opportunity.sourceExchange.name,
    opportunity.tokenPair,
    tradeSize
  );

  // Target chain swap
  totalGasUnits += await estimateSwapGas(
    opportunity.targetExchange.name,
    opportunity.tokenPair,
    tradeSize
  );

  // Cross-chain bridge (if needed)
  if (opportunity.sourceExchange.chainId !== opportunity.targetExchange.chainId) {
    totalGasUnits += 500000; // Estimated bridge gas
  }

  // MEV protection (if enabled)
  if (config.includeMEVProtection) {
    totalGasUnits += 100000; // Additional gas for MEV protection
  }

  // Gas optimization adjustments
  if (config.gasOptimization) {
    totalGasUnits = Math.floor(totalGasUnits * 0.9); // 10% reduction with optimization
  }

  // Calculate gas costs for source chain (where most operations occur)
  const gasPrice = await dexProvider.getGasPrice(opportunity.sourceExchange.chainId);
  return gasPrice.mul(totalGasUnits);
}

async function estimateSwapGas(
  exchangeName: string,
  tokenPair: string,
  tradeSize: BigNumber
): Promise<number> {
  // Gas estimates vary by DEX
  const gasEstimates: Record<string, number> = {
    'uniswap_v3': 150000,
    'uniswap_v2': 120000,
    'sushiswap': 130000,
    'curve': 200000,
    'balancer': 180000,
    'pancakeswap': 110000
  };

  const baseGas = gasEstimates[exchangeName] || 150000;

  // Adjust for complex routes or large trades
  const sizeMultiplier = parseFloat(ethers.utils.formatEther(tradeSize)) > 1000 ? 1.2 : 1.0;
  
  return Math.floor(baseGas * sizeMultiplier);
}

function calculateTradingFees(
  tradeSize: BigNumber,
  sourceFee: number,
  targetFee: number
): BigNumber {
  const totalFeePercentage = (sourceFee + targetFee) / 100;
  return tradeSize.mul(Math.floor(totalFeePercentage * 10000)).div(10000);
}

async function calculateBridgeFees(
  opportunity: ArbitrageOpportunity,
  tradeSize: BigNumber,
  dexProvider: DexProvider
): Promise<BigNumber> {
  if (opportunity.sourceExchange.chainId === opportunity.targetExchange.chainId) {
    return BigNumber.from(0);
  }

  // Get actual bridge fees from the DEX provider
  try {
    return await dexProvider.getBridgeFee(
      opportunity.sourceExchange.chainId,
      opportunity.targetExchange.chainId,
      tradeSize
    );
  } catch (error) {
    // Fallback to estimated fees
    const baseFeePercentage = 0.05; // 0.05% base bridge fee
    return tradeSize.mul(Math.floor(baseFeePercentage * 10000)).div(10000);
  }
}

function calculateSlippageCosts(
  tradeSize: BigNumber,
  slippagePercentage: number
): BigNumber {
  return tradeSize.mul(Math.floor(slippagePercentage * 10000)).div(10000);
}

function calculateMEVProtectionCosts(
  tradeSize: BigNumber,
  gasCosts: BigNumber
): BigNumber {
  // MEV protection typically costs extra gas (priority fees)
  return gasCosts.mul(20).div(100); // 20% of gas costs
}

function calculateGrossProfit(
  opportunity: ArbitrageOpportunity,
  tradeSize: BigNumber,
  priceImpact: { source: number; target: number; total: number }
): BigNumber {
  // Adjust prices for price impact
  const adjustedSourcePrice = opportunity.sourceExchange.price.mul(
    Math.floor((1 + priceImpact.source / 100) * 10000)
  ).div(10000);

  const adjustedTargetPrice = opportunity.targetExchange.price.mul(
    Math.floor((1 - priceImpact.target / 100) * 10000)
  ).div(10000);

  const priceDifference = adjustedTargetPrice.sub(adjustedSourcePrice);
  return priceDifference.mul(tradeSize).div(adjustedSourcePrice);
}

async function convertToUSD(
  amount: BigNumber,
  tokenPair: string,
  chainlinkProvider: ChainlinkProvider
): Promise<BigNumber> {
  try {
    // Get USD price for the token
    const priceData = await chainlinkProvider.getLatestPrice(tokenPair, 1); // Use Ethereum mainnet
    
    if (priceData) {
      return amount.mul(priceData.answer).div(ethers.utils.parseUnits('1', priceData.decimals));
    }
    
    // Fallback: assume 1:1 ratio if price unavailable
    return amount;

  } catch (error) {
    logger.warn('Failed to convert to USD, using fallback', {
      tokenPair,
      amount: ethers.utils.formatEther(amount),
      error: error instanceof Error ? error.message : String(error)
    });
    
    return amount;
  }
}

function calculateBreakEvenSize(
  opportunity: ArbitrageOpportunity,
  totalCosts: BigNumber
): BigNumber {
  const priceDifferenceRatio = opportunity.priceDifference.mul(ethers.utils.parseEther('1')).div(opportunity.sourceExchange.price);
  return totalCosts.mul(ethers.utils.parseEther('1')).div(priceDifferenceRatio);
}

function calculateRiskAdjustedReturn(
  netProfitPercentage: number,
  riskScore: number,
  confidence: number
): number {
  // Adjust return based on risk and confidence
  const riskAdjustment = (100 - riskScore) / 100; // Higher risk = lower adjustment
  const confidenceAdjustment = confidence; // Direct confidence multiplier
  
  return netProfitPercentage * riskAdjustment * confidenceAdjustment;
}

function estimateExecutionProbability(
  opportunity: ArbitrageOpportunity,
  liquidityData: any,
  slippageAnalysis: any,
  config: ProfitabilityConfig
): number {
  let probability = 0.8; // Base 80% probability

  // Adjust for liquidity
  const liquidityRatio = parseFloat(ethers.utils.formatEther(liquidityData.sourceDepth.add(liquidityData.targetDepth))) / 1000000; // $1M reference
  probability *= Math.min(1.0, liquidityRatio);

  // Adjust for slippage protection
  if (slippageAnalysis.protected) {
    probability += 0.1;
  } else {
    probability -= 0.2;
  }

  // Adjust for confidence
  probability *= opportunity.confidence;

  // Adjust for execution complexity
  if (opportunity.executionComplexity === 'complex') {
    probability *= 0.8;
  }

  return Math.min(1.0, Math.max(0.0, probability));
}

function estimateTimeToExecution(
  opportunity: ArbitrageOpportunity,
  tradeSize: BigNumber,
  dexProvider: DexProvider
): number {
  let baseTime = 30; // 30 seconds base execution time

  // Add time for cross-chain operations
  if (opportunity.sourceExchange.chainId !== opportunity.targetExchange.chainId) {
    baseTime += 300; // 5 minutes for bridge
  }

  // Add time based on trade size (larger trades take longer)
  const sizeMultiplier = Math.log(parseFloat(ethers.utils.formatEther(tradeSize)) + 1) / 10;
  baseTime += baseTime * sizeMultiplier;

  // Add time based on network congestion
  const blockTime = getBlockTime(opportunity.sourceExchange.chainId);
  const congestionMultiplier = blockTime > 10 ? 1.5 : 1.0;
  
  return Math.floor(baseTime * congestionMultiplier);
}
