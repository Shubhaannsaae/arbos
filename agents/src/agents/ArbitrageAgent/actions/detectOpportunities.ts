import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { AgentContext } from '../../../shared/types/agent';
import { ArbitrageOpportunity, TradingPair } from '../../../shared/types/market';
import { ARBITRAGE_THRESHOLDS } from '../../../shared/constants/thresholds';
import { getNetworkInfo } from '../../../shared/constants/networks';
import { DexProvider } from '../providers/dexProvider';
import { ChainlinkProvider } from '../providers/chainlinkProvider';

export interface OpportunityDetectionConfig {
  supportedDexes: string[];
  supportedChains: number[];
  minProfitThreshold: number;
  maxPositionSize: BigNumber;
  tokenPairs?: string[];
  maxOpportunities?: number;
}

export interface OpportunityMetrics {
  totalScanned: number;
  opportunitiesFound: number;
  avgPriceDifference: number;
  avgConfidence: number;
  scanDuration: number;
}

export async function detectOpportunities(
  dexProvider: DexProvider,
  chainlinkProvider: ChainlinkProvider,
  context: AgentContext,
  config: OpportunityDetectionConfig
): Promise<ArbitrageOpportunity[]> {
  const startTime = Date.now();
  const opportunities: ArbitrageOpportunity[] = [];
  
  logger.info('Starting arbitrage opportunity detection', {
    agentId: context.agentId,
    supportedChains: config.supportedChains,
    supportedDexes: config.supportedDexes,
    minProfitThreshold: config.minProfitThreshold
  });

  try {
    // Get default token pairs if not specified
    const tokenPairs = config.tokenPairs || await getDefaultTokenPairs(config.supportedChains);
    
    // Scan each token pair across all supported chains and DEXes
    for (const tokenPair of tokenPairs) {
      const pairOpportunities = await scanTokenPairOpportunities(
        tokenPair,
        dexProvider,
        chainlinkProvider,
        context,
        config
      );
      
      opportunities.push(...pairOpportunities);
      
      // Limit total opportunities to prevent overwhelming the system
      if (opportunities.length >= (config.maxOpportunities || 50)) {
        break;
      }
    }

    // Sort opportunities by potential profit (descending)
    opportunities.sort((a, b) => 
      b.potentialProfitPercentage - a.potentialProfitPercentage
    );

    const metrics: OpportunityMetrics = {
      totalScanned: tokenPairs.length * config.supportedChains.length * config.supportedDexes.length,
      opportunitiesFound: opportunities.length,
      avgPriceDifference: opportunities.length > 0 
        ? opportunities.reduce((sum, opp) => sum + opp.priceDifferencePercentage, 0) / opportunities.length
        : 0,
      avgConfidence: opportunities.length > 0
        ? opportunities.reduce((sum, opp) => sum + opp.confidence, 0) / opportunities.length
        : 0,
      scanDuration: Date.now() - startTime
    };

    logger.info('Arbitrage opportunity detection completed', {
      agentId: context.agentId,
      metrics,
      topOpportunities: opportunities.slice(0, 5).map(opp => ({
        tokenPair: opp.tokenPair,
        priceDifference: opp.priceDifferencePercentage,
        potentialProfit: opp.potentialProfitPercentage,
        confidence: opp.confidence
      }))
    });

    return opportunities;

  } catch (error) {
    logger.error('Failed to detect arbitrage opportunities', {
      agentId: context.agentId,
      error: error instanceof Error ? error.message : String(error),
      duration: Date.now() - startTime
    });
    
    return [];
  }
}

async function scanTokenPairOpportunities(
  tokenPair: string,
  dexProvider: DexProvider,
  chainlinkProvider: ChainlinkProvider,
  context: AgentContext,
  config: OpportunityDetectionConfig
): Promise<ArbitrageOpportunity[]> {
  const opportunities: ArbitrageOpportunity[] = [];
  
  try {
    // Get trading pairs data from all DEXes on all chains
    const tradingPairs = await getTradingPairsData(
      tokenPair,
      config.supportedChains,
      config.supportedDexes,
      dexProvider
    );

    if (tradingPairs.length < 2) {
      return opportunities; // Need at least 2 pairs for arbitrage
    }

    // Get Chainlink reference prices for validation
    const chainlinkPrices = await getChainlinkReferencePrices(
      tokenPair,
      config.supportedChains,
      chainlinkProvider
    );

    // Compare all pair combinations for arbitrage opportunities
    for (let i = 0; i < tradingPairs.length; i++) {
      for (let j = i + 1; j < tradingPairs.length; j++) {
        const pair1 = tradingPairs[i];
        const pair2 = tradingPairs[j];

        // Skip if same exchange and chain
        if (pair1.exchange === pair2.exchange && pair1.chainId === pair2.chainId) {
          continue;
        }

        const opportunity = await evaluatePairArbitrage(
          pair1,
          pair2,
          chainlinkPrices,
          context,
          config
        );

        if (opportunity && 
            opportunity.potentialProfitPercentage >= config.minProfitThreshold &&
            opportunity.confidence >= ARBITRAGE_THRESHOLDS.MIN_PRICE_CONFIDENCE) {
          opportunities.push(opportunity);
        }
      }
    }

    return opportunities;

  } catch (error) {
    logger.error('Failed to scan token pair opportunities', {
      tokenPair,
      error: error instanceof Error ? error.message : String(error)
    });
    
    return [];
  }
}

async function getTradingPairsData(
  tokenPair: string,
  chains: number[],
  dexes: string[],
  dexProvider: DexProvider
): Promise<Array<{
  tokenPair: string;
  exchange: string;
  chainId: number;
  price: BigNumber;
  liquidity: BigNumber;
  volume24h: BigNumber;
  fee: number;
  contractAddress: string;
}>> {
  const tradingPairs: any[] = [];

  for (const chainId of chains) {
    for (const dexName of dexes) {
      try {
        const pairData = await dexProvider.getTradingPairData(tokenPair, chainId, dexName);
        
        if (pairData && pairData.liquidity.gt(ARBITRAGE_THRESHOLDS.MIN_BRIDGE_AMOUNT)) {
          tradingPairs.push({
            tokenPair,
            exchange: dexName,
            chainId,
            price: pairData.price,
            liquidity: pairData.liquidity,
            volume24h: pairData.volume24h,
            fee: pairData.fee,
            contractAddress: pairData.contractAddress
          });
        }

      } catch (error) {
        logger.debug('Failed to get trading pair data', {
          tokenPair,
          chainId,
          dexName,
          error: error instanceof Error ? error.message : String(error)
        });
        // Continue with other pairs
      }
    }
  }

  return tradingPairs;
}

async function getChainlinkReferencePrices(
  tokenPair: string,
  chains: number[],
  chainlinkProvider: ChainlinkProvider
): Promise<Record<number, { price: BigNumber; confidence: number; timestamp: number }>> {
  const prices: Record<number, any> = {};

  for (const chainId of chains) {
    try {
      const priceData = await chainlinkProvider.getLatestPrice(tokenPair, chainId);
      
      if (priceData) {
        prices[chainId] = {
          price: priceData.answer,
          confidence: priceData.confidence || 1.0,
          timestamp: priceData.updatedAt
        };
      }

    } catch (error) {
      logger.debug('Failed to get Chainlink price', {
        tokenPair,
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });
      // Continue without Chainlink price for this chain
    }
  }

  return prices;
}

async function evaluatePairArbitrage(
  pair1: any,
  pair2: any,
  chainlinkPrices: Record<number, any>,
  context: AgentContext,
  config: OpportunityDetectionConfig
): Promise<ArbitrageOpportunity | null> {
  try {
    // Determine which pair has lower/higher price
    const price1 = parseFloat(ethers.utils.formatEther(pair1.price));
    const price2 = parseFloat(ethers.utils.formatEther(pair2.price));
    
    if (Math.abs(price1 - price2) / Math.min(price1, price2) < config.minProfitThreshold / 100) {
      return null; // Price difference too small
    }

    const [lowerPricePair, higherPricePair] = price1 < price2 ? [pair1, pair2] : [pair2, pair1];
    const priceDifference = higherPricePair.price.sub(lowerPricePair.price);
    const priceDifferencePercentage = (parseFloat(ethers.utils.formatEther(priceDifference)) / 
                                     parseFloat(ethers.utils.formatEther(lowerPricePair.price))) * 100;

    // Calculate maximum trade size based on liquidity
    const maxTradeSize = BigNumber.from(
      Math.min(
        parseFloat(ethers.utils.formatEther(lowerPricePair.liquidity)) * 0.1, // 10% of liquidity
        parseFloat(ethers.utils.formatEther(higherPricePair.liquidity)) * 0.1,
        parseFloat(ethers.utils.formatEther(config.maxPositionSize))
      ).toString()
    );

    // Estimate gas costs
    const estimatedGasCost = await estimateArbitrageGasCost(
      lowerPricePair,
      higherPricePair,
      context.gasPrice
    );

    // Calculate gross profit
    const grossProfit = priceDifference.mul(maxTradeSize).div(lowerPricePair.price);
    
    // Estimate total costs (gas + fees)
    const tradingFees = calculateTradingFees(maxTradeSize, lowerPricePair.fee, higherPricePair.fee);
    const bridgeFees = lowerPricePair.chainId !== higherPricePair.chainId 
      ? calculateBridgeFees(maxTradeSize, lowerPricePair.chainId, higherPricePair.chainId)
      : BigNumber.from(0);
    
    const totalCosts = estimatedGasCost.add(tradingFees).add(bridgeFees);
    const netProfit = grossProfit.sub(totalCosts);

    // Calculate confidence based on multiple factors
    const confidence = calculateOpportunityConfidence(
      lowerPricePair,
      higherPricePair,
      chainlinkPrices,
      priceDifferencePercentage
    );

    // Determine execution complexity
    const executionComplexity = lowerPricePair.chainId !== higherPricePair.chainId ? 'complex' : 'simple';

    const opportunity: ArbitrageOpportunity = {
      id: `arb_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      tokenPair: pair1.tokenPair,
      sourceExchange: {
        name: lowerPricePair.exchange,
        address: lowerPricePair.contractAddress,
        chainId: lowerPricePair.chainId,
        price: lowerPricePair.price,
        liquidity: lowerPricePair.liquidity,
        fee: lowerPricePair.fee
      },
      targetExchange: {
        name: higherPricePair.exchange,
        address: higherPricePair.contractAddress,
        chainId: higherPricePair.chainId,
        price: higherPricePair.price,
        liquidity: higherPricePair.liquidity,
        fee: higherPricePair.fee
      },
      priceDifference,
      priceDifferencePercentage,
      potentialProfit: grossProfit,
      potentialProfitPercentage: parseFloat(ethers.utils.formatEther(grossProfit)) / 
                                parseFloat(ethers.utils.formatEther(maxTradeSize)) * 100,
      maxTradeSize,
      estimatedGasCost,
      totalCosts,
      netProfit,
      confidence,
      riskScore: calculateRiskScore(lowerPricePair, higherPricePair, confidence),
      executionComplexity,
      requiredCapital: maxTradeSize,
      estimatedExecutionTime: estimateExecutionTime(lowerPricePair.chainId, higherPricePair.chainId),
      detectedAt: Date.now(),
      expiresAt: Date.now() + (ARBITRAGE_THRESHOLDS.OPPORTUNITY_EXPIRY * 1000),
      chainlinkData: {
        sourcePrice: chainlinkPrices[lowerPricePair.chainId]?.price || BigNumber.from(0),
        targetPrice: chainlinkPrices[higherPricePair.chainId]?.price || BigNumber.from(0),
        priceAge: Math.max(
          chainlinkPrices[lowerPricePair.chainId]?.timestamp ? 
            Date.now() / 1000 - chainlinkPrices[lowerPricePair.chainId].timestamp : 0,
          chainlinkPrices[higherPricePair.chainId]?.timestamp ? 
            Date.now() / 1000 - chainlinkPrices[higherPricePair.chainId].timestamp : 0
        ),
        feedReliability: Math.min(
          chainlinkPrices[lowerPricePair.chainId]?.confidence || 0,
          chainlinkPrices[higherPricePair.chainId]?.confidence || 0
        )
      }
    };

    return opportunity;

  } catch (error) {
    logger.error('Failed to evaluate pair arbitrage', {
      pair1: pair1.exchange,
      pair2: pair2.exchange,
      error: error instanceof Error ? error.message : String(error)
    });
    
    return null;
  }
}

async function estimateArbitrageGasCost(
  sourcePair: any,
  targetPair: any,
  gasPrice: BigNumber
): Promise<BigNumber> {
  // Base gas estimates for different operations
  const baseSwapGas = 150000; // Gas for a basic swap
  const bridgeGas = 300000;   // Additional gas for cross-chain operations
  const complexRouteGas = 50000; // Additional gas for complex routes

  let totalGas = baseSwapGas * 2; // Two swaps minimum

  // Add bridge gas if cross-chain
  if (sourcePair.chainId !== targetPair.chainId) {
    totalGas += bridgeGas;
  }

  // Add complexity gas based on DEX type
  if (sourcePair.exchange === 'curve' || targetPair.exchange === 'curve') {
    totalGas += complexRouteGas;
  }

  return gasPrice.mul(totalGas);
}

function calculateTradingFees(
  tradeSize: BigNumber,
  sourceFee: number,
  targetFee: number
): BigNumber {
  const totalFeePercentage = (sourceFee + targetFee) / 100;
  return tradeSize.mul(Math.floor(totalFeePercentage * 10000)).div(10000);
}

function calculateBridgeFees(
  amount: BigNumber,
  sourceChain: number,
  targetChain: number
): BigNumber {
  // Estimate bridge fees based on chain combination
  // These are rough estimates - real implementation would query actual bridge costs
  const baseBridgeFee = 0.01; // 1% base fee
  
  // Higher fees for certain chain combinations
  let feeMultiplier = 1;
  if (sourceChain === 1 || targetChain === 1) { // Ethereum involved
    feeMultiplier = 1.5;
  }
  
  const feePercentage = baseBridgeFee * feeMultiplier;
  return amount.mul(Math.floor(feePercentage * 10000)).div(10000);
}

function calculateOpportunityConfidence(
  sourcePair: any,
  targetPair: any,
  chainlinkPrices: Record<number, any>,
  priceDifferencePercentage: number
): number {
  let confidence = 0.5; // Base confidence

  // Increase confidence based on liquidity
  const minLiquidity = Math.min(
    parseFloat(ethers.utils.formatEther(sourcePair.liquidity)),
    parseFloat(ethers.utils.formatEther(targetPair.liquidity))
  );
  
  if (minLiquidity > 1000000) confidence += 0.2; // $1M+ liquidity
  else if (minLiquidity > 100000) confidence += 0.1; // $100k+ liquidity

  // Increase confidence based on volume
  const avgVolume = (
    parseFloat(ethers.utils.formatEther(sourcePair.volume24h)) +
    parseFloat(ethers.utils.formatEther(targetPair.volume24h))
  ) / 2;
  
  if (avgVolume > 10000000) confidence += 0.15; // $10M+ volume
  else if (avgVolume > 1000000) confidence += 0.1; // $1M+ volume

  // Increase confidence if Chainlink prices support the arbitrage
  const sourceChainlinkPrice = chainlinkPrices[sourcePair.chainId];
  const targetChainlinkPrice = chainlinkPrices[targetPair.chainId];
  
  if (sourceChainlinkPrice && targetChainlinkPrice) {
    const chainlinkPriceDiff = Math.abs(
      parseFloat(ethers.utils.formatEther(sourceChainlinkPrice.price)) -
      parseFloat(ethers.utils.formatEther(targetChainlinkPrice.price))
    );
    
    const dexPriceDiff = Math.abs(
      parseFloat(ethers.utils.formatEther(sourcePair.price)) -
      parseFloat(ethers.utils.formatEther(targetPair.price))
    );
    
    // If DEX prices align with Chainlink price difference, increase confidence
    if (Math.abs(chainlinkPriceDiff - dexPriceDiff) / Math.max(chainlinkPriceDiff, dexPriceDiff) < 0.1) {
      confidence += 0.2;
    }
  }

  // Decrease confidence for very large price differences (potential error)
  if (priceDifferencePercentage > 10) {
    confidence *= 0.5; // 50% confidence penalty for >10% difference
  }

  return Math.min(1.0, Math.max(0.0, confidence));
}

function calculateRiskScore(
  sourcePair: any,
  targetPair: any,
  confidence: number
): number {
  let riskScore = 30; // Base risk

  // Increase risk for low liquidity
  const minLiquidity = Math.min(
    parseFloat(ethers.utils.formatEther(sourcePair.liquidity)),
    parseFloat(ethers.utils.formatEther(targetPair.liquidity))
  );
  
  if (minLiquidity < 100000) riskScore += 30; // High risk for <$100k liquidity
  else if (minLiquidity < 1000000) riskScore += 15; // Medium risk for <$1M liquidity

  // Increase risk for cross-chain arbitrage
  if (sourcePair.chainId !== targetPair.chainId) {
    riskScore += 20;
  }

  // Increase risk for low confidence
  if (confidence < 0.7) {
    riskScore += 25;
  }

  // Increase risk for unknown DEXes
  const knownDexes = ['uniswap_v3', 'sushiswap', 'curve', 'balancer'];
  if (!knownDexes.includes(sourcePair.exchange) || !knownDexes.includes(targetPair.exchange)) {
    riskScore += 20;
  }

  return Math.min(100, Math.max(0, riskScore));
}

function estimateExecutionTime(sourceChain: number, targetChain: number): number {
  const baseExecutionTime = 30; // 30 seconds base
  
  if (sourceChain === targetChain) {
    return baseExecutionTime;
  }
  
  // Cross-chain execution times vary by chain combination
  const chainExecutionTimes: Record<number, number> = {
    1: 180,    // Ethereum - 3 minutes
    137: 60,   // Polygon - 1 minute
    42161: 30, // Arbitrum - 30 seconds
    43114: 60, // Avalanche - 1 minute
    56: 30     // BSC - 30 seconds
  };
  
  const sourceTime = chainExecutionTimes[sourceChain] || baseExecutionTime;
  const targetTime = chainExecutionTimes[targetChain] || baseExecutionTime;
  
  return Math.max(sourceTime, targetTime) + 120; // Add 2 minutes for bridge time
}

async function getDefaultTokenPairs(supportedChains: number[]): Promise<string[]> {
  // Return most liquid and commonly arbitraged token pairs
  return [
    'ETH/USD',
    'BTC/USD',
    'USDC/USDT',
    'ETH/USDC',
    'BTC/ETH',
    'LINK/USD',
    'LINK/ETH'
  ];
}
