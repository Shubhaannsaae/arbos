import { ethers, Provider } from 'ethers';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';
import { RouteOption } from './OptimalRouter';

export interface RouteCost {
  gasFee: bigint;
  bridgeFee: bigint;
  protocolFee: bigint;
  slippageCost: bigint;
  totalCost: bigint;
  costInUSD: number;
  gasPrice: bigint;
  gasUsed: number;
}

export interface GasPriceData {
  standard: bigint;
  fast: bigint;
  instant: bigint;
  timestamp: number;
}

export interface TokenPriceData {
  priceUSD: number;
  decimals: number;
  symbol: string;
  timestamp: number;
}

export class CostCalculator {
  private logger: Logger;
  private providers: Map<number, Provider> = new Map();
  private gasPriceCache: Map<number, GasPriceData> = new Map();
  private tokenPriceCache: Map<string, TokenPriceData> = new Map();
  
  // Protocol fee structures (basis points)
  private readonly PROTOCOL_FEES: { [protocol: string]: number } = {
    ccip: 0,        // CCIP charges gas only
    layerzero: 0,   // LayerZero charges gas only
    polygon: 0,     // Polygon PoS bridge is free
    arbitrum: 0,    // Arbitrum bridge charges gas only
    avalanche: 0    // Avalanche bridge charges gas only
  };

  // Bridge-specific fee structures
  private readonly BRIDGE_FEES: { [protocol: string]: { fixed: bigint; percentage: number } } = {
    ccip: { fixed: 0n, percentage: 0 },
    layerzero: { fixed: 0n, percentage: 0 },
    polygon: { fixed: 0n, percentage: 0 },
    arbitrum: { fixed: 0n, percentage: 0 },
    avalanche: { fixed: 1000000000000000n, percentage: 0 } // 0.001 ETH fixed fee example
  };

  constructor(providers: Map<number, Provider>) {
    this.providers = providers;
    
    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      transports: [
        new transports.File({ filename: 'logs/cost-calculator.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });
  }

  /**
   * Calculate total cost for a route
   */
  async calculateRouteCost(
    route: RouteOption,
    amount: bigint,
    token: string,
    priorityLevel: 'standard' | 'fast' | 'instant' = 'standard'
  ): Promise<RouteCost> {
    try {
      // Get current gas prices
      const gasPrices = await this.getGasPrices(route.sourceChain);
      const gasPrice = gasPrices[priorityLevel];

      // Calculate gas fee
      const gasFee = gasPrice * BigInt(route.gasLimit);

      // Calculate bridge fee
      const bridgeFee = this.calculateBridgeFee(route.protocol, amount);

      // Calculate protocol fee
      const protocolFee = this.calculateProtocolFee(route.protocol, amount);

      // Estimate slippage cost
      const slippageCost = await this.estimateSlippageCost(route, amount, token);

      // Calculate total cost
      const totalCost = gasFee + bridgeFee + protocolFee + slippageCost;

      // Convert to USD
      const costInUSD = await this.convertToUSD(totalCost, route.sourceChain);

      const result: RouteCost = {
        gasFee,
        bridgeFee,
        protocolFee,
        slippageCost,
        totalCost,
        costInUSD,
        gasPrice,
        gasUsed: route.gasLimit
      };

      this.logger.debug(`Cost calculated for route ${route.id}`, {
        totalCost: totalCost.toString(),
        costInUSD,
        breakdown: {
          gasFee: gasFee.toString(),
          bridgeFee: bridgeFee.toString(),
          protocolFee: protocolFee.toString(),
          slippageCost: slippageCost.toString()
        }
      });

      return result;

    } catch (error) {
      this.logger.error('Failed to calculate route cost', {
        route,
        amount: amount.toString(),
        token,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Compare costs between multiple routes
   */
  async compareCosts(
    routes: RouteOption[],
    amount: bigint,
    token: string
  ): Promise<{ route: RouteOption; cost: RouteCost }[]> {
    const costs = await Promise.all(
      routes.map(async route => ({
        route,
        cost: await this.calculateRouteCost(route, amount, token)
      }))
    );

    // Sort by total cost (ascending)
    return costs.sort((a, b) => 
      Number(a.cost.totalCost - b.cost.totalCost)
    );
  }

  /**
   * Get current gas prices for chain
   */
  async getGasPrices(chainId: number): Promise<GasPriceData> {
    // Check cache first
    const cached = this.gasPriceCache.get(chainId);
    if (cached && Date.now() - cached.timestamp < 30000) { // 30 second cache
      return cached;
    }

    try {
      const provider = this.providers.get(chainId);
      if (!provider) {
        throw new Error(`Provider not found for chain ${chainId}`);
      }

      const feeData = await provider.getFeeData();
      const baseGasPrice = feeData.gasPrice || 0n;

      const gasPrices: GasPriceData = {
        standard: baseGasPrice,
        fast: baseGasPrice * 120n / 100n,      // 20% higher
        instant: baseGasPrice * 150n / 100n,   // 50% higher
        timestamp: Date.now()
      };

      this.gasPriceCache.set(chainId, gasPrices);
      return gasPrices;

    } catch (error) {
      this.logger.error('Failed to get gas prices', {
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return fallback gas prices
      const fallbackPrice = this.getFallbackGasPrice(chainId);
      return {
        standard: fallbackPrice,
        fast: fallbackPrice * 120n / 100n,
        instant: fallbackPrice * 150n / 100n,
        timestamp: Date.now()
      };
    }
  }

  /**
   * Get token price in USD
   */
  async getTokenPrice(token: string): Promise<TokenPriceData> {
    // Check cache first
    const cached = this.tokenPriceCache.get(token);
    if (cached && Date.now() - cached.timestamp < 60000) { // 1 minute cache
      return cached;
    }

    try {
      // In production, integrate with price oracle or API
      // For now, return mock data based on known tokens
      const priceData = this.getMockTokenPrice(token);
      this.tokenPriceCache.set(token, priceData);
      return priceData;

    } catch (error) {
      this.logger.error('Failed to get token price', {
        token,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Estimate gas cost in USD
   */
  async estimateGasCostUSD(
    chainId: number,
    gasLimit: number,
    priorityLevel: 'standard' | 'fast' | 'instant' = 'standard'
  ): Promise<number> {
    const gasPrices = await this.getGasPrices(chainId);
    const gasPrice = gasPrices[priorityLevel];
    const gasCost = gasPrice * BigInt(gasLimit);

    return await this.convertToUSD(gasCost, chainId);
  }

  /**
   * Calculate optimal gas settings for transaction
   */
  async calculateOptimalGas(
    chainId: number,
    baseGasLimit: number,
    targetConfirmationTime: number // minutes
  ): Promise<{ gasLimit: number; gasPrice: bigint; estimatedTime: number }> {
    const gasPrices = await this.getGasPrices(chainId);
    
    // Select gas price based on target confirmation time
    let gasPrice: bigint;
    let estimatedTime: number;

    if (targetConfirmationTime <= 1) {
      gasPrice = gasPrices.instant;
      estimatedTime = 0.5;
    } else if (targetConfirmationTime <= 3) {
      gasPrice = gasPrices.fast;
      estimatedTime = 2;
    } else {
      gasPrice = gasPrices.standard;
      estimatedTime = 5;
    }

    // Add buffer to gas limit
    const gasLimit = Math.ceil(baseGasLimit * 1.2);

    return { gasLimit, gasPrice, estimatedTime };
  }

  /**
   * Calculate bridge fee for protocol
   */
  private calculateBridgeFee(protocol: string, amount: bigint): bigint {
    const feeStructure = this.BRIDGE_FEES[protocol];
    if (!feeStructure) {
      return 0n;
    }

    const percentageFee = amount * BigInt(feeStructure.percentage) / 10000n;
    return feeStructure.fixed + percentageFee;
  }

  /**
   * Calculate protocol fee
   */
  private calculateProtocolFee(protocol: string, amount: bigint): bigint {
    const feeRate = this.PROTOCOL_FEES[protocol] || 0;
    return amount * BigInt(feeRate) / 10000n;
  }

  /**
   * Estimate slippage cost
   */
  private async estimateSlippageCost(
    route: RouteOption,
    amount: bigint,
    token: string
  ): Promise<bigint> {
    // Simplified slippage calculation
    // In production, integrate with DEX APIs for real slippage estimates
    
    const baseSlippage = this.getBaseSlippage(route.protocol);
    const amountSlippage = this.calculateAmountBasedSlippage(amount);
    
    const totalSlippagePercent = baseSlippage + amountSlippage;
    return amount * BigInt(Math.floor(totalSlippagePercent * 100)) / 10000n;
  }

  /**
   * Convert amount to USD
   */
  private async convertToUSD(amount: bigint, chainId: number): Promise<number> {
    try {
      // Get native token price (ETH, AVAX, MATIC, etc.)
      const nativeTokenPrice = await this.getNativeTokenPrice(chainId);
      const amountInEther = Number(ethers.formatEther(amount));
      
      return amountInEther * nativeTokenPrice;
    } catch (error) {
      this.logger.warn('Failed to convert to USD', { amount: amount.toString(), chainId });
      return 0;
    }
  }

  /**
   * Get native token price for chain
   */
  private async getNativeTokenPrice(chainId: number): Promise<number> {
    // Mock prices - in production, use price oracle
    const nativePrices: { [chainId: number]: number } = {
      1: 3000,    // ETH
      43114: 40,  // AVAX
      137: 0.8,   // MATIC
      42161: 3000 // ETH (Arbitrum)
    };

    return nativePrices[chainId] || 1;
  }

  /**
   * Get mock token price data
   */
  private getMockTokenPrice(token: string): TokenPriceData {
    // Mock token prices - in production, use real price feed
    const mockPrices: { [token: string]: { price: number; decimals: number; symbol: string } } = {
      '0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8': { price: 1, decimals: 6, symbol: 'USDC' },
      '0x514910771AF9Ca656af840dff83E8264EcF986CA': { price: 15, decimals: 18, symbol: 'LINK' },
      '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599': { price: 45000, decimals: 8, symbol: 'WBTC' }
    };

    const mockData = mockPrices[token] || { price: 1, decimals: 18, symbol: 'UNKNOWN' };
    
    return {
      priceUSD: mockData.price,
      decimals: mockData.decimals,
      symbol: mockData.symbol,
      timestamp: Date.now()
    };
  }

  /**
   * Get fallback gas price for chain
   */
  private getFallbackGasPrice(chainId: number): bigint {
    const fallbackPrices: { [chainId: number]: bigint } = {
      1: 20000000000n,      // 20 gwei
      43114: 25000000000n,  // 25 gwei
      137: 30000000000n,    // 30 gwei
      42161: 100000000n     // 0.1 gwei
    };

    return fallbackPrices[chainId] || 20000000000n;
  }

  /**
   * Get base slippage for protocol
   */
  private getBaseSlippage(protocol: string): number {
    const baseSlippages: { [protocol: string]: number } = {
      ccip: 0.01,      // 0.01%
      layerzero: 0.02, // 0.02%
      polygon: 0.0,    // 0%
      arbitrum: 0.0,   // 0%
      avalanche: 0.01  // 0.01%
    };

    return baseSlippages[protocol] || 0.05;
  }

  /**
   * Calculate amount-based slippage
   */
  private calculateAmountBasedSlippage(amount: bigint): number {
    // Larger amounts may have higher slippage
    const amountInEther = Number(ethers.formatEther(amount));
    
    if (amountInEther > 1000000) return 0.1;      // 0.1% for > 1M
    if (amountInEther > 100000) return 0.05;      // 0.05% for > 100K
    if (amountInEther > 10000) return 0.02;       // 0.02% for > 10K
    
    return 0.01; // 0.01% for smaller amounts
  }

  /**
   * Update provider for chain
   */
  updateProvider(chainId: number, provider: Provider): void {
    this.providers.set(chainId, provider);
    this.logger.info(`Provider updated for chain ${chainId}`);
  }

  /**
   * Clear price caches
   */
  clearCaches(): void {
    this.gasPriceCache.clear();
    this.tokenPriceCache.clear();
    this.logger.info('Price caches cleared');
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    gasPriceCacheSize: number;
    tokenPriceCacheSize: number;
    oldestGasPriceEntry: number;
    oldestTokenPriceEntry: number;
  } {
    let oldestGasPrice = Date.now();
    let oldestTokenPrice = Date.now();

    for (const gasPrice of this.gasPriceCache.values()) {
      if (gasPrice.timestamp < oldestGasPrice) {
        oldestGasPrice = gasPrice.timestamp;
      }
    }

    for (const tokenPrice of this.tokenPriceCache.values()) {
      if (tokenPrice.timestamp < oldestTokenPrice) {
        oldestTokenPrice = tokenPrice.timestamp;
      }
    }

    return {
      gasPriceCacheSize: this.gasPriceCache.size,
      tokenPriceCacheSize: this.tokenPriceCache.size,
      oldestGasPriceEntry: oldestGasPrice,
      oldestTokenPriceEntry: oldestTokenPrice
    };
  }
}
