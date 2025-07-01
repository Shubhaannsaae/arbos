import { ethers, BigNumber, Contract } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { getNetworkConfig, getProvider, getWallet } from '../../../config/agentConfig';
import { getProtocolInfo } from '../../../shared/constants/protocols';

export interface SwapResult {
  hash: string;
  inputAmount: BigNumber;
  outputAmount: BigNumber;
  gasUsed: BigNumber;
  effectivePrice: BigNumber;
  slippage: number;
}

export interface OrderBookLevel {
  price: BigNumber;
  amount: BigNumber;
}

export interface OrderBook {
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
}

export interface TradingPairData {
  price: BigNumber;
  liquidity: BigNumber;
  volume24h: BigNumber;
  fee: number;
  contractAddress: string;
}

export class DexProvider {
  private supportedDexes: string[];
  private supportedChains: number[];
  private providers: Map<number, ethers.JsonRpcProvider> = new Map();
  private wallets: Map<number, ethers.Wallet> = new Map();
  private contracts: Map<string, Contract> = new Map();

  // Uniswap V3 ABI (minimal for our needs)
  private readonly UNISWAP_V3_ROUTER_ABI = [
    "function exactInputSingle((address tokenIn, address tokenOut, uint24 fee, address recipient, uint256 deadline, uint256 amountIn, uint256 amountOutMinimum, uint160 sqrtPriceLimitX96)) external payable returns (uint256 amountOut)",
    "function exactOutputSingle((address tokenIn, address tokenOut, uint24 fee, address recipient, uint256 deadline, uint256 amountOut, uint256 amountInMaximum, uint160 sqrtPriceLimitX96)) external payable returns (uint256 amountIn)"
  ];

  private readonly UNISWAP_V3_FACTORY_ABI = [
    "function getPool(address tokenA, address tokenB, uint24 fee) external view returns (address pool)"
  ];

  private readonly UNISWAP_V3_POOL_ABI = [
    "function slot0() external view returns (uint160 sqrtPriceX96, int24 tick, uint16 observationIndex, uint16 observationCardinality, uint16 observationCardinalityNext, uint8 feeProtocol, bool unlocked)",
    "function liquidity() external view returns (uint128)",
    "function fee() external view returns (uint24)",
    "function token0() external view returns (address)",
    "function token1() external view returns (address)"
  ];

  private readonly ERC20_ABI = [
    "function balanceOf(address owner) view returns (uint256)",
    "function transfer(address to, uint256 amount) returns (bool)",
    "function transferFrom(address from, address to, uint256 amount) returns (bool)",
    "function approve(address spender, uint256 amount) returns (bool)",
    "function allowance(address owner, address spender) view returns (uint256)",
    "function decimals() view returns (uint8)",
    "function symbol() view returns (string)",
    "function name() view returns (string)"
  ];

  constructor(supportedDexes: string[], supportedChains: number[]) {
    this.supportedDexes = supportedDexes;
    this.supportedChains = supportedChains;
    this.initializeProviders();
  }

  async initialize(): Promise<void> {
    logger.info('Initializing DEX provider', {
      supportedDexes: this.supportedDexes,
      supportedChains: this.supportedChains
    });

    await this.initializeContracts();
    await this.validateConnections();

    logger.info('DEX provider initialized successfully');
  }

  private initializeProviders(): void {
    this.supportedChains.forEach(chainId => {
      const provider = getProvider(chainId);
      const wallet = getWallet(chainId);
      
      this.providers.set(chainId, provider);
      this.wallets.set(chainId, wallet);
    });
  }

  private async initializeContracts(): Promise<void> {
    for (const chainId of this.supportedChains) {
      const provider = this.providers.get(chainId)!;
      const networkConfig = getNetworkConfig(chainId);

      // Initialize Uniswap V3 contracts
      if (networkConfig.dexContracts.uniswapV3Router) {
        const routerContract = new Contract(
          networkConfig.dexContracts.uniswapV3Router,
          this.UNISWAP_V3_ROUTER_ABI,
          provider
        );
        this.contracts.set(`uniswap_v3_router_${chainId}`, routerContract);

        const factoryContract = new Contract(
          networkConfig.dexContracts.uniswapV3Factory!,
          this.UNISWAP_V3_FACTORY_ABI,
          provider
        );
        this.contracts.set(`uniswap_v3_factory_${chainId}`, factoryContract);
      }

      // Initialize SushiSwap contracts
      if (networkConfig.dexContracts.sushiswapRouter) {
        const sushiContract = new Contract(
          networkConfig.dexContracts.sushiswapRouter,
          this.UNISWAP_V3_ROUTER_ABI, // SushiSwap uses similar interface
          provider
        );
        this.contracts.set(`sushiswap_router_${chainId}`, sushiContract);
      }
    }
  }

  private async validateConnections(): Promise<void> {
    const validationPromises = this.supportedChains.map(async (chainId) => {
      try {
        const provider = this.providers.get(chainId)!;
        const blockNumber = await provider.getBlockNumber();
        
        logger.debug('Chain connection validated', {
          chainId,
          blockNumber,
          network: getNetworkConfig(chainId).name
        });
        
        return true;
      } catch (error) {
        logger.error('Chain connection failed', {
          chainId,
          error: error instanceof Error ? error.message : String(error)
        });
        
        return false;
      }
    });

    const results = await Promise.all(validationPromises);
    const failedChains = this.supportedChains.filter((_, index) => !results[index]);

    if (failedChains.length > 0) {
      throw new Error(`Failed to connect to chains: ${failedChains.join(', ')}`);
    }
  }

  async getTradingPairData(
    tokenPair: string,
    chainId: number,
    dexName: string
  ): Promise<TradingPairData | null> {
    try {
      const [baseToken, quoteToken] = this.parseTokenPair(tokenPair);
      const tokenAddresses = await this.getTokenAddresses(baseToken, quoteToken, chainId);

      if (!tokenAddresses) {
        logger.warn('Token addresses not found', { tokenPair, chainId, dexName });
        return null;
      }

      switch (dexName.toLowerCase()) {
        case 'uniswap_v3':
          return await this.getUniswapV3PairData(tokenAddresses, chainId);
        case 'sushiswap':
          return await this.getSushiswapPairData(tokenAddresses, chainId);
        default:
          logger.warn('Unsupported DEX', { dexName });
          return null;
      }

    } catch (error) {
      logger.error('Failed to get trading pair data', {
        tokenPair,
        chainId,
        dexName,
        error: error instanceof Error ? error.message : String(error)
      });

      return null;
    }
  }

  async executeSwap(
    tokenPair: string,
    chainId: number,
    dexName: string,
    side: 'buy' | 'sell',
    amount: BigNumber,
    maxSlippage: number,
    mevProtection: boolean = true
  ): Promise<SwapResult> {
    const startTime = Date.now();
    
    logger.info('Executing swap', {
      tokenPair,
      chainId,
      dexName,
      side,
      amount: ethers.utils.formatEther(amount),
      maxSlippage
    });

    try {
      const [baseToken, quoteToken] = this.parseTokenPair(tokenPair);
      const tokenAddresses = await this.getTokenAddresses(baseToken, quoteToken, chainId);

      if (!tokenAddresses) {
        throw new Error(`Token addresses not found for ${tokenPair} on chain ${chainId}`);
      }

      const wallet = this.wallets.get(chainId)!;

      // Determine input and output tokens based on side
      const { tokenIn, tokenOut } = side === 'buy' 
        ? { tokenIn: tokenAddresses.quote, tokenOut: tokenAddresses.base }
        : { tokenIn: tokenAddresses.base, tokenOut: tokenAddresses.quote };

      // Get current gas price
      const gasPrice = await this.getGasPrice(chainId);

      // Calculate minimum output amount with slippage tolerance
      const currentPrice = await this.getCurrentPrice(tokenIn, tokenOut, chainId, dexName);
      const expectedOutput = this.calculateExpectedOutput(amount, currentPrice, side);
      const minOutputAmount = expectedOutput.mul(10000 - Math.floor(maxSlippage * 100)).div(10000);

      // Execute swap based on DEX
      let swapResult: SwapResult;

      switch (dexName.toLowerCase()) {
        case 'uniswap_v3':
          swapResult = await this.executeUniswapV3Swap(
            tokenIn,
            tokenOut,
            amount,
            minOutputAmount,
            chainId,
            wallet,
            gasPrice,
            mevProtection
          );
          break;

        case 'sushiswap':
          swapResult = await this.executeSushiswapSwap(
            tokenIn,
            tokenOut,
            amount,
            minOutputAmount,
            chainId,
            wallet,
            gasPrice,
            mevProtection
          );
          break;

        default:
          throw new Error(`Unsupported DEX: ${dexName}`);
      }

      logger.info('Swap executed successfully', {
        tokenPair,
        side,
        hash: swapResult.hash,
        actualOutput: ethers.utils.formatEther(swapResult.outputAmount),
        slippage: swapResult.slippage,
        duration: Date.now() - startTime
      });

      return swapResult;

    } catch (error) {
      logger.error('Swap execution failed', {
        tokenPair,
        chainId,
        dexName,
        side,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime
      });

      throw error;
    }
  }

  async getOrderBookDepth(
    tokenPair: string,
    chainId: number,
    dexName: string,
    levels: number = 10
  ): Promise<OrderBook> {
    try {
      const [baseToken, quoteToken] = this.parseTokenPair(tokenPair);
      const tokenAddresses = await this.getTokenAddresses(baseToken, quoteToken, chainId);

      if (!tokenAddresses) {
        return { bids: [], asks: [] };
      }

      // For AMMs, we simulate order book based on liquidity curves
      return await this.simulateOrderBook(tokenAddresses, chainId, dexName, levels);

    } catch (error) {
      logger.error('Failed to get order book depth', {
        tokenPair,
        chainId,
        dexName,
        error: error instanceof Error ? error.message : String(error)
      });

      return { bids: [], asks: [] };
    }
  }

  async calculatePriceImpact(
    tokenPair: string,
    tradeSize: BigNumber,
    chainId: number,
    dexName: string
  ): Promise<number> {
    try {
      const [baseToken, quoteToken] = this.parseTokenPair(tokenPair);
      const tokenAddresses = await this.getTokenAddresses(baseToken, quoteToken, chainId);

      if (!tokenAddresses) {
        return 0;
      }

      const currentPrice = await this.getCurrentPrice(
        tokenAddresses.base, 
        tokenAddresses.quote, 
        chainId, 
        dexName
      );

      // Get quote for the trade size
      const quote = await this.getSwapQuote(
        tokenAddresses.base,
        tokenAddresses.quote,
        tradeSize,
        chainId,
        dexName
      );

      if (quote.isZero()) {
        return 0;
      }

      // Calculate effective price and price impact
      const effectivePrice = quote.mul(ethers.utils.parseEther('1')).div(tradeSize);
      const priceImpact = Math.abs(
        parseFloat(ethers.utils.formatEther(effectivePrice.sub(currentPrice))) /
        parseFloat(ethers.utils.formatEther(currentPrice))
      ) * 100;

      return priceImpact;

    } catch (error) {
      logger.error('Failed to calculate price impact', {
        tokenPair,
        chainId,
        dexName,
        error: error instanceof Error ? error.message : String(error)
      });

      return 0;
    }
  }

  async getGasPrice(chainId: number): Promise<BigNumber> {
    try {
      const provider = this.providers.get(chainId)!;
      const feeData = await provider.getFeeData();
      
      // Use EIP-1559 if available, otherwise legacy gas price
      return feeData.maxFeePerGas || feeData.gasPrice || BigNumber.from('20000000000'); // 20 gwei fallback

    } catch (error) {
      logger.error('Failed to get gas price', {
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return reasonable fallback gas prices by chain
      const fallbackGasPrices: Record<number, string> = {
        1: '30000000000',    // 30 gwei for Ethereum
        137: '50000000000',  // 50 gwei for Polygon
        42161: '100000000',  // 0.1 gwei for Arbitrum
        43114: '25000000000', // 25 gwei for Avalanche
        56: '5000000000'     // 5 gwei for BSC
      };

      return BigNumber.from(fallbackGasPrices[chainId] || '20000000000');
    }
  }

  async getBridgeFee(
    sourceChain: number,
    targetChain: number,
    amount: BigNumber
  ): Promise<BigNumber> {
    try {
      // This would integrate with actual bridge protocols in production
      // For now, estimate based on chain combination and amount
      
      const baseFeePercentage = 0.05; // 0.05% base fee
      let multiplier = 1;

      // Ethereum mainnet involved = higher fees
      if (sourceChain === 1 || targetChain === 1) {
        multiplier = 2;
      }

      const feePercentage = baseFeePercentage * multiplier;
      return amount.mul(Math.floor(feePercentage * 10000)).div(10000);

    } catch (error) {
      logger.error('Failed to get bridge fee', {
        sourceChain,
        targetChain,
        error: error instanceof Error ? error.message : String(error)
      });

      // Fallback to 0.1% fee
      return amount.mul(10).div(10000);
    }
  }

  async checkAndApproveTokens(
    tokens: { base: string; quote: string },
    chainId: number,
    spenderAddress: string,
    amount: BigNumber
  ): Promise<Array<{ hash: string; gasUsed: BigNumber }>> {
    const approvals: Array<{ hash: string; gasUsed: BigNumber }> = [];
    const wallet = this.wallets.get(chainId)!;

    for (const tokenSymbol of [tokens.base, tokens.quote]) {
      try {
        const tokenAddress = await this.getTokenAddress(tokenSymbol, chainId);
        if (!tokenAddress) continue;

        const tokenContract = new Contract(tokenAddress, this.ERC20_ABI, wallet);
        
        // Check current allowance
        const currentAllowance = await tokenContract.allowance(wallet.address, spenderAddress);
        
        if (currentAllowance.lt(amount)) {
          // Approve maximum amount for gas efficiency
          const maxApproval = ethers.constants.MaxUint256;
          
          const tx = await tokenContract.approve(spenderAddress, maxApproval, {
            gasLimit: 100000,
            gasPrice: await this.getGasPrice(chainId)
          });

          const receipt = await tx.wait();
          
          approvals.push({
            hash: receipt.transactionHash,
            gasUsed: receipt.gasUsed
          });

          logger.info('Token approved', {
            token: tokenSymbol,
            chainId,
            spender: spenderAddress,
            hash: receipt.transactionHash
          });
        }

      } catch (error) {
        logger.error('Token approval failed', {
          token: tokenSymbol,
          chainId,
          spender: spenderAddress,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    return approvals;
  }

  async getNetworkCongestion(): Promise<Record<number, number>> {
    const congestion: Record<number, number> = {};

    for (const chainId of this.supportedChains) {
      try {
        const provider = this.providers.get(chainId)!;
        const [currentBlock, gasPrice] = await Promise.all([
          provider.getBlock('latest'),
          provider.getFeeData()
        ]);

        // Estimate congestion based on gas usage and prices
        const gasUsageRatio = currentBlock.gasUsed.mul(100).div(currentBlock.gasLimit);
        const gasPriceGwei = parseFloat(ethers.utils.formatUnits(
          gasPrice.gasPrice || BigNumber.from('20000000000'), 
          'gwei'
        ));

        // Simple congestion metric: gas usage + price pressure
        const congestionScore = Math.min(
          parseInt(gasUsageRatio.toString()) + Math.min(gasPriceGwei / 10, 50),
          100
        );

        congestion[chainId] = congestionScore;

      } catch (error) {
        logger.error('Failed to get network congestion', {
          chainId,
          error: error instanceof Error ? error.message : String(error)
        });

        congestion[chainId] = 50; // Default moderate congestion
      }
    }

    return congestion;
  }

  // Private helper methods

  private parseTokenPair(tokenPair: string): [string, string] {
    const [base, quote] = tokenPair.split('/');
    if (!base || !quote) {
      throw new Error(`Invalid token pair format: ${tokenPair}`);
    }
    return [base, quote];
  }

  private async getTokenAddresses(
    baseToken: string,
    quoteToken: string,
    chainId: number
  ): Promise<{ base: string; quote: string } | null> {
    try {
      const baseAddress = await this.getTokenAddress(baseToken, chainId);
      const quoteAddress = await this.getTokenAddress(quoteToken, chainId);

      if (!baseAddress || !quoteAddress) {
        return null;
      }

      return { base: baseAddress, quote: quoteAddress };

    } catch (error) {
      logger.error('Failed to get token addresses', {
        baseToken,
        quoteToken,
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });

      return null;
    }
  }

  private async getTokenAddress(tokenSymbol: string, chainId: number): Promise<string | null> {
    // Common token addresses by chain
    const tokenAddresses: Record<number, Record<string, string>> = {
      1: { // Ethereum
        'ETH': ethers.constants.AddressZero,
        'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        'USDC': '0xA0b86a33E6427d6D87EC7A7B7EEA7a3a7A6FE1a7',
        'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
        'BTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
        'LINK': '0x514910771AF9Ca656af840dff83E8264EcF986CA'
      },
      137: { // Polygon
        'MATIC': ethers.constants.AddressZero,
        'WMATIC': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
        'USDC': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
        'USDT': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
        'ETH': '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619',
        'BTC': '0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6'
      },
      42161: { // Arbitrum
        'ETH': ethers.constants.AddressZero,
        'WETH': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
        'USDC': '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',
        'USDT': '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9',
        'BTC': '0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f',
        'LINK': '0xf97f4df75117a78c1A5a0DBb814Af92458539FB4'
      },
      43114: { // Avalanche
        'AVAX': ethers.constants.AddressZero,
        'WAVAX': '0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7',
        'USDC': '0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E',
        'USDT': '0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7',
        'ETH': '0x49D5c2BdFfac6CE2BFdB6640F4F80f226bc10bAB',
        'BTC': '0x50b7545627a5162F82A992c33b87aDc75187B218'
      }
    };

    return tokenAddresses[chainId]?.[tokenSymbol] || null;
  }

  private async getUniswapV3PairData(
    tokenAddresses: { base: string; quote: string },
    chainId: number
  ): Promise<TradingPairData> {
    const factory = this.contracts.get(`uniswap_v3_factory_${chainId}`)!;
    
    // Try different fee tiers (0.05%, 0.3%, 1%)
    const feeTiers = [500, 3000, 10000];
    
    for (const fee of feeTiers) {
      try {
        const poolAddress = await factory.getPool(tokenAddresses.base, tokenAddresses.quote, fee);
        
        if (poolAddress !== ethers.constants.AddressZero) {
          const pool = new Contract(poolAddress, this.UNISWAP_V3_POOL_ABI, this.providers.get(chainId)!);
          
          const [slot0, liquidity] = await Promise.all([
            pool.slot0(),
            pool.liquidity()
          ]);

          // Calculate price from sqrtPriceX96
          const price = this.calculatePriceFromSqrtPriceX96(slot0.sqrtPriceX96);
          
          return {
            price,
            liquidity: BigNumber.from(liquidity),
            volume24h: BigNumber.from(0), // Would need additional API call
            fee: fee / 10000, // Convert to percentage
            contractAddress: poolAddress
          };
        }
      } catch (error) {
        // Continue to next fee tier
        continue;
      }
    }

    throw new Error('No Uniswap V3 pool found for token pair');
  }

  private async getSushiswapPairData(
    tokenAddresses: { base: string; quote: string },
    chainId: number
  ): Promise<TradingPairData> {
    // SushiSwap implementation would be similar to Uniswap V2/V3
    // This is a simplified version
    throw new Error('SushiSwap integration not implemented in this example');
  }

  private calculatePriceFromSqrtPriceX96(sqrtPriceX96: BigNumber): BigNumber {
    // Convert sqrtPriceX96 to actual price
    // Price = (sqrtPriceX96 / 2^96)^2
    const Q96 = BigNumber.from(2).pow(96);
    const priceX192 = sqrtPriceX96.mul(sqrtPriceX96);
    return priceX192.div(Q96).div(Q96);
  }

  private async getCurrentPrice(
    tokenIn: string,
    tokenOut: string,
    chainId: number,
    dexName: string
  ): Promise<BigNumber> {
    // Get current market price from the DEX
    const quote = await this.getSwapQuote(
      tokenIn,
      tokenOut,
      ethers.utils.parseEther('1'), // 1 unit
      chainId,
      dexName
    );

    return quote;
  }

  private async getSwapQuote(
    tokenIn: string,
    tokenOut: string,
    amountIn: BigNumber,
    chainId: number,
    dexName: string
  ): Promise<BigNumber> {
    // This would call the DEX's quoter contract
    // For now, return a mock quote
    return amountIn.mul(2000); // Mock 1:2000 ratio
  }

  private calculateExpectedOutput(
    inputAmount: BigNumber,
    price: BigNumber,
    side: 'buy' | 'sell'
  ): BigNumber {
    if (side === 'buy') {
      return inputAmount.mul(price).div(ethers.utils.parseEther('1'));
    } else {
      return inputAmount.mul(ethers.utils.parseEther('1')).div(price);
    }
  }

  private async executeUniswapV3Swap(
    tokenIn: string,
    tokenOut: string,
    amountIn: BigNumber,
    minAmountOut: BigNumber,
    chainId: number,
    wallet: ethers.Wallet,
    gasPrice: BigNumber,
    mevProtection: boolean
  ): Promise<SwapResult> {
    const router = this.contracts.get(`uniswap_v3_router_${chainId}`)!.connect(wallet);
    
    const params = {
      tokenIn,
      tokenOut,
      fee: 3000, // 0.3% fee tier
      recipient: wallet.address,
      deadline: Math.floor(Date.now() / 1000) + 300, // 5 minutes
      amountIn,
      amountOutMinimum: minAmountOut,
      sqrtPriceLimitX96: 0 // No price limit
    };

    const tx = await router.exactInputSingle(params, {
      gasLimit: 300000,
      gasPrice: mevProtection ? gasPrice.mul(110).div(100) : gasPrice // 10% priority if MEV protection
    });

    const receipt = await tx.wait();
    
    // Parse logs to get actual output amount
    const outputAmount = minAmountOut; // Simplified - would parse from logs
    const effectivePrice = outputAmount.mul(ethers.utils.parseEther('1')).div(amountIn);
    const slippage = 0; // Simplified - would calculate from expected vs actual

    return {
      hash: receipt.transactionHash,
      inputAmount: amountIn,
      outputAmount,
      gasUsed: receipt.gasUsed,
      effectivePrice,
      slippage
    };
  }

  private async executeSushiswapSwap(
    tokenIn: string,
    tokenOut: string,
    amountIn: BigNumber,
    minAmountOut: BigNumber,
    chainId: number,
    wallet: ethers.Wallet,
    gasPrice: BigNumber,
    mevProtection: boolean
  ): Promise<SwapResult> {
    // Similar to Uniswap implementation
    throw new Error('SushiSwap swap execution not implemented in this example');
  }

  private async simulateOrderBook(
    tokenAddresses: { base: string; quote: string },
    chainId: number,
    dexName: string,
    levels: number
  ): Promise<OrderBook> {
    // For AMMs, simulate order book from liquidity curves
    const bids: OrderBookLevel[] = [];
    const asks: OrderBookLevel[] = [];

    // This would query the actual AMM curves and simulate order book levels
    // For now, return empty order book
    return { bids, asks };
  }

  async updateConfiguration(config: {
    supportedDexes?: string[];
    supportedChains?: number[];
  }): Promise<void> {
    if (config.supportedDexes) {
      this.supportedDexes = config.supportedDexes;
    }
    
    if (config.supportedChains) {
      this.supportedChains = config.supportedChains;
      this.initializeProviders();
      await this.initializeContracts();
    }

    logger.info('DEX provider configuration updated', config);
  }
}
