import { ethers, BigNumber, Contract } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { Portfolio, PortfolioPosition, TokenInfo } from '../../../shared/types/market';
import { getNetworkConfig, getProvider, getWallet } from '../../../config/agentConfig';
import { PORTFOLIO_THRESHOLDS } from '../../../shared/constants/thresholds';

export interface TradeResult {
  transactionHash: string;
  inputAmount: BigNumber;
  outputAmount: BigNumber;
  gasUsed: BigNumber;
  effectivePrice: BigNumber;
  slippage: number;
}

export interface PortfolioUpdate {
  portfolioId: string;
  timestamp: number;
  previousValue: BigNumber;
  newValue: BigNumber;
  changes: Array<{
    action: 'add' | 'remove' | 'modify';
    position: PortfolioPosition;
  }>;
}

export class PortfolioProvider {
  private providers: Map<number, ethers.JsonRpcProvider> = new Map();
  private wallets: Map<number, ethers.Wallet> = new Map();
  private contracts: Map<string, Contract> = new Map();
  private portfolioDatabase: Map<string, Portfolio> = new Map();

  // ERC20 ABI for token operations
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

  // Uniswap V3 Router ABI (from @uniswap/v3-periphery)
  private readonly UNISWAP_V3_ROUTER_ABI = [
    "function exactInputSingle((address tokenIn, address tokenOut, uint24 fee, address recipient, uint256 deadline, uint256 amountIn, uint256 amountOutMinimum, uint160 sqrtPriceLimitX96)) external payable returns (uint256 amountOut)",
    "function exactOutputSingle((address tokenIn, address tokenOut, uint24 fee, address recipient, uint256 deadline, uint256 amountOut, uint256 amountInMaximum, uint160 sqrtPriceLimitX96)) external payable returns (uint256 amountIn)"
  ];

  // Chainlink Automation Registry ABI (from @chainlink/contracts v1.4.0)
  private readonly AUTOMATION_REGISTRY_ABI = [
    "function registerUpkeep(address target, uint32 gasLimit, address admin, bytes calldata checkData) external returns (uint256 id)",
    "function addFunds(uint256 id, uint96 amount) external",
    "function getUpkeep(uint256 id) external view returns (address target, uint32 executeGas, bytes memory checkData, uint96 balance, address lastKeeper, address admin, uint64 maxValidBlocknumber, uint96 amountSpent, bool paused, bytes memory offchainConfig)"
  ];

  // Portfolio Rebalancer Contract ABI (Custom implementation)
  private readonly PORTFOLIO_REBALANCER_ABI = [
    "function rebalancePortfolio(address[] calldata sellTokens, uint256[] calldata sellAmounts, address[] calldata buyTokens, uint256[] calldata buyAmounts, bytes calldata rebalanceData) external",
    "function calculateOptimalTrades(address[] calldata currentTokens, uint256[] calldata currentAmounts, uint256[] calldata targetPercentages) external view returns (address[] memory sellTokens, uint256[] memory sellAmounts, address[] memory buyTokens, uint256[] memory buyAmounts)",
    "function checkUpkeep(bytes calldata checkData) external view returns (bool upkeepNeeded, bytes memory performData)",
    "function performUpkeep(bytes calldata performData) external"
  ];

  constructor() {
    this.initializeProviders();
  }

  async initialize(): Promise<void> {
    logger.info('Initializing portfolio provider');

    await this.initializeContracts();
    await this.loadExistingPortfolios();
    await this.validateConnections();

    logger.info('Portfolio provider initialized successfully');
  }

  private initializeProviders(): void {
    const supportedChains = [1, 137, 42161, 43114]; // ETH, Polygon, Arbitrum, Avalanche

    supportedChains.forEach(chainId => {
      const provider = getProvider(chainId);
      const wallet = getWallet(chainId);
      
      this.providers.set(chainId, provider);
      this.wallets.set(chainId, wallet);
    });
  }

  private async initializeContracts(): Promise<void> {
    for (const chainId of [1, 137, 42161, 43114]) {
      try {
        const provider = this.providers.get(chainId)!;
        const wallet = this.wallets.get(chainId)!;
        const networkConfig = getNetworkConfig(chainId);

        // Initialize Uniswap V3 Router
        if (networkConfig.dexContracts?.uniswapV3Router) {
          const uniswapRouter = new Contract(
            networkConfig.dexContracts.uniswapV3Router,
            this.UNISWAP_V3_ROUTER_ABI,
            wallet
          );
          this.contracts.set(`uniswap_router_${chainId}`, uniswapRouter);
        }

        // Initialize Chainlink Automation Registry
        if (networkConfig.chainlinkContracts?.automationRegistry) {
          const automationRegistry = new Contract(
            networkConfig.chainlinkContracts.automationRegistry,
            this.AUTOMATION_REGISTRY_ABI,
            wallet
          );
          this.contracts.set(`automation_registry_${chainId}`, automationRegistry);
        }

        // Initialize Portfolio Rebalancer (Custom contract)
        const portfolioRebalancerAddress = await this.deployPortfolioRebalancer(chainId);
        if (portfolioRebalancerAddress) {
          const portfolioRebalancer = new Contract(
            portfolioRebalancerAddress,
            this.PORTFOLIO_REBALANCER_ABI,
            wallet
          );
          this.contracts.set(`portfolio_rebalancer_${chainId}`, portfolioRebalancer);
        }

        logger.debug('Contracts initialized for chain', {
          chainId,
          uniswapRouter: !!networkConfig.dexContracts?.uniswapV3Router,
          automationRegistry: !!networkConfig.chainlinkContracts?.automationRegistry
        });

      } catch (error) {
        logger.error('Failed to initialize contracts for chain', {
          chainId,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }
  }

  private async deployPortfolioRebalancer(chainId: number): Promise<string | null> {
    try {
      // In production, this would deploy or reference existing portfolio rebalancer contract
      // For now, return a placeholder address that would be the deployed contract
      const mockAddresses: Record<number, string> = {
        1: '0x1234567890123456789012345678901234567890', // Ethereum
        137: '0x2345678901234567890123456789012345678901', // Polygon
        42161: '0x3456789012345678901234567890123456789012', // Arbitrum
        43114: '0x4567890123456789012345678901234567890123' // Avalanche
      };

      return mockAddresses[chainId] || null;

    } catch (error) {
      logger.error('Failed to deploy portfolio rebalancer', {
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });
      return null;
    }
  }

  private async loadExistingPortfolios(): Promise<void> {
    try {
      // In production, this would load from a database or decentralized storage
      // For now, initialize with empty portfolio storage
      logger.debug('Portfolio storage initialized', {
        portfolioCount: this.portfolioDatabase.size
      });

    } catch (error) {
      logger.error('Failed to load existing portfolios', {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async validateConnections(): Promise<void> {
    const validationPromises = Array.from(this.providers.entries()).map(async ([chainId, provider]) => {
      try {
        await provider.getBlockNumber();
        return { chainId, success: true };
      } catch (error) {
        return { 
          chainId, 
          success: false, 
          error: error instanceof Error ? error.message : String(error) 
        };
      }
    });

    const results = await Promise.all(validationPromises);
    
    results.forEach(result => {
      if (!result.success) {
        logger.warn('Connection validation failed', result);
      }
    });
  }

  async getUserPortfolios(userAddress?: string): Promise<Portfolio[]> {
    try {
      if (userAddress) {
        // Filter portfolios by user address
        return Array.from(this.portfolioDatabase.values()).filter(
          portfolio => portfolio.owner === userAddress
        );
      }

      // Return all portfolios
      return Array.from(this.portfolioDatabase.values());

    } catch (error) {
      logger.error('Failed to get user portfolios', {
        userAddress,
        error: error instanceof Error ? error.message : String(error)
      });

      return [];
    }
  }

  async getPortfolio(portfolioId: string): Promise<Portfolio | null> {
    try {
      const portfolio = this.portfolioDatabase.get(portfolioId);
      
      if (portfolio) {
        // Update portfolio with current market values
        return await this.updatePortfolioValues(portfolio);
      }

      return null;

    } catch (error) {
      logger.error('Failed to get portfolio', {
        portfolioId,
        error: error instanceof Error ? error.message : String(error)
      });

      return null;
    }
  }

  async createPortfolio(
    owner: string,
    name: string,
    initialAllocation: Array<{
      token: TokenInfo;
      targetPercentage: number;
      initialAmount: BigNumber;
    }>,
    chainId: number = 1
  ): Promise<Portfolio> {
    try {
      const portfolioId = `portfolio_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // Calculate initial positions
      const positions: PortfolioPosition[] = [];
      let totalValue = BigNumber.from(0);

      for (const allocation of initialAllocation) {
        // Get current price for value calculation
        const currentPrice = await this.getTokenPrice(allocation.token.symbol, chainId);
        const value = allocation.initialAmount.mul(currentPrice).div(
          ethers.utils.parseUnits('1', allocation.token.decimals)
        );

        positions.push({
          token: allocation.token,
          amount: allocation.initialAmount,
          value,
          valueUsd: parseFloat(ethers.utils.formatEther(value)), // Simplified USD conversion
          percentage: 0, // Will be calculated after total value
          averageCost: currentPrice,
          unrealizedPnl: BigNumber.from(0),
          unrealizedPnlPercentage: 0,
          lastUpdated: Date.now()
        });

        totalValue = totalValue.add(value);
      }

      // Update percentages
      positions.forEach(position => {
        position.percentage = totalValue.gt(0) 
          ? parseFloat(ethers.utils.formatEther(position.value.mul(10000).div(totalValue))) / 100
          : 0;
      });

      const portfolio: Portfolio = {
        id: portfolioId,
        name,
        owner,
        chainId,
        positions,
        allocation: initialAllocation.map(alloc => ({
          token: alloc.token,
          targetPercentage: alloc.targetPercentage,
          minPercentage: Math.max(0, alloc.targetPercentage - 5),
          maxPercentage: alloc.targetPercentage + 5,
          rebalanceThreshold: 2.5
        })),
        totalValue,
        totalValueUsd: parseFloat(ethers.utils.formatEther(totalValue)),
        performance: {
          totalReturn: BigNumber.from(0),
          totalReturnPercentage: 0,
          annualizedReturn: 0,
          volatility: 0,
          sharpeRatio: 0,
          calmarRatio: 0,
          maxDrawdown: 0,
          winRate: 0,
          profitFactor: 0,
          beta: 1.0,
          alpha: 0,
          informationRatio: 0,
          trackingError: 0
        },
        riskMetrics: {
          overallRiskScore: 50,
          concentrationRisk: 0,
          liquidityRisk: 0,
          volatilityRisk: 0,
          correlationRisk: 0,
          valueAtRisk: {
            day1: BigNumber.from(0),
            week1: BigNumber.from(0),
            month1: BigNumber.from(0)
          },
          expectedShortfall: {
            day1: BigNumber.from(0),
            week1: BigNumber.from(0),
            month1: BigNumber.from(0)
          },
          riskContributions: []
        },
        rebalancing: {
          isEnabled: true,
          frequency: 'weekly',
          threshold: 5.0,
          lastRebalance: Date.now(),
          nextRebalance: Date.now() + (7 * 24 * 60 * 60 * 1000), // 1 week
          automationUpkeepId: null
        },
        createdAt: Date.now(),
        updatedAt: Date.now()
      };

      // Store portfolio
      this.portfolioDatabase.set(portfolioId, portfolio);

      logger.info('Portfolio created', {
        portfolioId,
        owner,
        name,
        totalValue: ethers.utils.formatEther(totalValue),
        positionCount: positions.length
      });

      return portfolio;

    } catch (error) {
      logger.error('Failed to create portfolio', {
        owner,
        name,
        error: error instanceof Error ? error.message : String(error)
      });

      throw error;
    }
  }

  async executeTrade(
    fromToken: string,
    toToken: string,
    amountIn: BigNumber,
    minAmountOut: BigNumber,
    maxSlippage: number,
    gasOptimization: boolean = true,
    chainId: number = 1
  ): Promise<TradeResult> {
    try {
      const wallet = this.wallets.get(chainId);
      const uniswapRouter = this.contracts.get(`uniswap_router_${chainId}`);

      if (!wallet || !uniswapRouter) {
        throw new Error(`Trading infrastructure not available for chain ${chainId}`);
      }

      // Get token addresses
      const fromTokenAddress = await this.getTokenAddress(fromToken, chainId);
      const toTokenAddress = await this.getTokenAddress(toToken, chainId);

      if (!fromTokenAddress || !toTokenAddress) {
        throw new Error(`Token addresses not found for ${fromToken}/${toToken} on chain ${chainId}`);
      }

      // Check and approve token if needed
      await this.ensureTokenApproval(fromTokenAddress, uniswapRouter.address, amountIn, chainId);

      // Prepare swap parameters
      const deadline = Math.floor(Date.now() / 1000) + 300; // 5 minutes
      const swapParams = {
        tokenIn: fromTokenAddress,
        tokenOut: toTokenAddress,
        fee: 3000, // 0.3% fee tier
        recipient: wallet.address,
        deadline,
        amountIn,
        amountOutMinimum: minAmountOut,
        sqrtPriceLimitX96: 0 // No price limit
      };

      // Estimate gas
      const gasEstimate = await uniswapRouter.estimateGas.exactInputSingle(swapParams);
      const gasPrice = await this.getOptimalGasPrice(chainId, gasOptimization);
      
      logger.info('Executing trade', {
        fromToken,
        toToken,
        amountIn: ethers.utils.formatEther(amountIn),
        minAmountOut: ethers.utils.formatEther(minAmountOut),
        gasEstimate: gasEstimate.toString(),
        gasPrice: ethers.utils.formatUnits(gasPrice, 'gwei')
      });

      // Execute swap
      const tx = await uniswapRouter.exactInputSingle(swapParams, {
        gasLimit: gasEstimate.mul(120).div(100), // 20% buffer
        gasPrice
      });

      const receipt = await tx.wait();

      // Parse swap result from logs
      const swapEvent = receipt.logs.find((log: any) => {
        try {
          const parsed = uniswapRouter.interface.parseLog(log);
          return parsed.name === 'Swap';
        } catch {
          return false;
        }
      });

      let actualOutputAmount = minAmountOut;
      if (swapEvent) {
        const parsed = uniswapRouter.interface.parseLog(swapEvent);
        actualOutputAmount = parsed.args.amount1 || parsed.args.amount0;
      }

      // Calculate effective price and slippage
      const effectivePrice = actualOutputAmount.mul(ethers.utils.parseEther('1')).div(amountIn);
      const expectedOutput = minAmountOut;
      const slippage = Math.abs(
        parseFloat(ethers.utils.formatEther(expectedOutput.sub(actualOutputAmount))) /
        parseFloat(ethers.utils.formatEther(expectedOutput))
      ) * 100;

      const result: TradeResult = {
        transactionHash: receipt.transactionHash,
        inputAmount: amountIn,
        outputAmount: actualOutputAmount,
        gasUsed: receipt.gasUsed,
        effectivePrice,
        slippage
      };

      logger.info('Trade executed successfully', {
        transactionHash: receipt.transactionHash,
        outputAmount: ethers.utils.formatEther(actualOutputAmount),
        slippage: slippage.toFixed(3),
        gasUsed: receipt.gasUsed.toString()
      });

      return result;

    } catch (error) {
      logger.error('Trade execution failed', {
        fromToken,
        toToken,
        amountIn: ethers.utils.formatEther(amountIn),
        error: error instanceof Error ? error.message : String(error)
      });

      throw error;
    }
  }

  async updatePortfolioPositions(
    portfolioId: string,
    newPositions: PortfolioPosition[]
  ): Promise<void> {
    try {
      const portfolio = this.portfolioDatabase.get(portfolioId);
      if (!portfolio) {
        throw new Error(`Portfolio ${portfolioId} not found`);
      }

      const previousValue = portfolio.totalValue;
      
      // Update positions
      portfolio.positions = newPositions;
      portfolio.totalValue = newPositions.reduce((sum, pos) => sum.add(pos.value), BigNumber.from(0));
      portfolio.totalValueUsd = parseFloat(ethers.utils.formatEther(portfolio.totalValue));
      portfolio.updatedAt = Date.now();

      // Update performance metrics
      const valueChange = portfolio.totalValue.sub(previousValue);
      const returnPercentage = previousValue.gt(0) 
        ? parseFloat(ethers.utils.formatEther(valueChange.mul(10000).div(previousValue))) / 100
        : 0;

      portfolio.performance.totalReturn = portfolio.performance.totalReturn.add(valueChange);
      portfolio.performance.totalReturnPercentage += returnPercentage;

      // Store updated portfolio
      this.portfolioDatabase.set(portfolioId, portfolio);

      logger.info('Portfolio positions updated', {
        portfolioId,
        newValue: ethers.utils.formatEther(portfolio.totalValue),
        valueChange: ethers.utils.formatEther(valueChange),
        returnPercentage: returnPercentage.toFixed(2)
      });

    } catch (error) {
      logger.error('Failed to update portfolio positions', {
        portfolioId,
        error: error instanceof Error ? error.message : String(error)
      });

      throw error;
    }
  }

  async registerRebalanceUpkeep(
    portfolioId: string,
    checkFrequency: number,
    gasLimit: number
  ): Promise<string> {
    try {
      const portfolio = this.portfolioDatabase.get(portfolioId);
      if (!portfolio) {
        throw new Error(`Portfolio ${portfolioId} not found`);
      }

      const automationRegistry = this.contracts.get(`automation_registry_${portfolio.chainId}`);
      if (!automationRegistry) {
        throw new Error(`Automation registry not available for chain ${portfolio.chainId}`);
      }

      const portfolioRebalancer = this.contracts.get(`portfolio_rebalancer_${portfolio.chainId}`);
      if (!portfolioRebalancer) {
        throw new Error(`Portfolio rebalancer not available for chain ${portfolio.chainId}`);
      }

      // Encode check data with portfolio information
      const checkData = ethers.utils.defaultAbiCoder.encode(
        ['string', 'uint256', 'uint256'],
        [portfolioId, PORTFOLIO_THRESHOLDS.REBALANCE_THRESHOLD * 100, checkFrequency]
      );

      // Register upkeep
      const tx = await automationRegistry.registerUpkeep(
        portfolioRebalancer.address,
        gasLimit,
        this.wallets.get(portfolio.chainId)!.address,
        checkData
      );

      const receipt = await tx.wait();

      // Extract upkeep ID from logs
      let upkeepId = '';
      for (const log of receipt.logs) {
        try {
          const parsed = automationRegistry.interface.parseLog(log);
          if (parsed.name === 'UpkeepRegistered') {
            upkeepId = parsed.args.id.toString();
            break;
          }
        } catch {
          continue;
        }
      }

      if (!upkeepId) {
        throw new Error('Failed to extract upkeep ID from transaction');
      }

      // Update portfolio with upkeep ID
      portfolio.rebalancing.automationUpkeepId = upkeepId;
      this.portfolioDatabase.set(portfolioId, portfolio);

      logger.info('Rebalance upkeep registered', {
        portfolioId,
        upkeepId,
        checkFrequency,
        gasLimit
      });

      return upkeepId;

    } catch (error) {
      logger.error('Failed to register rebalance upkeep', {
        portfolioId,
        error: error instanceof Error ? error.message : String(error)
      });

      throw error;
    }
  }

  async addUpkeepFunds(upkeepId: string, amount: BigNumber, chainId: number): Promise<void> {
    try {
      const automationRegistry = this.contracts.get(`automation_registry_${chainId}`);
      if (!automationRegistry) {
        throw new Error(`Automation registry not available for chain ${chainId}`);
      }

      const tx = await automationRegistry.addFunds(upkeepId, amount);
      await tx.wait();

      logger.info('Upkeep funds added', {
        upkeepId,
        amount: ethers.utils.formatEther(amount),
        chainId
      });

    } catch (error) {
      logger.error('Failed to add upkeep funds', {
        upkeepId,
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });

      throw error;
    }
  }

  async getPortfolioPerformance(portfolioId: string, period?: string): Promise<any> {
    try {
      const portfolio = this.portfolioDatabase.get(portfolioId);
      if (!portfolio) {
        throw new Error(`Portfolio ${portfolioId} not found`);
      }

      // In production, this would fetch detailed performance data from storage
      return {
        totalReturn: portfolio.performance.totalReturn,
        totalReturnPercentage: portfolio.performance.totalReturnPercentage,
        annualizedReturn: portfolio.performance.annualizedReturn,
        sharpeRatio: portfolio.performance.sharpeRatio,
        maxDrawdown: portfolio.performance.maxDrawdown,
        volatility: portfolio.performance.volatility,
        lastUpdated: portfolio.updatedAt
      };

    } catch (error) {
      logger.error('Failed to get portfolio performance', {
        portfolioId,
        error: error instanceof Error ? error.message : String(error)
      });

      throw error;
    }
  }

  // Private helper methods
  private async updatePortfolioValues(portfolio: Portfolio): Promise<Portfolio> {
    try {
      let totalValue = BigNumber.from(0);

      for (const position of portfolio.positions) {
        const currentPrice = await this.getTokenPrice(position.token.symbol, portfolio.chainId);
        const newValue = position.amount.mul(currentPrice).div(
          ethers.utils.parseUnits('1', position.token.decimals)
        );
        
        position.value = newValue;
        position.valueUsd = parseFloat(ethers.utils.formatEther(newValue));
        position.lastUpdated = Date.now();
        
        totalValue = totalValue.add(newValue);
      }

      // Update percentages
      portfolio.positions.forEach(position => {
        position.percentage = totalValue.gt(0) 
          ? parseFloat(ethers.utils.formatEther(position.value.mul(10000).div(totalValue))) / 100
          : 0;
      });

      portfolio.totalValue = totalValue;
      portfolio.totalValueUsd = parseFloat(ethers.utils.formatEther(totalValue));
      portfolio.updatedAt = Date.now();

      return portfolio;

    } catch (error) {
      logger.error('Failed to update portfolio values', {
        portfolioId: portfolio.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return portfolio;
    }
  }

  private async getTokenPrice(symbol: string, chainId: number): Promise<BigNumber> {
    try {
      // In production, this would use the MarketDataProvider
      // For now, return mock prices
      const mockPrices: Record<string, string> = {
        'ETH': '2000',
        'BTC': '45000',
        'LINK': '15',
        'USDC': '1',
        'USDT': '1',
        'MATIC': '0.8',
        'AVAX': '20',
        'UNI': '6',
        'AAVE': '100',
        'COMP': '50'
      };

      const price = mockPrices[symbol.toUpperCase()] || '1';
      return ethers.utils.parseEther(price);

    } catch (error) {
      logger.error('Failed to get token price', {
        symbol,
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });

      return ethers.utils.parseEther('1'); // Fallback price
    }
  }

  private async getTokenAddress(symbol: string, chainId: number): Promise<string | null> {
    // Token address mapping by chain
    const tokenAddresses: Record<number, Record<string, string>> = {
      1: { // Ethereum
        'ETH': ethers.constants.AddressZero,
        'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        'USDC': '0xA0b86a33E6427d6D87EC7A7B7EEA7a3a7A6FE1a7',
        'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
        'LINK': '0x514910771AF9Ca656af840dff83E8264EcF986CA',
        'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
        'AAVE': '0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9'
      },
      137: { // Polygon
        'MATIC': ethers.constants.AddressZero,
        'WMATIC': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
        'USDC': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
        'USDT': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
        'ETH': '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619'
      },
      42161: { // Arbitrum
        'ETH': ethers.constants.AddressZero,
        'WETH': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
        'USDC': '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',
        'USDT': '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9'
      },
      43114: { // Avalanche
        'AVAX': ethers.constants.AddressZero,
        'WAVAX': '0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7',
        'USDC': '0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E',
        'USDT': '0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7'
      }
    };

    return tokenAddresses[chainId]?.[symbol.toUpperCase()] || null;
  }

  private async ensureTokenApproval(
    tokenAddress: string,
    spenderAddress: string,
    amount: BigNumber,
    chainId: number
  ): Promise<void> {
    try {
      if (tokenAddress === ethers.constants.AddressZero) {
        return; // Native token doesn't need approval
      }

      const wallet = this.wallets.get(chainId)!;
      const tokenContract = new Contract(tokenAddress, this.ERC20_ABI, wallet);
      
      const currentAllowance = await tokenContract.allowance(wallet.address, spenderAddress);
      
      if (currentAllowance.lt(amount)) {
        const tx = await tokenContract.approve(spenderAddress, ethers.constants.MaxUint256);
        await tx.wait();

        logger.debug('Token approval completed', {
          tokenAddress,
          spenderAddress,
          chainId
        });
      }

    } catch (error) {
      logger.error('Token approval failed', {
        tokenAddress,
        spenderAddress,
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });

      throw error;
    }
  }

  private async getOptimalGasPrice(chainId: number, gasOptimization: boolean): Promise<BigNumber> {
    try {
      const provider = this.providers.get(chainId)!;
      const feeData = await provider.getFeeData();
      
      if (gasOptimization && feeData.maxFeePerGas) {
        // Use EIP-1559 with optimization
        return feeData.maxFeePerGas.mul(90).div(100); // 10% reduction
      }
      
      return feeData.gasPrice || BigNumber.from('20000000000'); // 20 gwei fallback

    } catch (error) {
      logger.error('Failed to get optimal gas price', {
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });

      return BigNumber.from('20000000000'); // 20 gwei fallback
    }
  }

  // Public utility methods
  async deletePortfolio(portfolioId: string): Promise<void> {
    const portfolio = this.portfolioDatabase.get(portfolioId);
    if (portfolio) {
      this.portfolioDatabase.delete(portfolioId);
      
      logger.info('Portfolio deleted', { portfolioId });
    }
  }

  async getPortfolioSummary(): Promise<{
    totalPortfolios: number;
    totalValue: BigNumber;
    averagePerformance: number;
    topPerformers: Array<{ id: string; performance: number }>;
  }> {
    const portfolios = Array.from(this.portfolioDatabase.values());
    
    const totalValue = portfolios.reduce((sum, p) => sum.add(p.totalValue), BigNumber.from(0));
    const avgPerformance = portfolios.length > 0 
      ? portfolios.reduce((sum, p) => sum + p.performance.totalReturnPercentage, 0) / portfolios.length
      : 0;

    const topPerformers = portfolios
      .sort((a, b) => b.performance.totalReturnPercentage - a.performance.totalReturnPercentage)
      .slice(0, 5)
      .map(p => ({ id: p.id, performance: p.performance.totalReturnPercentage }));

    return {
      totalPortfolios: portfolios.length,
      totalValue,
      averagePerformance: avgPerformance,
      topPerformers
    };
  }

  clearCache(): void {
    // Clear any cached data if needed
    logger.debug('Portfolio provider cache cleared');
  }
}
