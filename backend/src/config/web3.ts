import { ethers } from 'ethers';
import { logger } from '../utils/logger';
import { chainlinkConfig } from './chainlink';

interface Web3NetworkInfo {
  name: string;
  chainId: number;
  rpcUrl: string;
  blockExplorer: string;
  nativeCurrency: {
    name: string;
    symbol: string;
    decimals: number;
  };
  testnet: boolean;
  gasPrice?: {
    slow: number;
    standard: number;
    fast: number;
  };
}

interface ContractAddresses {
  [contractName: string]: {
    [chainId: number]: string;
  };
}

interface SignerConfig {
  [purpose: string]: {
    privateKey: string;
    supportedChains: number[];
  };
}

class Web3Config {
  private providers: Map<number, ethers.providers.JsonRpcProvider> = new Map();
  private signers: Map<string, Map<number, ethers.Wallet>> = new Map();
  private networkInfo: Map<number, Web3NetworkInfo> = new Map();
  private contractAddresses: ContractAddresses;
  private signerConfig: SignerConfig;

  constructor() {
    this.initializeNetworkInfo();
    this.contractAddresses = this.loadContractAddresses();
    this.signerConfig = this.loadSignerConfiguration();
    this.initializeProviders();
    this.initializeSigners();
  }

  /**
   * Initialize network information based on official documentation
   */
  private initializeNetworkInfo(): void {
    const networks: Web3NetworkInfo[] = [
      {
        name: 'Ethereum Mainnet',
        chainId: 1,
        rpcUrl: process.env.ETHEREUM_RPC_URL || 'https://eth-mainnet.g.alchemy.com/v2/your-api-key',
        blockExplorer: 'https://etherscan.io',
        nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
        testnet: false,
        gasPrice: { slow: 20, standard: 25, fast: 30 }
      },
      {
        name: 'Ethereum Sepolia',
        chainId: 11155111,
        rpcUrl: process.env.SEPOLIA_RPC_URL || 'https://eth-sepolia.g.alchemy.com/v2/your-api-key',
        blockExplorer: 'https://sepolia.etherscan.io',
        nativeCurrency: { name: 'Sepolia Ether', symbol: 'SepoliaETH', decimals: 18 },
        testnet: true,
        gasPrice: { slow: 1, standard: 2, fast: 3 }
      },
      {
        name: 'Avalanche C-Chain',
        chainId: 43114,
        rpcUrl: process.env.AVALANCHE_RPC_URL || 'https://api.avax.network/ext/bc/C/rpc',
        blockExplorer: 'https://snowtrace.io',
        nativeCurrency: { name: 'Avalanche', symbol: 'AVAX', decimals: 18 },
        testnet: false,
        gasPrice: { slow: 25, standard: 30, fast: 35 }
      },
      {
        name: 'Avalanche Fuji',
        chainId: 43113,
        rpcUrl: process.env.FUJI_RPC_URL || 'https://api.avax-test.network/ext/bc/C/rpc',
        blockExplorer: 'https://testnet.snowtrace.io',
        nativeCurrency: { name: 'Avalanche', symbol: 'AVAX', decimals: 18 },
        testnet: true,
        gasPrice: { slow: 25, standard: 30, fast: 35 }
      },
      {
        name: 'Polygon Mainnet',
        chainId: 137,
        rpcUrl: process.env.POLYGON_RPC_URL || 'https://polygon-rpc.com',
        blockExplorer: 'https://polygonscan.com',
        nativeCurrency: { name: 'Polygon', symbol: 'MATIC', decimals: 18 },
        testnet: false,
        gasPrice: { slow: 30, standard: 35, fast: 40 }
      },
      {
        name: 'Polygon Mumbai',
        chainId: 80001,
        rpcUrl: process.env.MUMBAI_RPC_URL || 'https://rpc-mumbai.maticvigil.com',
        blockExplorer: 'https://mumbai.polygonscan.com',
        nativeCurrency: { name: 'Polygon', symbol: 'MATIC', decimals: 18 },
        testnet: true,
        gasPrice: { slow: 1, standard: 2, fast: 3 }
      },
      {
        name: 'Arbitrum One',
        chainId: 42161,
        rpcUrl: process.env.ARBITRUM_RPC_URL || 'https://arb1.arbitrum.io/rpc',
        blockExplorer: 'https://arbiscan.io',
        nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
        testnet: false,
        gasPrice: { slow: 0.1, standard: 0.2, fast: 0.3 }
      },
      {
        name: 'Arbitrum Goerli',
        chainId: 421613,
        rpcUrl: process.env.ARBITRUM_GOERLI_RPC_URL || 'https://goerli-rollup.arbitrum.io/rpc',
        blockExplorer: 'https://testnet.arbiscan.io',
        nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
        testnet: true,
        gasPrice: { slow: 0.1, standard: 0.2, fast: 0.3 }
      },
      {
        name: 'Localhost',
        chainId: 31337,
        rpcUrl: 'http://localhost:8545',
        blockExplorer: 'http://localhost:8545',
        nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
        testnet: true,
        gasPrice: { slow: 1, standard: 2, fast: 3 }
      }
    ];

    // Store network info in map for quick access
    networks.forEach(network => {
      this.networkInfo.set(network.chainId, network);
    });
  }

  /**
   * Load contract addresses configuration
   */
  private loadContractAddresses(): ContractAddresses {
    return {
      // ArbOS Core Contracts
      arbitrageManager: {
        1: process.env.ARBITRAGE_MANAGER_MAINNET || '',
        11155111: process.env.ARBITRAGE_MANAGER_SEPOLIA || '',
        43114: process.env.ARBITRAGE_MANAGER_AVALANCHE || '',
        43113: process.env.ARBITRAGE_MANAGER_FUJI || '',
        137: process.env.ARBITRAGE_MANAGER_POLYGON || '',
        42161: process.env.ARBITRAGE_MANAGER_ARBITRUM || ''
      },
      portfolioManager: {
        1: process.env.PORTFOLIO_MANAGER_MAINNET || '',
        11155111: process.env.PORTFOLIO_MANAGER_SEPOLIA || '',
        43114: process.env.PORTFOLIO_MANAGER_AVALANCHE || '',
        43113: process.env.PORTFOLIO_MANAGER_FUJI || '',
        137: process.env.PORTFOLIO_MANAGER_POLYGON || '',
        42161: process.env.PORTFOLIO_MANAGER_ARBITRUM || ''
      },
      agentFactory: {
        1: process.env.AGENT_FACTORY_MAINNET || '',
        11155111: process.env.AGENT_FACTORY_SEPOLIA || '',
        43114: process.env.AGENT_FACTORY_AVALANCHE || '',
        43113: process.env.AGENT_FACTORY_FUJI || '',
        137: process.env.AGENT_FACTORY_POLYGON || '',
        42161: process.env.AGENT_FACTORY_ARBITRUM || ''
      },
      securityModule: {
        1: process.env.SECURITY_MODULE_MAINNET || '',
        11155111: process.env.SECURITY_MODULE_SEPOLIA || '',
        43114: process.env.SECURITY_MODULE_AVALANCHE || '',
        43113: process.env.SECURITY_MODULE_FUJI || '',
        137: process.env.SECURITY_MODULE_POLYGON || '',
        42161: process.env.SECURITY_MODULE_ARBITRUM || ''
      },
      // DEX Router Addresses
      uniswapV3Router: {
        1: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
        11155111: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
        137: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
        42161: '0xE592427A0AEce92De3Edee1F18E0157C05861564'
      },
      traderJoeRouter: {
        43114: '0x60aE616a2155Ee3d9A68541Ba4544862310933d4',
        43113: '0x60aE616a2155Ee3d9A68541Ba4544862310933d4'
      },
      pangolinRouter: {
        43114: '0xE54Ca86531e17Ef3616d22Ca28b0D458b6C89106',
        43113: '0xE54Ca86531e17Ef3616d22Ca28b0D458b6C89106'
      },
      quickSwapRouter: {
        137: '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff'
      },
      // Token Addresses
      weth: {
        1: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        11155111: '0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14',
        137: '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619',
        42161: '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1'
      },
      wavax: {
        43114: '0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7',
        43113: '0xd00ae08403B9bbb9124bB305C09058E32C39A48c'
      },
      wmatic: {
        137: '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
        80001: '0x9c3C9283D3e44854697Cd22D3Faa240Cfb032889'
      },
      usdc: {
        1: '0xA0b86a33E6441caa7e1f56fec8eaecE10e61E7a11',
        11155111: '0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238',
        43114: '0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E',
        43113: '0x5425890298aed601595a70AB815c96711a31Bc65',
        137: '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
        42161: '0xaf88d065e77c8cC2239327C5EDb3A432268e5831'
      },
      usdt: {
        1: '0xdAC17F958D2ee523a2206206994597C13D831ec7',
        43114: '0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7',
        137: '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
        42161: '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9'
      }
    };
  }

  /**
   * Load signer configuration from environment
   */
  private loadSignerConfiguration(): SignerConfig {
    return {
      arbitrage: {
        privateKey: process.env.ARBITRAGE_PRIVATE_KEY || '',
        supportedChains: [1, 11155111, 43114, 43113, 137, 42161]
      },
      portfolio: {
        privateKey: process.env.PORTFOLIO_PRIVATE_KEY || '',
        supportedChains: [1, 11155111, 43114, 43113, 137, 42161]
      },
      agent: {
        privateKey: process.env.AGENT_PRIVATE_KEY || '',
        supportedChains: [1, 11155111, 43114, 43113, 137, 42161]
      },
      security: {
        privateKey: process.env.SECURITY_PRIVATE_KEY || '',
        supportedChains: [1, 11155111, 43114, 43113, 137, 42161]
      },
      ccip: {
        privateKey: process.env.CCIP_PRIVATE_KEY || '',
        supportedChains: [1, 11155111, 43114, 43113, 137, 42161]
      },
      functions: {
        privateKey: process.env.FUNCTIONS_PRIVATE_KEY || '',
        supportedChains: [11155111, 43113, 80001]
      },
      vrf: {
        privateKey: process.env.VRF_PRIVATE_KEY || '',
        supportedChains: [1, 11155111, 43114, 43113, 137, 42161]
      },
      automation: {
        privateKey: process.env.AUTOMATION_PRIVATE_KEY || '',
        supportedChains: [1, 11155111, 43114, 43113, 137, 42161]
      }
    };
  }

  /**
   * Initialize providers for all supported networks
   */
  private initializeProviders(): void {
    const chainlinkNetworks = chainlinkConfig.getAllSupportedChains();
    
    for (const chainId of chainlinkNetworks) {
      const networkInfo = this.networkInfo.get(chainId);
      if (!networkInfo) {
        logger.warn(`Network info not found for chain ${chainId}`);
        continue;
      }

      try {
        const provider = new ethers.providers.JsonRpcProvider({
          url: networkInfo.rpcUrl,
          timeout: 30000
        }, {
          chainId: networkInfo.chainId,
          name: networkInfo.name
        });

        // Configure provider with retry logic
        provider.pollingInterval = 4000; // 4 seconds
        
        this.providers.set(chainId, provider);
        
        logger.debug(`Provider initialized for ${networkInfo.name}`, {
          chainId,
          rpcUrl: networkInfo.rpcUrl.split('/').pop() // Log only the last part for security
        });
      } catch (error) {
        logger.error(`Failed to initialize provider for chain ${chainId}:`, error);
      }
    }
  }

  /**
   * Initialize signers for all purposes and supported chains
   */
  private initializeSigners(): void {
    for (const [purpose, config] of Object.entries(this.signerConfig)) {
      if (!config.privateKey) {
        logger.warn(`No private key configured for ${purpose} signer`);
        continue;
      }

      const signerMap = new Map<number, ethers.Wallet>();

      for (const chainId of config.supportedChains) {
        const provider = this.providers.get(chainId);
        if (!provider) {
          logger.warn(`No provider available for ${purpose} signer on chain ${chainId}`);
          continue;
        }

        try {
          const wallet = new ethers.Wallet(config.privateKey, provider);
          signerMap.set(chainId, wallet);
          
          logger.debug(`Signer initialized for ${purpose} on chain ${chainId}`, {
            address: wallet.address
          });
        } catch (error) {
          logger.error(`Failed to initialize ${purpose} signer for chain ${chainId}:`, error);
        }
      }

      this.signers.set(purpose, signerMap);
    }
  }

  /**
   * Get provider for specific chain
   */
  public getProvider(chainId: number): ethers.providers.JsonRpcProvider | null {
    return this.providers.get(chainId) || null;
  }

  /**
   * Get signer for specific purpose and chain
   */
  public getSignerForChain(purpose: string, chainId: number): ethers.Wallet | null {
    const purposeSigners = this.signers.get(purpose);
    if (!purposeSigners) {
      logger.warn(`No signers configured for purpose: ${purpose}`);
      return null;
    }

    return purposeSigners.get(chainId) || null;
  }

  /**
   * Get all signers for a specific purpose
   */
  public getSignersForPurpose(purpose: string): Map<number, ethers.Wallet> | null {
    return this.signers.get(purpose) || null;
  }

  /**
   * Get network information
   */
  public getNetworkInfo(chainId: number): Web3NetworkInfo | null {
    return this.networkInfo.get(chainId) || null;
  }

  /**
   * Get contract address
   */
  public getContractAddress(contractName: string, chainId: number): string | null {
    return this.contractAddresses[contractName]?.[chainId] || null;
  }

  /**
   * Get gas price for chain
   */
  public async getGasPrice(chainId: number, speed: 'slow' | 'standard' | 'fast' = 'standard'): Promise<ethers.BigNumber> {
    const provider = this.getProvider(chainId);
    if (!provider) {
      throw new Error(`No provider available for chain ${chainId}`);
    }

    try {
      const gasPrice = await provider.getGasPrice();
      const networkInfo = this.getNetworkInfo(chainId);
      
      if (networkInfo?.gasPrice) {
        const multiplier = {
          slow: 0.8,
          standard: 1.0,
          fast: 1.2
        }[speed];
        
        return gasPrice.mul(Math.floor(multiplier * 100)).div(100);
      }

      return gasPrice;
    } catch (error) {
      logger.error(`Error getting gas price for chain ${chainId}:`, error);
      throw error;
    }
  }

  /**
   * Get current block number
   */
  public async getBlockNumber(chainId: number): Promise<number> {
    const provider = this.getProvider(chainId);
    if (!provider) {
      throw new Error(`No provider available for chain ${chainId}`);
    }

    return await provider.getBlockNumber();
  }

  /**
   * Get transaction receipt
   */
  public async getTransactionReceipt(chainId: number, txHash: string): Promise<ethers.providers.TransactionReceipt | null> {
    const provider = this.getProvider(chainId);
    if (!provider) {
      throw new Error(`No provider available for chain ${chainId}`);
    }

    return await provider.getTransactionReceipt(txHash);
  }

  /**
   * Wait for transaction confirmation
   */
  public async waitForTransaction(
    chainId: number, 
    txHash: string, 
    confirmations: number = 1,
    timeout: number = 60000
  ): Promise<ethers.providers.TransactionReceipt> {
    const provider = this.getProvider(chainId);
    if (!provider) {
      throw new Error(`No provider available for chain ${chainId}`);
    }

    return await provider.waitForTransaction(txHash, confirmations, timeout);
  }

  /**
   * Get balance for address
   */
  public async getBalance(purpose: string, chainId: number, tokenAddress?: string): Promise<ethers.BigNumber> {
    const signer = this.getSignerForChain(purpose, chainId);
    if (!signer) {
      throw new Error(`No signer available for ${purpose} on chain ${chainId}`);
    }

    if (!tokenAddress) {
      // Get native token balance
      return await signer.getBalance();
    }

    // Get ERC20 token balance
    const tokenContract = new ethers.Contract(
      tokenAddress,
      ['function balanceOf(address) view returns (uint256)'],
      signer
    );

    return await tokenContract.balanceOf(signer.address);
  }

  /**
   * Get nonce for signer
   */
  public async getNonce(purpose: string, chainId: number): Promise<number> {
    const signer = this.getSignerForChain(purpose, chainId);
    if (!signer) {
      throw new Error(`No signer available for ${purpose} on chain ${chainId}`);
    }

    return await signer.getTransactionCount('pending');
  }

  /**
   * Estimate gas for transaction
   */
  public async estimateGas(
    purpose: string,
    chainId: number,
    transaction: ethers.providers.TransactionRequest
  ): Promise<ethers.BigNumber> {
    const signer = this.getSignerForChain(purpose, chainId);
    if (!signer) {
      throw new Error(`No signer available for ${purpose} on chain ${chainId}`);
    }

    return await signer.estimateGas(transaction);
  }

  /**
   * Send transaction
   */
  public async sendTransaction(
    purpose: string,
    chainId: number,
    transaction: ethers.providers.TransactionRequest
  ): Promise<ethers.providers.TransactionResponse> {
    const signer = this.getSignerForChain(purpose, chainId);
    if (!signer) {
      throw new Error(`No signer available for ${purpose} on chain ${chainId}`);
    }

    // Add gas estimation if not provided
    if (!transaction.gasLimit) {
      transaction.gasLimit = await this.estimateGas(purpose, chainId, transaction);
    }

    // Add gas price if not provided
    if (!transaction.gasPrice) {
      transaction.gasPrice = await this.getGasPrice(chainId);
    }

    return await signer.sendTransaction(transaction);
  }

  /**
   * Check if chain is supported
   */
  public isChainSupported(chainId: number): boolean {
    return this.providers.has(chainId);
  }

  /**
   * Get all supported chain IDs
   */
  public getSupportedChains(): number[] {
    return Array.from(this.providers.keys());
  }

  /**
   * Get signer address for purpose
   */
  public getSignerAddress(purpose: string): string | null {
    const purposeSigners = this.signers.get(purpose);
    if (!purposeSigners || purposeSigners.size === 0) {
      return null;
    }

    // Return address from first available signer (all should have same address)
    const firstSigner = purposeSigners.values().next().value;
    return firstSigner ? firstSigner.address : null;
  }

  /**
   * Health check for providers
   */
  public async healthCheck(): Promise<{ [chainId: number]: boolean }> {
    const health: { [chainId: number]: boolean } = {};

    await Promise.all(
      Array.from(this.providers.entries()).map(async ([chainId, provider]) => {
        try {
          await provider.getBlockNumber();
          health[chainId] = true;
        } catch (error) {
          logger.warn(`Health check failed for chain ${chainId}:`, error);
          health[chainId] = false;
        }
      })
    );

    return health;
  }

  /**
   * Get configuration for logging (without sensitive data)
   */
  public getConfigForLogging(): any {
    return {
      supportedChains: this.getSupportedChains(),
      providersCount: this.providers.size,
      signersCount: this.signers.size,
      configuredPurposes: Array.from(this.signers.keys()),
      contractsConfigured: Object.keys(this.contractAddresses).length
    };
  }
}

// Create singleton instance
export const web3Config = new Web3Config();

// Export configuration class
export { Web3Config };

// Export types
export type { Web3NetworkInfo, ContractAddresses, SignerConfig };

// Default export
export default web3Config;
