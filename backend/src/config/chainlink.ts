import { logger } from '../utils/logger';

interface ChainlinkNetworkConfig {
  name: string;
  enabled: boolean;
  rpcUrl: string;
  chainId: number;
  linkToken: string;
  router?: string;
  coordinator?: string;
  registrar?: string;
  registry?: string;
  gasLane?: string;
  subscriptionId?: string;
  callbackGasLimit?: number;
  requestConfirmations?: number;
  numWords?: number;
  donId?: string;
  gatewayUrls?: string[];
}

interface ChainlinkServiceConfig {
  dataFeeds: {
    enabled: boolean;
    feeds: { [pair: string]: { [chainId: number]: string } };
    heartbeatToleranceMs: number;
    priceDeviationThreshold: number;
  };
  automation: {
    enabled: boolean;
    registryAddress: { [chainId: number]: string };
    registrarAddress: { [chainId: number]: string };
    linkAddress: { [chainId: number]: string };
    minUpkeepSpend: string;
    maxUpkeepSpend: string;
    gasLimit: number;
  };
  vrf: {
    enabled: boolean;
    coordinatorAddress: { [chainId: number]: string };
    keyHash: { [chainId: number]: string };
    subscriptionId: string;
    requestConfirmations: number;
    callbackGasLimit: number;
    numWords: number;
  };
  functions: {
    enabled: boolean;
    routerAddress: { [chainId: number]: string };
    donId: string;
    subscriptionId: string;
    gasLimit: number;
    secrets: {
      slotId: number;
      version: number;
    };
  };
  ccip: {
    enabled: boolean;
    router: { [chainId: number]: string };
    supportedChains: number[];
    gasLimit: number;
    extraArgs: string;
  };
}

interface ChainlinkConfiguration {
  networks: { [chainId: string]: ChainlinkNetworkConfig };
  services: ChainlinkServiceConfig;
  global: {
    environment: 'mainnet' | 'testnet' | 'local';
    apiTimeout: number;
    retryAttempts: number;
    retryDelay: number;
  };
}

class ChainlinkConfig {
  private config: ChainlinkConfiguration;

  constructor() {
    this.config = this.loadChainlinkConfiguration();
    this.validateConfiguration();
  }

  /**
   * Load Chainlink configuration based on official documentation
   */
  private loadChainlinkConfiguration(): ChainlinkConfiguration {
    const environment = (process.env.CHAINLINK_ENV || 'testnet') as 'mainnet' | 'testnet' | 'local';

    return {
      networks: this.getNetworkConfigurations(environment),
      services: this.getServiceConfigurations(environment),
      global: {
        environment,
        apiTimeout: parseInt(process.env.CHAINLINK_API_TIMEOUT || '30000'),
        retryAttempts: parseInt(process.env.CHAINLINK_RETRY_ATTEMPTS || '3'),
        retryDelay: parseInt(process.env.CHAINLINK_RETRY_DELAY || '1000')
      }
    };
  }

  /**
   * Get network configurations based on official Chainlink documentation
   */
  private getNetworkConfigurations(environment: string): { [chainId: string]: ChainlinkNetworkConfig } {
    const networks: { [chainId: string]: ChainlinkNetworkConfig } = {};

    if (environment === 'mainnet' || environment === 'testnet') {
      // Ethereum Mainnet
      networks['1'] = {
        name: 'Ethereum Mainnet',
        enabled: environment === 'mainnet',
        rpcUrl: process.env.ETHEREUM_RPC_URL || 'https://eth-mainnet.g.alchemy.com/v2/your-api-key',
        chainId: 1,
        linkToken: '0x514910771AF9Ca656af840dff83E8264EcF986CA',
        router: '0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D', // CCIP Router
        coordinator: '0x271682DEB8C4E0901D1a1550aD2e64D568E69909', // VRF Coordinator
        registrar: '0xDb8e8e2ccb5C033938736aa89Fe104b6e7EFE863', // Automation Registrar
        registry: '0x02777053d6764996e594c3E88AF1D58D5363a2e6', // Automation Registry
        gasLane: '0x8af398995b04c28e9951adb9721ef74c74f93e6a478f39e7e0777be13527e7ef',
        subscriptionId: process.env.VRF_SUBSCRIPTION_ID_MAINNET,
        callbackGasLimit: 100000,
        requestConfirmations: 3,
        numWords: 1
      };

      // Ethereum Sepolia Testnet
      networks['11155111'] = {
        name: 'Ethereum Sepolia',
        enabled: environment === 'testnet',
        rpcUrl: process.env.SEPOLIA_RPC_URL || 'https://eth-sepolia.g.alchemy.com/v2/your-api-key',
        chainId: 11155111,
        linkToken: '0x779877A7B0D9E8603169DdbD7836e478b4624789',
        router: '0x0BF3dE8c5D3e8A2B34D2BEeB17ABfCeBaf363A59', // CCIP Router
        coordinator: '0x8103B0A8A00be2DDC778e6e7eaa21791Cd364625', // VRF Coordinator
        registrar: '0xb0E49c5D0d05cbc241d68c05BC5BA1d1B7B72976', // Automation Registrar
        registry: '0x86EFBD0b6736Bed994962f9797049422A3A8E8Ad', // Automation Registry
        gasLane: '0x474e34a077df58807dbe9c96d3c009b23b3c6d0cce433e59bbf5b34f823bc56c',
        subscriptionId: process.env.VRF_SUBSCRIPTION_ID_SEPOLIA,
        callbackGasLimit: 100000,
        requestConfirmations: 3,
        numWords: 1,
        donId: 'fun-ethereum-sepolia-1'
      };

      // Avalanche Mainnet
      networks['43114'] = {
        name: 'Avalanche Mainnet',
        enabled: true,
        rpcUrl: process.env.AVALANCHE_RPC_URL || 'https://api.avax.network/ext/bc/C/rpc',
        chainId: 43114,
        linkToken: '0x5947BB275c521040051D82396192181b413227A3',
        router: '0xF4c7E640EdA248ef95972845a62bdC74237805dB', // CCIP Router
        coordinator: '0xd5D517aBE5cF79B7e95eC98dB0f0277788aFF634', // VRF Coordinator
        gasLane: '0x06eb0e2ea7cca202fc7c8258397a36f33d6d52c0ea43ae2fc61a4c8c01fae1e8',
        subscriptionId: process.env.VRF_SUBSCRIPTION_ID_AVALANCHE,
        callbackGasLimit: 100000,
        requestConfirmations: 1,
        numWords: 1,
        donId: 'fun-avalanche-mainnet-1'
      };

      // Avalanche Fuji Testnet
      networks['43113'] = {
        name: 'Avalanche Fuji',
        enabled: true,
        rpcUrl: process.env.FUJI_RPC_URL || 'https://api.avax-test.network/ext/bc/C/rpc',
        chainId: 43113,
        linkToken: '0x0b9d5D9136855f6FEc3c0993feE6E9CE8a297846',
        router: '0xF694E193200268f9a4868e4Aa017A0118C9a8177', // CCIP Router
        coordinator: '0x2eD832Ba664535e5886b75D64C46EB9a228C2610', // VRF Coordinator
        registrar: '0x819B58A646CDd8289275A87653a2aA4902b14fe6', // Automation Registrar
        registry: '0x819B58A646CDd8289275A87653a2aA4902b14fe6', // Automation Registry
        gasLane: '0x354d2f95da55398f44b7cff77da56283d9c6c829a4bdf1bbcaf2ad6a4d081f61',
        subscriptionId: process.env.VRF_SUBSCRIPTION_ID_FUJI,
        callbackGasLimit: 100000,
        requestConfirmations: 1,
        numWords: 1,
        donId: 'fun-avalanche-fuji-1'
      };

      // Polygon Mainnet
      networks['137'] = {
        name: 'Polygon Mainnet',
        enabled: environment === 'mainnet',
        rpcUrl: process.env.POLYGON_RPC_URL || 'https://polygon-rpc.com',
        chainId: 137,
        linkToken: '0xb0897686c545045aFc77CF20eC7A532E3120E0F1',
        router: '0x849c5ED5a80F5B408Dd4969b78c2C8fdf0565Bfe', // CCIP Router
        coordinator: '0xAE975071Be8F8eE67addBC1A82488F1C24858067', // VRF Coordinator
        gasLane: '0x6e099d640cde6de9d40ac749b4b594126b0169747122711109c9985d47751f93',
        subscriptionId: process.env.VRF_SUBSCRIPTION_ID_POLYGON,
        callbackGasLimit: 100000,
        requestConfirmations: 3,
        numWords: 1
      };

      // Arbitrum One
      networks['42161'] = {
        name: 'Arbitrum One',
        enabled: environment === 'mainnet',
        rpcUrl: process.env.ARBITRUM_RPC_URL || 'https://arb1.arbitrum.io/rpc',
        chainId: 42161,
        linkToken: '0xf97f4df75117a78c1A5a0DBb814Af92458539FB4',
        router: '0x141fa059441E0ca23ce184B6A78bafD2A517DdE8', // CCIP Router
        coordinator: '0x41034678D6C633D8a95c75e1138A360a28bA15d1', // VRF Coordinator
        gasLane: '0x68d24f9a037a649944964c2a1ebd0b2918f4a243d2a99701cc22b548cf2daff0',
        subscriptionId: process.env.VRF_SUBSCRIPTION_ID_ARBITRUM,
        callbackGasLimit: 100000,
        requestConfirmations: 1,
        numWords: 1
      };
    }

    // Local development network
    if (environment === 'local') {
      networks['31337'] = {
        name: 'Localhost',
        enabled: true,
        rpcUrl: 'http://localhost:8545',
        chainId: 31337,
        linkToken: '0x326C977E6efc84E512bB9C30f76E30c160eD06FB',
        subscriptionId: '1',
        callbackGasLimit: 100000,
        requestConfirmations: 1,
        numWords: 1
      };
    }

    return networks;
  }

  /**
   * Get service configurations based on official Chainlink services
   */
  private getServiceConfigurations(environment: string): ChainlinkServiceConfig {
    return {
      dataFeeds: {
        enabled: true,
        feeds: this.getDataFeedAddresses(environment),
        heartbeatToleranceMs: 300000, // 5 minutes
        priceDeviationThreshold: 0.5 // 0.5%
      },
      automation: {
        enabled: true,
        registryAddress: {
          1: '0x02777053d6764996e594c3E88AF1D58D5363a2e6',
          11155111: '0x86EFBD0b6736Bed994962f9797049422A3A8E8Ad',
          43114: '0x02777053d6764996e594c3E88AF1D58D5363a2e6',
          43113: '0x819B58A646CDd8289275A87653a2aA4902b14fe6',
          137: '0x02777053d6764996e594c3E88AF1D58D5363a2e6',
          42161: '0x75c0530885F385721fddA23C539AF3701d6183D4'
        },
        registrarAddress: {
          1: '0xDb8e8e2ccb5C033938736aa89Fe104b6e7EFE863',
          11155111: '0xb0E49c5D0d05cbc241d68c05BC5BA1d1B7B72976',
          43113: '0x819B58A646CDd8289275A87653a2aA4902b14fe6'
        },
        linkAddress: {
          1: '0x514910771AF9Ca656af840dff83E8264EcF986CA',
          11155111: '0x779877A7B0D9E8603169DdbD7836e478b4624789',
          43114: '0x5947BB275c521040051D82396192181b413227A3',
          43113: '0x0b9d5D9136855f6FEc3c0993feE6E9CE8a297846',
          137: '0xb0897686c545045aFc77CF20eC7A532E3120E0F1',
          42161: '0xf97f4df75117a78c1A5a0DBb814Af92458539FB4'
        },
        minUpkeepSpend: '5000000000000000000', // 5 LINK
        maxUpkeepSpend: '500000000000000000000', // 500 LINK
        gasLimit: 500000
      },
      vrf: {
        enabled: true,
        coordinatorAddress: {
          1: '0x271682DEB8C4E0901D1a1550aD2e64D568E69909',
          11155111: '0x8103B0A8A00be2DDC778e6e7eaa21791Cd364625',
          43114: '0xd5D517aBE5cF79B7e95eC98dB0f0277788aFF634',
          43113: '0x2eD832Ba664535e5886b75D64C46EB9a228C2610',
          137: '0xAE975071Be8F8eE67addBC1A82488F1C24858067',
          42161: '0x41034678D6C633D8a95c75e1138A360a28bA15d1'
        },
        keyHash: {
          1: '0x8af398995b04c28e9951adb9721ef74c74f93e6a478f39e7e0777be13527e7ef',
          11155111: '0x474e34a077df58807dbe9c96d3c009b23b3c6d0cce433e59bbf5b34f823bc56c',
          43114: '0x06eb0e2ea7cca202fc7c8258397a36f33d6d52c0ea43ae2fc61a4c8c01fae1e8',
          43113: '0x354d2f95da55398f44b7cff77da56283d9c6c829a4bdf1bbcaf2ad6a4d081f61',
          137: '0x6e099d640cde6de9d40ac749b4b594126b0169747122711109c9985d47751f93',
          42161: '0x68d24f9a037a649944964c2a1ebd0b2918f4a243d2a99701cc22b548cf2daff0'
        },
        subscriptionId: process.env.CHAINLINK_VRF_SUBSCRIPTION_ID || '1',
        requestConfirmations: 3,
        callbackGasLimit: 100000,
        numWords: 1
      },
      functions: {
        enabled: true,
        routerAddress: {
          11155111: '0xb83E47C2bC239B3bf370bc41e1459A34b41238D0',
          43113: '0xA9d587a00A31A52Ed70D6026794a8FC5E2F5dCb0',
          80001: '0x6E2dc0F9DB014aE19888F539E59285D2Ea04244C'
        },
        donId: process.env.CHAINLINK_FUNCTIONS_DON_ID || 'fun-avalanche-fuji-1',
        subscriptionId: process.env.CHAINLINK_FUNCTIONS_SUBSCRIPTION_ID || '1',
        gasLimit: 300000,
        secrets: {
          slotId: parseInt(process.env.CHAINLINK_FUNCTIONS_SLOT_ID || '0'),
          version: parseInt(process.env.CHAINLINK_FUNCTIONS_VERSION || '1')
        }
      },
      ccip: {
        enabled: true,
        router: {
          1: '0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D',
          11155111: '0x0BF3dE8c5D3e8A2B34D2BEeB17ABfCeBaf363A59',
          43114: '0xF4c7E640EdA248ef95972845a62bdC74237805dB',
          43113: '0xF694E193200268f9a4868e4Aa017A0118C9a8177',
          137: '0x849c5ED5a80F5B408Dd4969b78c2C8fdf0565Bfe',
          42161: '0x141fa059441E0ca23ce184B6A78bafD2A517DdE8'
        },
        supportedChains: [1, 11155111, 43114, 43113, 137, 42161],
        gasLimit: 2000000,
        extraArgs: '0x'
      }
    };
  }

  /**
   * Get Data Feed addresses based on official Chainlink documentation
   */
  private getDataFeedAddresses(environment: string): { [pair: string]: { [chainId: number]: string } } {
    return {
      'ETH/USD': {
        1: '0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419',
        11155111: '0x694AA1769357215DE4FAC081bf1f309aDC325306',
        43114: '0x976B3D034E162d8bD72D6b9C989d545b839003b0',
        43113: '0x86d67c3D38D2bCeE722E601025C25a575021c6EA',
        137: '0xF9680D99D6C9589e2a93a78A04A279e509205945',
        42161: '0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612'
      },
      'BTC/USD': {
        1: '0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c',
        11155111: '0x1b44F3514812d835EB1BDB0acB33d3fA3351Ee43',
        43114: '0x2779D32d5166BAaa2B2b658333bA7e6Ec0C65743',
        43113: '0x31CF013A08c6Ac228C94551d535d5BAfE19c602a',
        137: '0xc907E116054Ad103354f2D350FD2514433D57F6f',
        42161: '0x6ce185860a4963106506C203335A2910413708e9'
      },
      'AVAX/USD': {
        43114: '0x0A77230d17318075983913bC2145DB16C7366156',
        43113: '0x5498BB86BC934c8D34FDA08E81D444153d0D06aD'
      },
      'LINK/USD': {
        1: '0x2c1d072e956AFFC0D435Cb7AC38EF18d24d9127c',
        11155111: '0xc59E3633BAAC79493d908e63626716e204A45EdF',
        43114: '0x49ccd9ca821EfEab2b98c60dC60F518E765EDe9a',
        43113: '0x79c91fd4F8b3DaBEe17d286EB11cEE4D83521775',
        137: '0xd9FFdb71EbE7496cC440152d43986Aae0AB76665',
        42161: '0x86E53CF1B870786351Da77A57575e79CB55812CB'
      },
      'MATIC/USD': {
        137: '0xAB594600376Ec9fD91F8e885dADF0CE036862dE0'
      }
    };
  }

  /**
   * Validate Chainlink configuration
   */
  private validateConfiguration(): void {
    // Validate environment
    if (!['mainnet', 'testnet', 'local'].includes(this.config.global.environment)) {
      throw new Error('Invalid Chainlink environment. Must be mainnet, testnet, or local');
    }

    // Validate enabled networks
    const enabledNetworks = Object.values(this.config.networks).filter(n => n.enabled);
    if (enabledNetworks.length === 0) {
      throw new Error('At least one Chainlink network must be enabled');
    }

    // Validate required addresses for enabled networks
    for (const [chainId, network] of Object.entries(this.config.networks)) {
      if (network.enabled) {
        if (!network.linkToken) {
          throw new Error(`LINK token address required for chain ${chainId}`);
        }
        if (!network.rpcUrl) {
          throw new Error(`RPC URL required for chain ${chainId}`);
        }
      }
    }

    logger.info('Chainlink configuration validated successfully', {
      environment: this.config.global.environment,
      enabledNetworks: Object.keys(this.config.networks).filter(id => this.config.networks[id].enabled),
      enabledServices: Object.entries(this.config.services)
        .filter(([, service]) => service.enabled)
        .map(([name]) => name)
    });
  }

  /**
   * Get complete Chainlink configuration
   */
  public getChainlinkConfig(): ChainlinkConfiguration {
    return this.config;
  }

  /**
   * Get network configuration by chain ID
   */
  public getNetworkConfig(chainId: number): ChainlinkNetworkConfig | null {
    return this.config.networks[chainId.toString()] || null;
  }

  /**
   * Get all supported chain IDs
   */
  public getAllSupportedChains(): number[] {
    return Object.keys(this.config.networks)
      .filter(chainId => this.config.networks[chainId].enabled)
      .map(chainId => parseInt(chainId));
  }

  /**
   * Get Data Feed address for specific pair and chain
   */
  public getDataFeedAddress(pair: string, chainId: number): string | null {
    return this.config.services.dataFeeds.feeds[pair]?.[chainId] || null;
  }

  /**
   * Check if service is enabled
   */
  public isServiceEnabled(service: keyof ChainlinkServiceConfig): boolean {
    return this.config.services[service].enabled;
  }

  /**
   * Get CCIP router address for chain
   */
  public getCCIPRouter(chainId: number): string | null {
    return this.config.services.ccip.router[chainId] || null;
  }

  /**
   * Get VRF coordinator address for chain
   */
  public getVRFCoordinator(chainId: number): string | null {
    return this.config.services.vrf.coordinatorAddress[chainId] || null;
  }

  /**
   * Get Automation registry address for chain
   */
  public getAutomationRegistry(chainId: number): string | null {
    return this.config.services.automation.registryAddress[chainId] || null;
  }

  /**
   * Get Functions router address for chain
   */
  public getFunctionsRouter(chainId: number): string | null {
    return this.config.services.functions.routerAddress[chainId] || null;
  }

  /**
   * Get configuration for logging (without sensitive data)
   */
  public getConfigForLogging(): any {
    return {
      environment: this.config.global.environment,
      enabledNetworks: Object.keys(this.config.networks).filter(id => this.config.networks[id].enabled),
      services: Object.entries(this.config.services).reduce((acc, [name, service]) => {
        acc[name] = { enabled: service.enabled };
        return acc;
      }, {} as any)
    };
  }
}

// Create singleton instance
export const chainlinkConfig = new ChainlinkConfig();

// Export configuration class
export { ChainlinkConfig };

// Export types
export type { ChainlinkConfiguration, ChainlinkNetworkConfig, ChainlinkServiceConfig };

// Default export
export default chainlinkConfig;
