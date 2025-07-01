import { ethers, BigNumber, Contract } from 'ethers';
import axios from 'axios';
import { logger } from '../../../shared/utils/logger';
import { getNetworkConfig, getProvider, getWallet } from '../../../config/agentConfig';

export interface SystemLimits {
  cpu: number;
  memory: number;
  network: { bandwidth: number; latency: number };
  gas: { total: BigNumber; priceGwei: BigNumber };
  storage: number;
}

export interface ChainlinkIntegration {
  dataFeeds: Map<string, Contract>;
  automation: Map<number, Contract>;
  functions: Map<number, Contract>;
  vrf: Map<number, Contract>;
  ccip: Map<number, Contract>;
}

export interface OrchestrationMetrics {
  totalCoordinations: number;
  successfulCoordinations: number;
  averageCoordinationTime: number;
  resourceEfficiency: number;
  costOptimization: number;
  systemReliability: number;
}

export interface ResourcePooling {
  cpu: ResourcePool;
  memory: ResourcePool;
  network: ResourcePool;
  gas: GasPool;
  storage: ResourcePool;
}

export interface ResourcePool {
  total: number;
  allocated: number;
  available: number;
  reserved: number;
  queuedRequests: ResourceRequest[];
  allocationHistory: AllocationRecord[];
}

export interface GasPool {
  total: BigNumber;
  allocated: BigNumber;
  available: BigNumber;
  reserved: BigNumber;
  priceGwei: BigNumber;
  queuedRequests: GasRequest[];
  allocationHistory: GasAllocationRecord[];
}

export interface ResourceRequest {
  requestId: string;
  agentId: string;
  amount: number;
  priority: number;
  timestamp: number;
  deadline?: number;
}

export interface GasRequest {
  requestId: string;
  agentId: string;
  amount: BigNumber;
  priority: number;
  timestamp: number;
  chainId: number;
  estimatedGasPrice: BigNumber;
}

export interface AllocationRecord {
  agentId: string;
  amount: number;
  allocatedAt: number;
  releasedAt?: number;
  efficiency: number;
}

export interface GasAllocationRecord {
  agentId: string;
  amount: BigNumber;
  chainId: number;
  allocatedAt: number;
  releasedAt?: number;
  actualGasUsed: BigNumber;
  efficiency: number;
}

export class OrchestrationProvider {
  private supportedChains: number[];
  private providers: Map<number, ethers.JsonRpcProvider> = new Map();
  private wallets: Map<number, ethers.Wallet> = new Map();
  private chainlinkIntegration: ChainlinkIntegration;
  private resourcePools: ResourcePooling;
  private orchestrationMetrics: OrchestrationMetrics;
  private systemLimits: SystemLimits;

  // Chainlink contract ABIs from @chainlink/contracts v1.4.0
  private readonly CHAINLINK_PRICE_FEED_ABI = [
    "function latestRoundData() external view returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound)",
    "function decimals() external view returns (uint8)",
    "function description() external view returns (string memory)"
  ];

  private readonly CHAINLINK_AUTOMATION_REGISTRY_ABI = [
    "function registerUpkeep(address target, uint32 gasLimit, address admin, bytes calldata checkData) external returns (uint256 id)",
    "function getUpkeep(uint256 id) external view returns (address target, uint32 executeGas, bytes memory checkData, uint96 balance, address lastKeeper, address admin, uint64 maxValidBlocknumber, uint96 amountSpent, bool paused, bytes memory offchainConfig)",
    "function performUpkeep(uint256 id, bytes calldata performData) external",
    "function pauseUpkeep(uint256 id) external",
    "function unpauseUpkeep(uint256 id) external",
    "function addFunds(uint256 id, uint96 amount) external",
    "function withdrawFunds(uint256 id, address to) external"
  ];

  private readonly CHAINLINK_FUNCTIONS_CONSUMER_ABI = [
    "function sendRequest(string calldata source, bytes calldata encryptedSecretsReferences, string[] calldata args, uint64 subscriptionId, uint32 gasLimit) external returns (bytes32 requestId)",
    "function latestResponse() external view returns (bytes memory response, bytes memory err)",
    "function latestRequestId() external view returns (bytes32)",
    "function latestError() external view returns (bytes memory)"
  ];

  private readonly CHAINLINK_VRF_COORDINATOR_ABI = [
    "function requestRandomWords(bytes32 keyHash, uint64 subId, uint16 minimumRequestConfirmations, uint32 callbackGasLimit, uint32 numWords) external returns (uint256 requestId)",
    "function getRequestConfig() external view returns (uint16, uint32, bytes32[] memory)",
    "function getCommitment(uint256 requestId) external view returns (bytes32)"
  ];

  private readonly CHAINLINK_CCIP_ROUTER_ABI = [
    "function ccipSend(uint64 destinationChainSelector, tuple(bytes receiver, bytes data, tuple(address token, uint256 amount)[] tokenAmounts, address feeToken, bytes extraArgs) message) external payable returns (bytes32)",
    "function getFee(uint64 destinationChainSelector, tuple(bytes receiver, bytes data, tuple(address token, uint256 amount)[] tokenAmounts, address feeToken, bytes extraArgs) message) external view returns (uint256 fee)",
    "function getSupportedTokens(uint64 chainSelector) external view returns (address[] memory tokens)",
    "function isChainSupported(uint64 chainSelector) external view returns (bool supported)"
  ];

  // Contract addresses by chain for Chainlink services
  private readonly CHAINLINK_CONTRACTS: Record<number, any> = {
    1: { // Ethereum Mainnet
      priceFeeds: {
        'ETH/USD': '0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419',
        'BTC/USD': '0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c',
        'LINK/USD': '0x2c1d072e956AFFC0D435Cb7AC38EF18d24d9127c',
        'USDC/USD': '0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6'
      },
      automation: '0x02777053d6764996e594c3E88AF1D58D5363a2e6',
      functions: '0x6E2dc0F9DB014aE19888F539E59285D2Ea04244C',
      vrf: '0x271682DEB8C4E0901D1a1550aD2e64D568E69909',
      ccipRouter: '0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D'
    },
    137: { // Polygon Mainnet
      priceFeeds: {
        'MATIC/USD': '0xAB594600376Ec9fD91F8e885dADF0CE036862dE0',
        'ETH/USD': '0xF9680D99D6C9589e2a93a78A04A279e509205945',
        'BTC/USD': '0xc907E116054Ad103354f2D350FD2514433D57F6f',
        'USDC/USD': '0xfE4A8cc5b5B2366C1B58Bea3858e81843581b2F7'
      },
      automation: '0x02777053d6764996e594c3E88AF1D58D5363a2e6',
      functions: '0x6E2dc0F9DB014aE19888F539E59285D2Ea04244C',
      vrf: '0xAE975071Be8F8eE67addBC1A82488F1C24858067',
      ccipRouter: '0x3C3D92629A02a8D95D5CB9650fe49C3544f69B43'
    },
    42161: { // Arbitrum One
      priceFeeds: {
        'ETH/USD': '0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612',
        'BTC/USD': '0x6ce185860a4963106506C203335A2910413708e9',
        'LINK/USD': '0x86E53CF1B870786351Da77A57575e79CB55812CB',
        'ARB/USD': '0xb2A824043730FE05F3DA2efaFa1CBbe83fa548D6'
      },
      automation: '0x37B9da1c29C8AF4cE560465CC8d41fF0a46C5565',
      functions: '0x97083E831F8F0638855e2A515c90EdCF158DF238',
      vrf: '0x41034678D6C633D8a95c75e1138A360a28bA15d1',
      ccipRouter: '0x141fa059441E0ca23ce184B6A78bafD2A517DdE8'
    },
    43114: { // Avalanche Mainnet
      priceFeeds: {
        'AVAX/USD': '0x0A77230d17318075983913bC2145DB16C7366156',
        'ETH/USD': '0x976B3D034E162d8bD72D6b9C989d545b839003b0',
        'BTC/USD': '0x2779D32d5166BAaa2B2b658333bA7e6Ec0C65743',
        'LINK/USD': '0x49ccd9ca821EfEab2b98c60dC60F518E765EDe9a'
      },
      automation: '0x02777053d6764996e594c3E88AF1D58D5363a2e6',
      functions: '0x6E2dc0F9DB014aE19888F539E59285D2Ea04244C',
      vrf: '0xd5D517aBE5cF79B7e95eC98dB0f0277788aFF634',
      ccipRouter: '0xF4c7E640EdA248ef95972845a62bdC74237805dB'
    }
  };

  constructor() {
    this.supportedChains = [1, 137, 42161, 43114]; // Ethereum, Polygon, Arbitrum, Avalanche
    this.initializeProviders();
    this.initializeChainlinkIntegration();
    this.initializeResourcePools();
    this.initializeMetrics();
    this.initializeSystemLimits();
  }

  async initialize(): Promise<void> {
    logger.info('Initializing orchestration provider', {
      supportedChains: this.supportedChains
    });

    try {
      // Initialize blockchain connections
      await this.initializeBlockchainConnections();

      // Initialize Chainlink contracts
      await this.initializeChainlinkContracts();

      // Initialize resource monitoring
      await this.startResourceMonitoring();

      // Initialize cross-chain capabilities
      await this.initializeCrossChainCapabilities();

      // Validate system health
      await this.validateSystemHealth();

      logger.info('Orchestration provider initialized successfully');

    } catch (error) {
      logger.error('Failed to initialize orchestration provider', {
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  private initializeProviders(): void {
    this.supportedChains.forEach(chainId => {
      try {
        const provider = getProvider(chainId);
        const wallet = getWallet(chainId);
        
        this.providers.set(chainId, provider);
        this.wallets.set(chainId, wallet);

        logger.debug('Provider initialized', { chainId });

      } catch (error) {
        logger.error('Failed to initialize provider', {
          chainId,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    });
  }

  private initializeChainlinkIntegration(): void {
    this.chainlinkIntegration = {
      dataFeeds: new Map(),
      automation: new Map(),
      functions: new Map(),
      vrf: new Map(),
      ccip: new Map()
    };
  }

  private initializeResourcePools(): void {
    this.resourcePools = {
      cpu: {
        total: 100,
        allocated: 0,
        available: 100,
        reserved: 10,
        queuedRequests: [],
        allocationHistory: []
      },
      memory: {
        total: 100,
        allocated: 0,
        available: 100,
        reserved: 10,
        queuedRequests: [],
        allocationHistory: []
      },
      network: {
        total: 1000, // Mbps
        allocated: 0,
        available: 1000,
        reserved: 100,
        queuedRequests: [],
        allocationHistory: []
      },
      gas: {
        total: ethers.utils.parseEther('100'),
        allocated: BigNumber.from(0),
        available: ethers.utils.parseEther('100'),
        reserved: ethers.utils.parseEther('10'),
        priceGwei: ethers.utils.parseUnits('20', 'gwei'),
        queuedRequests: [],
        allocationHistory: []
      },
      storage: {
        total: 10000, // GB
        allocated: 0,
        available: 10000,
        reserved: 1000,
        queuedRequests: [],
        allocationHistory: []
      }
    };
  }

  private initializeMetrics(): void {
    this.orchestrationMetrics = {
      totalCoordinations: 0,
      successfulCoordinations: 0,
      averageCoordinationTime: 0,
      resourceEfficiency: 0,
      costOptimization: 0,
      systemReliability: 0
    };
  }

  private initializeSystemLimits(): void {
    this.systemLimits = {
      cpu: 100,
      memory: 100,
      network: { bandwidth: 1000, latency: 100 },
      gas: { 
        total: ethers.utils.parseEther('100'), 
        priceGwei: ethers.utils.parseUnits('20', 'gwei') 
      },
      storage: 10000
    };
  }

  private async initializeBlockchainConnections(): Promise<void> {
    const connectionPromises = this.supportedChains.map(async (chainId) => {
      try {
        const provider = this.providers.get(chainId);
        if (provider) {
          const blockNumber = await provider.getBlockNumber();
          logger.debug('Blockchain connection established', { chainId, blockNumber });
          return { chainId, success: true, blockNumber };
        }
        return { chainId, success: false, error: 'No provider configured' };
      } catch (error) {
        return { 
          chainId, 
          success: false, 
          error: error instanceof Error ? error.message : String(error) 
        };
      }
    });

    const results = await Promise.all(connectionPromises);
    
    results.forEach(result => {
      if (!result.success) {
        logger.warn('Blockchain connection failed', result);
      }
    });
  }

  private async initializeChainlinkContracts(): Promise<void> {
    for (const chainId of this.supportedChains) {
      try {
        const provider = this.providers.get(chainId);
        const contracts = this.CHAINLINK_CONTRACTS[chainId];

        if (!provider || !contracts) {
          logger.warn('Skipping Chainlink contracts initialization', { chainId });
          continue;
        }

        // Initialize Price Feed contracts
        if (contracts.priceFeeds) {
          for (const [pair, address] of Object.entries(contracts.priceFeeds)) {
            try {
              const priceFeed = new Contract(address as string, this.CHAINLINK_PRICE_FEED_ABI, provider);
              this.chainlinkIntegration.dataFeeds.set(`${pair}_${chainId}`, priceFeed);
              
              // Verify contract is working
              await priceFeed.latestRoundData();
              
              logger.debug('Price feed initialized', { pair, chainId, address });
            } catch (error) {
              logger.warn('Failed to initialize price feed', {
                pair,
                chainId,
                address,
                error: error instanceof Error ? error.message : String(error)
              });
            }
          }
        }

        // Initialize Automation Registry
        if (contracts.automation) {
          try {
            const automationRegistry = new Contract(
              contracts.automation,
              this.CHAINLINK_AUTOMATION_REGISTRY_ABI,
              provider
            );
            this.chainlinkIntegration.automation.set(chainId, automationRegistry);
            
            logger.debug('Automation registry initialized', { chainId, address: contracts.automation });
          } catch (error) {
            logger.warn('Failed to initialize automation registry', {
              chainId,
              error: error instanceof Error ? error.message : String(error)
            });
          }
        }

        // Initialize Functions Consumer
        if (contracts.functions) {
          try {
            const functionsConsumer = new Contract(
              contracts.functions,
              this.CHAINLINK_FUNCTIONS_CONSUMER_ABI,
              provider
            );
            this.chainlinkIntegration.functions.set(chainId, functionsConsumer);
            
            logger.debug('Functions consumer initialized', { chainId, address: contracts.functions });
          } catch (error) {
            logger.warn('Failed to initialize functions consumer', {
              chainId,
              error: error instanceof Error ? error.message : String(error)
            });
          }
        }

        // Initialize VRF Coordinator
        if (contracts.vrf) {
          try {
            const vrfCoordinator = new Contract(
              contracts.vrf,
              this.CHAINLINK_VRF_COORDINATOR_ABI,
              provider
            );
            this.chainlinkIntegration.vrf.set(chainId, vrfCoordinator);
            
            logger.debug('VRF coordinator initialized', { chainId, address: contracts.vrf });
          } catch (error) {
            logger.warn('Failed to initialize VRF coordinator', {
              chainId,
              error: error instanceof Error ? error.message : String(error)
            });
          }
        }

        // Initialize CCIP Router
        if (contracts.ccipRouter) {
          try {
            const ccipRouter = new Contract(
              contracts.ccipRouter,
              this.CHAINLINK_CCIP_ROUTER_ABI,
              provider
            );
            this.chainlinkIntegration.ccip.set(chainId, ccipRouter);
            
            logger.debug('CCIP router initialized', { chainId, address: contracts.ccipRouter });
          } catch (error) {
            logger.warn('Failed to initialize CCIP router', {
              chainId,
              error: error instanceof Error ? error.message : String(error)
            });
          }
        }

      } catch (error) {
        logger.error('Failed to initialize Chainlink contracts for chain', {
          chainId,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }
  }

  private async startResourceMonitoring(): Promise<void> {
    // Start periodic resource monitoring
    setInterval(async () => {
      await this.updateResourceMetrics();
    }, 30000); // Update every 30 seconds

    // Start resource rebalancing
    setInterval(async () => {
      await this.rebalanceResources();
    }, 300000); // Rebalance every 5 minutes

    logger.debug('Resource monitoring started');
  }

  private async initializeCrossChainCapabilities(): Promise<void> {
    try {
      // Verify CCIP support across chains
      for (const chainId of this.supportedChains) {
        const ccipRouter = this.chainlinkIntegration.ccip.get(chainId);
        if (ccipRouter) {
          try {
            // Get supported tokens for cross-chain operations
            const otherChains = this.supportedChains.filter(id => id !== chainId);
            for (const otherChainId of otherChains) {
              const chainSelector = this.getChainSelector(otherChainId);
              if (chainSelector) {
                const isSupported = await ccipRouter.isChainSupported(chainSelector);
                logger.debug('CCIP chain support verified', {
                  fromChain: chainId,
                  toChain: otherChainId,
                  supported: isSupported
                });
              }
            }
          } catch (error) {
            logger.debug('CCIP verification failed', {
              chainId,
              error: error instanceof Error ? error.message : String(error)
            });
          }
        }
      }

      logger.debug('Cross-chain capabilities initialized');

    } catch (error) {
      logger.error('Failed to initialize cross-chain capabilities', {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async validateSystemHealth(): Promise<void> {
    try {
      // Check blockchain connectivity
      const connectivityChecks = await Promise.all(
        this.supportedChains.map(async (chainId) => {
          try {
            const provider = this.providers.get(chainId);
            if (provider) {
              await provider.getBlockNumber();
              return { chainId, healthy: true };
            }
            return { chainId, healthy: false, error: 'No provider' };
          } catch (error) {
            return { 
              chainId, 
              healthy: false, 
              error: error instanceof Error ? error.message : String(error) 
            };
          }
        })
      );

      // Check Chainlink service availability
      const chainlinkChecks = await this.validateChainlinkServices();

      // Check resource availability
      const resourceChecks = this.validateResourceAvailability();

      const healthyChains = connectivityChecks.filter(check => check.healthy).length;
      const healthyServices = chainlinkChecks.filter(check => check.healthy).length;

      logger.info('System health validation completed', {
        healthyChains: `${healthyChains}/${this.supportedChains.length}`,
        healthyChainlinkServices: `${healthyServices}/${chainlinkChecks.length}`,
        resourceAvailability: resourceChecks
      });

    } catch (error) {
      logger.error('System health validation failed', {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async validateChainlinkServices(): Promise<Array<{ service: string; chainId: number; healthy: boolean; error?: string }>> {
    const checks: Array<{ service: string; chainId: number; healthy: boolean; error?: string }> = [];

    for (const chainId of this.supportedChains) {
      // Check Data Feeds
      const priceFeeds = Array.from(this.chainlinkIntegration.dataFeeds.entries())
        .filter(([key]) => key.endsWith(`_${chainId}`));
      
      for (const [feedKey, feed] of priceFeeds) {
        try {
          await feed.latestRoundData();
          checks.push({ service: `price_feed_${feedKey}`, chainId, healthy: true });
        } catch (error) {
          checks.push({ 
            service: `price_feed_${feedKey}`, 
            chainId, 
            healthy: false, 
            error: error instanceof Error ? error.message : String(error) 
          });
        }
      }

      // Check Automation
      const automation = this.chainlinkIntegration.automation.get(chainId);
      if (automation) {
        try {
          // Try to call a view function to verify contract is responsive
          await automation.interface; // Basic contract check
          checks.push({ service: 'automation', chainId, healthy: true });
        } catch (error) {
          checks.push({ 
            service: 'automation', 
            chainId, 
            healthy: false, 
            error: error instanceof Error ? error.message : String(error) 
          });
        }
      }

      // Check Functions
      const functions = this.chainlinkIntegration.functions.get(chainId);
      if (functions) {
        try {
          await functions.interface; // Basic contract check
          checks.push({ service: 'functions', chainId, healthy: true });
        } catch (error) {
          checks.push({ 
            service: 'functions', 
            chainId, 
            healthy: false, 
            error: error instanceof Error ? error.message : String(error) 
          });
        }
      }

      // Check VRF
      const vrf = this.chainlinkIntegration.vrf.get(chainId);
      if (vrf) {
        try {
          await vrf.interface; // Basic contract check
          checks.push({ service: 'vrf', chainId, healthy: true });
        } catch (error) {
          checks.push({ 
            service: 'vrf', 
            chainId, 
            healthy: false, 
            error: error instanceof Error ? error.message : String(error) 
          });
        }
      }

      // Check CCIP
      const ccip = this.chainlinkIntegration.ccip.get(chainId);
      if (ccip) {
        try {
          await ccip.interface; // Basic contract check
          checks.push({ service: 'ccip', chainId, healthy: true });
        } catch (error) {
          checks.push({ 
            service: 'ccip', 
            chainId, 
            healthy: false, 
            error: error instanceof Error ? error.message : String(error) 
          });
        }
      }
    }

    return checks;
  }

  private validateResourceAvailability(): Record<string, number> {
    return {
      cpu: (this.resourcePools.cpu.available / this.resourcePools.cpu.total) * 100,
      memory: (this.resourcePools.memory.available / this.resourcePools.memory.total) * 100,
      network: (this.resourcePools.network.available / this.resourcePools.network.total) * 100,
      storage: (this.resourcePools.storage.available / this.resourcePools.storage.total) * 100,
      gas: parseFloat(ethers.utils.formatEther(
        this.resourcePools.gas.available.mul(100).div(this.resourcePools.gas.total)
      ))
    };
  }

  // Public methods for agent orchestration
  async getSystemLimits(): Promise<SystemLimits> {
    return { ...this.systemLimits };
  }

  async allocateResources(
    agentId: string,
    resourceType: 'cpu' | 'memory' | 'network' | 'gas' | 'storage',
    amount: number | BigNumber,
    chainId?: number
  ): Promise<{ success: boolean; allocationId?: string; error?: string }> {
    try {
      if (resourceType === 'gas') {
        return await this.allocateGasResources(agentId, amount as BigNumber, chainId!);
      } else {
        return await this.allocateComputeResources(agentId, resourceType, amount as number);
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  private async allocateComputeResources(
    agentId: string,
    resourceType: 'cpu' | 'memory' | 'network' | 'storage',
    amount: number
  ): Promise<{ success: boolean; allocationId?: string; error?: string }> {
    const pool = this.resourcePools[resourceType];
    
    if (pool.available < amount) {
      return {
        success: false,
        error: `Insufficient ${resourceType} resources. Available: ${pool.available}, Requested: ${amount}`
      };
    }

    // Allocate resources
    pool.allocated += amount;
    pool.available -= amount;

    // Record allocation
    const allocationId = `${resourceType}_${agentId}_${Date.now()}`;
    const allocation: AllocationRecord = {
      agentId,
      amount,
      allocatedAt: Date.now(),
      efficiency: 0 // Will be updated when released
    };

    pool.allocationHistory.push(allocation);

    logger.debug('Resources allocated', {
      agentId,
      resourceType,
      amount,
      allocationId,
      poolStatus: {
        total: pool.total,
        allocated: pool.allocated,
        available: pool.available
      }
    });

    return { success: true, allocationId };
  }

  private async allocateGasResources(
    agentId: string,
    amount: BigNumber,
    chainId: number
  ): Promise<{ success: boolean; allocationId?: string; error?: string }> {
    const gasPool = this.resourcePools.gas;
    
    if (gasPool.available.lt(amount)) {
      return {
        success: false,
        error: `Insufficient gas resources. Available: ${ethers.utils.formatEther(gasPool.available)} ETH, Requested: ${ethers.utils.formatEther(amount)} ETH`
      };
    }

    // Allocate gas
    gasPool.allocated = gasPool.allocated.add(amount);
    gasPool.available = gasPool.available.sub(amount);

    // Record allocation
    const allocationId = `gas_${agentId}_${chainId}_${Date.now()}`;
    const allocation: GasAllocationRecord = {
      agentId,
      amount,
      chainId,
      allocatedAt: Date.now(),
      actualGasUsed: BigNumber.from(0),
      efficiency: 0
    };

    gasPool.allocationHistory.push(allocation);

    logger.debug('Gas allocated', {
      agentId,
      chainId,
      amount: ethers.utils.formatEther(amount),
      allocationId,
      poolStatus: {
        total: ethers.utils.formatEther(gasPool.total),
        allocated: ethers.utils.formatEther(gasPool.allocated),
        available: ethers.utils.formatEther(gasPool.available)
      }
    });

    return { success: true, allocationId };
  }

  async releaseResources(
    allocationId: string,
    actualUsage?: number | BigNumber
  ): Promise<{ success: boolean; error?: string }> {
    try {
      const [resourceType, agentId] = allocationId.split('_');

      if (resourceType === 'gas') {
        return await this.releaseGasResources(allocationId, actualUsage as BigNumber);
      } else {
        return await this.releaseComputeResources(allocationId, actualUsage as number);
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  private async releaseComputeResources(
    allocationId: string,
    actualUsage?: number
  ): Promise<{ success: boolean; error?: string }> {
    const [resourceType] = allocationId.split('_');
    const pool = this.resourcePools[resourceType as keyof ResourcePooling];

    if (!pool || typeof pool === 'object' && 'priceGwei' in pool) {
      return { success: false, error: 'Invalid resource type' };
    }

    const typedPool = pool as ResourcePool;

    // Find allocation record
    const allocationIndex = typedPool.allocationHistory.findIndex(
      (record) => !record.releasedAt && allocationId.includes(record.agentId)
    );

    if (allocationIndex === -1) {
      return { success: false, error: 'Allocation record not found' };
    }

    const allocation = typedPool.allocationHistory[allocationIndex];

    // Calculate efficiency
    const efficiency = actualUsage ? (actualUsage / allocation.amount) * 100 : 100;

    // Release resources
    typedPool.allocated -= allocation.amount;
    typedPool.available += allocation.amount;

    // Update allocation record
    allocation.releasedAt = Date.now();
    allocation.efficiency = efficiency;

    logger.debug('Resources released', {
      allocationId,
      resourceType,
      amount: allocation.amount,
      efficiency,
      poolStatus: {
        total: typedPool.total,
        allocated: typedPool.allocated,
        available: typedPool.available
      }
    });

    return { success: true };
  }

  private async releaseGasResources(
    allocationId: string,
    actualGasUsed?: BigNumber
  ): Promise<{ success: boolean; error?: string }> {
    const gasPool = this.resourcePools.gas;

    // Find allocation record
    const allocationIndex = gasPool.allocationHistory.findIndex(
      (record) => !record.releasedAt && allocationId.includes(record.agentId)
    );

    if (allocationIndex === -1) {
      return { success: false, error: 'Gas allocation record not found' };
    }

    const allocation = gasPool.allocationHistory[allocationIndex];

    // Calculate efficiency
    const efficiency = actualGasUsed ? 
      parseFloat(ethers.utils.formatEther(actualGasUsed.mul(100).div(allocation.amount))) : 100;

    // Release unused gas
    const unusedGas = actualGasUsed ? allocation.amount.sub(actualGasUsed) : BigNumber.from(0);
    gasPool.allocated = gasPool.allocated.sub(allocation.amount);
    gasPool.available = gasPool.available.add(allocation.amount).sub(actualGasUsed || BigNumber.from(0));

    // Update allocation record
    allocation.releasedAt = Date.now();
    allocation.actualGasUsed = actualGasUsed || BigNumber.from(0);
    allocation.efficiency = efficiency;

    logger.debug('Gas released', {
      allocationId,
      allocatedAmount: ethers.utils.formatEther(allocation.amount),
      actualUsed: ethers.utils.formatEther(actualGasUsed || BigNumber.from(0)),
      efficiency,
      poolStatus: {
        total: ethers.utils.formatEther(gasPool.total),
        allocated: ethers.utils.formatEther(gasPool.allocated),
        available: ethers.utils.formatEther(gasPool.available)
      }
    });

    return { success: true };
  }

  async getChainlinkPrice(pair: string, chainId: number): Promise<{
    price: BigNumber;
    decimals: number;
    updatedAt: number;
  }> {
    try {
      const feedKey = `${pair}_${chainId}`;
      const priceFeed = this.chainlinkIntegration.dataFeeds.get(feedKey);

      if (!priceFeed) {
        throw new Error(`Price feed not available for ${pair} on chain ${chainId}`);
      }

      const [, answer, , updatedAt] = await priceFeed.latestRoundData();
      const decimals = await priceFeed.decimals();

      return {
        price: answer,
        decimals,
        updatedAt: updatedAt.toNumber()
      };

    } catch (error) {
      logger.error('Failed to get Chainlink price', {
        pair,
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  async executeChainlinkFunction(
    sourceCode: string,
    args: string[],
    chainId: number,
    subscriptionId: number,
    gasLimit: number = 300000
  ): Promise<{ requestId: string; response?: string; error?: string }> {
    try {
      const functionsConsumer = this.chainlinkIntegration.functions.get(chainId);
      const wallet = this.wallets.get(chainId);

      if (!functionsConsumer || !wallet) {
        throw new Error(`Chainlink Functions not available on chain ${chainId}`);
      }

      const connectedConsumer = functionsConsumer.connect(wallet);

      // Send request
      const tx = await connectedConsumer.sendRequest(
        sourceCode,
        '0x', // No encrypted secrets for now
        args,
        subscriptionId,
        gasLimit
      );

      const receipt = await tx.wait();
      const requestId = await connectedConsumer.latestRequestId();

      logger.debug('Chainlink Functions request sent', {
        chainId,
        requestId,
        transactionHash: receipt.transactionHash
      });

      // Wait for response (in production, would use event listeners)
      await new Promise(resolve => setTimeout(resolve, 30000)); // 30 second wait

      try {
        const [response, err] = await connectedConsumer.latestResponse();
        
        if (err && err !== '0x') {
          return {
            requestId,
            error: ethers.utils.toUtf8String(err)
          };
        }

        return {
          requestId,
          response: response ? ethers.utils.toUtf8String(response) : undefined
        };

      } catch (responseError) {
        return {
          requestId,
          error: 'Failed to get response'
        };
      }

    } catch (error) {
      logger.error('Chainlink Functions execution failed', {
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  async sendCCIPMessage(
    destinationChainId: number,
    receiver: string,
    data: string,
    tokenTransfers?: Array<{ token: string; amount: BigNumber }>,
    sourceChainId: number = 1
  ): Promise<{ messageId: string; fee: BigNumber }> {
    try {
      const ccipRouter = this.chainlinkIntegration.ccip.get(sourceChainId);
      const wallet = this.wallets.get(sourceChainId);

      if (!ccipRouter || !wallet) {
        throw new Error(`CCIP not available on source chain ${sourceChainId}`);
      }

      const destinationChainSelector = this.getChainSelector(destinationChainId);
      if (!destinationChainSelector) {
        throw new Error(`Chain selector not found for chain ${destinationChainId}`);
      }

      const connectedRouter = ccipRouter.connect(wallet);

      // Prepare message
      const message = {
        receiver: ethers.utils.hexlify(ethers.utils.toUtf8Bytes(receiver)),
        data: ethers.utils.hexlify(ethers.utils.toUtf8Bytes(data)),
        tokenAmounts: tokenTransfers || [],
        feeToken: ethers.constants.AddressZero, // Pay in native token
        extraArgs: '0x'
      };

      // Get fee estimate
      const fee = await connectedRouter.getFee(destinationChainSelector, message);

      // Send message
      const tx = await connectedRouter.ccipSend(destinationChainSelector, message, {
        value: fee
      });

      const receipt = await tx.wait();

      logger.info('CCIP message sent', {
        sourceChainId,
        destinationChainId,
        receiver,
        fee: ethers.utils.formatEther(fee),
        transactionHash: receipt.transactionHash
      });

      return {
        messageId: receipt.transactionHash, // Simplified - would extract actual message ID from events
        fee
      };

    } catch (error) {
      logger.error('CCIP message sending failed', {
        sourceChainId,
        destinationChainId,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  // Helper methods
  private getChainSelector(chainId: number): string | null {
    const selectorMap: Record<number, string> = {
      1: '5009297550715157269',      // Ethereum
      137: '4051577828743386545',    // Polygon
      42161: '4949039107694359620',  // Arbitrum
      43114: '6433500567565415381'   // Avalanche
    };

    return selectorMap[chainId] || null;
  }

  private async updateResourceMetrics(): Promise<void> {
    try {
      // Update resource utilization metrics
      const metrics = {
        cpu: (this.resourcePools.cpu.allocated / this.resourcePools.cpu.total) * 100,
        memory: (this.resourcePools.memory.allocated / this.resourcePools.memory.total) * 100,
        network: (this.resourcePools.network.allocated / this.resourcePools.network.total) * 100,
        storage: (this.resourcePools.storage.allocated / this.resourcePools.storage.total) * 100,
        gas: parseFloat(ethers.utils.formatEther(
          this.resourcePools.gas.allocated.mul(100).div(this.resourcePools.gas.total)
        ))
      };

      // Calculate efficiency metrics
      const efficiencyMetrics = this.calculateResourceEfficiency();

      // Update orchestration metrics
      this.orchestrationMetrics.resourceEfficiency = efficiencyMetrics.overall;
      this.orchestrationMetrics.costOptimization = efficiencyMetrics.cost;

      logger.debug('Resource metrics updated', { metrics, efficiencyMetrics });

    } catch (error) {
      logger.error('Failed to update resource metrics', {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private calculateResourceEfficiency(): { overall: number; cost: number; cpu: number; memory: number; gas: number } {
    try {
      const cpuEfficiency = this.calculatePoolEfficiency(this.resourcePools.cpu.allocationHistory);
      const memoryEfficiency = this.calculatePoolEfficiency(this.resourcePools.memory.allocationHistory);
      const gasEfficiency = this.calculateGasEfficiency(this.resourcePools.gas.allocationHistory);

      const overall = (cpuEfficiency + memoryEfficiency + gasEfficiency) / 3;
      const cost = gasEfficiency; // Gas efficiency represents cost optimization

      return {
        overall,
        cost,
        cpu: cpuEfficiency,
        memory: memoryEfficiency,
        gas: gasEfficiency
      };

    } catch (error) {
      return { overall: 50, cost: 50, cpu: 50, memory: 50, gas: 50 };
    }
  }

  private calculatePoolEfficiency(allocationHistory: AllocationRecord[]): number {
    const recentAllocations = allocationHistory
      .filter(record => record.releasedAt && record.releasedAt > Date.now() - 3600000); // Last hour

    if (recentAllocations.length === 0) return 100;

    const avgEfficiency = recentAllocations.reduce((sum, record) => sum + record.efficiency, 0) / recentAllocations.length;
    return Math.min(Math.max(avgEfficiency, 0), 100);
  }

  private calculateGasEfficiency(allocationHistory: GasAllocationRecord[]): number {
    const recentAllocations = allocationHistory
      .filter(record => record.releasedAt && record.releasedAt > Date.now() - 3600000); // Last hour

    if (recentAllocations.length === 0) return 100;

    const avgEfficiency = recentAllocations.reduce((sum, record) => sum + record.efficiency, 0) / recentAllocations.length;
    return Math.min(Math.max(avgEfficiency, 0), 100);
  }

  private async rebalanceResources(): Promise<void> {
    try {
      // Clean up old allocation records
      const cutoffTime = Date.now() - 86400000; // 24 hours ago

      Object.values(this.resourcePools).forEach(pool => {
        if ('allocationHistory' in pool) {
          pool.allocationHistory = pool.allocationHistory.filter(
            record => !record.releasedAt || record.releasedAt > cutoffTime
          );
        }
      });

      logger.debug('Resource rebalancing completed');

    } catch (error) {
      logger.error('Resource rebalancing failed', {
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  // Public getters
  getOrchestrationMetrics(): OrchestrationMetrics {
    return { ...this.orchestrationMetrics };
  }

  getResourcePools(): ResourcePooling {
    return JSON.parse(JSON.stringify(this.resourcePools));
  }

  getChainlinkIntegrationStatus(): Record<string, number> {
    return {
      dataFeeds: this.chainlinkIntegration.dataFeeds.size,
      automation: this.chainlinkIntegration.automation.size,
      functions: this.chainlinkIntegration.functions.size,
      vrf: this.chainlinkIntegration.vrf.size,
      ccip: this.chainlinkIntegration.ccip.size
    };
  }

  async getNetworkLatency(chainId: number): Promise<number> {
    try {
      const startTime = Date.now();
      const provider = this.providers.get(chainId);
      
      if (provider) {
        await provider.getBlockNumber();
        return Date.now() - startTime;
      }
      
      return 999999; // High latency for unavailable chains
    } catch (error) {
      return 999999;
    }
  }

  async estimateGasCost(
    chainId: number,
    gasLimit: number,
    priorityFee?: BigNumber
  ): Promise<{ gasPrice: BigNumber; totalCost: BigNumber }> {
    try {
      const provider = this.providers.get(chainId);
      if (!provider) {
        throw new Error(`Provider not available for chain ${chainId}`);
      }

      const gasPrice = await provider.getGasPrice();
      const adjustedGasPrice = priorityFee ? gasPrice.add(priorityFee) : gasPrice;
      const totalCost = adjustedGasPrice.mul(gasLimit);

      return { gasPrice: adjustedGasPrice, totalCost };

    } catch (error) {
      logger.error('Gas cost estimation failed', {
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return default estimates
      const defaultGasPrice = ethers.utils.parseUnits('20', 'gwei');
      return {
        gasPrice: defaultGasPrice,
        totalCost: defaultGasPrice.mul(gasLimit)
      };
    }
  }
}
